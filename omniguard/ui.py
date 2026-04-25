from __future__ import annotations

import base64
import json
from pathlib import Path

import gradio as gr

from .benchmark import BenchmarkRunner
from .dataset_generation import SyntheticDatasetBuilder
from .editing import EditingResult
from .image_ops import overlay_mask
from .schemas import AnalysisResult, AttackResult, ProtectionResult
from .service import OmniGuardEngine


BENCHMARK_COLUMNS = (
    ("Атака", "attack_name"),
    ("Payload: подпись OK", "payload_auth_ok"),
    ("Payload: document_id совпал", "payload_document_match"),
    ("Точность бит payload", "payload_bit_accuracy"),
    ("Средний tamper score", "tamper_score_mean"),
    ("Максимальный tamper score", "tamper_score_max"),
    ("Доля подозрительных пикселей", "tamper_ratio"),
    ("IoU маски", "mask_iou"),
    ("Dice маски", "mask_dice"),
)
BENCHMARK_HEADERS = [title for title, _ in BENCHMARK_COLUMNS]


def _json_dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _logo_html(logo_path: Path) -> str:
    if not logo_path.exists():
        return "<h1>OmniGuard 2.0</h1>"
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return (
        "<div style='display:flex;align-items:center;justify-content:center;gap:16px;padding:12px;'>"
        f"<img src='data:image/png;base64,{encoded}' alt='Logo' style='height:56px;'>"
        "<div><div style='font-size:34px;font-weight:700;'>OmniGuard 2.0</div>"
        "<div style='font-size:15px;'>Защита изображения, локализация правок и проверка payload</div></div>"
        "</div>"
    )


def _format_metric(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return round(value, 4)
    return value


def _protect_summary(result: ProtectionResult) -> dict:
    summary = {
        "что_сделано": "В изображение встроены tamper-sensitive watermark и 100-битный payload.",
        "зачем_document_id": "document_id хешируется и встраивается в payload, чтобы потом проверить принадлежность изображения.",
        "что_анализировать_дальше": (
            "После этого можно либо сразу анализировать защищенное изображение, либо отредактировать его "
            "и посмотреть, как изменятся карта подозрительных областей и payload."
        ),
    }
    summary.update(result.to_dict())
    summary["payload"]["record"]["issued_at_utc"] = result.payload.record.issued_at_utc.isoformat()
    return summary


def _edit_summary(result: EditingResult) -> dict:
    backend_note = {
        "opencv-inpaint": (
            "OpenCV inpaint: область заполняется по соседним пикселям. "
            "Это не генеративная модель, поэтому результат обычно выглядит как локальное восстановление текстуры."
        ),
        "diffusers-inpaint": (
            "Diffusers inpaint: генеративная модель пытается перерисовать выделенную область по prompt."
        ),
    }.get(result.backend_name, "Использован backend редактирования без дополнительного описания.")
    return {
        "editor_backend": result.backend_name,
        "что_это_значит": backend_note,
        "prompt": result.prompt,
    }


def _analysis_verdict(result: AnalysisResult, threshold: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not result.payload.auth_ok:
        reasons.append("контрольная подпись payload не сошлась")
    if result.payload.document_match is False:
        reasons.append("встроенный document_id не совпал с ожидаемым")
    if result.tamper_ratio >= 0.02:
        reasons.append("доля подозрительных пикселей заметная")
    if result.tamper_score_max >= threshold * 4.0:
        reasons.append("есть локальная область с высоким tamper score")

    if reasons:
        return "Вероятно, изображение было изменено.", reasons

    weak_reasons: list[str] = []
    if result.tamper_ratio >= 0.005:
        weak_reasons.append("есть небольшая подозрительная область")
    if result.tamper_score_max >= threshold * 2.0:
        weak_reasons.append("локальный tamper score выше базового порога")
    if weak_reasons:
        return "Есть слабые признаки локального изменения, но без уверенного вывода.", weak_reasons

    return "Сильных признаков изменения не обнаружено.", [
        "payload проходит проверку",
        "карта подозрительных областей не содержит выраженных пиков",
    ]


def _analysis_markdown(result: AnalysisResult, threshold: float) -> str:
    verdict, reasons = _analysis_verdict(result, threshold)
    payload = result.payload
    reason_lines = "\n".join(f"- {reason}" for reason in reasons)
    document_match = "не проверялось" if payload.document_match is None else ("да" if payload.document_match else "нет")
    warnings = "\n".join(f"- {warning}" for warning in payload.warnings) if payload.warnings else "- предупреждений нет"
    return (
        f"**Вердикт:** {verdict}\n\n"
        f"**Почему система так считает:**\n{reason_lines}\n\n"
        f"**Ключевые сигналы:**\n"
        f"- `payload.auth_ok`: {'да' if payload.auth_ok else 'нет'}\n"
        f"- `payload.document_match`: {document_match}\n"
        f"- `payload.corrected_errors`: {payload.corrected_errors}\n"
        f"- `tamper_score_mean`: {result.tamper_score_mean:.4f}\n"
        f"- `tamper_score_max`: {result.tamper_score_max:.4f}\n"
        f"- `tamper_ratio`: {result.tamper_ratio:.4f}\n\n"
        f"**Как читать визуализацию:**\n"
        f"- `Heatmap / overlay` показывает, где модель видит аномалию. Чем ярче область, тем выше локальный score.\n"
        f"- `Бинарная маска` получается порогованием heatmap по `threshold={threshold:.2f}`.\n"
        f"- `tamper_ratio` это доля пикселей, попавших в бинарную маску.\n\n"
        f"**Предупреждения декодера:**\n{warnings}"
    )


def _analysis_report(result: AnalysisResult) -> dict:
    summary = result.to_dict()
    if result.payload.record is not None:
        summary["payload"]["record"]["issued_at_utc"] = result.payload.record.issued_at_utc.isoformat()
    summary["как_читать"] = {
        "tamper_score_mean": "Среднее значение карты аномалий по всему изображению.",
        "tamper_score_max": "Максимальная локальная аномалия. Полезно для поиска небольшой правки.",
        "tamper_ratio": "Доля пикселей, у которых score выше порога и которые попали в бинарную маску.",
        "payload.auth_ok": "Проверка HMAC-тега встроенного payload.",
        "payload.document_match": "Совпадение decoded document_id hash с ожидаемым document_id.",
    }
    return summary


def _benchmark_summary(results: list[AttackResult], report_path: Path) -> str:
    payload_fail = [
        result.attack_name for result in results if result.metrics.get("payload_auth_ok") is False
    ]
    document_fail = [
        result.attack_name for result in results if result.metrics.get("payload_document_match") is False
    ]
    strong_localization = [
        result.attack_name for result in results if (result.metrics.get("tamper_ratio") or 0.0) >= 0.02
    ]
    return (
        "**Как читать benchmark:**\n"
        "- `Payload: подпись OK` показывает, пережил ли payload атаку без потери целостности.\n"
        "- `Payload: document_id совпал` показывает, совпал ли извлеченный идентификатор с ожидаемым.\n"
        "- `Точность бит payload` это доля правильно восстановленных битов из 100.\n"
        "- `Средний/максимальный tamper score` показывают, насколько сильной была реакция локализатора.\n"
        "- `Доля подозрительных пикселей` показывает размер итоговой бинарной маски.\n"
        "- `IoU` и `Dice` считаются только там, где у атаки есть известная эталонная маска правки.\n\n"
        f"**Краткий итог текущего прогона:**\n"
        f"- атаки, где сломалась подпись payload: {', '.join(payload_fail) if payload_fail else 'нет'}\n"
        f"- атаки, где не совпал document_id: {', '.join(document_fail) if document_fail else 'нет'}\n"
        f"- атаки с заметной локализацией правок: {', '.join(strong_localization) if strong_localization else 'нет'}\n"
        f"- JSON-отчет сохранен в: `{report_path}`"
    )


def _dataset_summary(records_count: int, manifest_path: Path) -> str:
    return _json_dump(
        {
            "что_создано": (
                "Синтетический датасет для экспериментов. Для каждого исходного изображения создаются "
                "защищенная версия, одна или несколько искусственных правок и маска измененной области."
            ),
            "records_created": records_count,
            "manifest_path": str(manifest_path),
            "что_лежит_в_manifest": [
                "document_id",
                "attack_name",
                "original_path",
                "protected_path",
                "edited_path",
                "mask_path",
            ],
        }
    )


def create_app(engine: OmniGuardEngine | None = None) -> gr.Blocks:
    engine = engine or OmniGuardEngine()
    benchmark_runner = BenchmarkRunner(engine)
    dataset_builder = SyntheticDatasetBuilder(engine)

    def protect_ui(image, document_id):
        if image is None:
            raise gr.Error("Загрузите изображение для защиты.")
        if not document_id.strip():
            raise gr.Error("Укажите document_id для встраивания payload.")
        clean_document_id = document_id.strip()
        result = engine.protect_image(image, clean_document_id)
        return (
            result.protected_image,
            result.protected_image,
            result.protected_image,
            clean_document_id,
            _json_dump(_protect_summary(result)),
            result.payload.encoded_bits,
        )

    def edit_ui(sketch_data, prompt):
        if sketch_data is None or sketch_data.get("image") is None:
            raise gr.Error("Подайте защищенное изображение в редактор.")
        if sketch_data.get("mask") is None:
            raise gr.Error("Нарисуйте маску области редактирования.")
        result = engine.edit_image(sketch_data["image"], sketch_data["mask"], prompt)
        return result.image, _json_dump(_edit_summary(result))

    def analyze_ui(image, expected_document_id, reference_bits):
        if image is None:
            raise gr.Error("Нет изображения для анализа.")
        result = engine.analyze_image(
            image,
            expected_document_id=expected_document_id.strip() or None,
            reference_bits=reference_bits or None,
        )
        overlay = overlay_mask(image, result.tamper_heatmap)
        report = _analysis_report(result)
        explanation = _analysis_markdown(result, engine.settings.tamper_mask_threshold)
        return overlay, result.binary_mask, _json_dump(report), explanation

    def benchmark_ui(image, document_id):
        if image is None:
            raise gr.Error("Загрузите изображение для benchmark.")
        if not document_id.strip():
            raise gr.Error("Укажите document_id.")
        results, report_path = benchmark_runner.run(image, document_id.strip())
        rows = []
        for result in results:
            row = []
            for _, key in BENCHMARK_COLUMNS:
                if key == "attack_name":
                    row.append(result.attack_name)
                else:
                    row.append(_format_metric(result.metrics.get(key)))
            rows.append(row)
        return rows, str(report_path), _benchmark_summary(results, report_path)

    def dataset_ui(input_dir, output_dir, limit):
        if not input_dir.strip():
            raise gr.Error("Укажите директорию с исходными изображениями.")
        records, manifest_path = dataset_builder.build(
            input_dir=input_dir.strip(),
            output_dir=output_dir.strip() or None,
            limit=int(limit) if limit else None,
        )
        return _dataset_summary(len(records), manifest_path)

    with gr.Blocks(title="OmniGuard 2.0") as demo:
        payload_bits_state = gr.State([])
        gr.HTML(_logo_html(engine.settings.logo_path))
        gr.Markdown(
            "OmniGuard 2.0 объединяет три сценария: защиту изображения, анализ возможных правок "
            "и экспериментальную оценку устойчивости модели."
        )
        with gr.Tabs():
            with gr.TabItem("1. Защита, редактирование и анализ"):
                gr.Markdown(
                    "**Что делает вкладка:**\n"
                    "1. `Защитить изображение` встраивает два сигнала: tamper-sensitive watermark и payload с `document_id`.\n"
                    "2. `Редактирование` меняет выделенную область. Если доступен только OpenCV backend, это не генерация по смыслу, а локальное восстановление текстуры.\n"
                    "3. `Анализ` декодирует payload и строит карту подозрительных областей.\n\n"
                    "**Как пользоваться:** загрузите исходное изображение, задайте `document_id`, нажмите защиту, затем либо сразу анализируйте, либо сначала отредактируйте область."
                )
                with gr.Row():
                    with gr.Column():
                        source_image = gr.Image(label="Исходное изображение", type="numpy")
                        document_id = gr.Textbox(
                            label="Document ID",
                            placeholder="diploma-demo-001",
                        )
                        protect_button = gr.Button("1. Защитить изображение")
                    with gr.Column():
                        protected_image = gr.Image(label="Защищенное изображение", type="numpy")
                        protection_json = gr.Textbox(label="Что произошло на шаге защиты", lines=18)
                with gr.Row():
                    with gr.Column():
                        editor_canvas = gr.Image(
                            label="2. Выделите область, которую хотите изменить",
                            type="numpy",
                            tool="sketch",
                        )
                        prompt = gr.Textbox(
                            label="Prompt для генеративного редактирования",
                            placeholder="remove the selected object and fill the area naturally",
                        )
                        edit_button = gr.Button("2. Выполнить редактирование")
                    with gr.Column():
                        analysis_input_image = gr.Image(
                            label="Изображение, которое пойдет в анализ",
                            type="numpy",
                        )
                        edit_json = gr.Textbox(label="Что произошло на шаге редактирования", lines=9)
                with gr.Row():
                    with gr.Column():
                        expected_document_id = gr.Textbox(
                            label="Ожидаемый document_id для проверки",
                            placeholder="diploma-demo-001",
                        )
                        analyze_button = gr.Button("3. Проанализировать изображение")
                    with gr.Column():
                        analysis_overlay = gr.Image(
                            label="Heatmap, наложенная на изображение",
                            type="numpy",
                        )
                        analysis_mask = gr.Image(
                            label="Бинарная маска подозрительных областей",
                            type="numpy",
                        )
                        analysis_json = gr.Textbox(label="Технический отчет анализа", lines=18)
                analysis_explanation = gr.Markdown()
                protect_button.click(
                    protect_ui,
                    inputs=[source_image, document_id],
                    outputs=[
                        protected_image,
                        editor_canvas,
                        analysis_input_image,
                        expected_document_id,
                        protection_json,
                        payload_bits_state,
                    ],
                )
                edit_button.click(
                    edit_ui,
                    inputs=[editor_canvas, prompt],
                    outputs=[analysis_input_image, edit_json],
                )
                analyze_button.click(
                    analyze_ui,
                    inputs=[analysis_input_image, expected_document_id, payload_bits_state],
                    outputs=[analysis_overlay, analysis_mask, analysis_json, analysis_explanation],
                )
            with gr.TabItem("2. Benchmark и метрики"):
                gr.Markdown(
                    "**Что делает вкладка:** система сначала защищает изображение, затем прогоняет набор атак и после каждой атаки снова делает анализ.\n\n"
                    "**Зачем это нужно:** так можно проверить, переживает ли watermark JPEG-сжатие, blur, resize и более явные правки вроде copy-move и inpainting."
                )
                with gr.Row():
                    with gr.Column():
                        benchmark_image = gr.Image(label="Изображение для benchmark", type="numpy")
                        benchmark_document_id = gr.Textbox(label="Document ID", placeholder="benchmark-001")
                        benchmark_button = gr.Button("Запустить benchmark")
                    with gr.Column():
                        benchmark_table = gr.Dataframe(
                            headers=BENCHMARK_HEADERS,
                            value=[[""] * len(BENCHMARK_HEADERS)],
                        )
                        benchmark_report_path = gr.Textbox(label="Путь к JSON-отчету")
                benchmark_explanation = gr.Markdown()
                benchmark_button.click(
                    benchmark_ui,
                    inputs=[benchmark_image, benchmark_document_id],
                    outputs=[benchmark_table, benchmark_report_path, benchmark_explanation],
                )
            with gr.TabItem("3. Синтетический датасет"):
                gr.Markdown(
                    "**Что делает вкладка:** берет папку с изображениями и автоматически создает экспериментальный датасет.\n\n"
                    "Для каждого исходного изображения система создает защищенную версию, искусственно вносит правку, сохраняет маску измененной области и пишет `manifest.jsonl` для обучения или оценки."
                )
                dataset_input_dir = gr.Textbox(
                    label="Папка с исходными изображениями",
                    placeholder=r"C:\images",
                )
                dataset_output_dir = gr.Textbox(
                    label="Папка, куда сохранить датасет",
                    placeholder=r"C:\datasets\omniguard",
                )
                dataset_limit = gr.Number(label="Сколько изображений обработать", value=10, precision=0)
                dataset_button = gr.Button("Сгенерировать синтетический датасет")
                dataset_summary = gr.Textbox(label="Результат генерации", lines=14)
                dataset_button.click(
                    dataset_ui,
                    inputs=[dataset_input_dir, dataset_output_dir, dataset_limit],
                    outputs=[dataset_summary],
                )
    return demo


def launch_ui(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    app = create_app()
    app.launch(server_name=host, server_port=port, share=share)
