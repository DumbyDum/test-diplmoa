from __future__ import annotations

import base64
import json
from pathlib import Path

import gradio as gr

from .benchmark import BenchmarkRunner
from .editing import EDITOR_DESCRIPTIONS, EditingResult, editor_choices, editor_help_markdown
from .image_ops import overlay_mask
from .schemas import AnalysisResult, AttackResult, ProtectionResult
from .service import OmniGuardEngine


CUSTOM_CSS = """
.og-title {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.og-section {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 4px 0 8px 0;
  font-size: 20px;
  font-weight: 700;
}

.og-section-subtitle {
  margin: 0 0 12px 0;
  color: #556170;
  font-size: 14px;
}

.og-tooltip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 999px;
  background: #e5eefc;
  color: #1f5fbf;
  font-size: 12px;
  font-weight: 700;
  cursor: help;
  user-select: none;
}

.og-note {
  padding: 10px 12px;
  border-radius: 8px;
  background: #f5f8fc;
  border: 1px solid #d9e3f0;
  margin-bottom: 12px;
  font-size: 14px;
}
"""

BENCHMARK_COLUMNS = (
    ("Атака", "attack_name"),
    ("Payload: подпись OK", "payload_auth_ok"),
    ("Payload: document_id совпал", "payload_document_match"),
    ("Точность бит payload", "payload_bit_accuracy"),
    ("PSNR", "psnr_protected_vs_attacked"),
    ("SSIM", "ssim_protected_vs_attacked"),
    ("Макс. tamper score", "tamper_score_max"),
    ("Доля маски", "tamper_ratio"),
    ("IoU", "mask_iou"),
    ("Dice", "mask_dice"),
)
BENCHMARK_HEADERS = [title for title, _ in BENCHMARK_COLUMNS]
ANALYSIS_METRIC_HEADERS = ["Метрика", "Значение", "Что это значит"]


def _json_dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _logo_html(logo_path: Path) -> str:
    if not logo_path.exists():
        return "<h1>OmniGuard 2.0</h1>"
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return (
        "<div class='og-title'>"
        f"<img src='data:image/png;base64,{encoded}' alt='Logo' style='height:56px;'>"
        "<div><div style='font-size:34px;font-weight:700;'>OmniGuard 2.0</div>"
        "<div style='font-size:15px;'>Защита изображения, локализация изменений и проверка payload</div></div>"
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


def _doc_markdown(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


def _tooltip_icon(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return f"<span class='og-tooltip' title=\"{escaped}\">?</span>"


def _section_block(title: str, tooltip: str, subtitle: str | None = None) -> str:
    subtitle_html = f"<div class='og-section-subtitle'>{subtitle}</div>" if subtitle else ""
    return f"<div class='og-section'>{title}{_tooltip_icon(tooltip)}</div>{subtitle_html}"


def _note_block(text: str) -> str:
    return f"<div class='og-note'>{text}</div>"


def _protect_summary(result: ProtectionResult) -> dict:
    summary = {
        "что_сделано": "В изображение встроены tamper-sensitive watermark и 100-битный payload.",
        "зачем_document_id": "document_id хешируется и встраивается в payload, чтобы потом проверить принадлежность изображения.",
        "что_делать_дальше": "После защиты можно либо сразу анализировать изображение, либо изменить выделенную область и проверить реакцию системы.",
    }
    summary.update(result.to_dict())
    summary["payload"]["record"]["issued_at_utc"] = result.payload.record.issued_at_utc.isoformat()
    return summary


def _edit_summary(result: EditingResult) -> dict:
    backend_note = EDITOR_DESCRIPTIONS.get(
        result.backend_name,
        "Использован backend редактирования без дополнительного описания.",
    )
    return {
        "editor_backend": result.backend_name,
        "editor_backend_label": result.backend_label,
        "что_это_значит": backend_note,
        "prompt": result.prompt,
        "model_id": result.model_id,
        "num_inference_steps": result.num_inference_steps,
        "guidance_scale": result.guidance_scale,
        "allow_download": result.allow_download,
    }


def _analysis_verdict(result: AnalysisResult) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not result.payload.auth_ok:
        reasons.append("контрольная подпись payload не сошлась")
    if result.payload.document_match is False:
        reasons.append("встроенный document_id не совпал с ожидаемым")
    if result.tamper_ratio >= 0.02:
        reasons.append("маска покрывает заметную область изображения")
    if result.tamper_score_max >= 0.55 or (
        result.tamper_score_max >= 0.35
        and (
            result.tamper_ratio >= 0.005
            or result.comparison_metrics.get("changed_pixel_ratio_vs_reference", 0.0) >= 0.005
        )
    ):
        reasons.append("локальный tamper score высокий")
    if result.comparison_metrics.get("changed_pixel_ratio_vs_reference", 0.0) >= 0.01:
        reasons.append("по сравнению с опорной версией изменилось заметное число пикселей")

    if reasons:
        return "Вероятно, изображение было изменено.", reasons

    weak_reasons: list[str] = []
    if result.tamper_ratio >= 0.005:
        weak_reasons.append("есть небольшая подозрительная область")
    if result.tamper_score_max >= 0.2:
        weak_reasons.append("локальный tamper score выше фонового уровня")
    if weak_reasons:
        return "Есть слабые признаки локального изменения, но без уверенного вывода.", weak_reasons

    return "Сильных признаков изменения не обнаружено.", [
        "payload проходит проверку",
        "выраженных аномальных областей не найдено",
    ]


def _analysis_markdown(result: AnalysisResult) -> str:
    verdict, reasons = _analysis_verdict(result)
    payload = result.payload
    reason_lines = "\n".join(f"- {reason}" for reason in reasons)
    document_match = "не проверялось" if payload.document_match is None else ("да" if payload.document_match else "нет")
    warnings = "\n".join(f"- {warning}" for warning in payload.warnings) if payload.warnings else "- предупреждений нет"
    mode = result.metadata.get("analysis_mode", "hybrid")
    reference_used = "да" if result.metadata.get("reference_image_used") else "нет"
    return (
        f"**Вердикт:** {verdict}\n\n"
        f"**Почему система так считает:**\n{reason_lines}\n\n"
        f"**Режим анализа:** `{mode}`\n"
        f"**Использована защищенная опорная версия:** {reference_used}\n\n"
        f"**Ключевые сигналы:**\n"
        f"- `payload.auth_ok`: {'да' if payload.auth_ok else 'нет'}\n"
        f"- `payload.document_match`: {document_match}\n"
        f"- `payload.corrected_errors`: {payload.corrected_errors}\n"
        f"- `tamper_score_mean`: {result.tamper_score_mean:.4f}\n"
        f"- `tamper_score_max`: {result.tamper_score_max:.4f}\n"
        f"- `tamper_ratio`: {result.tamper_ratio:.4f}\n\n"
        f"**Как читать визуализацию:**\n"
        f"- `Heatmap` показывает силу аномалии по пикселям.\n"
        f"- `Бинарная маска` содержит только пиксели выше порога.\n"
        f"- Если доступна защищенная опорная версия, локализация становится заметно точнее.\n\n"
        f"**Предупреждения декодера:**\n{warnings}"
    )


def _analysis_report(result: AnalysisResult) -> dict:
    summary = result.to_dict()
    if result.payload.record is not None:
        summary["payload"]["record"]["issued_at_utc"] = result.payload.record.issued_at_utc.isoformat()
    summary["как_читать"] = {
        "tamper_score_mean": "Среднее значение карты аномалий по всему изображению.",
        "tamper_score_max": "Максимальная локальная аномалия. Особенно полезна для небольшой правки.",
        "tamper_ratio": "Доля пикселей, попавших в бинарную маску.",
        "payload.auth_ok": "Проверка HMAC-тега встроенного payload.",
        "payload.document_match": "Совпадение decoded document_id hash с ожидаемым document_id.",
        "comparison_metrics": "Дополнительные метрики сравнения с защищенной опорной версией.",
    }
    return summary


def _analysis_metrics_rows(result: AnalysisResult) -> list[list[object]]:
    rows = [
        ["payload_auth_ok", "да" if result.payload.auth_ok else "нет", "Прошла ли встроенная подпись payload проверку"],
        [
            "payload_document_match",
            "не проверялось" if result.payload.document_match is None else ("да" if result.payload.document_match else "нет"),
            "Совпал ли восстановленный document_id с ожидаемым",
        ],
        ["payload_corrected_errors", result.payload.corrected_errors, "Сколько ошибок исправил Hamming-декодер"],
        ["tamper_score_mean", round(result.tamper_score_mean, 4), "Средняя аномалия по всему изображению"],
        ["tamper_score_max", round(result.tamper_score_max, 4), "Самая сильная локальная аномалия"],
        ["tamper_ratio", round(result.tamper_ratio, 4), "Доля пикселей в итоговой подозрительной маске"],
    ]
    comparison_descriptions = {
        "mse_vs_reference": "Средняя квадратичная ошибка между текущим и защищенным изображением",
        "mae_vs_reference": "Среднее абсолютное отличие пикселей",
        "rmse_vs_reference": "Корень из средней квадратичной ошибки",
        "psnr_vs_reference": "Похожесть изображений по отношению сигнал/шум",
        "ssim_vs_reference": "Структурное сходство изображений",
        "changed_pixel_ratio_vs_reference": "Какая доля пикселей заметно изменилась относительно защищенной версии",
    }
    for key, value in result.comparison_metrics.items():
        rows.append([key, _format_metric(value), comparison_descriptions.get(key, "Дополнительная метрика сравнения")])
    return rows


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
        "- `PSNR` и `SSIM` описывают визуальную близость атакованного изображения к защищенному.\n"
        "- `Макс. tamper score` показывает силу самой заметной локальной аномалии.\n"
        "- `IoU` и `Dice` считаются только там, где у атаки есть известная эталонная маска правки.\n\n"
        f"**Краткий итог текущего прогона:**\n"
        f"- атаки, где сломалась подпись payload: {', '.join(payload_fail) if payload_fail else 'нет'}\n"
        f"- атаки, где не совпал document_id: {', '.join(document_fail) if document_fail else 'нет'}\n"
        f"- атаки с заметной локализацией правок: {', '.join(strong_localization) if strong_localization else 'нет'}\n"
        f"- JSON-отчет сохранен в: `{report_path}`"
    )


def create_app(engine: OmniGuardEngine | None = None) -> gr.Blocks:
    engine = engine or OmniGuardEngine()
    benchmark_runner = BenchmarkRunner(engine)

    interface_guide_md = _doc_markdown(
        engine.settings.docs_dir / "INTERFACE_GUIDE.md",
        "# Справка по интерфейсу\n\nФайл `docs/INTERFACE_GUIDE.md` не найден.",
    )
    metrics_reference_md = _doc_markdown(
        engine.settings.docs_dir / "METRICS_REFERENCE.md",
        "# Метрики и формулы\n\nФайл `docs/METRICS_REFERENCE.md` не найден.",
    )
    analysis_interpretation_md = _doc_markdown(
        engine.settings.docs_dir / "ANALYSIS_INTERPRETATION.md",
        "# Как интерпретировать результат анализа\n\nФайл `docs/ANALYSIS_INTERPRETATION.md` не найден.",
    )
    user_guide_md = _doc_markdown(
        engine.settings.docs_dir / "USER_GUIDE.md",
        "# Подробный гайд\n\nФайл `docs/USER_GUIDE.md` не найден.",
    )

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

    def edit_ui(sketch_data, prompt, editor_name, editor_model_id, allow_download):
        if sketch_data is None or sketch_data.get("image") is None:
            raise gr.Error("Подайте защищенное изображение в редактор.")
        if sketch_data.get("mask") is None:
            raise gr.Error("Нарисуйте маску области редактирования.")
        try:
            result = engine.edit_image(
                sketch_data["image"],
                sketch_data["mask"],
                prompt,
                editor_name=editor_name,
                editor_model_id=editor_model_id.strip() or None,
                allow_download=allow_download,
            )
        except Exception as exc:
            raise gr.Error(
                "Не удалось выполнить редактирование выбранным backend. "
                "Проверьте доступность модели или переключитесь на OpenCV."
            ) from exc
        return result.image, _json_dump(_edit_summary(result))

    def analyze_ui(image, expected_document_id, reference_bits, protected_reference, analysis_mode, threshold):
        if image is None:
            raise gr.Error("Нет изображения для анализа.")
        result = engine.analyze_image(
            image,
            expected_document_id=expected_document_id.strip() or None,
            reference_bits=reference_bits or None,
            reference_image=protected_reference,
            analysis_mode=analysis_mode,
            threshold_override=float(threshold),
        )
        overlay = overlay_mask(image, result.tamper_heatmap)
        report = _analysis_report(result)
        explanation = _analysis_markdown(result)
        rows = _analysis_metrics_rows(result)
        return overlay, result.binary_mask, _json_dump(report), explanation, rows

    def analyze_external_ui(external_image, protected_reference, expected_document_id, analysis_mode, threshold):
        if external_image is None:
            raise gr.Error("Загрузите внешне измененное изображение для анализа.")
        if analysis_mode == "reference" and protected_reference is None:
            raise gr.Error("Для режима 'Только сравнение с опорной версией' нужно загрузить защищенную опорную версию.")
        result = engine.analyze_image(
            external_image,
            expected_document_id=expected_document_id.strip() or None,
            reference_image=protected_reference,
            analysis_mode=analysis_mode,
            threshold_override=float(threshold),
        )
        overlay = overlay_mask(external_image, result.tamper_heatmap)
        report = _analysis_report(result)
        explanation = _analysis_markdown(result)
        rows = _analysis_metrics_rows(result)
        return overlay, result.binary_mask, _json_dump(report), explanation, rows

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

    with gr.Blocks(title="OmniGuard 2.0", css=CUSTOM_CSS) as demo:
        payload_bits_state = gr.State([])
        gr.HTML(_logo_html(engine.settings.logo_path))
        gr.HTML(
            _note_block(
                "Наводи курсор на значок <b>?</b> рядом с заголовками разделов — там быстрые подсказки. "
                "Для самой точной локализации изменений используй гибридный режим: он сравнивает текущую версию с защищенной опорной."
            )
        )
        with gr.Tabs():
            with gr.TabItem("1. Основной сценарий"):
                gr.HTML(
                    _section_block(
                        "Шаг 1. Защита изображения",
                        "На этом шаге изображение получает два скрытых слоя: tamper-sensitive watermark и payload с document_id.",
                        "Сначала защити изображение. После этого оно автоматически подставится и в редактор, и в анализ.",
                    )
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

                gr.HTML(
                    _section_block(
                        "Шаг 2. Локальное редактирование",
                        "Выдели область кистью. Теперь можно выбрать конкретный редактор: быстрые локальные OpenCV-варианты или более качественные diffusers-режимы.",
                        "Этот шаг нужен, чтобы смоделировать реальное вмешательство и потом проверить, увидит ли его анализатор.",
                    )
                )
                gr.Markdown(editor_help_markdown())
                with gr.Row():
                    with gr.Column():
                        editor_canvas = gr.Image(
                            label="Выделите область, которую хотите изменить",
                            type="numpy",
                            tool="sketch",
                        )
                        editor_name = gr.Dropdown(
                            choices=editor_choices(),
                            value="auto",
                            label="Редактор изображения",
                        )
                        editor_model_id = gr.Textbox(
                            label="Локальный путь / model id для diffusers",
                            placeholder=engine.settings.inpaint_model_id,
                        )
                        allow_model_download = gr.Checkbox(
                            label="Разрешить загрузку модели из интернета, если ее нет локально",
                            value=engine.settings.allow_inpaint_model_download,
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

                gr.HTML(
                    _section_block(
                        "Шаг 3. Анализ и локализация изменений",
                        "Гибридный режим использует и watermark, и сравнение с защищенной опорной версией. Он обычно выделяет локальные изменения заметно точнее.",
                        "Здесь система извлекает payload, оценивает целостность изображения и строит карту подозрительных областей.",
                    )
                )
                with gr.Row():
                    with gr.Column():
                        expected_document_id = gr.Textbox(
                            label="Ожидаемый document_id для проверки",
                            placeholder="diploma-demo-001",
                        )
                        analysis_mode = gr.Radio(
                            choices=[
                                ("Гибридный: watermark + опорная версия", "hybrid"),
                                ("Только watermark", "watermark"),
                                ("Только сравнение с опорной версией", "reference"),
                            ],
                            value="hybrid",
                            label="Режим локализации",
                        )
                        threshold_slider = gr.Slider(
                            minimum=0.01,
                            maximum=0.50,
                            step=0.01,
                            value=engine.settings.tamper_mask_threshold,
                            label="Порог бинарной маски",
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
                analysis_metrics_table = gr.Dataframe(
                    headers=ANALYSIS_METRIC_HEADERS,
                    value=[["", "", ""]],
                    label="Ключевые метрики текущего анализа",
                )
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
                    inputs=[editor_canvas, prompt, editor_name, editor_model_id, allow_model_download],
                    outputs=[analysis_input_image, edit_json],
                )
                analyze_button.click(
                    analyze_ui,
                    inputs=[
                        analysis_input_image,
                        expected_document_id,
                        payload_bits_state,
                        protected_image,
                        analysis_mode,
                        threshold_slider,
                    ],
                    outputs=[
                        analysis_overlay,
                        analysis_mask,
                        analysis_json,
                        analysis_explanation,
                        analysis_metrics_table,
                    ],
                )
            with gr.TabItem("2. Benchmark"):
                gr.HTML(
                    _section_block(
                        "Benchmark устойчивости",
                        "Здесь система сама защищает изображение, применяет набор атак и после каждой атаки считает метрики.",
                        "Эта вкладка нужна для оценки устойчивости, а не для единичного примера.",
                    )
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
            with gr.TabItem("3. Анализ внешнего файла"):
                gr.HTML(
                    _section_block(
                        "Анализ изображения, которое вы изменили вручную вне приложения",
                        "Используй эту вкладку, если ты сохранил защищенное изображение, отредактировал его в Photoshop, Paint, GIMP или другом редакторе и теперь хочешь проверить результат нашим кодом.",
                        "Лучший сценарий: загрузить измененный файл и отдельно загрузить защищенную опорную версию. Тогда гибридный режим даст самую точную локализацию.",
                    )
                )
                gr.HTML(
                    _note_block(
                        "Как это работает: сначала загружаешь измененный файл, потом при желании добавляешь защищенную опорную версию, "
                        "задаешь ожидаемый document_id и запускаешь анализ. В этом сценарии система проверяет payload и ищет области изменений "
                        "без использования встроенных редакторов."
                    )
                )
                with gr.Row():
                    with gr.Column():
                        external_analysis_image = gr.Image(
                            label="Внешне измененное изображение",
                            type="numpy",
                        )
                        external_reference_image = gr.Image(
                            label="Защищенная опорная версия (рекомендуется)",
                            type="numpy",
                        )
                        external_expected_document_id = gr.Textbox(
                            label="Ожидаемый document_id",
                            placeholder="diploma-demo-001",
                        )
                        external_analysis_mode = gr.Radio(
                            choices=[
                                ("Гибридный: watermark + опорная версия", "hybrid"),
                                ("Только watermark", "watermark"),
                                ("Только сравнение с опорной версией", "reference"),
                            ],
                            value="hybrid",
                            label="Режим локализации",
                        )
                        external_threshold_slider = gr.Slider(
                            minimum=0.01,
                            maximum=0.50,
                            step=0.01,
                            value=engine.settings.tamper_mask_threshold,
                            label="Порог бинарной маски",
                        )
                        external_analyze_button = gr.Button("Запустить анализ внешнего файла")
                    with gr.Column():
                        external_analysis_overlay = gr.Image(
                            label="Heatmap для внешнего файла",
                            type="numpy",
                        )
                        external_analysis_mask = gr.Image(
                            label="Бинарная маска подозрительных областей",
                            type="numpy",
                        )
                        external_analysis_json = gr.Textbox(
                            label="Технический отчет анализа",
                            lines=18,
                        )
                external_analysis_explanation = gr.Markdown()
                external_analysis_metrics_table = gr.Dataframe(
                    headers=ANALYSIS_METRIC_HEADERS,
                    value=[["", "", ""]],
                    label="Ключевые метрики анализа внешнего файла",
                )
                external_analyze_button.click(
                    analyze_external_ui,
                    inputs=[
                        external_analysis_image,
                        external_reference_image,
                        external_expected_document_id,
                        external_analysis_mode,
                        external_threshold_slider,
                    ],
                    outputs=[
                        external_analysis_overlay,
                        external_analysis_mask,
                        external_analysis_json,
                        external_analysis_explanation,
                        external_analysis_metrics_table,
                    ],
                )
            with gr.TabItem("4. Справка по разделам"):
                gr.Markdown(interface_guide_md)
            with gr.TabItem("5. Как интерпретировать анализ"):
                gr.Markdown(analysis_interpretation_md)
            with gr.TabItem("6. Метрики и формулы"):
                gr.Markdown(metrics_reference_md)
            with gr.TabItem("7. Подробный гайд"):
                gr.Markdown(user_guide_md)
    return demo


def launch_ui(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    app = create_app()
    app.launch(server_name=host, server_port=port, share=share)
