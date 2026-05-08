from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import gradio as gr

from .attacks import ATTACK_REFERENCE_MARKDOWN
from .basic_watermarking import METHOD_REFERENCE_MARKDOWN, method_choices
from .editing import EDITOR_DESCRIPTIONS, EditingResult, editor_choices, editor_help_markdown
from .image_ops import overlay_mask
from .paper_comparison import (
    PAPER_COMPARISON_HEADERS,
    PaperComparisonRunner,
    paper_degradation_choices,
    paper_local_edit_choices,
    rows_for_table as paper_rows_for_table,
)
from .requirement_experiments import RequirementExperimentRunner
from .schemas import AnalysisResult, ProtectionResult
from .service import OmniGuardEngine


CUSTOM_CSS = """
.og-title {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.og-note {
  padding: 10px 12px;
  border-radius: 8px;
  background: #f5f8fc;
  border: 1px solid #d9e3f0;
  margin: 8px 0 12px 0;
  font-size: 14px;
}

.og-section {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 8px 0 6px 0;
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

.og-theme-links {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 0 2px 0;
  color: #4b5563;
  font-size: 14px;
}

.og-theme-links a {
  color: #1f5fbf;
  text-decoration: none;
  border: 1px solid #c9d7ef;
  border-radius: 6px;
  padding: 4px 8px;
  background: #ffffff;
}

.og-theme-links a:hover {
  background: #eef5ff;
}
"""


ANALYSIS_METRIC_HEADERS = ["Метрика", "Значение", "Что показывает"]
METHOD_METRIC_HEADERS = ["Метрика", "Значение", "Как читать"]
REQUIREMENT_BENCHMARK_HEADERS = ["Изображение", "Метод", "Атака", "PSNR", "SSIM", "bpp", "BER", "Статус"]


METRIC_REFERENCE_MARKDOWN = """
# Метрики из требований

В требованиях используются четыре основные метрики: `PSNR`, `SSIM`, `bpp` и `BER`.

## PSNR

`PSNR` показывает, насколько изображение после встраивания похоже на исходное. Чем выше значение, тем менее заметен водяной знак.

Формулы:

`MSE = (1 / (H * W * C)) * sum((I(i,j,c) - Iw(i,j,c))^2)`

`PSNR = 10 * log10(MAX_I^2 / MSE)`

где `I` — исходное изображение, `Iw` — изображение с ЦВЗ, `H` и `W` — высота и ширина, `C` — число каналов, `MAX_I = 255`.

## SSIM

`SSIM` оценивает структурное сходство изображений. Значение ближе к `1` означает, что структура изображения почти не изменилась.

Формула:

`SSIM(x, y) = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))`

где `mu` — средняя яркость, `sigma` — дисперсия, `sigma_xy` — ковариация.

## bpp

`bpp` — bits per pixel, то есть сколько бит ЦВЗ приходится на один пиксель изображения.

Формула:

`bpp = N_bits / (H * W)`

где `N_bits` — число встроенных бит, `H * W` — количество пикселей.

## BER

`BER` — bit error rate. Метрика показывает, какая доля бит ЦВЗ была извлечена неправильно после атаки.

Формула:

`BER = N_error / N_bits`

где `N_error` — количество несовпавших бит между исходным и извлеченным ЦВЗ.

Интерпретация:

- `BER = 0` — все биты восстановлены правильно.
- `BER = 0.1` — 10% бит повреждены.
- `BER` около `0.5` — извлечение почти случайное.
"""


INTERFACE_REFERENCE_MARKDOWN = """
# Разделы интерфейса

## Основной сценарий

Используется для полного рабочего цикла OmniGuard: загрузка изображения, встраивание payload, локальное редактирование и анализ изменений.

## Методы встраивания ЦВЗ

Раздел показывает базовые алгоритмы из требований: OmniGuard, LSB и DCT. Здесь можно встроить текстовый payload в одно изображение и сразу увидеть `PSNR`, `SSIM`, `bpp` и `BER` без атаки.

## Пакетный benchmark

Раздел принимает много изображений, встраивает ЦВЗ выбранными методами, применяет атаки из требований и считает метрики.

## Сравнение со статьей

Раздел воспроизводит набор метрик из статьи OmniGuard: `Capacity`, `PSNR`, `SSIM`, `Bit Accuracy`, `F1` и `AUC`. Базовая и улучшенная ветки проверяются в одинаковых условиях: одно protected image, одна атака, одна деградация и одна ground-truth маска.

## Анализ внешнего файла

Нужен, если защищенное изображение было изменено вручную во внешнем редакторе. Сюда загружается измененная версия и, желательно, защищенная опорная версия.

## Справочник

Внутри собраны объяснения методов, атак, метрик и интерпретации анализа.
"""


ANALYSIS_REFERENCE_MARKDOWN = """
# Как читать результат анализа

Раздел анализа отвечает на два вопроса:

1. удалось ли корректно извлечь payload;
2. есть ли локальные области, похожие на изменение изображения.

## Payload

`payload_auth_ok` показывает, прошла ли HMAC-проверка встроенного payload. Если значение `нет`, значит извлеченные биты не согласуются с контрольной подписью.

`payload_document_match` показывает, совпал ли hash извлеченного `document_id` с ожидаемым `document_id`, который ввел пользователь.

`payload_corrected_errors` показывает, сколько одиночных битовых ошибок исправил Hamming-декодер.

## Heatmap

`Heatmap` — это карта подозрительности. Чем ярче пиксель, тем сильнее локальное отклонение.

В режиме `watermark` карта строится по нарушению скрытого watermark. В режиме `reference` карта строится по различию между текущим изображением и защищенной опорной версией. В режиме `hybrid` используются оба источника.

## Бинарная маска

Бинарная маска получается из heatmap по порогу. Все пиксели выше порога считаются подозрительными, остальные отбрасываются.

## Практическая интерпретация

- если `payload_auth_ok = нет`, встроенные данные повреждены или извлечены некорректно;
- если `payload_document_match = нет`, изображение не соответствует ожидаемому `document_id`;
- если `tamper_ratio` заметно больше нуля, система нашла область подозрительных пикселей;
- если `changed_pixel_ratio_vs_reference` заметно больше нуля, изображение отличается от опорной защищенной версии.
"""


def _json_dump(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _logo_html(logo_path: Path) -> str:
    if not logo_path.exists():
        return "<h1>OmniGuard 2.0</h1>"
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return (
        "<div class='og-title'>"
        f"<img src='data:image/png;base64,{encoded}' alt='Logo' style='height:56px;'>"
        "<div><div style='font-size:34px;font-weight:700;'>OmniGuard 2.0</div>"
        "<div style='font-size:15px;'>ЦВЗ, payload, атаки, метрики и пакетные эксперименты</div></div>"
        "</div>"
    )


def _theme_links_html() -> str:
    return (
        "<div class='og-theme-links'>"
        "<span>Тема интерфейса:</span>"
        "<a href='?__theme=light'>Светлая</a>"
        "<a href='?__theme=system'>Авто</a>"
        "<a href='?__theme=dark'>Темная</a>"
        "</div>"
    )


def _tooltip_icon(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return f"<span class='og-tooltip' title=\"{escaped}\">?</span>"


def _section_block(title: str, tooltip: str, subtitle: str | None = None) -> str:
    subtitle_html = f"<div class='og-section-subtitle'>{subtitle}</div>" if subtitle else ""
    return f"<div class='og-section'>{title}{_tooltip_icon(tooltip)}</div>{subtitle_html}"


def _note_block(text: str) -> str:
    return f"<div class='og-note'>{text}</div>"


def _format_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return round(value, 6)
    return value


def _doc_markdown(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


def _protect_summary(result: ProtectionResult) -> dict[str, Any]:
    summary = result.to_dict()
    summary["что_сделано"] = "В изображение встроены tamper-sensitive watermark и 100-битный payload."
    summary["payload"]["record"]["issued_at_utc"] = result.payload.record.issued_at_utc.isoformat()
    return summary


def _edit_summary(result: EditingResult) -> dict[str, Any]:
    return {
        "editor_backend": result.backend_name,
        "editor_backend_label": result.backend_label,
        "описание": EDITOR_DESCRIPTIONS.get(result.backend_name, ""),
        "prompt": result.prompt,
        "model_id": result.model_id,
        "num_inference_steps": result.num_inference_steps,
        "guidance_scale": result.guidance_scale,
        "allow_download": result.allow_download,
    }


def _analysis_verdict(result: AnalysisResult) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not result.payload.auth_ok:
        reasons.append("контрольная подпись payload не прошла проверку")
    if result.payload.document_match is False:
        reasons.append("извлеченный document_id не совпал с ожидаемым")
    if result.tamper_ratio >= 0.02:
        reasons.append("подозрительная маска занимает заметную часть изображения")
    if result.comparison_metrics.get("changed_pixel_ratio_vs_reference", 0.0) >= 0.01:
        reasons.append("изображение заметно отличается от защищенной опорной версии")
    if result.tamper_score_max >= 0.55:
        reasons.append("найдена сильная локальная аномалия watermark")

    if reasons:
        return "Вероятно, изображение было изменено.", reasons
    if result.tamper_ratio >= 0.005 or result.tamper_score_max >= 0.2:
        return "Есть слабые признаки локального изменения.", [
            "аномалии есть, но они недостаточно сильные для уверенного вывода"
        ]
    return "Сильных признаков изменения не обнаружено.", [
        "payload и карта аномалий не показывают выраженного вмешательства"
    ]


def _analysis_markdown(result: AnalysisResult) -> str:
    verdict, reasons = _analysis_verdict(result)
    reason_lines = "\n".join(f"- {reason}" for reason in reasons)
    document_match = (
        "не проверялось"
        if result.payload.document_match is None
        else ("да" if result.payload.document_match else "нет")
    )
    return (
        f"**Вердикт:** {verdict}\n\n"
        f"**Почему:**\n{reason_lines}\n\n"
        f"**Payload:**\n"
        f"- подпись payload OK: {'да' if result.payload.auth_ok else 'нет'}\n"
        f"- document_id совпал: {document_match}\n"
        f"- исправлено ошибок Hamming: {result.payload.corrected_errors}\n\n"
        f"**Локализация:**\n"
        f"- средний tamper score: `{result.tamper_score_mean:.4f}`\n"
        f"- максимальный tamper score: `{result.tamper_score_max:.4f}`\n"
        f"- доля бинарной маски: `{result.tamper_ratio:.4f}`"
    )


def _analysis_report(result: AnalysisResult) -> dict[str, Any]:
    report = result.to_dict()
    if result.payload.record is not None:
        report["payload"]["record"]["issued_at_utc"] = result.payload.record.issued_at_utc.isoformat()
    report["как_читать"] = {
        "payload.auth_ok": "показывает, прошла ли HMAC-проверка payload",
        "payload.document_match": "показывает, совпал ли document_id с ожидаемым",
        "tamper_score_max": "самая сильная локальная аномалия",
        "tamper_ratio": "доля пикселей, попавших в бинарную маску",
    }
    return report


def _analysis_metrics_rows(result: AnalysisResult) -> list[list[Any]]:
    rows = [
        ["payload_auth_ok", "да" if result.payload.auth_ok else "нет", "Целостность извлеченного payload"],
        [
            "payload_document_match",
            "не проверялось" if result.payload.document_match is None else ("да" if result.payload.document_match else "нет"),
            "Совпадение ожидаемого document_id",
        ],
        ["tamper_score_mean", round(result.tamper_score_mean, 6), "Средняя сила аномалии по изображению"],
        ["tamper_score_max", round(result.tamper_score_max, 6), "Самая сильная локальная аномалия"],
        ["tamper_ratio", round(result.tamper_ratio, 6), "Доля пикселей в бинарной маске"],
    ]
    for key, value in result.comparison_metrics.items():
        rows.append([key, _format_value(value), "Сравнение с защищенной опорной версией"])
    return rows


def _method_metric_rows(metrics: dict[str, Any], metadata: dict[str, Any]) -> list[list[Any]]:
    return [
        ["PSNR", _format_value(metrics.get("psnr")), "Чем выше, тем менее заметен ЦВЗ"],
        ["SSIM", _format_value(metrics.get("ssim")), "Чем ближе к 1, тем больше структурное сходство"],
        ["bpp", _format_value(metrics.get("bpp")), "Сколько бит ЦВЗ приходится на один пиксель"],
        ["BER без атаки", _format_value(metadata.get("clean_ber")), "Доля ошибочных бит сразу после извлечения"],
        ["Встроено бит", metadata.get("embedded_bits"), "Размер встроенного payload"],
    ]


def _benchmark_rows_for_table(rows: list[dict[str, Any]], limit: int = 500) -> list[list[Any]]:
    table: list[list[Any]] = []
    for row in rows[:limit]:
        table.append(
            [
                row.get("image", ""),
                row.get("method", ""),
                row.get("attack", ""),
                _format_value(row.get("psnr")),
                _format_value(row.get("ssim")),
                _format_value(row.get("bpp")),
                _format_value(row.get("ber")),
                row.get("status", ""),
            ]
        )
    return table


def _benchmark_summary(rows: list[dict[str, Any]], report_path: Path) -> str:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") != "ok"]
    avg_ber_by_method: dict[str, list[float]] = {}
    for row in ok_rows:
        value = row.get("ber")
        if isinstance(value, (int, float)):
            avg_ber_by_method.setdefault(str(row.get("method")), []).append(float(value))
    method_lines = []
    for method, values in avg_ber_by_method.items():
        avg = sum(values) / len(values) if values else 0.0
        method_lines.append(f"- `{method}`: средний BER = `{avg:.4f}`")
    method_block = "\n".join(method_lines) if method_lines else "- нет успешных строк"
    return (
        f"**Готово.** Обработано строк эксперимента: `{len(rows)}`.\n\n"
        f"**Ошибок:** `{len(error_rows)}`.\n\n"
        f"**Средний BER по методам:**\n{method_block}\n\n"
        f"**CSV-отчет:** `{report_path}`\n\n"
        "В таблице показаны только первые 500 строк, полный результат сохранен в CSV."
    )


def _paper_comparison_summary(rows: list[dict[str, Any]], report_path: Path) -> str:
    baseline = rows[0] if rows else {}
    enhanced = rows[1] if len(rows) > 1 else {}

    def delta(metric: str) -> str:
        left = baseline.get(metric)
        right = enhanced.get(metric)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return "н/д"
        return f"{right - left:+.6f}"

    return (
        "**Сравнение выполнено в одинаковых условиях.** Обе ветки получили одно и то же защищенное "
        "изображение, одну и ту же локальную атаку, одну глобальную деградацию, один `document_id`, "
        "одинаковые payload-биты и одну эталонную маску.\n\n"
        f"- изменение `F1`: `{delta('F1')}`\n"
        f"- изменение `AUC`: `{delta('AUC')}`\n"
        f"- изменение `Bit Accuracy, %`: `{delta('Bit Accuracy, %')}`\n\n"
        f"**JSON-отчет:** `{report_path}`"
    )


def create_app(engine: OmniGuardEngine | None = None) -> gr.Blocks:
    engine = engine or OmniGuardEngine()
    requirement_runner = RequirementExperimentRunner(engine)
    paper_runner = PaperComparisonRunner(engine)

    requirement_guide_md = _doc_markdown(
        engine.settings.docs_dir / "REQUIREMENTS_ATTACKS_METRICS_METHODS.md",
        "# Методология по требованиям\n\nФайл справки не найден.",
    )
    technical_pipeline_md = _doc_markdown(
        engine.settings.docs_dir / "PIPELINE_METRICS_HEATMAP_DEEP_DIVE.md",
        "# Пайплайн, метрики и heatmap\n\nФайл справки не найден.",
    )
    paper_comparison_md = _doc_markdown(
        engine.settings.docs_dir / "PAPER_METRICS_AND_COMPARISON.md",
        "# Метрики статьи и сравнение\n\nФайл справки не найден.",
    )

    def protect_ui(image, document_id):
        if image is None:
            raise gr.Error("Загрузите изображение для защиты.")
        if not document_id.strip():
            raise gr.Error("Укажите document_id.")
        result = engine.protect_image(image, document_id.strip())
        return (
            result.protected_image,
            result.protected_image,
            result.protected_image,
            document_id.strip(),
            _json_dump(_protect_summary(result)),
            result.payload.encoded_bits,
        )

    def edit_ui(sketch_data, prompt, editor_name, editor_model_id, allow_download):
        if sketch_data is None or sketch_data.get("image") is None:
            raise gr.Error("Сначала защитите изображение или передайте его в редактор.")
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
            raise gr.Error("Не удалось выполнить редактирование. Попробуйте OpenCV-редактор.") from exc
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
        return (
            overlay_mask(image, result.tamper_heatmap),
            result.binary_mask,
            _json_dump(_analysis_report(result)),
            _analysis_markdown(result),
            _analysis_metrics_rows(result),
        )

    def analyze_external_ui(external_image, protected_reference, expected_document_id, analysis_mode, threshold):
        if external_image is None:
            raise gr.Error("Загрузите внешне измененное изображение.")
        if analysis_mode == "reference" and protected_reference is None:
            raise gr.Error("Для режима сравнения нужна защищенная опорная версия.")
        result = engine.analyze_image(
            external_image,
            expected_document_id=expected_document_id.strip() or None,
            reference_image=protected_reference,
            analysis_mode=analysis_mode,
            threshold_override=float(threshold),
        )
        return (
            overlay_mask(external_image, result.tamper_heatmap),
            result.binary_mask,
            _json_dump(_analysis_report(result)),
            _analysis_markdown(result),
            _analysis_metrics_rows(result),
        )

    def method_demo_ui(image, payload_text, method_id):
        if image is None:
            raise gr.Error("Загрузите изображение.")
        if not payload_text.strip():
            raise gr.Error("Введите текст payload / document_id для встраивания.")
        bundle, explanation = requirement_runner.run_single(image, payload_text.strip(), method_id)
        return (
            bundle.watermarked_image,
            _method_metric_rows(bundle.metrics, bundle.metadata),
            _json_dump(
                {
                    "method": bundle.method_name,
                    "metadata": bundle.metadata,
                    "payload_bits_preview": bundle.payload_bits[:64],
                    "payload_bits_total": len(bundle.payload_bits),
                }
            ),
            explanation,
        )

    def batch_benchmark_ui(uploaded_files, folder_path, payload_text, method_ids):
        selected_methods = list(method_ids or [])
        if not selected_methods:
            raise gr.Error("Выберите хотя бы один метод встраивания.")
        image_paths = requirement_runner.collect_images(uploaded_files, folder_path)
        if not image_paths:
            raise gr.Error("Загрузите изображения или укажите папку с изображениями.")
        rows, report_path = requirement_runner.run_batch(
            image_paths=image_paths,
            payload_text=payload_text.strip() or "omniguard-demo",
            method_ids=selected_methods,
        )
        return _benchmark_rows_for_table(rows), str(report_path), _benchmark_summary(rows, report_path)

    def paper_comparison_ui(image, document_id, local_edit_id, degradation_id, threshold):
        if image is None:
            raise gr.Error("Загрузите изображение для сравнения.")
        if not document_id.strip():
            raise gr.Error("Укажите document_id.")
        result = paper_runner.run_generated(
            image,
            document_id.strip(),
            local_edit_id=local_edit_id,
            degradation_id=degradation_id,
            threshold=float(threshold),
        )
        return (
            result.protected_image,
            result.attacked_image,
            result.ground_truth_mask,
            result.baseline_overlay,
            result.baseline_mask,
            result.enhanced_overlay,
            result.enhanced_mask,
            paper_rows_for_table(result.rows),
            str(result.report_path),
            _paper_comparison_summary(result.rows, result.report_path),
            _json_dump(result.report),
        )

    with gr.Blocks(title="OmniGuard 2.0", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
        payload_bits_state = gr.State([])
        gr.HTML(_logo_html(engine.settings.logo_path))
        gr.HTML(_theme_links_html())
        gr.HTML(
            _note_block(
                "Интерфейс теперь разделен на практические сценарии и один общий справочник. "
                "Наводи курсор на значок <b>?</b>, чтобы увидеть быстрые подсказки."
            )
        )

        with gr.Tabs():
            with gr.TabItem("1. Основной сценарий"):
                gr.HTML(
                    _section_block(
                        "Шаг 1. Защита изображения",
                        "Встраивает в изображение tamper-sensitive watermark и payload с document_id.",
                        "Используй этот раздел для полного цикла: защита, редактирование, анализ.",
                    )
                )
                with gr.Row():
                    with gr.Column():
                        source_image = gr.Image(label="Исходное изображение", type="numpy")
                        document_id = gr.Textbox(label="Document ID", placeholder="diploma-demo-001")
                        protect_button = gr.Button("1. Защитить изображение")
                    with gr.Column():
                        protected_image = gr.Image(label="Защищенное изображение", type="numpy")
                        with gr.Accordion("Технический отчет о защите", open=False):
                            protection_json = gr.Textbox(label="JSON", lines=14)

                gr.HTML(
                    _section_block(
                        "Шаг 2. Локальное редактирование",
                        "Позволяет искусственно изменить выделенную область и проверить, найдет ли ее анализатор.",
                    )
                )
                gr.Markdown(editor_help_markdown())
                with gr.Row():
                    with gr.Column():
                        editor_canvas = gr.Image(label="Выделите область для редактирования", type="numpy", tool="sketch")
                        editor_name = gr.Dropdown(choices=editor_choices(), value="auto", label="Редактор")
                        editor_model_id = gr.Textbox(
                            label="Локальный путь / model id для diffusers",
                            placeholder=engine.settings.inpaint_model_id,
                        )
                        allow_model_download = gr.Checkbox(
                            label="Разрешить загрузку diffusers-модели из интернета",
                            value=engine.settings.allow_inpaint_model_download,
                        )
                        prompt = gr.Textbox(
                            label="Prompt для генеративного редактора",
                            placeholder="remove the selected object and fill the area naturally",
                        )
                        edit_button = gr.Button("2. Выполнить редактирование")
                    with gr.Column():
                        analysis_input_image = gr.Image(label="Изображение для анализа", type="numpy")
                        with gr.Accordion("Технический отчет о редактировании", open=False):
                            edit_json = gr.Textbox(label="JSON", lines=9)

                gr.HTML(
                    _section_block(
                        "Шаг 3. Анализ изменений",
                        "Извлекает payload, строит heatmap и бинарную маску подозрительных областей.",
                    )
                )
                with gr.Row():
                    with gr.Column():
                        expected_document_id = gr.Textbox(label="Ожидаемый document_id", placeholder="diploma-demo-001")
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
                        analyze_button = gr.Button("3. Проанализировать")
                    with gr.Column():
                        analysis_overlay = gr.Image(label="Heatmap поверх изображения", type="numpy")
                        analysis_mask = gr.Image(label="Бинарная маска изменений", type="numpy")
                        with gr.Accordion("Технический отчет анализа", open=False):
                            analysis_json = gr.Textbox(label="JSON", lines=16)
                analysis_explanation = gr.Markdown()
                analysis_metrics_table = gr.Dataframe(
                    headers=ANALYSIS_METRIC_HEADERS,
                    value=[["", "", ""]],
                    label="Ключевые показатели анализа",
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

            with gr.TabItem("2. Методы встраивания ЦВЗ"):
                gr.HTML(
                    _section_block(
                        "Сравнение базовых методов",
                        "Показывает, как работают OmniGuard, LSB и DCT на одном изображении.",
                        "Раздел показывает результат встраивания, базовые метрики и технические параметры выбранного метода.",
                    )
                )
                with gr.Row():
                    with gr.Column():
                        method_image = gr.Image(label="Изображение", type="numpy")
                        method_payload = gr.Textbox(label="Payload / document_id", value="diploma-demo-001")
                        method_id = gr.Radio(choices=method_choices(), value="lsb", label="Метод встраивания")
                        method_button = gr.Button("Встроить ЦВЗ")
                    with gr.Column():
                        method_output = gr.Image(label="Изображение с ЦВЗ", type="numpy")
                        method_metrics = gr.Dataframe(
                            headers=METHOD_METRIC_HEADERS,
                            value=[["", "", ""]],
                            label="Метрики метода",
                        )
                        with gr.Accordion("Технические детали метода", open=False):
                            method_json = gr.Textbox(label="JSON", lines=14)
                method_explanation = gr.Markdown()
                method_button.click(
                    method_demo_ui,
                    inputs=[method_image, method_payload, method_id],
                    outputs=[method_output, method_metrics, method_json, method_explanation],
                )

            with gr.TabItem("3. Пакетный benchmark"):
                gr.HTML(
                    _section_block(
                        "Эксперимент на датасете",
                        "Принимает много изображений, встраивает ЦВЗ выбранными методами, применяет атаки и считает PSNR, SSIM, bpp, BER.",
                        "Подходит для 50-100 изображений. OmniGuard может работать дольше LSB/DCT; для быстрого чернового прогона его можно снять в списке методов.",
                    )
                )
                with gr.Row():
                    with gr.Column():
                        benchmark_files = gr.File(
                            label="Пул изображений",
                            file_count="multiple",
                            type="file",
                            file_types=[".png", ".jpg", ".jpeg", ".bmp", ".webp"],
                        )
                        benchmark_folder = gr.Textbox(
                            label="Или путь к папке с изображениями",
                            placeholder=r"C:\Users\Mi\Downloads\dataset",
                        )
                        benchmark_payload = gr.Textbox(label="Payload / document_id", value="diploma-demo-001")
                        benchmark_methods = gr.CheckboxGroup(
                            choices=method_choices(),
                            value=["omniguard", "lsb", "dct"],
                            label="Методы для сравнения",
                        )
                        benchmark_button = gr.Button("Запустить пакетный benchmark")
                    with gr.Column():
                        benchmark_table = gr.Dataframe(
                            headers=REQUIREMENT_BENCHMARK_HEADERS,
                            value=[[""] * len(REQUIREMENT_BENCHMARK_HEADERS)],
                            label="Результаты benchmark",
                        )
                        benchmark_report_path = gr.Textbox(label="Путь к CSV-отчету")
                benchmark_explanation = gr.Markdown()
                benchmark_button.click(
                    batch_benchmark_ui,
                    inputs=[benchmark_files, benchmark_folder, benchmark_payload, benchmark_methods],
                    outputs=[benchmark_table, benchmark_report_path, benchmark_explanation],
                )

            with gr.TabItem("4. Сравнение со статьей"):
                gr.HTML(
                    _section_block(
                        "Paper-compatible сравнение",
                        "Считает метрики из статьи OmniGuard для базовой watermark-only локализации и улучшенной hybrid-локализации.",
                        "Обе ветки получают один и тот же protected image, одну и ту же атаку, один document_id и одну ground-truth маску.",
                    )
                )
                gr.Markdown(
                    "Здесь базовая версия означает режим `watermark-only residual`: анализ строит карту только по нарушению "
                    "встроенного локального watermark. Улучшенная версия использует гибридную карту: watermark + сравнение "
                    "с защищенной опорной версией + адаптивная бинаризация."
                )
                with gr.Row():
                    with gr.Column():
                        paper_image = gr.Image(label="Исходное изображение", type="numpy")
                        paper_document_id = gr.Textbox(label="Document ID", value="paper-demo-001")
                        paper_local_edit = gr.Dropdown(
                            choices=paper_local_edit_choices(),
                            value="opencv_inpaint_proxy",
                            label="Локальная атака",
                        )
                        paper_degradation = gr.Dropdown(
                            choices=paper_degradation_choices(),
                            value="clean",
                            label="Глобальная деградация после локальной атаки",
                        )
                        paper_threshold = gr.Slider(
                            minimum=0.01,
                            maximum=0.50,
                            step=0.01,
                            value=engine.settings.tamper_mask_threshold,
                            label="Порог бинарной маски для F1",
                        )
                        paper_button = gr.Button("Запустить сравнение")
                    with gr.Column():
                        paper_protected_image = gr.Image(label="Protected image", type="numpy")
                        paper_attacked_image = gr.Image(label="Attacked / received image", type="numpy")
                        paper_gt_mask = gr.Image(label="Ground-truth маска атаки", type="numpy")
                with gr.Row():
                    with gr.Column():
                        paper_baseline_overlay = gr.Image(label="Базовая версия: heatmap", type="numpy")
                        paper_baseline_mask = gr.Image(label="Базовая версия: binary mask", type="numpy")
                    with gr.Column():
                        paper_enhanced_overlay = gr.Image(label="Улучшенная версия: heatmap", type="numpy")
                        paper_enhanced_mask = gr.Image(label="Улучшенная версия: binary mask", type="numpy")
                paper_table = gr.Dataframe(
                    headers=PAPER_COMPARISON_HEADERS,
                    value=[[""] * len(PAPER_COMPARISON_HEADERS)],
                    label="Метрики из статьи OmniGuard",
                )
                paper_report_path = gr.Textbox(label="Путь к JSON-отчету")
                paper_summary = gr.Markdown()
                with gr.Accordion("Технический отчет сравнения", open=False):
                    paper_json = gr.Textbox(label="JSON", lines=18)
                paper_button.click(
                    paper_comparison_ui,
                    inputs=[paper_image, paper_document_id, paper_local_edit, paper_degradation, paper_threshold],
                    outputs=[
                        paper_protected_image,
                        paper_attacked_image,
                        paper_gt_mask,
                        paper_baseline_overlay,
                        paper_baseline_mask,
                        paper_enhanced_overlay,
                        paper_enhanced_mask,
                        paper_table,
                        paper_report_path,
                        paper_summary,
                        paper_json,
                    ],
                )

            with gr.TabItem("5. Анализ внешнего файла"):
                gr.HTML(
                    _section_block(
                        "Проверка изображения после ручного редактирования",
                        "Используй этот раздел, если изменил защищенное изображение в Photoshop, Paint, GIMP или другом редакторе.",
                    )
                )
                with gr.Row():
                    with gr.Column():
                        external_analysis_image = gr.Image(label="Внешне измененное изображение", type="numpy")
                        external_reference_image = gr.Image(label="Защищенная опорная версия", type="numpy")
                        external_expected_document_id = gr.Textbox(label="Ожидаемый document_id", placeholder="diploma-demo-001")
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
                        external_analyze_button = gr.Button("Запустить анализ")
                    with gr.Column():
                        external_analysis_overlay = gr.Image(label="Heatmap", type="numpy")
                        external_analysis_mask = gr.Image(label="Бинарная маска", type="numpy")
                        with gr.Accordion("Технический отчет анализа", open=False):
                            external_analysis_json = gr.Textbox(label="JSON", lines=16)
                external_analysis_explanation = gr.Markdown()
                external_analysis_metrics_table = gr.Dataframe(
                    headers=ANALYSIS_METRIC_HEADERS,
                    value=[["", "", ""]],
                    label="Ключевые показатели анализа",
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

            with gr.TabItem("6. Справочник"):
                gr.HTML(
                    _section_block(
                        "Все объяснения в одном месте",
                        "Внутри можно переключаться между подразделами: интерфейс, методы, атаки, метрики и интерпретация анализа.",
                    )
                )
                with gr.Tabs():
                    with gr.TabItem("Методология требований"):
                        gr.Markdown(requirement_guide_md)
                    with gr.TabItem("Пайплайн и heatmap"):
                        gr.Markdown(technical_pipeline_md)
                    with gr.TabItem("Метрики статьи OmniGuard"):
                        gr.Markdown(paper_comparison_md)
                    with gr.TabItem("Разделы интерфейса"):
                        gr.Markdown(INTERFACE_REFERENCE_MARKDOWN)
                    with gr.TabItem("Методы ЦВЗ"):
                        gr.Markdown(METHOD_REFERENCE_MARKDOWN)
                    with gr.TabItem("Атаки"):
                        gr.Markdown(ATTACK_REFERENCE_MARKDOWN)
                    with gr.TabItem("Метрики и формулы"):
                        gr.Markdown(METRIC_REFERENCE_MARKDOWN)
                    with gr.TabItem("Как читать анализ"):
                        gr.Markdown(ANALYSIS_REFERENCE_MARKDOWN)

    return demo


def launch_ui(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    app = create_app()
    app.launch(server_name=host, server_port=port, share=share)
