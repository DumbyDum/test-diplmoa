from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .image_ops import ensure_rgb_uint8
from .metrics import bpp, psnr, ssim


@dataclass(slots=True)
class WatermarkEmbedResult:
    method_id: str
    method_name: str
    watermarked_image: np.ndarray
    payload_bits: list[int]
    metrics: dict[str, float | None]
    metadata: dict[str, object]


@dataclass(slots=True)
class WatermarkExtractResult:
    bits: list[int]
    metadata: dict[str, object]


class WatermarkMethod(Protocol):
    method_id: str
    method_name: str

    def embed(self, image: np.ndarray, payload_text: str) -> WatermarkEmbedResult:
        ...

    def extract(self, image: np.ndarray, bit_count: int, metadata: dict[str, object]) -> WatermarkExtractResult:
        ...


def text_to_bits(text: str) -> list[int]:
    data = text.encode("utf-8")
    bits: list[int] = []
    for byte in data:
        bits.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
    return bits


def bits_to_text(bits: list[int]) -> str:
    bytes_out = bytearray()
    usable = len(bits) - (len(bits) % 8)
    for index in range(0, usable, 8):
        value = 0
        for bit in bits[index : index + 8]:
            value = (value << 1) | int(bit)
        bytes_out.append(value)
    return bytes(bytes_out).decode("utf-8", errors="replace")


def _embedding_metrics(original: np.ndarray, watermarked: np.ndarray, payload_bit_count: int) -> dict[str, float | None]:
    return {
        "psnr": psnr(original, watermarked),
        "ssim": ssim(original, watermarked),
        "bpp": bpp(payload_bit_count, original),
    }


class LSBWatermarkMethod:
    method_id = "lsb"
    method_name = "LSB: младший бит синего канала"

    def embed(self, image: np.ndarray, payload_text: str) -> WatermarkEmbedResult:
        source = ensure_rgb_uint8(image)
        bits = text_to_bits(payload_text)
        capacity = source.shape[0] * source.shape[1]
        if len(bits) > capacity:
            raise ValueError(
                f"Payload слишком большой для LSB: нужно {len(bits)} бит, доступно {capacity} бит."
            )
        watermarked = source.copy()
        blue = watermarked[..., 2].reshape(-1)
        bit_array = np.array(bits, dtype=np.uint8)
        blue[: len(bit_array)] = (blue[: len(bit_array)] & 0xFE) | bit_array
        metadata = {
            "capacity_bits": int(capacity),
            "embedded_bits": len(bits),
            "channel": "B/RGB index 2",
            "formula": "B'(i,j) = 2 * floor(B(i,j) / 2) + bit_k",
        }
        return WatermarkEmbedResult(
            method_id=self.method_id,
            method_name=self.method_name,
            watermarked_image=watermarked,
            payload_bits=bits,
            metrics=_embedding_metrics(source, watermarked, len(bits)),
            metadata=metadata,
        )

    def extract(self, image: np.ndarray, bit_count: int, metadata: dict[str, object]) -> WatermarkExtractResult:
        source = ensure_rgb_uint8(image)
        blue = source[..., 2].reshape(-1)
        available = min(bit_count, blue.size)
        bits = (blue[:available] & 1).astype(np.uint8).tolist()
        return WatermarkExtractResult(bits=bits, metadata={"extracted_bits": len(bits)})


class DCTWatermarkMethod:
    method_id = "dct"
    method_name = "DCT: квантование коэффициента 8x8"

    def __init__(self, coefficient: tuple[int, int] = (4, 3), quantization_step: float = 18.0):
        self.coefficient = coefficient
        self.quantization_step = quantization_step

    def _capacity(self, image: np.ndarray) -> int:
        height, width = image.shape[:2]
        return (height // 8) * (width // 8)

    def embed(self, image: np.ndarray, payload_text: str) -> WatermarkEmbedResult:
        source = ensure_rgb_uint8(image)
        bits = text_to_bits(payload_text)
        capacity = self._capacity(source)
        if len(bits) > capacity:
            raise ValueError(
                f"Payload слишком большой для DCT: нужно {len(bits)} бит, доступно {capacity} бит."
            )
        ycrcb = cv2.cvtColor(source, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        y_channel = ycrcb[..., 0]
        coeff_y, coeff_x = self.coefficient

        bit_index = 0
        for top in range(0, (source.shape[0] // 8) * 8, 8):
            if bit_index >= len(bits):
                break
            for left in range(0, (source.shape[1] // 8) * 8, 8):
                if bit_index >= len(bits):
                    break
                block = y_channel[top : top + 8, left : left + 8] - 128.0
                dct_block = cv2.dct(block)
                quantized = int(round(float(dct_block[coeff_y, coeff_x]) / self.quantization_step))
                target_bit = bits[bit_index]
                if (quantized & 1) != target_bit:
                    quantized += 1 if quantized >= 0 else -1
                dct_block[coeff_y, coeff_x] = quantized * self.quantization_step
                restored = cv2.idct(dct_block) + 128.0
                y_channel[top : top + 8, left : left + 8] = restored
                bit_index += 1

        ycrcb[..., 0] = np.clip(y_channel, 0.0, 255.0)
        watermarked = cv2.cvtColor(np.clip(ycrcb, 0.0, 255.0).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        metadata = {
            "capacity_bits": int(capacity),
            "embedded_bits": len(bits),
            "block_size": 8,
            "coefficient": list(self.coefficient),
            "quantization_step": self.quantization_step,
            "formula": "q = round(C(u,v) / Delta), q' parity = bit, C'(u,v) = q' * Delta",
        }
        return WatermarkEmbedResult(
            method_id=self.method_id,
            method_name=self.method_name,
            watermarked_image=watermarked,
            payload_bits=bits,
            metrics=_embedding_metrics(source, watermarked, len(bits)),
            metadata=metadata,
        )

    def extract(self, image: np.ndarray, bit_count: int, metadata: dict[str, object]) -> WatermarkExtractResult:
        source = ensure_rgb_uint8(image)
        ycrcb = cv2.cvtColor(source, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        y_channel = ycrcb[..., 0]
        coeff_y, coeff_x = tuple(metadata.get("coefficient", self.coefficient))
        quantization_step = float(metadata.get("quantization_step", self.quantization_step))
        bits: list[int] = []

        for top in range(0, (source.shape[0] // 8) * 8, 8):
            if len(bits) >= bit_count:
                break
            for left in range(0, (source.shape[1] // 8) * 8, 8):
                if len(bits) >= bit_count:
                    break
                block = y_channel[top : top + 8, left : left + 8] - 128.0
                dct_block = cv2.dct(block)
                quantized = int(round(float(dct_block[coeff_y, coeff_x]) / quantization_step))
                bits.append(quantized & 1)

        return WatermarkExtractResult(bits=bits, metadata={"extracted_bits": len(bits)})


WATERMARK_METHOD_DESCRIPTIONS: dict[str, str] = {
    "omniguard": (
        "OmniGuard встраивает нейросетевой tamper-sensitive watermark и 100-битный payload. "
        "В benchmark для BER сравниваются исходные 100 бит payload и биты, извлеченные после атаки."
    ),
    "lsb": (
        "LSB заменяет младший бит синего канала на бит ЦВЗ. Метод очень простой и почти незаметный, "
        "но плохо переживает JPEG, фильтры и изменение размера."
    ),
    "dct": (
        "DCT делит изображение на блоки 8x8, считает дискретное косинусное преобразование и кодирует бит "
        "через четность квантованного коэффициента. Обычно он устойчивее LSB к мягкому сжатию."
    ),
}


def get_basic_method(method_id: str) -> WatermarkMethod:
    if method_id == "lsb":
        return LSBWatermarkMethod()
    if method_id == "dct":
        return DCTWatermarkMethod()
    raise ValueError(f"Unknown basic watermark method: {method_id}")


def method_choices(include_omniguard: bool = True) -> list[tuple[str, str]]:
    choices = []
    if include_omniguard:
        choices.append(("OmniGuard: нейросетевой ЦВЗ + payload", "omniguard"))
    choices.extend(
        [
            ("LSB: младший бит пикселя", "lsb"),
            ("DCT: коэффициенты ДКП 8x8", "dct"),
        ]
    )
    return choices


METHOD_REFERENCE_MARKDOWN = """
### Базовые методы встраивания ЦВЗ

#### OmniGuard
OmniGuard использует уже существующую нейросетевую схему проекта. В изображение встраивается tamper-sensitive watermark и 100-битный payload. В эксперименте payload играет роль ЦВЗ, а метрика `BER` показывает, сколько бит payload исказилось после атаки.

#### LSB
LSB расшифровывается как Least Significant Bit — младший значащий бит. Для каждого бита ЦВЗ берется очередной пиксель и заменяется младший бит синего канала:

`B'(i,j) = 2 * floor(B(i,j) / 2) + b_k`

где `B(i,j)` — исходное значение синего канала, `b_k` — очередной бит водяного знака, `B'(i,j)` — новое значение. Если младший бит уже совпадает, пиксель не меняется; если не совпадает, значение меняется всего на 1.

#### DCT / ДКП
DCT — дискретное косинусное преобразование. Изображение переводится в цветовое пространство `YCrCb`, берется канал яркости `Y`, затем он делится на блоки `8x8`. В каждом блоке считается DCT, выбирается среднечастотный коэффициент `(4,3)`, и бит кодируется через четность квантованного коэффициента:

`q = round(C(u,v) / Delta)`

`q' mod 2 = b_k`

`C'(u,v) = q' * Delta`

где `C(u,v)` — DCT-коэффициент, `Delta` — шаг квантования, `b_k` — бит ЦВЗ. После изменения коэффициента выполняется обратное DCT.
"""
