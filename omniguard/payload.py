from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets

from .schemas import PayloadDecodeResult, PayloadEncodeResult, PayloadRecord


PAYLOAD_VERSION = 1
PAYLOAD_RAW_BITS = 56
PAYLOAD_CODE_BITS = 98
PAYLOAD_PAD_BITS = 2
PAYLOAD_TOTAL_BITS = PAYLOAD_CODE_BITS + PAYLOAD_PAD_BITS
PAYLOAD_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _int_to_bits(value: int, width: int) -> list[int]:
    return [int(bit) for bit in f"{value:0{width}b}"]


def _bits_to_int(bits: list[int]) -> int:
    if not bits:
        return 0
    return int("".join(str(bit) for bit in bits), 2)


def _hash_bits(document_id: str, width: int) -> list[int]:
    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") & ((1 << width) - 1)
    return _int_to_bits(value, width)


def _compute_auth_bits(secret_key: str, bits: list[int], width: int = 8) -> list[int]:
    message = "".join(str(bit) for bit in bits).encode("ascii")
    digest = hmac.new(secret_key.encode("utf-8"), message, hashlib.sha256).digest()
    value = digest[0] & ((1 << width) - 1)
    return _int_to_bits(value, width)


def _hamming_encode_nibble(nibble: list[int]) -> list[int]:
    d1, d2, d3, d4 = nibble
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]


def _hamming_decode_word(word: list[int]) -> tuple[list[int], int]:
    bits = word[:]
    s1 = bits[0] ^ bits[2] ^ bits[4] ^ bits[6]
    s2 = bits[1] ^ bits[2] ^ bits[5] ^ bits[6]
    s3 = bits[3] ^ bits[4] ^ bits[5] ^ bits[6]
    syndrome = s1 + (s2 << 1) + (s3 << 2)
    corrected = 0
    if syndrome:
        index = syndrome - 1
        if 0 <= index < len(bits):
            bits[index] ^= 1
            corrected = 1
    return [bits[2], bits[4], bits[5], bits[6]], corrected


def hamming_encode(bits: list[int]) -> tuple[list[int], list[int]]:
    if len(bits) % 4 != 0:
        raise ValueError("Bit stream for Hamming encoding must be divisible by 4.")
    encoded: list[int] = []
    for offset in range(0, len(bits), 4):
        encoded.extend(_hamming_encode_nibble(bits[offset : offset + 4]))
    padded = encoded + [0, 1]
    return encoded, padded


def hamming_decode(bits: list[int]) -> tuple[list[int], int]:
    if len(bits) < PAYLOAD_CODE_BITS:
        raise ValueError("Bit stream is too short for payload decoding.")
    corrected = 0
    decoded: list[int] = []
    for offset in range(0, PAYLOAD_CODE_BITS, 7):
        nibble, nibble_corrected = _hamming_decode_word(bits[offset : offset + 7])
        decoded.extend(nibble)
        corrected += nibble_corrected
    return decoded[:PAYLOAD_RAW_BITS], corrected


def build_payload_bits(
    document_id: str,
    secret_key: str,
    issued_at_utc: datetime | None = None,
    nonce: int | None = None,
) -> PayloadEncodeResult:
    issued_at_utc = issued_at_utc or datetime.now(timezone.utc)
    issued_hours = int((issued_at_utc - PAYLOAD_EPOCH).total_seconds() // 3600)
    if issued_hours < 0:
        raise ValueError("Payload timestamp is before the payload epoch.")
    if issued_hours >= 2**20:
        raise ValueError("Payload timestamp is out of range for the current payload format.")
    nonce = secrets.randbelow(2**8) if nonce is None else nonce
    base_bits = (
        _int_to_bits(PAYLOAD_VERSION, 4)
        + _int_to_bits(issued_hours, 20)
        + _hash_bits(document_id, 16)
        + _int_to_bits(nonce, 8)
    )
    auth_bits = _compute_auth_bits(secret_key, base_bits, width=8)
    raw_bits = base_bits + auth_bits
    encoded_bits, padded_bits = hamming_encode(raw_bits)
    record = PayloadRecord(
        version=PAYLOAD_VERSION,
        issued_at_utc=PAYLOAD_EPOCH + timedelta(hours=issued_hours),
        document_hash_hex=f"{_bits_to_int(base_bits[24:40]):04x}",
        nonce=nonce,
        auth_tag_hex=f"{_bits_to_int(auth_bits):02x}",
    )
    return PayloadEncodeResult(record=record, encoded_bits=padded_bits, raw_bits=raw_bits)


def decode_payload_bits(
    encoded_bits: list[int],
    secret_key: str,
    expected_document_id: str | None = None,
    reference_bits: list[int] | None = None,
) -> PayloadDecodeResult:
    raw_bits, corrected_errors = hamming_decode(encoded_bits)
    version = _bits_to_int(raw_bits[0:4])
    issued_hours = _bits_to_int(raw_bits[4:24])
    document_hash_bits = raw_bits[24:40]
    nonce = _bits_to_int(raw_bits[40:48])
    auth_bits = raw_bits[48:56]
    base_bits = raw_bits[:48]
    expected_auth = _compute_auth_bits(secret_key, base_bits, width=8)
    auth_ok = auth_bits == expected_auth
    document_hash_hex = f"{_bits_to_int(document_hash_bits):04x}"
    record = PayloadRecord(
        version=version,
        issued_at_utc=PAYLOAD_EPOCH + timedelta(hours=issued_hours),
        document_hash_hex=document_hash_hex,
        nonce=nonce,
        auth_tag_hex=f"{_bits_to_int(auth_bits):02x}",
    )
    document_match = None
    if expected_document_id is not None:
        document_match = document_hash_bits == _hash_bits(expected_document_id, 16)
    warnings: list[str] = []
    if version != PAYLOAD_VERSION:
        warnings.append(
            f"Unexpected payload version {version}; expected {PAYLOAD_VERSION}."
        )
    if not auth_ok:
        warnings.append("Payload authentication tag check failed.")
    bit_accuracy = None
    if reference_bits is not None:
        shared = min(len(reference_bits), len(encoded_bits))
        if shared:
            matches = sum(
                int(reference_bits[index] == encoded_bits[index]) for index in range(shared)
            )
            bit_accuracy = matches / shared
    return PayloadDecodeResult(
        record=record,
        decoded_bits=encoded_bits[:PAYLOAD_TOTAL_BITS],
        corrected_errors=corrected_errors,
        bit_accuracy=bit_accuracy,
        auth_ok=auth_ok,
        document_match=document_match,
        warnings=warnings,
    )
