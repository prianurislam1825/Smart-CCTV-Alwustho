"""Konversi WAV PCM ke G.711 A-law (PCMA) 8000 Hz mono.

Ekstraksi dari tests/test_vigi_audio_convert.py agar dapat dipakai
modul pengiriman audio VIGI, bukan hanya skrip uji.
"""

from __future__ import annotations

import math
import wave


def alaw_encode(sample: int) -> int:
    """Enkoding satu sampel PCM 16-bit menjadi satu byte A-law."""
    value = max(-32768, min(32767, int(sample)))
    mask = 0xD5 if value >= 0 else 0x55

    if value < 0:
        value = -value - 1

    value = min(value, 32767)

    if value < 256:
        encoded = value >> 4
    else:
        exponent = min(7, max(1, value.bit_length() - 8))
        mantissa = (value >> (exponent + 3)) & 0x0F
        encoded = (exponent << 4) | mantissa

    return encoded ^ mask


def wav_to_alaw(wav_path, target_rate: int = 8000) -> bytes:
    """Baca WAV PCM mono 16-bit, resample ke target_rate, dan rekam A-law."""
    with wave.open(str(wav_path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Sumber harus PCM mono 16-bit.")

        source_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    samples = [
        int.from_bytes(raw[i:i + 2], "little", signed=True)
        for i in range(0, len(raw), 2)
    ]

    output_length = round(len(samples) * target_rate / source_rate)
    encoded = bytearray()

    for i in range(output_length):
        position = i * source_rate / target_rate
        left = min(int(position), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        fraction = position - left

        sample = round(
            samples[left] * (1 - fraction) + samples[right] * fraction
        )
        encoded.append(alaw_encode(sample))

    return bytes(encoded)


def frame_alaw(payload: bytes, bytes_per_frame: int = 160) -> list[bytes]:
    """Pecah payload A-law menjadi frame 20 ms (160 byte @ 8000 Hz)."""
    frames = []

    for offset in range(0, len(payload), bytes_per_frame):
        chunk = payload[offset:offset + bytes_per_frame]

        if len(chunk) < bytes_per_frame:
            # Frame terakhir diisi nol agar panjang frame tetap.
            chunk = chunk + bytes(bytes_per_frame - len(chunk))

        frames.append(chunk)

    return frames
