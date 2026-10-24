
import math
import wave
from pathlib import Path

SOURCE = Path("tests/greeting.wav")
OUTPUT = Path("tests/greeting_alaw.raw")
TARGET_RATE = 8000


def alaw_encode(sample):
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


with wave.open(str(SOURCE), "rb") as wav:
    if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
        raise SystemExit("Sumber harus PCM mono 16-bit.")

    source_rate = wav.getframerate()
    raw = wav.readframes(wav.getnframes())

samples = [
    int.from_bytes(raw[i:i + 2], "little", signed=True)
    for i in range(0, len(raw), 2)
]

output_length = round(len(samples) * TARGET_RATE / source_rate)
encoded = bytearray()

for i in range(output_length):
    position = i * source_rate / TARGET_RATE
    left = min(int(position), len(samples) - 1)
    right = min(left + 1, len(samples) - 1)
    fraction = position - left

    sample = round(
        samples[left] * (1 - fraction) + samples[right] * fraction
    )
    encoded.append(alaw_encode(sample))

OUTPUT.write_bytes(encoded)

print("Konversi selesai.")
print("Codec: G.711 A-law (PCMA)")
print("Sample rate: 8000 Hz")
print("Channel: mono")
print(f"File: {OUTPUT.resolve()}")
print(f"Ukuran: {len(encoded):,} byte")
