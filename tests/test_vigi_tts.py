
from pathlib import Path
import pyttsx3

OUTPUT = Path("tests") / "greeting.wav"

engine = pyttsx3.init()
engine.setProperty("rate", 145)
engine.setProperty("volume", 1.0)

voices = engine.getProperty("voices")
for voice in voices:
    print(f"Voice tersedia: {voice.name}")

engine.save_to_file("Selamat datang. Semoga aktivitas hari ini berjalan lancar.", str(OUTPUT))
engine.runAndWait()
engine.stop()

if OUTPUT.exists() and OUTPUT.stat().st_size > 44:
    print(f"\nAudio berhasil dibuat: {OUTPUT.resolve()}")
    print(f"Ukuran: {OUTPUT.stat().st_size:,} byte")
else:
    print("GAGAL: file audio tidak terbentuk dengan benar.")
