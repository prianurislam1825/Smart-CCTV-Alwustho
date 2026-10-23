
from pathlib import Path
import os

# Harus diatur sebelum mengimpor Ultralytics.
BASE_DIR = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(BASE_DIR / "cache" / "ultralytics")

import torch
from ultralytics import YOLO

MODEL_PATH = BASE_DIR / "models" / "yolo26n.pt"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

print("PyTorch:", torch.__version__)
print("CUDA tersedia:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA tidak tersedia.")

print("GPU:", torch.cuda.get_device_name(0))

# Unduh bobot langsung ke folder models pada D.
if not MODEL_PATH.exists():
    import urllib.request

    model_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
    print("Mengunduh model ke:", MODEL_PATH)
    urllib.request.urlretrieve(model_url, str(MODEL_PATH))

print("Bobot model:", MODEL_PATH)

model = YOLO(str(MODEL_PATH))

# Tes inferensi menggunakan gambar contoh.
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device=0,
    imgsz=640,
    project=str(BASE_DIR / "runs"),
    name="gpu_test",
    exist_ok=True,
    verbose=True,
)

print("Inferensi selesai.")
print("Jumlah objek terdeteksi:", len(results[0].boxes))
print("Hasil tersimpan di:", BASE_DIR / "runs" / "gpu_test")
