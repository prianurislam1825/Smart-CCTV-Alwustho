
from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ["YOLO_CONFIG_DIR"] = str(BASE_DIR / "cache" / "ultralytics")

import torch

# Pastikan DLL CUDA/cuDNN dari PyTorch tersedia untuk ONNX Runtime.
_DLL_DIRECTORY = os.add_dll_directory(
    str(Path(torch.__file__).resolve().parent / "lib")
)

import cv2
import onnxruntime as ort
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

from src.capture.camera_manager import CameraManager


ENV_PATH = BASE_DIR / ".env"
MODEL_ROOT = BASE_DIR / "models" / "insightface"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main():
    load_dotenv(ENV_PATH)

    rtsp_url = os.getenv("OFFICE_CAMERA_RTSP")
    if not rtsp_url:
        raise RuntimeError(
            "OFFICE_CAMERA_RTSP belum diatur di file .env"
        )

    print("=" * 55)
    print("SMART CCTV - FACE DETECTION")
    print("=" * 55)
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "Tidak tersedia untuk PyTorch",
    )
    print("ONNX Runtime:", ort.__version__)
    print("Provider tersedia:", ort.get_available_providers())

    print("\nMemuat InsightFace buffalo_l...")

    face_app = FaceAnalysis(
        name="buffalo_l",
        root=str(MODEL_ROOT),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    face_app.prepare(
        ctx_id=0,
        det_size=(640, 640),
    )

    print("\nProvider setiap model:")
    for name, model in face_app.models.items():
        session = getattr(model, "session", None)
        if session is not None:
            print(f"- {name}: {session.get_providers()}")

    camera = CameraManager(
        camera_id="office_main",
        rtsp_url=rtsp_url,
    )

    camera.start()

    previous_time = time.perf_counter()
    fps = 0.0
    frame_count = 0
    report_start = time.perf_counter()

    print("\nMembuka CCTV. Tekan Q untuk keluar.")

    try:
        while True:
            frame = camera.get_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            inference_start = time.perf_counter()
            faces = face_app.get(frame)
            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000

            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                confidence = float(face.det_score)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 220, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Face {confidence:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 220, 0),
                    2,
                )

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now

            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = (
                    instant_fps
                    if fps == 0
                    else 0.8 * fps + 0.2 * instant_fps
                )

            frame_count += 1

            cv2.putText(
                frame,
                f"Faces: {len(faces)} | FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Inference: {inference_ms:.1f} ms",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow(
                "Smart CCTV - Face Detection (GPU)",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Laporan performa setiap 10 detik.
            if now - report_start >= 10:
                print(
                    f"FPS: {fps:.1f} | "
                    f"Inferensi: {inference_ms:.1f} ms | "
                    f"Wajah: {len(faces)}"
                )
                report_start = now

    except KeyboardInterrupt:
        print("\nPengujian dihentikan.")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Koneksi kamera ditutup.")


if __name__ == "__main__":
    main()
