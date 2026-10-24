
from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import cv2
import torch
from ultralytics import YOLO

MODEL_PATH = BASE_DIR / "models" / "yolo26n.pt"

IMAGE_SIZE = 640
CONFIDENCE = 0.40
WEBCAM_INDEX = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Memuat model YOLO26n...")

    model = YOLO(str(MODEL_PATH))

    camera = cv2.VideoCapture(WEBCAM_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "Webcam tidak dapat dibuka. "
            "Periksa indeks webcam atau tutup aplikasi lain."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam aktif. Tekan Q untuk keluar.")

    frame_counter = 0
    fps = 0.0
    fps_start = time.perf_counter()
    last_log = fps_start

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                logging.warning("Gagal membaca frame webcam.")
                time.sleep(0.05)
                continue

            inference_start = time.perf_counter()

            results = model.track(
                source=frame,
                device=0,
                imgsz=IMAGE_SIZE,
                classes=[0],
                conf=CONFIDENCE,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000

            result = results[0]
            annotated = result.plot(labels=False)

            detected_count = (
                len(result.boxes)
                if result.boxes is not None
                else 0
            )

            tracked_count = 0

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes.xyxy.cpu().tolist()
                ids = result.boxes.id.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()

                tracked_count = len(ids)

                for box, track_id, confidence in zip(
                    boxes, ids, confidences
                ):
                    x1, y1, x2, y2 = map(int, box)

                    label = f"ID {track_id} | {confidence:.2f}"
                    text_y = max(25, y1 - 10)

                    cv2.rectangle(
                        annotated,
                        (x1, max(0, text_y - 25)),
                        (x1 + 190, text_y + 5),
                        (255, 0, 0),
                        -1,
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            frame_counter += 1
            now = time.perf_counter()
            elapsed = now - fps_start

            if elapsed >= 1.0:
                fps = frame_counter / elapsed
                frame_counter = 0
                fps_start = now

            cv2.rectangle(
                annotated,
                (10, 10),
                (380, 125),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                annotated,
                f"Detected: {detected_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                f"Tracked: {tracked_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f} | Inference: {inference_ms:.0f} ms",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "Smart CCTV - Webcam Tracking",
                annotated,
            )

            if now - last_log >= 5.0:
                logging.info(
                    "Detected=%d | Tracked=%d | FPS=%.1f | "
                    "Inference=%.0f ms",
                    detected_count,
                    tracked_count,
                    fps,
                    inference_ms,
                )
                last_log = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logging.info("Program dihentikan.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Pengujian webcam selesai.")


if __name__ == "__main__":
    main()
