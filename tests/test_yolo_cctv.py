
from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

os.environ["YOLO_CONFIG_DIR"] = str(
    BASE_DIR / "cache" / "ultralytics"
)

import cv2
import torch
from dotenv import load_dotenv
from ultralytics import YOLO
from src.capture.camera_manager import CameraManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

MODEL_PATH = BASE_DIR / "models" / "yolo26n.pt"
ENV_PATH = BASE_DIR / ".env"

IMAGE_SIZE = 640
CONFIDENCE = 0.40
RECONNECT_DELAY = 3.0


def main():
    load_dotenv(ENV_PATH)
    rtsp_url = os.getenv("OFFICE_CAMERA_RTSP")

    if not rtsp_url:
        raise RuntimeError(
            "OFFICE_CAMERA_RTSP belum diisi di file .env"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Memuat model YOLO26n dengan ByteTrack...")

    model = YOLO(str(MODEL_PATH))

    camera = CameraManager(
        camera_id="office_main",
        rtsp_url=rtsp_url,
        reconnect_delay=RECONNECT_DELAY,
    )
    camera.start()

    print("Tracking aktif. Tekan Q pada jendela video untuk keluar.")

    previous_time = time.perf_counter()
    fps = 0.0
    frame_counter = 0
    fps_start = time.perf_counter()

    try:
        while True:
            frame = camera.get_frame()

            if frame is None:
                time.sleep(0.02)
                continue

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

            result = results[0]

            # Gambar kotak tanpa label bawaan agar tidak bertumpuk.
            annotated = result.plot(labels=False)

            tracked_count = 0

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().tolist()

                if result.boxes.id is not None:
                    ids = result.boxes.id.int().cpu().tolist()
                    confidences = (
                        result.boxes.conf.cpu().tolist()
                    )
                    tracked_count = len(ids)

                    for box, track_id, confidence in zip(
                        boxes, ids, confidences
                    ):
                        x1, y1, x2, y2 = map(int, box)

                        label = f"ID {track_id} | {confidence:.2f}"
                        text_y = max(25, y1 - 10)

                        # Latar label agar mudah dibaca.
                        (text_w, text_h), baseline = (
                            cv2.getTextSize(
                                label,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                2,
                            )
                        )

                        cv2.rectangle(
                            annotated,
                            (x1, max(0, text_y - text_h - baseline)),
                            (x1 + text_w, text_y + baseline),
                            (255, 0, 0),
                            -1,
                        )

                        cv2.putText(
                            annotated,
                            label,
                            (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2,
                        )

            # FPS pemrosesan aplikasi secara perkiraan.
            frame_counter += 1
            now = time.perf_counter()
            elapsed = now - fps_start

            if elapsed >= 1.0:
                fps = frame_counter / elapsed
                frame_counter = 0
                fps_start = now

            # Panel informasi.
            cv2.rectangle(
                annotated,
                (10, 10),
                (350, 100),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                annotated,
                f"Tracked people: {tracked_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                f"Processing FPS: {fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "Smart CCTV - Person Tracking",
                annotated,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logging.info("Program dihentikan dari keyboard.")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Tracking CCTV dihentikan.")


if __name__ == "__main__":
    main()
