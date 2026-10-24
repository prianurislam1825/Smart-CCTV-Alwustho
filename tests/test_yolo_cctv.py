
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
    format="%(asctime)s | %(levelname)s | %(message)s",
)

MODEL_PATH = BASE_DIR / "models" / "yolo26n.pt"
ENV_PATH = BASE_DIR / ".env"

IMAGE_SIZE = 640
CONFIDENCE = 0.40
RECONNECT_DELAY = 3.0
DEVICE = 0


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

    print("GPU:", torch.cuda.get_device_name(DEVICE))
    print("Memuat YOLO dan ByteTrack...")

    model = YOLO(str(MODEL_PATH))

    camera = CameraManager(
        camera_id="office_main",
        rtsp_url=rtsp_url,
        reconnect_delay=RECONNECT_DELAY,
    )
    camera.start()

    print("Tracking aktif. Tekan Q untuk keluar.")

    fps = 0.0
    frame_counter = 0
    fps_start = time.perf_counter()
    last_log = fps_start

    try:
        while True:
            frame = camera.get_frame()

            if frame is None:
                time.sleep(0.02)
                continue

            inference_start = time.perf_counter()

            results = model.track(
                source=frame,
                device=DEVICE,
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

                    (text_w, text_h), baseline = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        2,
                    )

                    top = max(0, text_y - text_h - baseline)

                    cv2.rectangle(
                        annotated,
                        (x1, top),
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
                (440, 145),
                (0, 0, 0),
                -1,
            )

            lines = [
                f"Detected people: {detected_count}",
                f"Tracked people: {tracked_count}",
                f"Processing FPS: {fps:.1f}",
                f"Inference latency: {inference_ms:.1f} ms",
            ]

            for index, line in enumerate(lines):
                cv2.putText(
                    annotated,
                    line,
                    (20, 38 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(
                "Smart CCTV - Multi Person Tracking",
                annotated,
            )

            if now - last_log >= 5.0:
                logging.info(
                    "detected=%d tracked=%d fps=%.1f latency=%.1fms",
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
        camera.stop()
        cv2.destroyAllWindows()
        print("Tracking dihentikan.")


if __name__ == "__main__":
    main()
