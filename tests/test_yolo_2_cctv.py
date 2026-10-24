
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
import numpy as np
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
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


def main():
    load_dotenv(ENV_PATH)

    camera_configs = [
        ("office_main", "CCTV 1", os.getenv("OFFICE_CAMERA_RTSP")),
        ("office_second", "CCTV 2", os.getenv("SECOND_CAMERA_RTSP")),
    ]

    if not all(url for _, _, url in camera_configs):
        raise RuntimeError(
            "Isi OFFICE_CAMERA_RTSP dan SECOND_CAMERA_RTSP "
            "di file .env terlebih dahulu."
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia.")

    print("GPU:", torch.cuda.get_device_name(DEVICE))
    print("Memuat model YOLO...")

    # Model terpisah untuk memastikan state tracker setiap kamera terisolasi.
    models = {
        camera_id: YOLO(str(MODEL_PATH))
        for camera_id, _, _ in camera_configs
    }

    cameras = {
        camera_id: CameraManager(
            camera_id=camera_id,
            rtsp_url=url,
            reconnect_delay=RECONNECT_DELAY,
        )
        for camera_id, _, url in camera_configs
    }

    stats = {
        camera_id: {
            "fps": 0.0,
            "frames": 0,
            "start": time.perf_counter(),
            "latency": 0.0,
            "detected": 0,
            "tracked": 0,
        }
        for camera_id, _, _ in camera_configs
    }

    for camera in cameras.values():
        camera.start()

    print("Tes 2 CCTV berjalan. Tekan Q untuk keluar.")

    try:
        while True:
            panels = []

            for camera_id, camera_name, _ in camera_configs:
                frame = cameras[camera_id].get_frame()
                stat = stats[camera_id]

                if frame is None:
                    panel = np.zeros(
                        (360, 640, 3), dtype=np.uint8
                    )
                    cv2.putText(
                        panel,
                        f"{camera_name}: Menunggu frame...",
                        (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    panels.append(panel)
                    continue

                start = time.perf_counter()

                results = models[camera_id].track(
                    source=frame,
                    device=DEVICE,
                    imgsz=IMAGE_SIZE,
                    classes=[0],
                    conf=CONFIDENCE,
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False,
                )

                stat["latency"] = (
                    time.perf_counter() - start
                ) * 1000

                result = results[0]
                annotated = result.plot(labels=False)

                stat["detected"] = (
                    len(result.boxes)
                    if result.boxes is not None else 0
                )

                stat["tracked"] = 0

                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):
                    boxes = result.boxes.xyxy.cpu().tolist()
                    ids = result.boxes.id.int().cpu().tolist()
                    confs = result.boxes.conf.cpu().tolist()
                    stat["tracked"] = len(ids)

                    for box, track_id, conf in zip(
                        boxes, ids, confs
                    ):
                        x1, y1, x2, y2 = map(int, box)
                        label = f"ID {track_id} | {conf:.2f}"
                        y = max(25, y1 - 8)

                        cv2.putText(
                            annotated,
                            label,
                            (x1, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2,
                        )

                stat["frames"] += 1
                now = time.perf_counter()
                elapsed = now - stat["start"]

                if elapsed >= 1.0:
                    stat["fps"] = stat["frames"] / elapsed
                    stat["frames"] = 0
                    stat["start"] = now

                cv2.rectangle(
                    annotated,
                    (10, 10),
                    (460, 150),
                    (0, 0, 0),
                    -1,
                )

                lines = [
                    camera_name,
                    f"Detected: {stat['detected']} | "
                    f"Tracked: {stat['tracked']}",
                    f"FPS: {stat['fps']:.1f}",
                    f"Latency: {stat['latency']:.1f} ms",
                ]

                for i, line in enumerate(lines):
                    cv2.putText(
                        annotated,
                        line,
                        (20, 35 + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                panel = cv2.resize(
                    annotated, (640, 360)
                )
                panels.append(panel)

            combined = np.hstack(panels)
            cv2.imshow("Smart CCTV - 2 Camera Test", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        for camera in cameras.values():
            camera.stop()
        cv2.destroyAllWindows()
        print("Tes 2 CCTV dihentikan.")


if __name__ == "__main__":
    main()
