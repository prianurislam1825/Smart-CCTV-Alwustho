from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch

# Membantu ONNX Runtime menemukan DLL CUDA/cuDNN dari PyTorch.
os.add_dll_directory(
    str(Path(torch.__file__).resolve().parent / "lib")
)

import cv2
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

from src.capture.camera_manager import CameraManager
from src.vision.face_recognizer import FaceRecognizer
from src.vision.face_track_binder import FaceTrackBinder
from src.vision.tracking_face_pipeline import TrackingFacePipeline
from src.greeting.greeting_manager import GreetingManager

ENV_PATH = BASE_DIR / ".env"
MODEL_ROOT = BASE_DIR / "models" / "insightface"
REGISTRY_PATH = BASE_DIR / "data" / "face_registry.json"
GREETING_CONFIG_PATH = BASE_DIR / "config" / "greetings.json"

SIMILARITY_THRESHOLD = 0.45
MIN_CONSECUTIVE_MATCHES = 3
MIN_FACE_SIZE = 45
MIN_DETECTION_SCORE = 0.50


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger("greeting-cctv")


def build_speaker(args, logger):
    """Bangun speaker. Default: mode uji tanpa mengirim audio ke kamera."""
    speaker_impl = None

    if args.speaker == "vigi" or args.speaker == "vigi-dry":
        load_dotenv(ENV_PATH)

        rtsp_url = os.getenv("OFFICE_CAMERA_RTSP")
        camera_ip = os.getenv("VIGI_CAMERA_IP")

        if not rtsp_url or not camera_ip:
            logger.warning(
                "OFFICE_CAMERA_RTSP/VIGI_CAMERA_IP belum diatur; "
                "greeting hanya akan dilog."
            )
            return None

        if args.speaker == "vigi":
            from src.audio.vigi_speaker import speaker_from_env

            speaker_impl = speaker_from_env(
                send_audio=True,
                camera_ip=camera_ip,
                rtsp_url=rtsp_url,
            )
        else:
            # dry-run: jalur TTS->A-law dijalankan, audio tidak dikirim.
            from src.audio.vigi_talk_client import VigiTalkClient, \
                credentials_from_rtsp_url

            username, password = credentials_from_rtsp_url(rtsp_url)

            if not username:
                logger.warning("URL RTSP tidak memuat kredensial; mode dry-run gelog.")
                return None

            talk_client = VigiTalkClient(
                host=camera_ip,
                username=username,
                password=password,
            )

            from src.audio.vigi_speaker import VigiGreetingSpeaker

            speaker_impl = VigiGreetingSpeaker(
                talk_client, send_audio=False
            )

        speaker_impl.start()
        logger.info("Speaker VIGI dijalankan: mode=%s", args.speaker)

    return speaker_impl


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Alur lengkap: deteksi wajah -> pengenalan -> greeting "
            "melalui speaker VIGI C340."
        )
    )
    parser.add_argument(
        "--speaker",
        choices=("log", "vigi-dry", "vigi"),
        default="log",
        help=(
            "log: greeting hanya dilog; "
            "vigi-dry: TTS+konversi dijalankan, tidak dikirim ke kamera; "
            "vigi: kirim audio ke speaker kamera (AKTIFKAN SUARA)."
        ),
    )
    args = parser.parse_args()

    logger = setup_logging()
    load_dotenv(ENV_PATH)

    rtsp_url = os.getenv("OFFICE_CAMERA_RTSP")

    if not rtsp_url:
        raise RuntimeError("OFFICE_CAMERA_RTSP belum diatur dalam .env.")

    recognizer = FaceRecognizer(
        registry_path=REGISTRY_PATH,
        similarity_threshold=SIMILARITY_THRESHOLD,
        min_face_size=MIN_FACE_SIZE,
        min_detection_score=MIN_DETECTION_SCORE,
    )

    pipeline = TrackingFacePipeline(
        face_recognizer=recognizer,
        min_consecutive_matches=MIN_CONSECUTIVE_MATCHES,
    )
    binder = FaceTrackBinder()
    speaker_impl = build_speaker(args, logger)

    greeting_manager = GreetingManager(
        config_path=GREETING_CONFIG_PATH,
        speaker=speaker_impl,
    )

    logger.info("Identitas greeting terdaftar: %s", list(greeting_manager._people))

    if torch.cuda.is_available():
        logger.info("GPU aktif: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("CUDA tidak tersedia; memakai CPU.")

    face_app = FaceAnalysis(
        name="buffalo_l",
        root=str(MODEL_ROOT),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    camera = CameraManager(camera_id="office_main", rtsp_url=rtsp_url)
    camera.start()

    logger.info("Mulai; tekan Q di jendela untuk keluar.")

    try:
        while True:
            frame = camera.get_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            faces = face_app.get(frame)
            track_ids = binder.bind(faces)

            verifications = pipeline.update(
                [
                    {"track_id": track_id, "face": face}
                    for track_id, face in zip(track_ids, faces)
                ]
            )

            for verification in verifications:
                if verification["status"] == "verified":
                    logger.info(
                        "Id=%s track=%s terverifikasi (sim=%.3f).",
                        verification["person_id"],
                        verification["track_id"],
                        verification["similarity"],
                    )

            # Keputusan greeting (dan audio bila diaktifkan).
            greeting_manager.evaluate_frame(verifications)

            # Overlay sederhana.
            for verification, face in zip(verifications, faces):
                x1, y1, x2, y2 = map(int, face.bbox)
                color = (
                    (0, 200, 0)
                    if verification["status"] == "verified"
                    else (0, 165, 255)
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.imshow("Smart CCTV - Greeting", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Dihentikan pengguna.")

    finally:
        if speaker_impl is not None:
            speaker_impl.stop()

        camera.stop()
        cv2.destroyAllWindows()
        logger.info("Selesai.")


if __name__ == "__main__":
    main()
