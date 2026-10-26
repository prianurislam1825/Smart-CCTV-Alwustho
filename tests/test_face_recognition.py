from __future__ import annotations

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
from src.vision.face_track_binder import FaceTrackBinder
from src.vision.tracking_face_pipeline import TrackingFacePipeline

ENV_PATH = BASE_DIR / ".env"
MODEL_ROOT = BASE_DIR / "models" / "insightface"
REGISTRY_PATH = BASE_DIR / "data" / "face_registry.json"

# Ambang eksperimen, belum layak dijadikan aturan absensi final.
SIMILARITY_THRESHOLD = 0.45
MIN_CONSECUTIVE_MATCHES = 3

# Pemeriksaan kualitas awal; akan dievaluasi dari hasil CCTV.
MIN_FACE_SIZE = 45
MIN_DETECTION_SCORE = 0.50


def main():
    load_dotenv(ENV_PATH)

    rtsp_url = os.getenv("OFFICE_CAMERA_RTSP")
    if not rtsp_url:
        raise RuntimeError(
            "OFFICE_CAMERA_RTSP belum diatur dalam file .env."
        )

    # FaceRecognizer memuat registry sendiri tanpa menulis file.
    from src.vision.face_recognizer import FaceRecognizer

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

    print("=" * 60)
    print("SMART CCTV - FACE RECOGNITION DIAGNOSTIC")
    print("=" * 60)
    print("Identitas terdaftar:")
    for person_id, person in recognizer.people.items():
        print(f"- {person_id}: {person['name']}")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA tidak tersedia; memakai CPU.")

    print("Memuat InsightFace buffalo_l...")

    face_app = FaceAnalysis(
        name="buffalo_l",
        root=str(MODEL_ROOT),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    print("\nProvider model:")
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
    last_report = 0.0

    stats = {
        "no_face": 0,
        "low_quality": 0,
        "unknown": 0,
        "candidate": 0,
        "verified": 0,
    }

    print("\nCCTV aktif. Tekan Q untuk keluar.")
    print("Mode diagnostik: absensi belum dicatat.")

    try:
        while True:
            frame = camera.get_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            start = time.perf_counter()
            faces = face_app.get(frame)
            inference_ms = (time.perf_counter() - start) * 1000

            frame_status = "WAJAH TIDAK TERDETEKSI"

            if not faces:
                stats["no_face"] += 1

            # Ikat wajah ke track_id yang stabil antar frame.
            track_ids = binder.bind(faces)

            tracked_people = [
                {"track_id": track_id, "face": face}
                for track_id, face in zip(track_ids, faces)
            ]

            verifications = pipeline.update(tracked_people)

            for verification, face in zip(verifications, faces):
                x1, y1, x2, y2 = map(int, face.bbox)

                status = verification["status"]

                if status == "no_face" or status == "low_quality":
                    label = "WAJAH TERDETEKSI - KUALITAS RENDAH"
                    color = (0, 165, 255)
                    frame_status = "WAJAH TERDETEKSI - KUALITAS RENDAH"
                    stats["low_quality"] += 1
                elif status == "unknown":
                    similarity = verification.get("similarity", -1.0)
                    label = f"TIDAK DIKENAL | sim {similarity:.2f}"
                    color = (0, 0, 255)
                    frame_status = "WAJAH TERDETEKSI - BELUM COCOK"
                    stats["unknown"] += 1
                elif status == "candidate":
                    similarity = verification.get("similarity", -1.0)
                    matches = verification.get("consecutive_matches", 0)
                    label = (
                        f"Verifikasi {verification['name']} "
                        f"({matches}) | sim {similarity:.2f}"
                    )
                    color = (0, 165, 255)
                    frame_status = "WAJAH TERDETEKSI - VERIFIKASI"
                    stats["candidate"] += 1
                else:  # verified
                    similarity = verification.get("similarity", -1.0)
                    label = (
                        f"{verification['name']} "
                        f"| sim {similarity:.2f}"
                    )
                    color = (0, 200, 0)
                    frame_status = f"{verification['name']} - TERVERIFIKASI"
                    stats["verified"] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
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

            cv2.putText(
                frame,
                frame_status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Faces: {len(faces)} | FPS: {fps:.1f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Inference: {inference_ms:.1f} ms",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Smart CCTV - Face Recognition Diagnostic", frame)

            if now - last_report >= 10:
                print(
                    f"FPS={fps:.1f} | inferensi={inference_ms:.1f} ms | "
                    f"wajah={len(faces)} | status={frame_status}"
                )
                print("Statistik:", stats)
                last_report = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nPengujian dihentikan.")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("\nPengujian selesai.")
        print("Statistik akhir:", stats)


if __name__ == "__main__":
    main()
