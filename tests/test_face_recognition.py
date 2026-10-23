
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch

# Membantu ONNX Runtime menemukan DLL CUDA/cuDNN dari PyTorch.
_DLL_DIRECTORY = os.add_dll_directory(
    str(Path(torch.__file__).resolve().parent / "lib")
)

import cv2
import numpy as np
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

from src.capture.camera_manager import CameraManager


ENV_PATH = BASE_DIR / ".env"
MODEL_ROOT = BASE_DIR / "models" / "insightface"
REGISTRY_PATH = BASE_DIR / "data" / "face_registry.json"

PERSON_ID = "EMP001"
PERSON_NAME = "Pria Nur Islam"

# Ambang eksperimen, belum layak dijadikan aturan absensi final.
SIMILARITY_THRESHOLD = 0.45
MIN_CONSECUTIVE_MATCHES = 3

# Pemeriksaan kualitas awal; akan dievaluasi dari hasil CCTV.
MIN_FACE_SIZE = 45
MIN_DETECTION_SCORE = 0.50


def load_registry():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            "Registry tidak ditemukan. Jalankan registrasi wajah dahulu."
        )

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    people = {}

    for person in registry.get("people", []):
        person_id = person.get("person_id")
        name = person.get("name", "Tidak diketahui")
        vectors = []

        for raw in person.get("embeddings", []):
            vector = np.asarray(raw, dtype=np.float32)
            norm = float(np.linalg.norm(vector))

            if norm > 1e-12 and np.all(np.isfinite(vector)):
                vectors.append(vector / norm)

        if person_id and vectors:
            if person_id == PERSON_ID:
                name = PERSON_NAME
                person["name"] = PERSON_NAME

            people[person_id] = {
                "name": name,
                "embeddings": vectors,
            }

    if PERSON_ID not in people:
        raise RuntimeError(
            f"Identitas {PERSON_ID} tidak ditemukan dalam registry."
        )

    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(registry, file, ensure_ascii=False, indent=2)

    return people


def match_face(embedding, people):
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))

    if norm <= 1e-12:
        return None, "Tidak dikenal", -1.0

    vector /= norm

    best_id = None
    best_name = "Tidak dikenal"
    best_score = -1.0

    for person_id, person in people.items():
        for registered_vector in person["embeddings"]:
            score = float(np.dot(vector, registered_vector))

            if score > best_score:
                best_score = score
                best_id = person_id
                best_name = person["name"]

    if best_score >= SIMILARITY_THRESHOLD:
        return best_id, best_name, best_score

    return None, "Tidak dikenal", best_score


def main():
    load_dotenv(ENV_PATH)

    rtsp_url = os.getenv("OFFICE_CAMERA_RTSP")
    if not rtsp_url:
        raise RuntimeError(
            "OFFICE_CAMERA_RTSP belum diatur dalam file .env."
        )

    people = load_registry()

    print("=" * 60)
    print("SMART CCTV - FACE RECOGNITION DIAGNOSTIC")
    print("=" * 60)
    print("Identitas terdaftar:")
    for person_id, person in people.items():
        print(f"- {person_id}: {person['name']}")

    print("\nGPU:", torch.cuda.get_device_name(0))
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
    consecutive_matches = {}
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

            visible_ids = set()
            frame_status = "WAJAH TIDAK TERDETEKSI"

            if not faces:
                stats["no_face"] += 1

            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                detection_score = float(face.det_score)

                if (
                    min(width, height) < MIN_FACE_SIZE
                    or detection_score < MIN_DETECTION_SCORE
                ):
                    label = "WAJAH TERDETEKSI - KUALITAS RENDAH"
                    color = (0, 165, 255)
                    frame_status = "WAJAH TERDETEKSI - KUALITAS RENDAH"
                    stats["low_quality"] += 1

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
                    continue

                person_id, name, similarity = match_face(
                    face.normed_embedding, people
                )

                if person_id is None:
                    label = f"TIDAK DIKENAL | sim {similarity:.2f}"
                    color = (0, 0, 255)
                    frame_status = "WAJAH TERDETEKSI - BELUM COCOK"
                    stats["unknown"] += 1

                else:
                    visible_ids.add(person_id)
                    consecutive_matches[person_id] = (
                        consecutive_matches.get(person_id, 0) + 1
                    )

                    if (
                        consecutive_matches[person_id]
                        >= MIN_CONSECUTIVE_MATCHES
                    ):
                        label = f"{name} | sim {similarity:.2f}"
                        color = (0, 200, 0)
                        frame_status = f"{name} - TERVERIFIKASI"
                        stats["verified"] += 1
                    else:
                        label = f"Verifikasi {name} | sim {similarity:.2f}"
                        color = (0, 165, 255)
                        frame_status = "WAJAH TERDETEKSI - VERIFIKASI"
                        stats["candidate"] += 1

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

            for person_id in list(consecutive_matches):
                if person_id not in visible_ids:
                    consecutive_matches[person_id] = 0

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
