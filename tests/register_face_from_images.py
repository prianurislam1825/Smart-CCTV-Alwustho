from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch

# Gunakan DLL CUDA/cuDNN yang tersedia dari PyTorch.
os.add_dll_directory(
    str(Path(torch.__file__).resolve().parent / "lib")
)

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src.vision.registry import load_registry, save_registry_atomic

SAMPLES_DIR = BASE_DIR / "data" / "face_samples"
DATA_DIR = BASE_DIR / "data"
REGISTRY_PATH = DATA_DIR / "face_registry.json"
MODEL_ROOT = BASE_DIR / "models" / "insightface"

# Identitas resmi. Nama ini harus konsisten dengan config/greetings.json.
PERSON_ID = "EMP001"
PERSON_NAME = "Pria Nur Islam"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not SAMPLES_DIR.exists():
        raise FileNotFoundError(
            f"Folder foto tidak ditemukan: {SAMPLES_DIR}"
        )

    image_paths = sorted(
        path for path in SAMPLES_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    if not image_paths:
        raise RuntimeError(
            f"Tidak ada foto di folder: {SAMPLES_DIR}"
        )

    existing = None
    if REGISTRY_PATH.exists():
        registry = load_registry(REGISTRY_PATH)
        for person in registry["people"]:
            if person.get("person_id") == PERSON_ID:
                existing = person
    else:
        registry = {"people": []}

    # Jika ID sudah terdaftar, hanya nama yang diperbarui bila berbeda;
    # embedding yang valid tidak pernah diganti tanpa registrasi ulang.
    if existing is not None:
        if existing.get("name") == PERSON_NAME:
            print(f"ID {PERSON_ID} sudah terdaftar dengan nama yang sama.")
            print("Tidak ada perubahan registry.")
            return

        old_name = existing.get("name")
        existing["name"] = PERSON_NAME
        save_registry_atomic(REGISTRY_PATH, registry)

        print(f"Nama diperbarui: '{old_name}' -> '{PERSON_NAME}'.")
        print(f"Embedding tetap utuh: {len(existing.get('embeddings', []))}")
        return

    print("Memuat InsightFace buffalo_l...")

    # GPU hanya dianggap aktif jika CUDA benar-benar tersedia.
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        print("CUDA tidak tersedia; memakai CPU.")
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    print("Folder foto:", SAMPLES_DIR)
    print("Jumlah foto:", len(image_paths))

    face_app = FaceAnalysis(
        name="buffalo_l",
        root=str(MODEL_ROOT),
        providers=providers,
    )
    face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    embeddings = []
    failed_images = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[GAGAL] Tidak bisa membaca: {image_path.name}")
            failed_images.append(image_path.name)
            continue

        faces = face_app.get(image)

        if len(faces) != 1:
            print(
                f"[DITOLAK] {image_path.name}: "
                f"terdeteksi {len(faces)} wajah; harus tepat satu."
            )
            failed_images.append(image_path.name)
            continue

        face = faces[0]

        if float(face.det_score) < 0.60:
            print(
                f"[DITOLAK] {image_path.name}: "
                f"deteksi wajah kurang meyakinkan."
            )
            failed_images.append(image_path.name)
            continue

        embedding = np.asarray(
            face.normed_embedding, dtype=np.float32
        )

        if not np.all(np.isfinite(embedding)):
            print(f"[DITOLAK] Embedding tidak valid: {image_path.name}")
            failed_images.append(image_path.name)
            continue

        embeddings.append(embedding.tolist())
        print(
            f"[OK] {image_path.name} "
            f"(deteksi {float(face.det_score):.2f})"
        )

    if len(embeddings) < 2:
        raise RuntimeError(
            f"Registrasi dibatalkan: hanya {len(embeddings)} "
            "embedding valid. Periksa kualitas dan sudut foto."
        )

    from datetime import datetime

    registry["people"].append({
        "person_id": PERSON_ID,
        "name": PERSON_NAME,
        "embeddings": embeddings,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })

    save_registry_atomic(REGISTRY_PATH, registry)

    print("\nRegistrasi selesai.")
    print("ID:", PERSON_ID)
    print("Nama:", PERSON_NAME)
    print("Embedding tersimpan:", len(embeddings))
    print("Foto ditolak:", len(failed_images))
    print("File registry:", REGISTRY_PATH)
    print(
        "\nCatatan: data ini baru terdaftar; "
        "pengenalan identitas belum diuji."
    )


if __name__ == "__main__":
    main()
