
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch

# Gunakan DLL CUDA/cuDNN yang tersedia dari PyTorch.
_DLL_DIRECTORY = os.add_dll_directory(
    str(Path(torch.__file__).resolve().parent / "lib")
)

import cv2
import numpy as np
from insightface.app import FaceAnalysis


SAMPLES_DIR = BASE_DIR / "data" / "face_samples"
DATA_DIR = BASE_DIR / "data"
REGISTRY_PATH = DATA_DIR / "face_registry.json"
MODEL_ROOT = BASE_DIR / "models" / "insightface"

# Ubah nilai ini sesuai identitas yang ingin didaftarkan.
PERSON_ID = "EMP001"
PERSON_NAME = "Nanda Ari Wahyu Widagdo"


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

    if REGISTRY_PATH.exists():
        with REGISTRY_PATH.open("r", encoding="utf-8") as file:
            registry = json.load(file)
    else:
        registry = {"people": []}

    if any(
        person.get("person_id") == PERSON_ID
        for person in registry["people"]
    ):
        raise RuntimeError(
            f"ID {PERSON_ID} sudah terdaftar. "
            "Gunakan ID lain atau periksa registry terlebih dahulu."
        )

    print("Memuat InsightFace buffalo_l...")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Folder foto:", SAMPLES_DIR)
    print("Jumlah foto:", len(image_paths))

    face_app = FaceAnalysis(
        name="buffalo_l",
        root=str(MODEL_ROOT),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

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

    registry["people"].append({
        "person_id": PERSON_ID,
        "name": PERSON_NAME,
        "embeddings": embeddings,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })

    temp_path = REGISTRY_PATH.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(registry, file, ensure_ascii=False, indent=2)
    temp_path.replace(REGISTRY_PATH)

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
