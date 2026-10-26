"""Unit test FaceRecognizer dan modul registry (tanpa kamera / model)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.vision.face_recognizer import FaceRecognizer
from src.vision.registry import (
    RegistryError,
    load_registry,
    rename_person,
    save_registry_atomic,
)

BASE_DIR = Path(__file__).resolve().parents[1]
REAL_REGISTRY_PATH = BASE_DIR / "data" / "face_registry.json"
GREETING_CONFIG_PATH = BASE_DIR / "config" / "greetings.json"

OFFICIAL_EMP001_NAME = "Pria Nur Islam"


def make_vector(seed: int, dim: int = 512) -> list[float]:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return (vector / np.linalg.norm(vector)).tolist()


def write_registry(path: Path, registry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(registry, ensure_ascii=False), encoding="utf-8"
    )


# ----------------------------------------------------------------------
# 1. Registry tidak ditemukan
# ----------------------------------------------------------------------

def test_registry_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        FaceRecognizer(registry_path=tmp_path / "missing.json")


# load_registry juga harus melapor file hilang.
def test_registry_helper_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "missing.json")


# ----------------------------------------------------------------------
# 2. Registry kosong / JSON rusak
# ----------------------------------------------------------------------

def test_registry_empty(tmp_path):
    path = tmp_path / "registry.json"
    write_registry(path, {"people": []})

    with pytest.raises(RuntimeError):
        FaceRecognizer(registry_path=path)


def test_registry_corrupt_json(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text("{ bukan json", encoding="utf-8")

    with pytest.raises(Exception):
        FaceRecognizer(registry_path=path)


# ----------------------------------------------------------------------
# 3. Registry nyata: EMP001 bernama Pria Nur Islam (skipid bila file absen)
# ----------------------------------------------------------------------

@pytest.mark.skipif(
    not REAL_REGISTRY_PATH.is_file(),
    reason="Registry lokal tidak ada di lingkungan ini.",
)
def test_real_registry_emp001_official_name():
    registry = load_registry(REAL_REGISTRY_PATH)

    emp001 = next(
        (
            person
            for person in registry["people"]
            if person["person_id"] == "EMP001"
        ),
        None,
    )

    assert emp001 is not None, "EMP001 tidak ada di registry nyata."
    assert emp001["name"] == OFFICIAL_EMP001_NAME
    assert len(emp001["embeddings"]) >= 1, (
        "EMP001 tanpa embedding; pengenalan tidak akan bekerja."
    )


def test_real_greeting_config_matches_official_name():
    assert GREETING_CONFIG_PATH.is_file()

    config = json.loads(GREETING_CONFIG_PATH.read_text(encoding="utf-8"))

    emp001 = config["people"]["EMP001"]
    assert emp001["name"] == OFFICIAL_EMP001_NAME
    assert OFFICIAL_EMP001_NAME in emp001["greeting"]


# ----------------------------------------------------------------------
# 4. Rename tidak menggandakan identitas / tidak menghapus embedding
# ----------------------------------------------------------------------

def test_rename_person_keeps_embeddings(tmp_path):
    path = tmp_path / "registry.json"
    embeddings = [make_vector(1), make_vector(2)]

    write_registry(path, {
        "people": [
            {
                "person_id": "EMP001",
                "name": "Nama Lama",
                "embeddings": embeddings,
            }
        ]
    })

    result = rename_person(path, "EMP001", OFFICIAL_EMP001_NAME)

    assert result["changed"] is True
    assert result["n_embeddings"] == 2

    registry = load_registry(path)

    assert len(registry["people"]) == 1, "Rename jangan menambah entri."
    emp001 = registry["people"][0]

    assert emp001["name"] == OFFICIAL_EMP001_NAME
    assert emp001["embeddings"] == embeddings, (
        "Embedding tidak boleh berubah saat rename."
    )


def test_rename_identical_is_noop(tmp_path):
    path = tmp_path / "registry.json"
    embeddings = [make_vector(3)]

    write_registry(path, {
        "people": [
            {
                "person_id": "EMP001",
                "name": OFFICIAL_EMP001_NAME,
                "embeddings": embeddings,
            }
        ]
    })

    result = rename_person(path, "EMP001", OFFICIAL_EMP001_NAME)

    assert result["changed"] is False

    registry = load_registry(path)

    assert len(registry["people"]) == 1


def test_rename_unknown_identity_raises(tmp_path):
    path = tmp_path / "registry.json"
    write_registry(path, {
        "people": [
            {
                "person_id": "OTHER01",
                "name": "Orang Lain",
                "embeddings": [make_vector(4)],
            }
        ]
    })

    with pytest.raises(KeyError):
        rename_person(path, "EMP001", OFFICIAL_EMP001_NAME)

    # Identitas lain tetap utuh.
    registry = load_registry(path)
    assert registry["people"][0]["person_id"] == "OTHER01"
    assert registry["people"][0]["name"] == "Orang Lain"


def test_rename_validates(tmp_path):
    path = tmp_path / "registry.json"

    # Registry belum ada: rename tidak boleh membuat file palsu.
    with pytest.raises(FileNotFoundError):
        rename_person(path, "EMP001", OFFICIAL_EMP001_NAME)

    assert not path.exists()


# ----------------------------------------------------------------------
# 5-7. Pencocokan embedding
# ----------------------------------------------------------------------

def make_recognizer(tmp_path, embeddings, name=OFFICIAL_EMP001_NAME):
    path = tmp_path / "registry.json"
    write_registry(path, {
        "people": [
            {
                "person_id": "EMP001",
                "name": name,
                "embeddings": embeddings,
            }
        ]
    })

    return FaceRecognizer(registry_path=path, similarity_threshold=0.45)


def test_valid_embedding_matches_correct_identity(tmp_path):
    vector = make_vector(11)
    recognizer = make_recognizer(tmp_path, [vector])

    result = recognizer.match_embedding(np.asarray(vector))

    assert result["recognized"] is True
    assert result["person_id"] == "EMP001"
    assert result["name"] == OFFICIAL_EMP001_NAME


def test_invalid_embedding_rejected(tmp_path):
    recognizer = make_recognizer(tmp_path, [make_vector(12)])

    # Vektor nol: norm 0 -> ditolak.
    result = recognizer.match_embedding(np.zeros(512, dtype=np.float32))
    assert result["recognized"] is False

    # Vektor NaN/inf: ditolak.
    bad = np.full(512, np.nan, dtype=np.float32)
    result = recognizer.match_embedding(bad)
    assert result["recognized"] is False

    # Dimensi salah: tidak boleh crash, cukup tidak dikenali.
    result = recognizer.match_embedding(np.ones(64, dtype=np.float32))
    assert result["recognized"] is False


def test_below_threshold_is_unknown(tmp_path):
    target = make_vector(13)
    recognizer = make_recognizer(tmp_path, [target])

    # Vektor yang berbeda jauh: cosine dibangkitkan acak hingga hamper
    # tegak lurus. Kita buat vektor ortogonal + sedikit noise kecil.
    rng = np.random.default_rng(0)
    noise = rng.normal(size=512, scale=0.02).astype(np.float32)
    orthogonal = rng.normal(size=512).astype(np.float32)
    orthogonal /= np.linalg.norm(orthogonal)

    # Orthogonal cukup: sim ~ 0 -> di bawah threshold.
    result = recognizer.match_embedding(orthogonal)

    assert result["recognized"] is False
    assert result["person_id"] is None
    assert result["name"] == "Tidak dikenal"


def test_low_quality_face_returns_status(tmp_path):
    class FakeFace:
        bbox = (0, 0, 20, 20)  # sangat kecil -> kualitas rendah
        det_score = 0.55
        normed_embedding = np.asarray(make_vector(14))

    recognizer = make_recognizer(tmp_path, [make_vector(14)])
    result = recognizer.recognize_face(FakeFace())

    assert result["recognized"] is False
    assert result["status"] == "low_quality"
    assert result["person_id"] is None
