"""Unit test TrackingFacePipeline (mock FaceRecognizer)."""

from __future__ import annotations

import pytest

from src.vision.tracking_face_pipeline import TrackingFacePipeline


class MockRecognizer:
    """Recognizer tiruan: selalu mengenali EMP001 bila embedding valid."""

    def __init__(self):
        self.low_quality = False

    def recognize_face(self, face):
        if self.low_quality:
            return {
                "person_id": None,
                "name": "Kualitas wajah rendah",
                "similarity": -1.0,
                "recognized": False,
                "status": "low_quality",
            }

        if getattr(face, "valid", False):
            return {
                "person_id": "EMP001",
                "name": "Pria Nur Islam",
                "similarity": 0.80,
                "recognized": True,
                "status": "recognized",
            }

        return {
            "person_id": None,
            "name": "Tidak dikenal",
            "similarity": 0.10,
            "recognized": False,
            "status": "unknown",
        }


class Face:
    def __init__(self, valid=True):
        self.valid = valid


# ----------------------------------------------------------------------
# Wajah baru TIDAK langsung terverifikasi
# ----------------------------------------------------------------------

def test_new_track_is_candidate_first():
    pipeline = TrackingFacePipeline(
        face_recognizer=MockRecognizer(),
        min_consecutive_matches=3,
    )

    first = pipeline.update([{"track_id": 1, "face": Face(valid=True)}])

    assert first[0]["status"] == "candidate"
    assert first[0]["person_id"] is None


def test_requires_consecutive_matches():
    pipeline = TrackingFacePipeline(
        face_recognizer=MockRecognizer(),
        min_consecutive_matches=3,
    )

    statuses = []

    for _ in range(3):
        result = pipeline.update([{"track_id": 1, "face": Face(valid=True)}])
        statuses.append(result[0]["status"])

    assert statuses == ["candidate", "candidate", "verified"]
    assert statuses[-1] == "verified"


def test_verification_broken_by_unknown_frame():
    pipeline = TrackingFacePipeline(
        face_recognizer=MockRecognizer(),
        min_consecutive_matches=3,
    )

    pipeline.update([{"track_id": 1, "face": Face(valid=True)}])
    pipeline.update([{"track_id": 1, "face": Face(valid=True)}])

    # Satu frame tidak dikenal: counter harus reset.
    broken = pipeline.update([{"track_id": 1, "face": Face(valid=False)}])

    assert broken[0]["status"] == "unknown"

    # Verifikasi harus mulai dari nol lagi.
    again = pipeline.update([{"track_id": 1, "face": Face(valid=True)}])

    assert again[0]["status"] == "candidate"
    assert again[0]["consecutive_matches"] == 1


# ----------------------------------------------------------------------
# Kualitas rendah
# ----------------------------------------------------------------------

def test_low_quality_face_is_not_verified():
    recognizer = MockRecognizer()
    pipeline = TrackingFacePipeline(
        face_recognizer=recognizer,
        min_consecutive_matches=3,
    )

    recognizer.low_quality = True
    result = pipeline.update([{"track_id": 1, "face": Face(valid=True)}])

    assert result[0]["status"] == "low_quality"
    assert result[0]["person_id"] is None


# ----------------------------------------------------------------------
# Dua track tidak saling menukar kandidatnya
# ----------------------------------------------------------------------

def test_two_tracks_count_consecutive_matches_independently():
    pipeline = TrackingFacePipeline(
        face_recognizer=MockRecognizer(),
        min_consecutive_matches=3,
    )

    pipeline.update([
        {"track_id": 1, "face": Face(valid=True)},
        {"track_id": 2, "face": Face(valid=True)},
    ])
    pipeline.update([
        {"track_id": 1, "face": Face(valid=True)},
        {"track_id": 2, "face": Face(valid=True)},
    ])

    # Track 1 gagal satu frame; track 2 tetap beruntun.
    broken = pipeline.update([
        {"track_id": 1, "face": Face(valid=False)},
        {"track_id": 2, "face": Face(valid=True)},
    ])

    statuses = {row["track_id"]: row["status"] for row in broken}

    assert statuses[1] == "unknown"
    assert statuses[2] == "verified"


# ----------------------------------------------------------------------
# Reset
# ----------------------------------------------------------------------

def test_disappeared_track_is_reset():
    pipeline = TrackingFacePipeline(
        face_recognizer=MockRecognizer(),
        min_consecutive_matches=2,
    )

    pipeline.update([{"track_id": 1, "face": Face(valid=True)}])
    pipeline.update([])  # track hilang

    result = pipeline.update([{"track_id": 1, "face": Face(valid=True)}])

    assert result[0]["status"] == "candidate"
    assert result[0]["consecutive_matches"] == 1


def test_reset_clears_all_state():
    pipeline = TrackingFacePipeline(
        face_recognizer=MockRecognizer(),
        min_consecutive_matches=1,
    )

    pipeline.update([{"track_id": 1, "face": Face(valid=True)}])
    pipeline.reset()
    assert pipeline._track_state == {}


def test_invalid_min_matches_rejected():
    with pytest.raises(ValueError):
        TrackingFacePipeline(
            face_recognizer=MockRecognizer(),
            min_consecutive_matches=0,
        )
