
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.vision.face_recognizer import FaceRecognizer


class TrackingFacePipeline:
    """
    Menghubungkan track ID YOLO/ByteTrack dengan pengenalan wajah.

    Pemanggilan update() dilakukan untuk setiap frame.
    Pipeline ini tidak menyimpan absensi.
    """

    def __init__(
        self,
        face_recognizer: FaceRecognizer,
        min_consecutive_matches: int = 3,
    ):
        if min_consecutive_matches < 1:
            raise ValueError(
                "min_consecutive_matches minimal bernilai 1."
            )

        self.face_recognizer = face_recognizer
        self.min_consecutive_matches = min_consecutive_matches

        # Status sementara setiap track ID.
        self._track_state: dict[int, dict[str, Any]] = {}

    def update(self, tracked_people: list[dict[str, Any]]) -> list[dict]:
        """
        tracked_people berisi:
        [
            {
                "track_id": 12,
                "face": insightface_face_object atau None
            }
        ]

        Satu track tanpa wajah yang valid tidak otomatis dikenali.
        """
        results = []
        visible_track_ids = set()

        for person in tracked_people:
            track_id = person.get("track_id")
            face = person.get("face")

            if track_id is None:
                continue

            track_id = int(track_id)
            visible_track_ids.add(track_id)

            state = self._track_state.setdefault(
                track_id,
                {
                    "candidate_id": None,
                    "consecutive_matches": 0,
                    "person_id": None,
                    "name": "Tidak dikenal",
                },
            )

            if face is None:
                state["candidate_id"] = None
                state["consecutive_matches"] = 0

                results.append({
                    "track_id": track_id,
                    "person_id": None,
                    "name": "Tidak dikenal",
                    "status": "no_face",
                })
                continue

            match = self.face_recognizer.recognize_face(face)

            if match["status"] == "low_quality":
                state["candidate_id"] = None
                state["consecutive_matches"] = 0

                results.append({
                    "track_id": track_id,
                    "person_id": None,
                    "name": "Kualitas wajah rendah",
                    "status": "low_quality",
                    "similarity": match["similarity"],
                })
                continue

            candidate_id = match["person_id"]

            if candidate_id is None:
                state["candidate_id"] = None
                state["consecutive_matches"] = 0

                results.append({
                    "track_id": track_id,
                    "person_id": None,
                    "name": "Tidak dikenal",
                    "status": "unknown",
                    "similarity": match["similarity"],
                })
                continue

            if state["candidate_id"] == candidate_id:
                state["consecutive_matches"] += 1
            else:
                state["candidate_id"] = candidate_id
                state["consecutive_matches"] = 1

            verified = (
                state["consecutive_matches"]
                >= self.min_consecutive_matches
            )

            if verified:
                state["person_id"] = candidate_id
                state["name"] = match["name"]

            results.append({
                "track_id": track_id,
                "person_id": candidate_id if verified else None,
                "name": match["name"] if verified else "Memverifikasi...",
                "status": "verified" if verified else "candidate",
                "similarity": match["similarity"],
                "consecutive_matches": state["consecutive_matches"],
            })

        # Track yang tidak muncul pada frame ini dihapus dari status
        # kandidat agar kemunculan kembali tidak dianggap beruntun.
        for track_id in list(self._track_state):
            if track_id not in visible_track_ids:
                self._track_state.pop(track_id, None)

        return results

    def reset(self) -> None:
        """Menghapus seluruh status tracking sementara."""
        self._track_state.clear()
