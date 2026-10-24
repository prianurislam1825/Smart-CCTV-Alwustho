
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FaceRecognizer:
    """Memuat registry dan mencocokkan embedding wajah."""

    def __init__(
        self,
        registry_path: Path,
        similarity_threshold: float = 0.45,
        min_face_size: int = 45,
        min_detection_score: float = 0.50,
    ):
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold harus 0 sampai 1.")

        self.registry_path = Path(registry_path)
        self.similarity_threshold = similarity_threshold
        self.min_face_size = min_face_size
        self.min_detection_score = min_detection_score

        self.people = {}
        self.load_registry()

    @staticmethod
    def _normalize(vector):
        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            return None

        norm = float(np.linalg.norm(vector))

        if norm <= 1e-12:
            return None

        return vector / norm

    def load_registry(self):
        if not self.registry_path.is_file():
            raise FileNotFoundError(
                f"Registry tidak ditemukan: {self.registry_path}"
            )

        with self.registry_path.open("r", encoding="utf-8") as file:
            registry = json.load(file)

        people = {}

        for person in registry.get("people", []):
            person_id = person.get("person_id")
            name = person.get("name")
            raw_embeddings = person.get("embeddings", [])

            if not person_id or not name:
                continue

            embeddings = []

            for raw in raw_embeddings:
                vector = self._normalize(raw)

                if vector is not None:
                    embeddings.append(vector)

            if embeddings:
                people[person_id] = {
                    "name": name,
                    "embeddings": embeddings,
                }

        if not people:
            raise RuntimeError(
                "Registry tidak memiliki embedding wajah yang valid."
            )

        self.people = people
        return len(self.people)

    def match_embedding(self, embedding):
        query = self._normalize(embedding)

        if query is None:
            return {
                "person_id": None,
                "name": "Tidak dikenal",
                "similarity": -1.0,
                "recognized": False,
            }

        best_id = None
        best_name = "Tidak dikenal"
        best_score = -1.0

        for person_id, person in self.people.items():
            for registered in person["embeddings"]:
                if registered.shape != query.shape:
                    continue

                score = float(np.dot(query, registered))

                if score > best_score:
                    best_id = person_id
                    best_name = person["name"]
                    best_score = score

        recognized = (
            best_id is not None
            and best_score >= self.similarity_threshold
        )

        return {
            "person_id": best_id if recognized else None,
            "name": best_name if recognized else "Tidak dikenal",
            "similarity": best_score,
            "recognized": recognized,
        }

    def recognize_face(self, face):
        x1, y1, x2, y2 = face.bbox
        width = max(0, int(x2 - x1))
        height = max(0, int(y2 - y1))
        detection_score = float(face.det_score)

        if (
            min(width, height) < self.min_face_size
            or detection_score < self.min_detection_score
        ):
            return {
                "person_id": None,
                "name": "Kualitas wajah rendah",
                "similarity": -1.0,
                "recognized": False,
                "status": "low_quality",
            }

        result = self.match_embedding(face.normed_embedding)
        result["status"] = (
            "recognized" if result["recognized"] else "unknown"
        )

        return result
