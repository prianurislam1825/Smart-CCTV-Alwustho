"""Binding stabil wajah per-frame ke track ID sederhana.

Dipakai pada jalur diagnostik yang memakai deteksi InsightFace tanpa
ByteTrack. Pencocokan memakai IoU antar bbox frame-berurutan (greedy),
sehingga dua wajah dalam satu frame tidak saling tertukar yaitu selama
bbox mereka tidak saling tumpuk. Pipeline verifikasi tetap
TrackingFacePipeline; binder hanya menyediakan track_id.
"""

from __future__ import annotations


def _bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


class FaceTrackBinder:
    """Ikat daftar wajah (objek dengan .bbox) ke ID track yang stabil."""

    def __init__(self, iou_threshold: float = 0.30):
        if not 0.0 <= iou_threshold < 1.0:
            raise ValueError("iou_threshold harus 0 sampai < 1.")

        self.iou_threshold = iou_threshold
        self._previous: dict[int, tuple] = {}  # track_id -> bbox
        self._next_track_id = 1

    def bind(self, faces) -> list[int]:
        """
        Mengembalikan list track_id dengan urutan sama dengan `faces`.
        Wajah tumpang tindih dengan bbox track lama memakai ID tersebut;
        sisanya dapat ID baru.
        """
        bboxes = [
            tuple(float(v) for v in face.bbox)
            for face in faces
            if face is not None and getattr(face, "bbox", None) is not None
        ]

        assignments: list[int] = [-1] * len(bboxes)

        # Pasangan (iou, [index_lama, index_baru]) urut dari terbesar.
        pairs: list[tuple[float, int, int]] = []

        prev_ids = list(self._previous)
        for new_index, bbox in enumerate(bboxes):
            for prev_index, prev_id in enumerate(prev_ids):
                iou = _bbox_iou(self._previous[prev_id], bbox)

                if iou >= self.iou_threshold:
                    pairs.append((iou, prev_index, new_index))

        pairs.sort(key=lambda item: item[0], reverse=True)

        used_previous: set[int] = set()
        used_new: set[int] = set()

        for iou, prev_index, new_index in pairs:
            if prev_index in used_previous or new_index in used_new:
                continue

            track_id = prev_ids[prev_index]
            assignments[new_index] = track_id
            used_previous.add(prev_index)
            used_new.add(new_index)

        # Sisa wajah tanpa pasangan: track baru.
        updated_previous: dict[int, tuple] = {}

        for new_index, bbox in enumerate(bboxes):
            if new_index in used_new:
                track_id = assignments[new_index]
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                assignments[new_index] = track_id

            updated_previous[track_id] = bbox

        self._previous = updated_previous
        return assignments

    def reset(self) -> None:
        """Lupakan seluruh track saat ini."""
        self._previous.clear()
