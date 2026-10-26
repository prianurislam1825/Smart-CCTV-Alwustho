"""Unit test FaceTrackBinder: identitas per track tidak tertukar."""

from __future__ import annotations

import pytest

from src.vision.face_track_binder import FaceTrackBinder, _bbox_iou


class FakeFace:
    def __init__(self, bbox):
        self.bbox = tuple(bbox)


def test_iou_identical_is_one():
    assert _bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_is_zero():
    assert _bbox_iou((0, 0, 10, 10), (20, 20, 40, 40)) == 0.0


# ----------------------------------------------------------------------
# Track ID stabil antar frame
# ----------------------------------------------------------------------

def test_same_face_gets_stable_track_id():
    binder = FaceTrackBinder()

    first = binder.bind([FakeFace((100, 100, 200, 200))])
    second = binder.bind([FakeFace((104, 102, 206, 204))])

    assert first == second
    assert first[0] >= 1


def test_new_face_gets_new_track_id():
    binder = FaceTrackBinder()

    first_ids = binder.bind([FakeFace((10, 10, 100, 100))])
    second_ids = binder.bind([FakeFace((400, 400, 500, 500))])

    assert second_ids[0] != first_ids[0]


# ----------------------------------------------------------------------
# Dua wajah tidak saling tertukar saat mereka sedikit bergerak
# ----------------------------------------------------------------------

def test_two_moving_faces_keep_their_own_ids():
    binder = FaceTrackBinder()

    ids_frame1 = binder.bind([
        FakeFace((0, 0, 100, 100)),
        FakeFace((300, 0, 400, 100)),
    ])

    # Kedua wajah bergerak sedikit (mitigasi jitter).
    ids_frame2 = binder.bind([
        FakeFace((8, 5, 106, 103)),
        FakeFace((310, 3, 408, 101)),
    ])

    # Track kiri tetap kiri; kanan tetap kanan.
    assert ids_frame2[0] == ids_frame1[0]
    assert ids_frame2[1] == ids_frame1[1]
    assert ids_frame1[0] != ids_frame1[1]


def test_partial_frame_keeps_matching():
    binder = FaceTrackBinder()

    binder.bind([
        FakeFace((0, 0, 100, 100)),
        FakeFace((300, 0, 400, 100)),
    ])

    # Satu wajah tidak terdeteksi frame ini: track lain tetap konsisten.
    ids = binder.bind([FakeFace((305, 2, 403, 100))])

    assert len(ids) == 1


def test_reset_clears_tracks():
    binder = FaceTrackBinder()

    binder.bind([FakeFace((0, 0, 100, 100))])
    binder.reset()

    ids = binder.bind([FakeFace((0, 0, 100, 100))])

    # Setelah reset, wajah mendapat track baru (ID berbeda).
    assert ids[0] != 1


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        FaceTrackBinder(iou_threshold=1.0)
