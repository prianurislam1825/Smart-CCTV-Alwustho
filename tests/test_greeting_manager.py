"""Unit test GreetingManager (mock speaker, tanpa perangkat keras)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.greeting.greeting_manager import GreetingManager

BASE_DIR = Path(__file__).resolve().parents[1]
REAL_CONFIG = BASE_DIR / "config" / "greetings.json"


class MockSpeaker:
    """Speaker tiruan: mencatat panggilan speak() dan status sibuk."""

    def __init__(self, fail=False):
        self.calls = []
        self.busy = False
        self.fail = fail

    def is_busy(self) -> bool:
        return self.busy

    def speak(self, event: dict) -> bool:
        if self.fail:
            raise RuntimeError("Audio gagal (simulasi).")

        self.calls.append(event)
        return True


def make_manager(speaker=None, **kwargs) -> GreetingManager:
    return GreetingManager(
        config_path=REAL_CONFIG,
        speaker=speaker,
        **kwargs,
    )


def verified(track_id=1, person_id="EMP001", similarity=0.75, name=None):
    verification = {
        "track_id": track_id,
        "person_id": person_id,
        "name": name or person_id,
        "status": "verified",
        "similarity": similarity,
        "consecutive_matches": 3,
    }
    return verification


# ----------------------------------------------------------------------
# 8. Wajah berkualitas rendah tidak memicu greeting
# ----------------------------------------------------------------------

def test_low_quality_does_not_trigger_greeting():
    speaker = MockSpeaker()
    manager = make_manager(speaker)

    events = manager.evaluate_frame([
        {
            "track_id": 1,
            "person_id": None,
            "name": "Kualitas wajah rendah",
            "status": "low_quality",
            "similarity": -1.0,
        }
    ])

    assert events == []
    assert speaker.calls == []


# ----------------------------------------------------------------------
# 9. Identitas kandidat belum memicu greeting
# ----------------------------------------------------------------------

def test_candidate_does_not_trigger_greeting():
    speaker = MockSpeaker()
    manager = make_manager(speaker)

    for matches in (1, 2):
        events = manager.evaluate_frame([
            {
                "track_id": 1,
                "person_id": "EMP001",
                "name": "Memverifikasi...",
                "status": "candidate",
                "similarity": 0.80,
                "consecutive_matches": matches,
            }
        ])

        assert events == []
        assert speaker.calls == []


# ----------------------------------------------------------------------
# 10. Identitas terverifikasi memicu greeting yang benar
# ----------------------------------------------------------------------

def test_verified_triggers_correct_greeting():
    speaker = MockSpeaker()
    clock = {"now": 1000.0}
    manager = make_manager(speaker, time_fn=lambda: clock["now"])

    events = manager.evaluate_frame([verified(track_id=7)])

    assert len(events) == 1
    event = events[0]

    assert event["person_id"] == "EMP001"
    assert event["greeting"] == (
        "Selamat datang, Pria Nur Islam. "
        "Semoga aktivitas hari ini berjalan lancar."
    )
    assert event["track_id"] == 7

    # speak() juga dipanggil pada speaker.
    assert len(speaker.calls) == 1
    assert speaker.calls[0]["person_id"] == "EMP001"


def test_verified_unknown_identity_without_config_is_skipped():
    """Identitas terverifikasi tapi tanpa konfigurasi: tidak diputar."""
    speaker = MockSpeaker()
    manager = make_manager(speaker)

    events = manager.evaluate_frame([
        verified(track_id=2, person_id="OTHER09")
    ])

    assert events == []
    assert speaker.calls == []


# ----------------------------------------------------------------------
# 11. Tidak berulang pada frame/sesi kunjungan yang sama
# ----------------------------------------------------------------------

def test_no_repeat_within_same_visit():
    speaker = MockSpeaker()
    clock = {"now": 1000.0}
    manager = make_manager(speaker, time_fn=lambda: clock["now"])

    for _ in range(10):
        manager.evaluate_frame([verified(track_id=1)])

    assert len(speaker.calls) == 1

    # Track hilang sedetik lalu datang kembali: kunjungan baru,
    # tetapi masih dalam cooldown identity-level, jadi tidak diputar lagi.
    clock["now"] = 1010.0
    manager.evaluate_frame([])  # tracker reset kunjungan
    events = manager.evaluate_frame([verified(track_id=1)])

    assert events == []
    assert len(speaker.calls) == 1


def test_repeats_allowed_after_cooldown_expires_between_visits():
    speaker = MockSpeaker()
    clock = {"now": 1000.0}
    manager = make_manager(speaker, time_fn=lambda: clock["now"])

    manager.evaluate_frame([verified(track_id=1)])
    assert len(speaker.calls) == 1

    # Kunjungan baru setelah cooldown (120 detik) lewat:
    # track harus hilang dulu agar sesi kunjungan di-reset.
    clock["now"] = 1000.0 + 121.0
    manager.evaluate_frame([])

    events = manager.evaluate_frame([verified(track_id=1)])

    assert len(events) == 1
    assert len(speaker.calls) == 2


# ----------------------------------------------------------------------
# 12. Identitas berbeda menerima greeting masing-masing
# ----------------------------------------------------------------------

def test_different_identities_each_get_their_own_greeting():
    speaker = MockSpeaker()
    clock = {"now": 1000.0}
    manager = make_manager(speaker, time_fn=lambda: clock["now"])

    # Identitas kedua ditambahkan lewat konfigurasi runtime.
    manager.add_person(
        "EMP002",
        "Rekan Kerja",
        "Selamat datang, Rekan Kerja. Silakan masuk.",
        cooldown_seconds=120,
    )

    events = manager.evaluate_frame([
        verified(track_id=1, person_id="EMP001"),
        verified(track_id=2, person_id="EMP002"),
    ])

    assert {event["person_id"] for event in events} == {"EMP001", "EMP002"}
    texts = {event["person_id"]: event["greeting"] for event in events}

    assert "Pria Nur Islam" in texts["EMP001"]
    assert "Rekan Kerja" in texts["EMP002"]
    assert len(speaker.calls) == 2


# ----------------------------------------------------------------------
# Cooldown seperti dites; audio sibuk menunda, lalu mencoba lagi.
# ----------------------------------------------------------------------

def test_busy_audio_defers_then_delivers_on_next_frame():
    speaker = MockSpeaker()
    clock = {"now": 1000.0}
    manager = make_manager(speaker, time_fn=lambda: clock["now"])

    speaker.busy = True
    events = manager.evaluate_frame([verified(track_id=1)])

    assert events == []
    assert speaker.calls == []

    # Audio selesai pada frame selanjutnya: greeting terkirim.
    speaker.busy = False
    events = manager.evaluate_frame([verified(track_id=1)])

    assert len(events) == 1
    assert len(speaker.calls) == 1


def test_track_disappearance_resets_visit():
    speaker = MockSpeaker()
    clock = {"now": 1000.0}
    manager = make_manager(speaker, time_fn=lambda: clock["now"])

    manager.evaluate_frame([verified(track_id=1)])
    assert len(speaker.calls) == 1

    # Track hilang -> sesi kunjungan reset.
    clock["now"] = 1100.0  # cooldown masih aktif
    manager.evaluate_frame([])

    # Kunjungan baru: masih dalam cooldown, tidak diputar.
    events = manager.evaluate_frame([verified(track_id=1)])
    assert events == []


def test_identity_change_on_same_track_fresh_session():
    speaker = MockSpeaker()
    manager = make_manager(speaker)
    manager.add_person("EMP002", "B", "Selamat datang, B.", 0.0001)

    manager.evaluate_frame([verified(track_id=1, person_id="EMP001")])
    assert len(speaker.calls) == 1

    # Track yang sama kini menunjuk identitas lain: sesi baru.
    events = manager.evaluate_frame([
        verified(track_id=1, person_id="EMP002")
    ])

    # Greeting kedua boleh tampil karena cooldown EMP002 hampir nol.
    assert len(events) == 1
    assert events[0]["person_id"] == "EMP002"


# ----------------------------------------------------------------------
# 14. Kegagalan speaker tidak merusak evaluasi (dan pipeline induknya)
# ----------------------------------------------------------------------

def test_speaker_failure_is_isolated():
    speaker = MockSpeaker(fail=True)
    clock = {"now": 1000.0}
    manager = make_manager(speaker, time_fn=lambda: clock["now"])

    # Tidak boleh melempar exception meski speaker gagal.
    events = manager.evaluate_frame([verified(track_id=1)])

    assert len(events) == 1
    assert events[0]["delivered"] is False

    # Kunjungan ini sudah diproses; tidak mecoba ulang tiap frame.
    for _ in range(5):
        manager.evaluate_frame([verified(track_id=1)])

    assert len(speaker.calls) == 0  # speak() selalu error


def test_config_file_structure_valid():
    config = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))

    assert config["people"]["EMP001"]["name"] == "Pria Nur Islam"
    assert config["defaults"]["cooldown_seconds"] >= 0
