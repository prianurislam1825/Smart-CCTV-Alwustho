"""Greeting manager: memutuskan kapan greeting boleh diputar.

Tugas modul (terpisah dari pengenalan wajah):
- Menerima hasil verifikasi per track dari TrackingFacePipeline.
- Hanya identitas TERVERIFIKASI dengan konfigurasi greeting yang memicu audio.
- Satu greeting per kunjungan (per track) + cooldown per identitas.
- Menghormati status antrean/audio yang sedang dimainkan.
- Log berisi identitas, similarity, status verifikasi, alasan, dan status
  pengiriman audio -- tanpa data biometrik mentah.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Status pipeline yang tidak boleh memicu greeting.
_NON_VERIFIED_STATUSES = {"no_face", "low_quality", "unknown", "candidate"}


class GreetingSpeaker(Protocol):
    """Kontrak speaker yang diterima GreetingManager (bisa mock / VIGI)."""

    def speak(self, event: dict) -> object:
        ...

    def is_busy(self) -> bool:
        ...


class GreetingManager:
    """Keputusan greeting per frame, berbasis sesi kunjungan per track."""

    def __init__(
        self,
        config_path: Path,
        speaker: GreetingSpeaker | None = None,
        time_fn=time.time,
    ):
        self.config_path = Path(config_path)
        self.speaker = speaker
        self.time_fn = time_fn

        self._load_config()

        # cooldown per identitas: person_id -> waktu terakhir greeting.
        self._last_greeted_at: dict[str, float] = {}

        # sesi kunjungan per track: track_id -> {"person_id", "greeted"}.
        self._track_sessions: dict[Any, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Konfigurasi
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        with self.config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        if not isinstance(config, dict):
            raise ValueError(f"Konfigurasi greeting tidak valid: {self.config_path}")

        self._defaults = config.get("defaults") or {}
        self._people = config.get("people") or {}

        self._cooldowns: dict[str, float] = {}
        for person_id, entry in self._people.items():
            entry = entry or {}
            self._cooldowns[person_id] = float(
                entry.get(
                    "cooldown_seconds",
                    self._defaults.get("cooldown_seconds", 120.0),
                )
            )

            greeting_text = entry.get("greeting")
            if greeting_text:
                # Sanity: greeting dikonfigurasi menyebut nama orang yang sama.
                entry_name = entry.get("name")
                if entry_name and entry_name not in greeting_text:
                    logger.warning(
                        "Greeting %s tidak memuat nama %s.",
                        person_id,
                        entry_name,
                    )

    def add_person(
        self,
        person_id: str,
        name: str,
        greeting: str,
        cooldown_seconds: float | None = None,
    ) -> None:
        """Menambah identitas baru tanpa mengubah kode utama."""
        self._people[person_id] = {
            "name": name,
            "greeting": greeting,
        }
        seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else self._defaults.get("cooldown_seconds", 120.0)
        )
        self._cooldowns[person_id] = float(seconds)

    # ------------------------------------------------------------------
    # Evaluasi per frame
    # ------------------------------------------------------------------

    def evaluate_frame(self, verifications: list[dict]) -> list[dict]:
        """
        verifications: hasil TrackingFacePipeline.update() untuk satu frame,
        yakni daftar dict berisi track_id, person_id, name, status, similarity.
        Mengembalikan daftar event greeting yang benar-benar dipicu frame ini.
        """
        visible_track_ids: set[Any] = set()
        events: list[dict] = []

        for verification in verifications or []:
            track_id = verification.get("track_id")

            if track_id is None:
                continue

            track_id = int(track_id)
            visible_track_ids.add(track_id)
            status = verification.get("status")

            session = self._track_sessions.get(track_id)

            if status in _NON_VERIFIED_STATUSES:
                if session is not None and verification.get("person_id") is None:
                    # Identitas tidak lagi terdeteksi pada track ini:
                    # kunjungan baru harus verifikasi ulang.
                    session["person_id"] = None
                    session["greeted"] = False
                continue

            person_id = verification.get("person_id")

            if not person_id:
                continue

            entry = self._people.get(person_id)

            if entry is None:
                logger.info(
                    "track=%s id=%s terverifikasi tetapi tidak punya "
                    "konfigurasi greeting; dilewati.",
                    track_id,
                    person_id,
                )
                continue

            session = self._track_sessions.setdefault(
                track_id,
                {"person_id": None, "greeted": False},
            )

            if session["person_id"] != person_id:
                # Track ditempati orang lain (atau kunjungan baru):
                # sesi greeting dimulai dari nol.
                session["person_id"] = person_id
                session["greeted"] = False

            similarity = float(verification.get("similarity", -1.0))

            if session["greeted"]:
                logger.debug(
                    "id=%s track=%s sudah disapa kunjungan ini; skip.",
                    person_id,
                    track_id,
                )
                continue

            if self._is_in_cooldown(person_id):
                logger.info(
                    "id=%s (%s) track=%s dalam cooldown; greeting "
                    "dilewati. similarity=%.3f",
                    person_id,
                    entry.get("name"),
                    track_id,
                    similarity,
                )
                continue

            busy = False
            if self.speaker is not None:
                try:
                    busy = bool(self.speaker.is_busy())
                except Exception:
                    logger.exception("Gagal membaca status speaker.")

            if busy:
                # Tidak ditandai sudah disapa agar bisa dicoba lagi
                # di frame berikutnya setelah audio selesai.
                logger.info(
                    "id=%s (%s) track=%s; audio sedang dimainkan, "
                    "greeting menunggu. similarity=%.3f",
                    person_id,
                    entry.get("name"),
                    track_id,
                    similarity,
                )
                continue

            event = {
                "person_id": person_id,
                "name": entry.get("name") or person_id,
                "greeting": entry.get("greeting", ""),
                "track_id": track_id,
                "similarity": similarity,
                "reason": "identity_verified",
            }

            # Kegagalan speaker tidak boleh merusak loop kamera.
            delivered = None
            if self.speaker is not None:
                try:
                    delivered = self.speaker.speak(event)
                except Exception:
                    logger.exception(
                        "id=%s track=%s pengiriman audio gagal.",
                        person_id,
                        track_id,
                    )
                    delivered = False

            event["delivered"] = (
                True if delivered is True or delivered is None else False
            )

            session["greeted"] = True
            self._last_greeted_at[person_id] = self.time_fn()

            logger.info(
                "Greeting dipicu: id=%s track=%s reason=%s "
                "similarity=%.3f delivered=%s",
                person_id,
                track_id,
                event["reason"],
                similarity,
                event["delivered"],
            )
            events.append(event)

        # Track yang tidak lagi muncul: kunjungan berakhir.
        for track_id in list(self._track_sessions):
            if track_id not in visible_track_ids:
                self._track_sessions.pop(track_id, None)

        return events

    def _is_in_cooldown(self, person_id: str) -> bool:
        last = self._last_greeted_at.get(person_id)

        if last is None:
            return False

        return (self.time_fn() - last) < self._cooldowns.get(person_id, 120.0)

    def reset(self) -> None:
        """Hapus sesi kunjungan dan cooldown (misal untuk pengujian)."""
        self._track_sessions.clear()
        self._last_greeted_at.clear()
