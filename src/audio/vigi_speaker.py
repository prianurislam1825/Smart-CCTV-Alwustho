"""Pemutar greeting ke speaker bawaan VIGI C340.

Aliran:
  teks greeting -> pyttsx3 (WAV lokal) -> G.711 A-law 8000 Hz mono
  -> klien talk VIGI -> speaker kamera.

Kebijakan antrean: satu audio pada satu waktu. Bila sedang memutar,
permintaan baru ditolak (is_busy == True) dan tidak mengantri panjang,
agar greeting tidak menumpuk saat orang menatap kamera lebih lama.

Thread pemutar dipisah dari loop kamera sehingga kegagalan audio
tidak menghentikan deteksi/pengenalan wajah.
"""

from __future__ import annotations

import logging
import queue
import tempfile
import threading
from pathlib import Path

from src.audio import alaw
from src.audio.vigi_talk_client import VigiTalkClient, credentials_from_rtsp_url

logger = logging.getLogger(__name__)

# Param TTS yang dipakai greeting.
TTS_RATE = 145


class VigiGreetingSpeaker:
    """Antrean greeting satu-per-satu dengan worker thread."""

    def __init__(
        self,
        talk_client: VigiTalkClient,
        send_audio: bool = True,
        queue_size: int = 1,
    ):
        """
        talk_client: klien sesi talk ke kamera.
        send_audio: False untuk mode uji tanpa mengirim ke kamera
                    (registrasi/test greeting lokal).
        """
        self.talk_client = talk_client
        self.send_audio = send_audio

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._playing = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # API untuk GreetingManager
    # ------------------------------------------------------------------

    def is_busy(self) -> bool:
        return self._playing.is_set() or not self._queue.empty()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker,
            name="vigi-audio-queue",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        try:
            self.talk_client.close_session()
        except Exception:
            logger.exception("Gagal menutup sesi talk saat berhenti.")

    def speak(self, event: dict) -> bool:
        """
        Masukkan greeting ke antrean.

        Mengembalikan True bila diterima antrean; False bila speaker
        sedang sibuk sehingga pemanggil bisa mencoba di frame berikut.
        """
        if self.is_busy():
            logger.info(
                "Antrean audio penuh; greeting id=%s ditolak sementara.",
                event.get("person_id"),
            )
            return False

        try:
            self._queue.put_nowait(event)
        except queue.Full:
            return False

        logger.info(
            "Greeting id=%s masuk antrean audio.",
            event.get("person_id"),
        )
        return True

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            self._playing.set()

            try:
                text = event.get("greeting", "")

                if not text:
                    logger.info(
                        "Greeting tanpa teks; dilewati (id=%s).",
                        event.get("person_id"),
                    )
                    continue

                alaw_frames = self._text_to_frames(text)

                if not alaw_frames:
                    logger.info(
                        "Audio kosong setelah konversi; tidak dikirim."
                    )
                    continue

                if not self.send_audio:
                    logger.info(
                        "Mode uji: audio greeting id=%s disiapkan "
                        "(%d frame) tetapi tidak dikirim ke kamera.",
                        event.get("person_id"),
                        len(alaw_frames),
                    )
                    continue

                self._send_frames(alaw_frames)

                logger.info(
                    "Audio terkirim: id=%s track=%s durasi_frame=%d.",
                    event.get("person_id"),
                    event.get("track_id"),
                    len(alaw_frames),
                )

            except Exception:
                # TANGGAPAN kegagalan audio: cukup log, jangan hentikan
                # worker maupun pipeline kamera.
                logger.exception(
                    "Pemutaran greeting gagal (id=%s); sistem lanjut.",
                    event.get("person_id"),
                )
            finally:
                self._playing.clear()

    def _text_to_frames(self, text: str) -> list[bytes]:
        """Sintesis teks dan konversi ke frame A-law 20 ms."""
        temp_dir = Path(tempfile.gettempdir())
        wav_path = temp_dir / "smart_cctv_greeting.wav"

        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", 1.0)
        engine.save_to_file(text, str(wav_path))
        engine.runAndWait()
        engine.stop()

        if not wav_path.is_file() or wav_path.stat().st_size <= 44:
            raise RuntimeError("TTS gagal menghasilkan file audio.")

        payload = alaw.wav_to_alaw(wav_path, target_rate=8000)
        return alaw.frame_alaw(payload, bytes_per_frame=160)

    def _send_frames(self, frames: list[bytes]) -> None:
        """Buka sesi talk, kirim frame A-law, dan tutup sesi."""
        with self.talk_client as client:
            sock = client._sock

            if sock is None:
                raise RuntimeError("Sesi talk tidak memiliki koneksi.")

            for index, frame in enumerate(frames):
                if self._stop_event.is_set():
                    break

                sent = 0

                while sent < len(frame):
                    sent += sock.send(frame[sent:])

                # Interval kirim ~20 ms per frame 160 byte @ 8000 Hz.
                # Angka ini bagian format payload yang perlu
                # diverifikasi langsung pada kamera.
                import time

                time.sleep(0.02)

            logger.info(
                "Frame terkirim: %d (0..%d).", len(frames), max(0, len(frames) - 1)
            )


def speaker_from_env(
    send_audio: bool,
    camera_ip: str,
    rtsp_url: str,
) -> VigiGreetingSpeaker:
    """Bangun speaker dari konfigurasi environment; tanpa hardcode."""
    from src.audio.vigi_talk_client import DEFAULT_RTSP_PORT

    username, password = credentials_from_rtsp_url(rtsp_url)

    if not username:
        raise ValueError(
            "OFFICE_CAMERA_RTSP belum memuat username; tidak bisa "
            "membangun klien talk VIGI."
        )

    talk_client = VigiTalkClient(
        host=camera_ip,
        username=username,
        password=password,
        port=DEFAULT_RTSP_PORT,
    )

    return VigiGreetingSpeaker(talk_client, send_audio=send_audio)
