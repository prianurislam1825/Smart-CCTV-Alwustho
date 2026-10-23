
import logging
import threading
import time

import cv2


logger = logging.getLogger(__name__)


class CameraManager:
    """Mengelola koneksi RTSP dan pembacaan frame CCTV."""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        reconnect_delay: float = 3.0,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay

        self._capture = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

        self.connected = False
        self.last_frame_time = None

    def start(self):
        """Menjalankan pembacaan kamera di thread terpisah."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"camera-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def _connect(self):
        logger.info("[%s] Menghubungkan ke RTSP...", self.camera_id)

        capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not capture.isOpened():
            capture.release()
            logger.warning("[%s] Koneksi RTSP gagal.", self.camera_id)
            return False

        self._capture = capture
        self.connected = True
        logger.info("[%s] RTSP terhubung.", self.camera_id)
        return True

    def _capture_loop(self):
        while not self._stop_event.is_set():
            if self._capture is None or not self._capture.isOpened():
                self.connected = False

                if not self._connect():
                    self._stop_event.wait(self.reconnect_delay)
                    continue

            ok, frame = self._capture.read()

            if not ok:
                logger.warning(
                    "[%s] Frame gagal dibaca. Mencoba reconnect.",
                    self.camera_id,
                )
                self._release_capture()
                self._stop_event.wait(self.reconnect_delay)
                continue

            with self._lock:
                self._latest_frame = frame
                self.last_frame_time = time.time()

        self._release_capture()

    def get_frame(self):
        """Mengembalikan frame terbaru atau None jika belum tersedia."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _release_capture(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self.connected = False

    def stop(self):
        """Menghentikan pembacaan dan melepaskan koneksi kamera."""
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        self._release_capture()
        logger.info("[%s] Kamera dihentikan.", self.camera_id)
