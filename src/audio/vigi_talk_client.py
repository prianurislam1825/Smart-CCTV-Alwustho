"""Klien talk (dua arah) ke speaker bawaan TP-Link VIGI C340.

Protokol yang dipakai hanya mengikuti bukti yang sudah diuji di
tests/test_vigi_speaker.py: request bergaya RTSP/1.0 dengan method
MULTITRANS ke rtsp://<ip>/multitrans, body JSON, authentikasi Digest
(qop=auth, MD5 atau SHA-256) atas challenge kamera.

CATATAN VERIFIKASI PERANGKAT: pembentukan sesi talk dan pengiriman
audio frames belum diverifikasi pada kamera fisik saat pengembangan
ini; format payload audio setelah sesi dibuka harus dikonfirmasi
dengan tes berjenjang di tests/ sebelum dipakai produksi.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import socket
import time
from urllib.parse import unquote, urlparse

ALGORITHMS = ("MD5", "SHA256")
DEFAULT_RTSP_PORT = 554


class VigiTalkError(RuntimeError):
    """Kegagalan sesi talk VIGI tanpa mengekspos kredensial."""


def digest_hash(value: str, algorithm: str) -> str:
    normalized = algorithm.upper().replace("-", "")

    if normalized not in ALGORITHMS:
        raise ValueError(f"Algoritma tidak didukung: {algorithm}")

    return hashlib.new(normalized, value.encode("utf-8")).hexdigest()


def parse_digest_challenge(header: str) -> dict | None:
    """Ambil challenge Digest dari header WWW-Authenticate."""
    line = next(
        (
            item for item in header.splitlines()
            if item.lower().startswith("www-authenticate:")
        ),
        None,
    )

    if not line:
        return None

    value = line.split(":", 1)[1].strip()
    fields = dict(re.findall(r'(\w+)="([^"]*)"', value))

    algorithm = re.search(
        r"\balgorithm\s*=\s*\"?([A-Za-z0-9_-]+)\"?",
        value,
        re.I,
    )
    qop = re.search(r'\bqop\s*=\s*"([^"]+)"', value, re.I)

    fields["algorithm"] = algorithm.group(1) if algorithm else "MD5"
    fields["qop"] = qop.group(1) if qop else ""
    return fields


class VigiTalkClient:
    """Sesi talk ke VIGI C340 melalui Stream Interface (port 554)."""

    def __init__(
        self,
        host: str,
        username: str | None = None,
        password: str | None = None,
        port: int = DEFAULT_RTSP_PORT,
        timeout: float = 5.0,
        talk_mode: str = "half_duplex",
    ):
        if not host:
            raise ValueError("Alamat kamera tidak boleh kosong.")

        self.host = host
        self.port = int(port)
        self.username = username or ""
        self._password = password or ""
        self.timeout = float(timeout)
        self.talk_mode = talk_mode

        self._sock: socket.socket | None = None
        self._cseq = 0
        self.session_id: str | None = None

    # ------------------------------------------------------------------
    # Koneksi
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        if self._sock is not None:
            return

        try:
            self._sock = socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout,
            )
            self._sock.settimeout(self.timeout)
        except OSError as exc:
            self._sock = None

            # Jangan masukkan username/password ke laporan.
            raise VigiTalkError(
                f"Gagal tersambung ke {self.host}:{self.port}: "
                f"{type(exc).__name__}"
            ) from exc

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

        self._sock = None

    # ------------------------------------------------------------------
    # Request/response
    # ------------------------------------------------------------------

    def _request(self, params: dict, authorization: str | None = None):
        """Kirim satu request RTSP-style ke Stream Interface."""
        if self._sock is None:
            raise VigiTalkError("Koneksi belum dibuka.")

        self._cseq += 1
        uri = f"rtsp://{self.host}/multitrans"
        body = json.dumps(
            params, separators=(",", ":")
        ).encode("utf-8")

        lines = [
            f"MULTITRANS {uri} RTSP/1.0",
            f"CSeq: {self._cseq}",
            "Content-Type: application/json",
            f"Content-Length: {len(body)}",
        ]

        if authorization:
            lines.append(f"Authorization: {authorization}")

        packet = "\r\n".join(lines).encode("utf-8") + b"\r\n\r\n" + body

        try:
            self._sock.sendall(packet)
            return self._read_response()
        except socket.timeout as exc:
            raise VigiTalkError(
                "Timeout menunggu respons kamera."
            ) from exc
        except OSError as exc:
            raise VigiTalkError(
                f"Koneksi ke kamera terputus: {type(exc).__name__}"
            ) from exc

    def _read_response(self):
        try:
            data = bytearray()

            while b"\r\n\r\n" not in data:
                chunk = self._sock.recv(4096)

                if not chunk:
                    raise VigiTalkError(
                        "Koneksi ditutup kamera saat menunggu respons."
                    )

                data.extend(chunk)

                if len(data) > 65536:
                    raise VigiTalkError("Header respons terlalu besar.")

            split_at = data.find(b"\r\n\r\n")
            header = data[:split_at].decode("utf-8", "replace")
            body = bytearray(data[split_at + 4:])

            content_length = 0

            for line in header.splitlines():
                if line.lower().startswith("content-length:"):
                    content_length = int(
                        line.split(":", 1)[1].strip()
                    )

            while len(body) < content_length:
                chunk = self._sock.recv(
                    content_length - len(body)
                )

                if not chunk:
                    break

                body.extend(chunk)

            return header, bytes(body)
        except socket.timeout as exc:
            raise VigiTalkError(
                "Timeout membaca respons kamera."
            ) from exc
        except OSError as exc:
            raise VigiTalkError(
                f"Koneksi ke kamera terputus: {type(exc).__name__}"
            ) from exc

    # ------------------------------------------------------------------
    # Autentikasi Digest
    # ------------------------------------------------------------------

    def _make_authorization(self, challenge: dict, params: dict) -> str:
        """Buat header Authorization dari challenge kamera."""
        algorithm = challenge["algorithm"]
        realm = challenge.get("realm")
        nonce = challenge.get("nonce")

        qop_values = [
            item.strip().lower()
            for item in challenge.get("qop", "").split(",")
        ]

        if not realm or not nonce:
            raise VigiTalkError("Challenge tidak memuat realm/nonce.")

        if "auth" not in qop_values:
            raise VigiTalkError("Challenge tidak menawarkan qop=auth.")

        nc = "00000001"
        cnonce = secrets.token_hex(8)
        uri = f"rtsp://{self.host}/multitrans"
        method = "MULTITRANS"

        ha1 = digest_hash(f"{self.username}:{realm}:{self._password}", algorithm)
        ha2 = digest_hash(f"{method}:{uri}", algorithm)
        response_hash = digest_hash(
            f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}",
            algorithm,
        )

        # Header dibangun tanpa menampilkannya ke log.
        return (
            f"Digest username=\"{self.username}\", "
            f"realm=\"{realm}\", "
            f"nonce=\"{nonce}\", "
            f"uri=\"{uri}\", "
            f"algorithm={algorithm}, "
            f"qop=auth, "
            f"nc={nc}, "
            f"cnonce=\"{cnonce}\", "
            f"response=\"{response_hash}\""
        )

    def _prepare_payload(self, params: dict) -> dict:
        """Bungkus params menjadi format body MULTITRANS yang sudah terbukti."""
        return {
            "type": "request",
            "seq": str(int(time.time() * 1000) % 100000 or 1),
            "params": params,
        }

    # ------------------------------------------------------------------
    # Sesi talk
    # ------------------------------------------------------------------

    def open_session(self) -> str:
        """
        Autentikasi Digest lalu minta sesi talk.

        Mengembalikan session_id dari kamera bila sesi berhasil.
        """
        params = {
            "method": "get",
            "talk": {"mode": self.talk_mode},
        }
        payload = self._prepare_payload(params)
        self._connect()
        self._cseq = 0

        try:
            header, body = self._request(payload)

            if "401" not in header.splitlines()[0]:
                raise VigiTalkError(
                    "Kamera tidak memberi challenge Digest "
                    f"({header.splitlines()[0] if header else 'respons kosong'})."
                )

            challenge = parse_digest_challenge(header)

            if not challenge:
                raise VigiTalkError(
                    "Challenge Digest tidak ditemukan di respons."
                )

            header, body = self._request(
                payload,
                self._make_authorization(challenge, payload),
            )
        except VigiTalkError:
            self._close()
            raise

        status_line = header.splitlines()[0] if header else ""

        if " 401" in status_line or " 403" in status_line:
            self._close()

            # Tidak menampilkan nonce/hash atau kredensial.
            raise VigiTalkError(
                "Autentikasi ditolak kamera "
                f"({status_line.strip()})."
            )

        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            self._close()
            raise VigiTalkError(
                "Body respons bukan JSON yang valid."
            ) from exc

        response_params = parsed.get("params", parsed)
        error_code = response_params.get("error_code")
        session_id = response_params.get("session_id")

        if error_code != 0 or session_id in (None, "", 0, "0"):
            self._close()
            raise VigiTalkError(
                f"Sesi talk gagal dibuat (error_code={error_code})."
            )

        self.session_id = str(session_id)
        return self.session_id

    def close_session(self) -> None:
        """Tutup sesi koneksi talk; aman dipanggil dua kali."""
        self._close()
        self.session_id = None

    def __enter__(self) -> "VigiTalkClient":
        self.open_session()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_session()


def credentials_from_rtsp_url(rtsp_url: str):
    """Ambil username/password ter-URL-encode dari URL RTSP."""
    if not rtsp_url:
        return None, None

    parsed = urlparse(rtsp_url)

    if not parsed.username:
        return None, None

    return unquote(parsed.username), unquote(parsed.password or "")
