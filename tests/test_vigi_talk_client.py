"""Unit test VigiTalkClient dengan mock jaringan (tanpa kamera nyata)."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from src.audio import alaw
from src.audio.vigi_talk_client import (
    VigiTalkClient,
    VigiTalkError,
    credentials_from_rtsp_url,
    digest_hash,
)


# ----------------------------------------------------------------------
# Helper respons kamera tiruan
# ----------------------------------------------------------------------

def rtsp_response(status, body=b"", www_auth=None):
    lines = [status, "CSeq: 1"]

    if www_auth:
        lines.append(f"WWW-Authenticate: {www_auth}")

    lines.append(f"Content-Length: {len(body)}")
    header = "\r\n".join(lines)
    return header.encode("utf-8") + b"\r\n\r\n" + body


CHALLENGE = (
    'Digest realm="TP-Link Camera", '
    'nonce="abc123", '
    'algorithm=MD5, '
    'qop="auth"'
)

SUCCESS_BODY = (
    b'{"type":"response","seq":"1","params":'
    b'{"error_code":0,"session_id":"8423"}}'
)


class FakeSocket:
    """Socket tiruan yang memutar urutan respons yang diprogram."""

    def __init__(self, responses, timeout_on=None):
        self.sent = []
        self.responses = list(responses)
        self.index = 0
        self.timeout_on = set(timeout_on or [])
        self.closed = False

    def settimeout(self, value):
        pass

    def sendall(self, data):
        self.sent.append(data)

    def recv(self, size):
        if self.index in self.timeout_on:
            raise socket.timeout()

        if self.index >= len(self.responses):
            raise socket.timeout()

        response = self.responses[self.index]
        self.index += 1
        return response

    def close(self):
        self.closed = True


def build_client():
    return VigiTalkClient(
        host="10.0.0.9",
        username="user",
        password="secret",
        port=554,
        timeout=1.0,
    )


# ----------------------------------------------------------------------
# Autentikasi gagal
# ----------------------------------------------------------------------

def test_auth_rejected_raises_and_closes():
    client = build_client()
    fake = FakeSocket([
        rtsp_response("RTSP/1.0 401 Unauthorized", www_auth=CHALLENGE),
        rtsp_response("RTSP/1.0 401 Unauthorized", www_auth=CHALLENGE),
    ])

    with patch("socket.create_connection", return_value=fake):
        with pytest.raises(VigiTalkError) as excinfo:
            client.open_session()

    assert "otolak" in str(excinfo.value).lower() or "401" in str(excinfo.value)
    assert client.session_id is None
    assert fake.closed


def test_missing_digest_challenge_raises():
    client = build_client()
    fake = FakeSocket([rtsp_response("RTSP/1.0 401 Unauthorized")])

    with patch("socket.create_connection", return_value=fake):
        with pytest.raises(VigiTalkError, match="Digest"):
            client.open_session()

    assert fake.closed


# ----------------------------------------------------------------------
# Timeout
# ----------------------------------------------------------------------

def test_timeout_raises_wrapped_error():
    client = build_client()
    fake = FakeSocket(
        [rtsp_response("RTSP/1.0 401 Unauthorized", www_auth=CHALLENGE)],
        timeout_on={1},
    )

    with patch("socket.create_connection", return_value=fake):
        with pytest.raises(VigiTalkError, match="[Tt]imeout"):
            client.open_session()

    assert client.session_id is None


def test_connection_refused_raises_wrapped_error():
    client = build_client()

    with patch(
        "socket.create_connection",
        side_effect=ConnectionRefusedError(),
    ):
        with pytest.raises(VigiTalkError, match="Gagal tersambung"):
            client.open_session()


# ----------------------------------------------------------------------
# Sesi berhasil dibuat -> session_id tersimpan
# ----------------------------------------------------------------------

def test_successful_session_open():
    client = build_client()
    fake = FakeSocket([
        rtsp_response("RTSP/1.0 401 Unauthorized", www_auth=CHALLENGE),
        rtsp_response("RTSP/1.0 200 OK", body=SUCCESS_BODY),
    ])

    with patch("socket.create_connection", return_value=fake):
        session_id = client.open_session()

    assert session_id == "8423"
    assert client.session_id == "8423"

    assert len(fake.sent) == 2

    # Header kedua memuat Authorization, body sama lengkap.
    second = fake.sent[1].decode("utf-8", "replace")
    assert "Authorization: Digest" in second
    assert "response=" in second

    # Kredensial tidak pernah dikirim dalam teks polos.
    assert "secret" not in second


def test_session_failure_error_code():
    client = build_client()
    fake = FakeSocket([
        rtsp_response("RTSP/1.0 401 Unauthorized", www_auth=CHALLENGE),
        rtsp_response(
            "RTSP/1.0 200 OK",
            body=b'{"params":{"error_code":-2}}',
        ),
    ])

    with patch("socket.create_connection", return_value=fake):
        with pytest.raises(VigiTalkError, match="error_code"):
            client.open_session()

    assert client.session_id is None


def test_invalid_json_body_raises():
    client = build_client()
    fake = FakeSocket([
        rtsp_response("RTSP/1.0 401 Unauthorized", www_auth=CHALLENGE),
        rtsp_response("RTSP/1.0 200 OK", body=b"bukan json"),
    ])

    with patch("socket.create_connection", return_value=fake):
        with pytest.raises(VigiTalkError, match="JSON"):
            client.open_session()


def test_unsupported_hash_algorithm():
    with pytest.raises(ValueError):
        digest_hash("x", "SHAKE256")


# ----------------------------------------------------------------------
# Kredensial dari URL RTSP
# ----------------------------------------------------------------------

def test_credentials_from_rtsp_url():
    username, password = credentials_from_rtsp_url(
        "rtsp://ad%40min:p%40ss@10.0.0.9:554/stream1"
    )

    assert username == "ad@min"
    assert password == "p@ss"


def test_credentials_missing_returns_none():
    assert credentials_from_rtsp_url("rtsp://10.0.0.9:554/stream1") == (
        None,
        None,
    )


def test_empty_host_rejected():
    with pytest.raises(ValueError):
        VigiTalkClient(host="")


# ----------------------------------------------------------------------
# 14. Framing audio A-law
# ----------------------------------------------------------------------

def test_alaw_framing_pads_last_frame():
    payload = bytes(range(100))  # 100 byte -> 1 frame + padding

    frames = alaw.frame_alaw(payload, bytes_per_frame=160)

    assert len(frames) == 1
    assert len(frames[0]) == 160
    assert frames[0][:100] == payload


def test_alaw_framing_multiple_frames():
    payload = b"\x00" * 480
    frames = alaw.frame_alaw(payload, bytes_per_frame=160)

    assert len(frames) == 3
    assert all(len(frame) == 160 for frame in frames)
