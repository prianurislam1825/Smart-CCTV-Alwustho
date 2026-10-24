
from __future__ import annotations

import hashlib
import json
import ssl
import getpass
import urllib.request
import urllib.error

IP = "192.168.20.97"
USERNAME = "admin"
BASE_URL = f"https://{IP}:20443"

TLS_CONTEXT = ssl._create_unverified_context()


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        context=TLS_CONTEXT,
        timeout=10,
    ) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def authenticate(password: str) -> str | None:
    challenge = post_json(
        BASE_URL,
        {"method": "doAuth", "params": None},
    )

    auth = challenge.get("authenticate")

    if not auth:
        print("Challenge autentikasi tidak ditemukan.")
        print("Respons:", challenge)
        return None

    if auth.get("algorithm", "").upper() != "SHA-256":
        print("Algoritma tidak didukung:", auth.get("algorithm"))
        return None

    realm = auth["realm"]
    nonce = auth["nonce"]
    method = auth["method"]
    uri = auth["uri"]

    a1 = sha256(f"{USERNAME}:{realm}:{password}")
    a2 = sha256(f"{method}:{uri}")
    response_hash = sha256(f"{a1}:{nonce}:{a2}")

    result = post_json(
        BASE_URL,
        {
            "method": "doAuth",
            "params": {
                "nonce": nonce,
                "response": response_hash,
            },
        },
    )

    if result.get("errCode") != 0 or not result.get("stok"):
        print("Autentikasi gagal.")
        print("errCode:", result.get("errCode"))
        return None

    print("Autentikasi berhasil.")
    return str(result["stok"])


def test_stream_port(stok: str) -> None:
    # Mengikuti pola URL sesi yang digunakan OpenAPI VIGI.
    session_url = f"{BASE_URL}/stok={stok}"

    print("\nMenguji endpoint getStreamPort...")

    try:
        result = post_json(
            session_url,
            {"method": "getStreamPort"},
        )

        # Jangan menampilkan token sesi jika respons memuatnya.
        print("Respons API:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if result.get("errCode") == 0:
            print("\nPermintaan API berhasil.")
        else:
            print("\nAPI merespons, tetapi permintaan belum berhasil.")
            print("Periksa format endpoint dan parameter sesuai")
            print("dokumentasi OpenAPI untuk firmware kamera ini.")

    except urllib.error.HTTPError as exc:
        print("HTTP error:", exc.code)
        print(exc.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        print("Pengujian endpoint gagal:", type(exc).__name__, str(exc))


def main() -> None:
    print("=== VIGI OpenAPI Test ===")
    print("Kamera:", IP)
    print("Port API: 20443")

    password = getpass.getpass("Password CCTV: ")

    try:
        stok = authenticate(password)

        if stok:
            test_stream_port(stok)

    except urllib.error.HTTPError as exc:
        print("HTTP error:", exc.code)
        print(exc.read().decode("utf-8", errors="replace"))
    except urllib.error.URLError as exc:
        print("Koneksi gagal:", exc.reason)
    except Exception as exc:
        print("Terjadi kesalahan:", type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
