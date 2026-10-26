
import getpass
import hashlib
import json
import re
import secrets
import socket

CAMERA_IP = "192.168.20.152"
CAMERA_PORT = 554
TIMEOUT = 5

REQUEST_URI = f"rtsp://{CAMERA_IP}/multitrans"
REQUEST_METHOD = "MULTITRANS"


def digest_hash(value, algorithm):
    normalized = algorithm.upper().replace("-", "")
    if normalized not in ("MD5", "SHA256"):
        raise ValueError(f"Algoritma tidak didukung: {algorithm}")
    return hashlib.new(normalized, value.encode("utf-8")).hexdigest()


def read_response(sock):
    data = bytearray()

    while b"\r\n\r\n" not in data:
        chunk = sock.recv(1)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > 65536:
            raise RuntimeError("Header respons terlalu besar.")

    split_at = data.find(b"\r\n\r\n")
    if split_at < 0:
        return data.decode("utf-8", "replace"), b""

    header = data[:split_at].decode("utf-8", "replace")
    body = bytearray(data[split_at + 4:])

    content_length = 0
    for line in header.splitlines():
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())

    while len(body) < content_length:
        chunk = sock.recv(content_length - len(body))
        if not chunk:
            break
        body.extend(chunk)

    return header, bytes(body)


def parse_digest(header):
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
        r'\balgorithm\s*=\s*"?([A-Za-z0-9_-]+)"?',
        value,
        re.I,
    )
    qop = re.search(r'\bqop\s*=\s*"([^"]+)"', value, re.I)

    fields["algorithm"] = algorithm.group(1) if algorithm else "MD5"
    fields["qop"] = qop.group(1) if qop else ""
    return fields


def make_authorization(username, password, challenge):
    algorithm = challenge["algorithm"]
    realm = challenge.get("realm")
    nonce = challenge.get("nonce")
    qop_values = [
        item.strip().lower()
        for item in challenge.get("qop", "").split(",")
    ]

    if not realm or not nonce:
        raise ValueError("Challenge tidak memuat realm/nonce.")
    if "auth" not in qop_values:
        raise ValueError("Challenge tidak menawarkan qop=auth.")

    nc = "00000001"
    cnonce = secrets.token_hex(8)

    ha1 = digest_hash(
        f"{username}:{realm}:{password}",
        algorithm,
    )
    ha2 = digest_hash(
        f"{REQUEST_METHOD}:{REQUEST_URI}",
        algorithm,
    )
    response_hash = digest_hash(
        f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}",
        algorithm,
    )

    return (
        f'Digest username="{username}", '
        f'realm="{realm}", '
        f'nonce="{nonce}", '
        f'uri="{REQUEST_URI}", '
        f'algorithm={algorithm}, '
        f'qop=auth, '
        f'nc={nc}, '
        f'cnonce="{cnonce}", '
        f'response="{response_hash}"'
    )


def send_request(sock, payload, authorization=None):
    body = json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")

    lines = [
        f"{REQUEST_METHOD} {REQUEST_URI} RTSP/1.0",
        "CSeq: 1",
        "Content-Type: application/json",
        f"Content-Length: {len(body)}",
    ]

    if authorization:
        lines.append(f"Authorization: {authorization}")

    packet = (
        "\r\n".join(lines) + "\r\n\r\n"
    ).encode("utf-8") + body

    sock.sendall(packet)
    return read_response(sock)


def show_response(header, body):
    print("\nStatus:", header.splitlines()[0] if header else "(kosong)")

    # Jangan tampilkan header Authorization, nonce, atau hash Digest.
    for line in header.splitlines():
        if line.lower().startswith("www-authenticate:"):
            challenge = parse_digest(header)
            if challenge:
                print("Digest algorithm:", challenge.get("algorithm"))
                print("Digest qop:", challenge.get("qop"))

    parsed = None
    if body:
        try:
            parsed = json.loads(body)
            print("Body:", json.dumps(parsed, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("Body bukan JSON yang valid.")

    return parsed


def main():
    print("=== VIGI Stream Interface Diagnostic ===")
    print(f"Kamera: {CAMERA_IP}:{CAMERA_PORT}")
    print("Tidak mengirim audio atau mengubah konfigurasi kamera.")

    username = input("Username akun RTSP: ").strip()
    password = getpass.getpass("Password akun RTSP: ")

    payload = {
        "type": "request",
        "seq": "1",
        "params": {
            "method": "get",
            "talk": {"mode": "half_duplex"},
        },
    }

    try:
        with socket.create_connection(
            (CAMERA_IP, CAMERA_PORT),
            timeout=TIMEOUT,
        ) as sock:
            sock.settimeout(TIMEOUT)

            print("\n[1] Request awal...")
            header, body = send_request(sock, payload)
            show_response(header, body)

            if "401" not in header:
                print("\nTidak menerima challenge 401; pengujian dihentikan.")
                return

            challenge = parse_digest(header)
            if not challenge:
                print("Challenge Digest tidak ditemukan.")
                return

            print("\n[2] Menghitung Digest dari challenge kamera...")
            authorization = make_authorization(
                username,
                password,
                challenge,
            )

            print("[3] Request autentikasi (CSeq tetap 1)...")
            header, body = send_request(sock, payload, authorization)
            result = show_response(header, body)

            if "401" in header:
                print(
                    "\nHASIL: Masih 401. Format CSeq bukan satu-satunya "
                    "penyebab yang mungkin."
                )
                print(
                    "Belum bisa memastikan apakah penyebabnya kredensial, "
                    "hak akses Stream Interface, atau variasi protokol."
                )
                return

            if not isinstance(result, dict):
                print("\nRespons belum cukup untuk memastikan hasil.")
                return

            params = result.get("params", result)
            error_code = params.get("error_code")
            session_id = params.get("session_id")

            print("\n=== Ringkasan ===")
            print("error_code:", error_code)
            print("session_id tersedia:", bool(session_id))

            if error_code == 0 and session_id not in (None, "", "0", 0):
                print("Sesi talk berhasil dibuat. Audio tetap tidak dikirim.")
            else:
                print("Sesi talk belum terkonfirmasi berhasil. Audio tidak dikirim.")

    except socket.timeout:
        print("Timeout saat menunggu respons kamera.")
    except OSError as exc:
        print("Kesalahan jaringan:", exc)
    except ValueError as exc:
        print("Kesalahan autentikasi:", exc)


if __name__ == "__main__":
    main()
