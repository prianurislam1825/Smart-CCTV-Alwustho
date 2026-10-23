import os

import cv2

RTSP_URL_ENV = "OFFICE_CAMERA_RTSP"

url = os.getenv(RTSP_URL_ENV)

if not url:
    print(f"Variabel environment {RTSP_URL_ENV} belum diset.")
    print("Salin .env.example menjadi .env lalu isi URL RTSP kamera.")
    raise SystemExit(1)

print("=== SMART CCTV - RTSP TEST ===")
print("Menghubungkan ke CCTV kantor...")

camera = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not camera.isOpened():
    print("GAGAL membuka RTSP.")
    print("Periksa jaringan, kredensial, dan path stream.")
    raise SystemExit(1)

print("RTSP berhasil dibuka!")
print("Tekan Q pada jendela video untuk keluar.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Frame gagal dibaca atau stream terputus.")
            break

        cv2.imshow("Smart CCTV - Live Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
