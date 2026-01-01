import cv2
import os
# -----------------------------
# AYARLAR
# -----------------------------
BASE_DIR = os.getcwd()
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_face", "me")

FPS_EXTRACT = 2
FACE_SIZE = (112, 112)
CROP_SCALE = 0.6   # merkezin %60'ını al

os.makedirs(OUTPUT_DIR, exist_ok=True)
saved_count = 0
# -----------------------------
# TÜM VİDEOLARI GEZ
# -----------------------------
for video_name in sorted(os.listdir(VIDEO_DIR)):
    if not video_name.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_name))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(video_fps / FPS_EXTRACT))

    frame_count = 0
    print(f" İşleniyor: {video_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            h, w, _ = frame.shape

            # Merkezden kırp
            cx, cy = w // 2, h // 2
            cw, ch = int(w * CROP_SCALE), int(h * CROP_SCALE)

            x1 = max(0, cx - cw // 2)
            y1 = max(0, cy - ch // 2)
            x2 = min(w, cx + cw // 2)
            y2 = min(h, cy + ch // 2)

            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, FACE_SIZE)

            save_path = os.path.join(
                OUTPUT_DIR, f"face_{saved_count:05d}.jpg"
            )
            cv2.imwrite(save_path, face)
            saved_count += 1

        frame_count += 1

    cap.release()

print(f"\n Toplam kaydedilen yüz sayisi: {saved_count}")
