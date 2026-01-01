import cv2
import numpy as np
import threading
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# ---------------------------
# 1. Asenkron Kamera Okuma
# ---------------------------
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        # Kamera çözünürlüğünü sabitlemek istersen:
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ---------------------------
# 2. Yardımcı Fonksiyonlar
# ---------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ---------------------------
# 3. Model ve Ayarlar
# ---------------------------
app = FaceAnalysis(name="buffalo_s") 

# det_size: 320x320 işlem yükünü hafifletir.
app.prepare(ctx_id=0, det_size=(320, 320))

registered_embedding = np.load("ref_embedding.npy")
registered_embedding = registered_embedding / norm(registered_embedding)

# Global Durumlar
faces = []
is_processing = False
THRESHOLD = 0.45 
PROC_SCALE = 0.5 

def process_faces(frame_to_process):
    global faces, is_processing
    # Arka planda analiz
    rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)
    is_processing = False

# ---------------------------
# 4. Ana Döngü
# ---------------------------
vs = VideoStream(src=0).start()
print("Sistem Calisiyor.")

while True:
    frame = vs.read()
    if frame is None: continue

    # A. Modeli Tetikle (Boştaysa)
    if not is_processing:
        is_processing = True
        # Görüntüyü küçültüp işleme gönder (Hız dopingi)
        small_frame = cv2.resize(frame, (0, 0), fx=PROC_SCALE, fy=PROC_SCALE)
        t = threading.Thread(target=process_faces, args=(small_frame,), daemon=True)
        t.start()

    # B. Mevcut Sonuçları Çiz
    for face in faces:
        # Koordinatları tekrar orijinal boyuta çek
        bbox = (face.bbox / PROC_SCALE).astype(int)
        
        emb = face.embedding
        emb = emb / norm(emb)
        similarity = cosine_similarity(registered_embedding, emb)

        if similarity > THRESHOLD:
            label = f"ONAYLANDI ({similarity:.2f})"
            color = (0, 255, 0)
        else:
            label = f"YABANCI"
            color = (0, 0, 255)

        # Görselleştirme
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # C. Ekrana Bas
    cv2.imshow("Real-Time Face ID", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

vs.stop()
cv2.destroyAllWindows()