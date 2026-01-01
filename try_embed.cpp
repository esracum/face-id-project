#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// Colab'da oluşturduğun 512 boyutlu embedding dizisini içeren dosya
#include "my_face.h" 

/* ========= MATEMATİKSEL YARDIMCILAR ========= */
inline void l2Normalize(float* v, int len = 512) {
    float norm = 0.0f;
    for (int i = 0; i < len; i++) norm += v[i] * v[i];
    norm = std::sqrt(norm) + 1e-6f;
    for (int i = 0; i < len; i++) v[i] /= norm;
}

float cosineSimilarity(const float* a, const float* b, int len = 512) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < len; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-6f);
}

int main() {
    try {
        /* ===== 1. KURULUMLAR ===== */
        cv::CascadeClassifier face_detector;
        if (!face_detector.load("/home/esocan/Desktop/face_samples/haarcascade_frontalcatface.xml")) {
            throw std::runtime_error("Haarcascade dosyasi bulunamadi!");
        }

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FaceID");
        const char* model_path = "/home/esocan/Desktop/face_samples/w600k_r50.onnx"; // İndirdiğin ArcFace modeli
        Ort::Session session(env, model_path, Ort::SessionOptions{});
        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        const char* input_names[] = { input_name.get() };
        const char* output_names[] = { output_name.get() };

        cv::VideoCapture cap(0);
        if (!cap.isOpened()) throw std::runtime_error("Kamera acilamadi!");

        /* ===== 2. DEĞİŞKENLER (Akıcılık İçin) ===== */
        cv::Mat frame;
        std::vector<cv::Rect> last_faces;
        std::string current_label = "Taratiliyor...";
        float current_score = 0.0f;
        cv::Scalar current_color = cv::Scalar(255, 255, 0); // Varsayılan Sarı
        int frame_counter = 0;

        std::cout << "Sistem Baslatildi. ESC ile cikin." << std::endl;

        while (cap.read(frame)) {
            frame_counter++;

            /* 3. OPTİMİZASYON: Yüz Tespitini 3 karede bir yap */
            if (frame_counter % 3 == 0) {
                cv::Mat gray, small_gray;
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                // Tespit hızını artırmak için görüntüyü yarı yarıya küçült
                cv::resize(gray, small_gray, cv::Size(), 0.5, 0.5);
                face_detector.detectMultiScale(small_gray, last_faces, 1.2, 5, 0, cv::Size(60, 60));
                
                // Koordinatları orijinal boyuta geri çek
                for (auto& r : last_faces) { r.x *= 2; r.y *= 2; r.width *= 2; r.height *= 2; }
            }

            /* Yüz Tanımayı 3 karede bir yap */
            if (frame_counter % 3 == 0 && !last_faces.empty()) {
                // Sadece kameraya en yakın veya ilk bulunan yüzü işle
                cv::Rect roi = last_faces[0] & cv::Rect(0, 0, frame.cols, frame.rows);
                cv::Mat face_roi = frame(roi);

                // Ön İşleme (ArcFace NCHW Standardı)
                cv::Mat resized, face_rgb;
                cv::resize(face_roi, resized, cv::Size(112, 112));
                cv::cvtColor(resized, face_rgb, cv::COLOR_BGR2RGB);
                face_rgb.convertTo(face_rgb, CV_32FC3, 1.0 / 127.5, -1.0); // [-1, 1] Normalizasyonu

                std::vector<float> input_tensor_values(1 * 3 * 112 * 112);
                std::vector<cv::Mat> channels(3);
                for (int i = 0; i < 3; ++i) {
                    channels[i] = cv::Mat(112, 112, CV_32FC1, &input_tensor_values[i * 112 * 112]);
                }
                cv::split(face_rgb, channels);

                // ONNX Inference
                std::array<int64_t, 4> input_shape{1, 3, 112, 112};
                auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

                auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
                float* emb = outputs[0].GetTensorMutableData<float>();
                l2Normalize(emb);

                current_score = cosineSimilarity(my_face_embedding, emb);
                
                // Karar ve Görsel Güncelleme
                if (current_score > 0.40f) { // Threshold: 0.40 - 0.50 arası test et
                    current_label = "ESCOCAN";
                    current_color = cv::Scalar(0, 255, 0); // Yeşil
                } else {
                    current_label = "BILINMEYEN";
                    current_color = cv::Scalar(0, 0, 255); // Kırmızı
                }
            }

            /* 5. GÖRSELLEŞTİRME */
            for (const auto& rect : last_faces) {
                cv::rectangle(frame, rect, current_color, 2);
                std::string info = current_label + " " + std::to_string(current_score).substr(0,4);
                cv::putText(frame, info, {rect.x, rect.y - 10}, cv::FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2);
            }

            cv::imshow("Fluid FaceID System", frame);
            if (cv::waitKey(1) == 27) break;
        }
    } catch (const std::exception& e) {
        std::cerr << "HATA: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
