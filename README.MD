# Yüz Tanıma Sistemi (Face Recognition System)

Bu proje, videolardan veri seti hazırlama, Google Colab üzerinde öznitelik (embedding) çıkarma ve elde edilen verileri hem **Python** hem de **C++** ortamlarında koşturma sürecini kapsar.

## Sistem Bileşenleri ve Dosyalar

| Dosya Adı | Açıklama |
| :--- | :--- |
| `extract_video.py` | Seçilen videolardan kareler yakalayarak veri seti oluşturur. |
| `face_detection.ipynb` | Colab ortamında yüz tespiti yapar ve embedding vektörlerini üretir. |
| `ref_embedding.npy` | Python uygulamasında yüz tespiti için gerekli olan referans kaynak kod. |
| `my_face.h` | C++ uygulaması için gerekli olan header formatındaki yüz tanıma embeddingi|
| `w600k_r50.onnx` | Yüz tanıma için kullanılan ana model dosyası. |

---

## Kurulum ve Çalıştırma

### 1. Modelin Temini
Sistemin çalışması için `w600k_r50.onnx` model dosyasını aşağıdaki bağlantıdan indirip proje ana dizinine yerleştirin:
[InsightFace ONNX Model Linki](https://huggingface.co/public-data/insightface/blob/main/models/buffalo_l/w600k_r50.onnx)

### 2. Veri Seti ve Embedding Hazırlığı
1.  `extract_video.py` dosyasını çalıştırarak videolarınızdan yüzleri ayıklayın. Embeddinge uygun hale getirin.
2.  `face_detection.ipynb` dosyasını Google Colab'da açın ve ayıklanan yüzleri buraya yükleyin.
3.  İşlem sonunda oluşan;
    * Python için: `ref_embedding.npy`
    * C++ için: `my_face.h` dosyalarını indirin.

### 3. C++ Projesini Derleme (Linux/Ubuntu)
Uygulamayı derlemek için terminali açın ve aşağıdaki komutları sırasıyla takip edin:

```bash
# Proje dizinine gidin
cd ~/Desktop/face_samples 

# Eğer daha önceden derleme yaptıysanız temizleyin ve yeni build klasörü oluşturun
rm -rf build && mkdir build && cd build

# CMake yapılandırmasını kurun ve derleyin
cmake ..
make -j$(nproc)

# Uygulamayı çalıştırın
./faceid

## !!! Dikkat Edilmesi Gerekenler !!!
### Python Kullanıcıları: ref_embedding.npy dosyasının script ile aynı dizinde olduğundan emin olun.
### insightface kütüphanesi 3.10 üzeri python sürümünde hata verebilir.
### C++ Kullanıcıları: my_face.h dosyasını projeye dahil ettikten sonra make komutunu tekrar çalıştırmayı unutmayın.
### Dosya Yolları: Kod içerisindeki .onnx model yolunun, dosyayı indirdiğiniz konumla eşleştiğini kontrol edin.
