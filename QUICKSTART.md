# Hızlı Başlangıç Kılavuzu

**Orman Yangını Tahmin Sistemi** - Tüm ML pipeline'ını 2 adımda çalıştırın!

---

## 1. Bağımlılıkları Kurma (İlk Kez)

Terminal'de:

```bash
# Virtual environment aktif et
.\venv\Scripts\activate

# Paketleri yükle
pip install -r requirements.txt
```

**Gerekli paketler:**
- `tensorflow>=2.12.0` - Derin öğrenme modeli
- `pandas`, `numpy`, `scikit-learn` - Veri işleme
- `folium` - İnteraktif haritalar
- `matplotlib` - Grafik görselleştirme

---

## 2. Tam Pipeline'ı Çalıştır

Terminal'de:

```bash
python src/main.py
```

**Bu 4 adımı otomatik olarak çalıştırır:**

| Adım | İşlem | Çıktı |
|------|-------|-------|
| 1 | Veri ön işleme | `data/processed/processed_data.csv` |
| 2 | Model eğitimi | `models/yangin_model.keras` |
| 3 | Tahmin yapma | `outputs/predictions.csv` |
| 4 | Harita oluşturma | `outputs/maps/fire_risk_map.html` |

**Beklenen süre:** ~5-10 dakika (maliyete bağlı olarak)

---

## Çıktılar

Pipeline tamamlandığında bu dosyalar oluşturulur:

```
data/processed/
├── processed_data.csv              # Temizlenmiş eğitim verisi (23K satır)

models/
├── yangin_model.keras              # Eğitilmiş TensorFlow modeli
└── scaler.pkl                      # Veri normalizasyon aracı

outputs/
├── predictions.csv                 # Tüm lokasyonlar + risk skorları
├── test_predictions.csv            # Test seti tahminleri + doğruluk
├── reports/
│   ├── training_history.png        # Eğitim grafiği (4 panel)
│   ├── feature_importance.csv      # Özellik önem sırası
│   ├── classification_report.txt   # Model metrikleri
│   └── data_quality_report.txt     # Veri kalite raporu
└── maps/
    └── fire_risk_map.html          # İnteraktif harita (32 MB)
```

---

## İnteraktif Harita Görüntüle

Eğitim tamamlandıktan sonra:

```bash
# Tarayıcıda aç
start outputs/maps/fire_risk_map.html
```

**Harita özellikleri:**
- ~5,000 CokmuzMarker ile optimized (performans için sampling)
- Kırmızı = Yüksek risk, Turuncu = Orta, Yeşil = Düşük
- Her noktaya tıklarken risk skoru (basit popup)
- Isı haritası katmanı (heatmap) - tüm 23K veri
- **Marker Clustering** - yakın noktaları gruplandırır
- **Test Doğruluk Katmanı** - doğru/yanlış çerçeve
- **Hızmı:** 32 MB → 6 MB harita, tarayıcıda anında yüklenir

---

## Yapılandırmayı Özelleştir

### Eğitim parametreleri

`src/config.py` içinde değiştir:

```python
# Model
MODEL_CONFIG = {
    'layers': [64, 32, 16, 1],      # ← Katman boyutları
    'dropout': [0.3, 0.2],            # ← Regularizasyon
    'batch_norm': True,               # ← Batch normalization
}

# Eğitim
EPOCHS = 100                          # ← Eğitim turları
BATCH_SIZE = 32                       # ← Batch büyüklüğü
LEARNING_RATE = 0.001                # ← Öğrenme hızı

# Veri
TEST_SIZE = 0.2                       # ← Test seti oranı
```

### Harita Optimizasyonu

`src/config.py` içinde harita performansını kontrol et:

```python
# Harita Optimizasyonu
MAX_MARKERS_ON_MAP = 5000          # Gösterilecek max nokta (daha az = daha hızlı)
USE_MARKER_CLUSTERING = True        # Cluster ekle (önerilen, hızlı)
SIMPLIFIED_POPUPS = True            # Basit popup'lar (daha hızlı render)
HEATMAP_ONLY_MODE = False           # True = sadece ısı haritası (çok hızlı)
```

**Hızlandırma Tüyoları:**
- Yavaş tarayıcı? → `MAX_MARKERS_ON_MAP` değerini 3000'e düşür
- Hiç nokta istemiyor? → `MAX_MARKERS_ON_MAP = 0` (sadece heatmap)
- Isı haritası istemiyorsanız: Python kodlarında `add_heatmap=False` kullan

### Sadece spesifik adımları çalıştır

```python
# src/main.py içinde:

# Sadece tahmin yap (model zaten var)
python -c "from src.predict import predict_fire_risk; predict_fire_risk()"

# Sadece harita oluştur
python -c "from src.map_visualization import create_risk_map; create_risk_map()"
```

---

## Veri Formatı

Input CSV (`data/raw/yangin_model_verisi.csv`):

```
NDVI,bagil_nem,sicaklik,yagis,latitude,longitude,.geo,etiket,yangin_var
5234,65.3,298.15,2.5,37.71,37.72,"{...}",0,1
...
```

Taraflı özelliklere ve yüksek yangın riski için, bu veri modeli kendiliğinden öğrenecektir.

---

## Sorun Giderme

| Hata | Çözüm |
|------|-------|
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow` |
| `FileNotFoundError: data/raw/...csv` | CSV dosyasının `data/raw/` klasöründe olduğunu kontrol et |
| `CUDA error` (GPU sorunları) | TensorFlow CPU sürümünü kullan: `pip install tensorflow-cpu` |
| Çok fazla RAM kullanılıyor | `BATCH_SIZE` değerini azalt (32 → 16) |

---

## Gelişmiş Kullanım

### Sadece preprocess
```bash
python -c "from src.preprocess import preprocess_data; preprocess_data()"
```

### Sadece train
```bash
python -c "from src.train import train_fire_prediction_model; train_fire_prediction_model()"
```

### Hızlı Harita (model varsa)
```bash
python src/main.py --quick
```
Eğitimi atlar, sadece tahmin + harita oluşturur.

### Kendi verilerinle tahmin
```python
from src.predict import predict_fire_risk
from src.map_visualization import create_risk_map

# Yeni CSV ile tahmin yap
predict_fire_risk('/path/to/your/data.csv')

# Harita oluştur (optimize)
create_risk_map(
    add_clusters=True,          # Marker cluster
    max_markers=3000,           # Daha hızlı (5000 default)
    simplified_popups=True      # Basit popup
)
```

---

## Proje Yapısı

```
ignis/
├── src/
│   ├── main.py                 # Ana pipeline
│   ├── config.py               # Yapılandırma
│   ├── preprocess.py           # Veri temizleme
│   ├── train.py                # Model eğitimi
│   ├── predict.py              # Tahlı
│   ├── map_visualization.py    # Folium harita
│   ├── utils.py                # Yardımcı fonksiyonlar
│   └── gee_data_processor.py   # GEE entegrasyonu (isteğe bağlı)
│
├── data/
│   ├── raw/                    # Orijinal CSV'ler
│   └── processed/              # Temizlenmiş veri
│
├── models/                     # Kaydedilen ML modeller
├── outputs/                    # Tahminler ve haritalar
├── requirements.txt            # Python paketleri
└── README.md                   # Detaylı dokümantasyon
```

---

## Komut Özeti

```bash
# İlk kurulum
.\venv\Scripts\activate
pip install -r requirements.txt

# Tüm pipeline'ı çalıştır
python src/main.py

# Harita aç
start outputs/maps/fire_risk_map.html

# Sonuçları kontrol et
type outputs/predictions.csv | Select-Object -First 5
```

---

**Sorular? [README.md](./README.md) detaylı dokümantasyon içerir!**
