# 🔥 Google Earth Engine - Yangin Veri İşleme (VS Code)

Colab notebook'dan VS Code'a uyarlanmış tamamen lokal çalışan versiyon.

## 📋 İçindekiler
1. [Kurulum](#kurulum)
2. [Konfigürasyon](#konfigürasyon)
3. [Kullanım](#kullanım)
4. [Dosya Yapısı](#dosya-yapısı)
5. [Sorun Giderme](#sorun-giderme)

---

## 🚀 Kurulum

### 1. Bağımlılıkları Yükle

```bash
# Virtual environment'i etkinleştir
.\venv\Scripts\activate

# Paketleri yükle
pip install -r requirements.txt
```

### 2. Google Earth Engine'i Kurulum

Google Earth Engine API'sini kullanabilmek için kimlik doğrulaması yapmanız gerekir:

```bash
# GEE kimlik doğrulama (browser'da açılacak)
earthengine authenticate
```

Bu komut çalıştırıldıktan sonra:
- Browser'da Google hesabınızla giriş yapın
- Yetkilendirmeyi onaylayın
- Kimlik bilgileri lokal olarak kaydedilecek

---

## ⚙️ Konfigürasyon

### `src/gee_config.py`

Konfigürasyon dosyasında değiştirebileceğiniz ayarlar:

```python
# Proje ID (Google Cloud Project)
PROJECT_ID = 'ignisai-496207'

# Örnekleme parametreleri
SAMPLING = {
    'num_points': 1000,      # Kaç nokta örneklenecek
    'scale': 10000,          # Çözünürlük (meter)
    'seed': 42               # Tekrarlanabilirlik
}

# Tarih aralıkları
DATE_RANGES = {
    'full': {
        'start': '2019-06-01',
        'end': '2025-01-12',
    }
}
```

---

## 💻 Kullanım

### Temel Kullanım

```bash
# VS Code terminal'de
python src/gee_data_processor.py
```

### Beklenen Çıktı

```
============================================================
🔥 GOOGLE EARTH ENGINE - YANGIN VERİ İŞLEYİCİ
============================================================

✅ Google Earth Engine başarıyla bağlandı!

📅 Tarih aralığı: 2019-06-01 - 2025-01-12

🔥 Yangın verileri toplanıyor: 2019-06-01 - 2025-01-12
🌿 NDVI verileri toplanıyor...
🌡️ Sıcaklık verileri toplanıyor...
...
📥 Veriler çekiliyor...
✅ Veri kaydedildi: ../data/yangin_rastgele_verisi.csv
   Satır sayısı: 1000
   Sütun sayısı: 6
```

### Python'da Kullanım

```python
from src.gee_data_processor import (
    initialize_gee,
    get_fire_data,
    get_ndvi_data,
    sample_points,
    export_to_csv
)

# Başlat
initialize_gee()

# Yangın verilerini al
fire_data, turkiye = get_fire_data('2022-01-01', '2022-12-31')

# NDVI verilerini al
ndvi_data = get_ndvi_data('2022-01-01', '2022-12-31', turkiye)

# Vb...
```

---

## 📁 Dosya Yapısı

```
ignis/
├── src/
│   ├── gee_data_processor.py      # ⭐ Ana işleme dosyası
│   └── gee_config.py               # Konfigürasyon
│
├── data/
│   ├── yangin_model_verisi.csv     # Eğitim verisi
│   └── yangin_rastgele_verisi.csv  # İşlemci tarafından oluşturulan
│
├── outputs/                        # İşlem sonuçları
├── models/                         # Kaydedilmiş modeller
├── noteboks/
│   └── colab_notebook.ipynb        # Orijinal Colab kodu (referans)
│
├── ilkoglus.py                     # İlk test dosyası
├── requirements.txt                # Paket bağımlılıkları
└── README.md                       # Bu dosya
```

---

## 📊 Veri Kaynakları

İşleme komut dosyasında kullanılan veri kaynakları:

| Veri Tipi | Kaynak | Çözünürlük | Açıklama |
|-----------|:------:|:-----------:|----------|
| 🔥 Yangın | MODIS MCD64A1 | 500m | Yanık bölge haritaları |
| 🌿 NDVI | MODIS MOD13Q1 | 250m | Bitki örtüsü indeksi |
| 🌡️ Sıcaklık | ERA5 Daily | ~25km | Hava sıcaklığı |
| 💧 Çiğlenme | ERA5 Daily | ~25km | Çiğlenme noktası |
| 🌡️ LST | MODIS MOD11A1 | 1km | Arazi yüzey sıcaklığı |
| 🌧️ Yağış | CHIRPS Daily | 5km | Günlük yağış |

---

## 🛠️ Sorun Giderme

### ❌ "earthengine: command not found"

```bash
# Python paketi olarak GEE API'sini kurun
pip install earthengine-api
```

### ❌ "Authentication failed"

```bash
# Yeniden kimlik doğrulaması yapın
earthengine authenticate --clear-credentials
earthengine authenticate
```

### ❌ "Collection not available for this region"

- Tarih aralığını kontrol edin
- Veri kaynağının Türkiye'yi kapsadığını doğrulayın
- GEE Catalog'unda veri mevcudiyetini kontrol edin: https://developers.google.com/earth-engine/datasets

### ❌ "Memory error" - Çok data

```python
# gee_config.py'da örnekleme sayısını azaltın
SAMPLING = {
    'num_points': 500,  # 1000'den 500'e azalt
    'scale': 10000,
}
```

---

## 📝 Notlar

- ⏱️ İlk çalıştırma 5-10 dakika sürebilir (veri indiriliyor)
- 💾 Veriler `data/` klasörüne kaydediliyor
- 🔄 Tekrarlanabilir sonuçlar için `seed=42` kullanılıyor
- 🌍 Sadece Türkiye bölgesine filtre uygulanıyor

---

## 🔗 Kaynaklar

- [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
- [EE Python API](https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api)
- [GEE Veri Kataloğu](https://developers.google.com/earth-engine/datasets)

---

**Sorular? Issues?** Lütfen bildirin! 🚀
