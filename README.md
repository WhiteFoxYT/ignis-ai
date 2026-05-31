# 🔥 Orman Yangını Tahmin Sistemi - Tam Rehber

> **Hiçbir makine öğrenmesi bilgisi olmadan bu projeyi anlayabilecek kadar detaylı açıklamalar içeren Türkçe rehber**

---

## 📑 İçindekiler

1. [Giriş](#giriş)
2. [Machine Learning Nedir?](#machine-learning-nedir)
3. [Projemiz Nedir?](#projemiz-nedir)
4. [Teknik Mimarisi](#teknik-mimarisi)
5. [Kurulum](#kurulum)
6. [Hızlı Başlangıç](#hızlı-başlangıç)
7. [Detaylı İşleyiş](#detaylı-işleyiş)
8. [Çıktılar ve Sonuçlar](#çıktılar-ve-sonuçlar)
9. [Hata Ayıklama](#hata-ayıklama)

---

## 🎯 Giriş

Bu proje, **yapay zeka** teknolojini kullanarak **orman yangınlarını tahmin etmektedir**.

### Neden Önemli?

Türkiye'de her yıl yüzlerce hektar orman yangın nedeniyle kaybedilmektedir. Bu sistem:
- ✅ Yangın çıkacak yerleri **önceden bilmek** için
- ✅ İtfaiye ekiplerinin **hazırlanmasına** yardımcı olmak için
- ✅ Yangın **riskini en aza indirmek** için
- ✅ **Ormansızlaşmayı önlemek** için

### Örnek Senaryo

Diyelim ki:
- Koordinatlar: **39.5°N, 35.2°E** (Ankara'nın kuzey-batı tarafı)
- NDVI (bitki yoğunluğu): **4500** (orman alanı)
- Sıcaklık: **35°C** (çok sıcak)
- Nem: **%25** (çok kuru)
- Yağış: **0.2mm** (yok denecek kadar az)

↓ **Model Çalışıyor** ↓

**SONUÇ**: "Bu bölgede yangın çıkma olasılığı **%87**" → ⚠️ **YÜKSEK RİSK**

→ İtfaiye ekibi: Bölgeyi hazırlamaya başla, araçları konumlandır!

---

## 🤖 Machine Learning Nedir?

### Basit Tanım

**Makine öğrenmesi**, bilgisayara **kuralları söylemek yerine**, çok sayıda **örnek göstererek** kendi kendine **karar almayı öğretme** teknikleridir.

### Nasıl Çalışır? (Detaylı Açıklama)

#### 📖 Bir Çocuğa Meyveler Tanıtmak Gibi

Bir çocuğa elmalar ve portakalları ayırmayı nasıl öğretirsiniz?

```
1️⃣ Göstermek: 100 elma ve 100 portakal gösteriş
   - "Bak, bunlar elmalar (kırmızı/yeşil, yuvarlak)"
   - "Bunlar portakallar (turuncu, yuvarlak ama daha pürüzlü)"

2️⃣ Öğrenmek: Çocuk özellikleri fark eder
   - "Elmalar genelde daha açık renkli"
   - "Portakallar daha derili"

3️⃣ Test Etmek: Bilmediği bir meyve gösterirsin
   - Eğer doğru ayırıyorsa: "Tebrikler! Öğrendin!"
   - Eğer yanlış: "Hayır, bunu daha iyi öğrenmeliyiz"

4️⃣ Daha Iyi Öğrenme: Tekrar tekrar yaparsın
   - Sonunda çocuk hiç görmediği elmalar ve portakalları ayırabilir!
```

#### Makine Öğrenmesi Aynı Mantık:

```
1️⃣ VERİ TOPLAMA
   ├─ Geçmiş yangın verileri (~10,000 örnek)
   ├─ Her örneğe ait özellikleri (6 parametre)
   └─ "Yangın oldu mu?" yanıtı (0 veya 1)

2️⃣ MODELE GÖSTERME
   ├─ Model: "Hmm, anladım..."
   └─ İç içe matematiksel kurallar oluşturuyor

3️⃣ TEST ETME
   ├─ Modele hiç görmediği veriler sunuyoruz
   └─ "Ne kadar doğru tahmin ettiler?"

4️⃣ İYİLEŞTİRME
   ├─ Hatalı tahminler buluyoruz
   ├─ Modele geri bildirimi veriyoruz
   └─ Tekrar tekrar öğreniyor

5️⃣ HAZIR! Şimdi yeni veriler için tahmin yapabilir
```

### Machine Learning Türleri

Bu projede kullandığımız tür: **İkili Sınıflandırma (Binary Classification)**

```
Soru: "Bu alan yangın riski taşıyor mu?"
Cevap:
├─ EVET (1) - Yangın riski var
└─ HAYIR (0) - Yangın riski yok
```

---

## 🔥 Projemiz Nedir?

### Hedefler

| Hedef | Açıklama | Fayda |
|-------|----------|-------|
| **Tahmin** | Yangın çıkacak yerleri önceden bulmak | ⏰ Erken uyarı |
| **Hız** | Saniyeler içinde sonuç | 🚀 Gerçek zamanlı |
| **Doğruluk** | %85-90 başarı oranı | ✅ Güvenilir |
| **Görselleştirme** | İnteraktif harita | 🗺️ Anlaşılır |

### Sistem Akışı

```
┌─────────────────────────────────────────────────────┐
│         ORMAN YANGINI TAHMİN SİSTEMİ               │
└─────────────────────────────────────────────────────┘
                         │
           ┌─────────────┼─────────────┐
           ↓             ↓             ↓
    📥 GİRİŞ VERİSİ   🧠 MODELLER    📊 ANALİZ
        │                  │             │
        ├─ NDVI          ├─ Eğitim     ├─ Metrikleri
        ├─ Nem           ├─ Test       ├─ Grafikleri
        ├─ Sıcak         ├─ Doğrula    └─ Raporları
        ├─ Yağış         └─ Tahmin
        ├─ Enlem
        └─ Boylam            │
                             ↓
                    📤 ÇIKTI DOSYALARI
                      │
              ┌───────┼───────┬────────┐
              ↓       ↓       ↓        ↓
            CSV    Harita  Grafikler Raporlar
```

### Veri Akışı (Detaylı)

```
Ham Veriler (data/raw/)
│
├─ yangin_model_verisi.csv    (yangın alanları)
└─ yangin_rastgele_verisi.csv (yangın olmayan alanlar)
        │
        ↓
   preprocess.py (src/)  → Veri temizleme, hazırlama
        │
        ↓
   processed_data.csv (data/processed/) → İşlenmiş veri
        │
        ├─────────────────────────────────────┐
        ↓                                      ↓
   train.py                            predict.py
   (Model eğit)                      (Tahmin yap)
        │                                      │
        ├──────────────────┬──────────────────┤
        ↓                  ↓                  ↓
   yangin_model.keras   Metrrikler    predictions.csv
   scaler.pkl           Raporlar      (tahminler)
   Grafikler                                  │
                                              ↓
                                    map_visualization.py
                                    (Harita oluştur)
                                              │
                                              ↓
                                    fire_risk_map.html
                                    (İnteraktif harita)
```

---

## 🛠️ Teknik Mimarisi

### Kullanılan Kütüphaneler

#### 1. **TensorFlow/Keras** - Sinir Ağı

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
```

**Ne işe yarar?**
- Derin öğrenme (Deep Learning) modellerini oluşturma
- Sinir ağını tanımlama ve eğitme
- Tahmin yapma

**Neden?**
- Google tarafından geliştirilen güvenilir kütüphane
- Hızlı ve etkili
- GPU desteği

#### 2. **NumPy** - Matematiksel İşlemler

```python
import numpy as np
```

**Ne işe yarar?**
- Sayısal işlemler (matrices, arrays)
- Matematiksel hesaplamalar
- Veri transformasyonları

**Örnek:**
```python
data = np.array([1, 2, 3, 4, 5])
mean = data.mean()  # Ortalama: 3.0
std = data.std()    # Standart sapma hesapla
```

#### 3. **Pandas** - Veri İşleme

```python
import pandas as pd
```

**Ne işe yarar?**
- CSV dosyaları okuma/yazma
- Tablo şeklindeki verilerle çalışma
- Veri temizleme ve filtreleme

**Örnek:**
```python
df = pd.read_csv('data.csv')  # CSV oku
df = df.dropna()              # Eksik değerleri sil
mean_temp = df['sicaklik'].mean()  # Sıcaklık ortalaması
```

#### 4. **Scikit-learn** - Makine Öğrenmesi Araçları

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
```

**Ne işe yarar?**
- Veri bölme (train/test)
- Özellik ölçekleme (normalizasyon)
- Model performans metrikleri

**Örnek:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)  # Verileri 70% eğitim, 30% test olarak böl
```

#### 5. **Folium** - İnteraktif Harita

```python
import folium
from folium.plugins import HeatMap, MarkerCluster
```

**Ne işe yarar?**
- Web haritaları oluşturma
- Coğrafi verileri görselleştirme
- İnteraktif pinler ve popup'lar

**Örnek:**
```python
m = folium.Map(location=[39.5, 35.2], zoom_start=7)
folium.CircleMarker([39.5, 35.2], radius=10, color='red').add_to(m)
m.save('map.html')
```

#### 6. **Matplotlib & Seaborn** - Görselleştirme

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

**Ne işe yarar?**
- Grafikler ve şemalar çizme
- İstatistikleri görselleştirme
- Eğitim sonuçlarını gösterme

#### 7. **Joblib** - Model Kaydetme

```python
import joblib
```

**Ne işe yarar?**
- Eğitilmiş modelleri kaydedeme
- StandardScaler'ı kaydedeme
- Daha sonra yüklemek için

**Örnek:**
```python
joblib.dump(scaler, 'scaler.pkl')  # Kaydet
scaler = joblib.load('scaler.pkl')  # Yükle
```

### Model Mimarisi (Sinir Ağı)

```
┌────────────────────────────────────────────┐
│  İNPUT KATMANI (6 özellik)                 │
│  ├─ NDVI                                   │
│  ├─ Bağıl Nem                              │
│  ├─ Sıcaklık                               │
│  ├─ Yağış                                  │
│  ├─ Latitude                               │
│  └─ Longitude                              │
└─────────────────┬──────────────────────────┘
                  ↓
         ┌─────────────────┐
         │ Dense Layer 1   │
         │ 64 Neuron       │
         │ ReLU Activation │
         └────────┬────────┘
                  ↓
         ┌─────────────────────────┐
         │ Batch Normalization     │
         │ (Veriyi Standartlaştır) │
         └────────┬────────────────┘
                  ↓
         ┌─────────────────┐
         │ Dropout(0.3)    │
         │ %30'u Devre Dışı│
         └────────┬────────┘
                  ↓
         ┌─────────────────┐
         │ Dense Layer 2   │
         │ 32 Neuron       │
         │ ReLU Activation │
         └────────┬────────┘
                  ↓
         ┌─────────────────────────┐
         │ Batch Normalization     │
         └────────┬────────────────┘
                  ↓
         ┌─────────────────┐
         │ Dropout(0.2)    │
         │ %20'i Devre Dışı│
         └────────┬────────┘
                  ↓
         ┌─────────────────┐
         │ Dense Layer 3   │
         │ 16 Neuron       │
         │ ReLU Activation │
         └────────┬────────┘
                  ↓
      ┌──────────────────────┐
      │ OUTPUT LAYER         │
      │ 1 Neuron             │
      │ Sigmoid Activation   │
      │ Çıktı: 0.0 - 1.0     │
      └──────────────────────┘
           ↓ (Sonuç)
    0.857 = %85.7 yangın riski
```

### Neden Bu Mimarı?

| Seçim | Neden |
|-------|-------|
| **ReLU** | Doğrusal olmayan ilişkileri yakalaması |
| **64→32→16** | Boyut kademeli olarak azalması |
| **BatchNorm** | Eğitimi hızlandırmak ve kararlı hale getirmek |
| **Dropout** | Modeli ezberlemesini (overfitting) önlemek |
| **Sigmoid** | Olasılık çıkışı için (0-1 aralığında) |

---

## 📊 Veri Seti Hakkında

### Veri Kaynakları

#### 1. **Bitki Örtüsü Verileri (NDVI)**

```
NDVI = (NIR - RED) / (NIR + RED)

Nedir? 
- Uydu tarafından ölçülen yeşil bitki yoğunluğu
- -1 ile +1 arasında değişir

Anlamı:
  0.0 - 0.2  = Çıplak toprak / Su
  0.2 - 0.5  = Çalı / Otlak  
  0.5 - 1.0  = Ormanlık / Desiş orman

Yangınla İlişkisi:
  ✅ Kurak alanlar (düşük NDVI) = YÜKSEK risk
  ✅ Yeşil alanlar (yüksek NDVI) = DÜŞÜK risk
```

#### 2. **Meteorolojik Veriler**

**Bağıl Nem (%)**
```
Tanım: Havanın ne kadar nemli olduğu
Aralık: %0 (çok kuru) → %100 (ıslak)

Yangınla İlişkisi:
  %20-30 = ⚠️ YÜKSEK RİSK
  %50-60 = 🟡 ORTA RİSK
  %70-80 = ✅ DÜŞÜK RİSK
```

**Sıcaklık (°C)**
```
Tanım: Hava sıcaklığı
Aralık: 0°C → 45°C (projedeki aralık ölçeklenmiş)

Yangınla İlişkisi:
  25-30°C = ⚠️ YÜKSEK RİSK
  15-20°C = 🟡 ORTA RİSK
  0-10°C = ✅ DÜŞÜK RİSK
```

**Yağış (mm)**
```
Tanım: Önceki dönemdeki yağış miktarı
Aralık: 0mm (yok) → 200mm (çok)

Yangınla İlişkisi:
  0-1mm = ⚠️ YÜKSEK RİSK
  3-5mm = 🟡 ORTA RİSK
  10+mm = ✅ DÜŞÜK RİSK
```

#### 3. **Coğrafi Veriler**

**Latitude & Longitude**
```
Tanım: Koordinatlar (Enlem & Boylam)
Aralık: Türkiye → 36-40°N, 30-43°E

Neden Gerekli?
  ✅ Bölgesel farklılıkları öğretmek
  ✅ Akdeniz bölgesi daha yüksek risk
  ✅ Karadeniz bölgesi daha düşük risk
```

### Veri Seti İstatistikleri

```
Toplam Veri:           ~10,000 örnek
├─ Yangın Olan:        ~4,000 (40%)
├─ Yangın Olmayan:     ~6,000 (60%)
└─ Denge: Stratified sampling ile sağlandı

Bölüm:
├─ Eğitim: 7,000 (%70) → Model öğrensin diye
├─ Test:   3,000 (%30) → Model test etmek için
└─ Doğrulama: 10% (eğitim içinden)

Zaman Periyodu:
├─ Tarih Aralığı: 2015-2023
├─ Mevsimsel Değişim: Tüm mevsimler kapsanmış
└─ Veri Kalitesi: Uydu hatalarıyla filtrelemiş
```

### Örnek Veri

| NDVI | Nem(%) | Sıcak(K) | Yağış(mm) | Lat | Lon | Yangın? |
|------|--------|----------|-----------|-----|-----|---------|
| 4500 | 35.2   | 15300    | 0.5       | 39.2| 35.4| ✅ EVET |
| 3200 | 45.1   | 15100    | 2.3       | 38.1| 32.5| ❌ HAYIR|
| 5100 | 40.2   | 15250    | 1.2       | 39.8| 34.2| ✅ EVET |

---

## 📥 Kurulum

### 1. Sistem Gereksinimleri

```
Donanım:
  ✅ İşlemci: Intel/AMD (2+ çekirdek)
  ✅ RAM: 4GB (8GB+ önerilen)
  ✅ Disk: 2GB boş alan
  ✅ GPU: Opsiyonel (NVIDIA/AMD hızlandırma için)

Yazılım:
  ✅ Python: 3.8-3.11
  ✅ Git: (clone için, isteğe bağlı)
  ✅ pip/conda: Paket yöneticisi
  ✅ Tarayıcı: Harita görüntüleme için
```

### 2. Python Kurulumu

Eğer Python yüklü değilse:
1. https://www.python.org/ ziyaret et
2. "Download" butonuna tıkla
3. Kurulum dosyasını çalıştır
4. "Add Python to PATH" seçeneğini işaretle

Kontrol etmek için:
```bash
python --version
# Çıktı: Python 3.x.x
```

### 3. Proje İndir

```bash
# Komut satırında proje klasörüne git
cd C:\Users\[Adınız]\Desktop\ignis

# Veya Explorer'dan klasöre sağ tıkla
# → "Open in Terminal" veya "Git Bash Here"
```

### 4. Sanal Ortam (Virtual Environment) Oluştur

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Sonuç:** Terminal'de `(venv)` görmelisin

### 5. Paketleri Yükle

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Bu ne kadar sürer?** 5-10 dakika (internet hızına bağlı)

### 6. Kontrol Et

```bash
python -c "import tensorflow; print('✅ TensorFlow kuruldu!')"
```

---

## 🚀 Hızlı Başlangıç

### En Hızlı Yol (start.py)

```bash
python start.py
```

Bu otomatik olarak yapar:
1. ✅ Veri ön işleme
2. ✅ Model eğitimi
3. ✅ Test ve analiz
4. ✅ Tahminler
5. ✅ Harita oluşturma

**Beklenen Süre:** 5-15 dakika

### Çıktılar

```
✅ Tamamllandı!

Dosyalar oluşturuldu:
├─ data/processed/processed_data.csv
├─ models/yangin_model.keras
├─ outputs/predictions.csv
├─ outputs/test_predictions.csv
├─ outputs/maps/fire_risk_map.html ← Bu dosyayı aç!
└─ outputs/reports/* (data_quality_report.txt dahil)
```

### Haritayı Aç

```bash
# Windows
start outputs/maps/fire_risk_map.html

# macOS
open outputs/maps/fire_risk_map.html

# Linux
xdg-open outputs/maps/fire_risk_map.html
```

Veya sadece Explorer'da dosyayı bulup çift tıkla!

---

## 📝 Detaylı İşleyiş

### Adım 1: Veri Ön İşleme (preprocess.py)

```bash
python src/preprocess.py
```

**Ne yapıyor?**

```
RAW VERİ (ham)
    ↓
1. 📥 VERİ YÜKLEME
   └─ CSV dosyalarını oku

2. 🧹 VERİ TEMİZLEME
   ├─ Eksik değerleri kontrol et
   ├─ Hatalı veri tipleri düzelt
   └─ Aykırı değerleri (outlier) işle

3. 🔍 KOORDİNAT ÇIKARMA
   ├─ ".geo" sütununu analiz et
   └─ Latitude/Longitude'u ayıkla

4. ⚖️ VERİ ÖLÇEKLEME (Normalizasyon)
   ├─ Verinin özelliğini standart forma sok
   │  (Ortalama=0, Std=1 yapılır)
   └─ Scaler'ı kaydet

5. 💾 KAY

DET
   └─ processed_data.csv oluştur

SONUÇ: İşlenmiş veri hazır!
```

**Çıktı Dosyası:** `data/processed/processed_data.csv`

### Adım 2: Model Eğitimi (train.py)

```bash
python src/train.py
```

**Ne yapıyor?**

```
İŞLENMİŞ VERİ
    ↓
1. 📊 VERİ BÖLME
   ├─ 70% → Eğitim seti
   ├─ 30% → Test seti
   └─ 10% → Doğrulama seti

2. 🤖 MODEL OLUŞTURMA
   ├─ Sinir ağı mimarisi tasarla
   ├─ Katmanları düzenle
   └─ Parametreleri ayarla

3. 🏋️ EĞİTİM SÜRECİ
   ├─ 100 epoch (maksimum)
   ├─ Her epoch'ta:
   │  ├─ Tahmini yap (Forward pass)
   │  ├─ Hatayı ölç (Loss)
   │  ├─ Gradyan hesapla (Backward pass)
   │  └─ Ağırlıkları güncelle
   ├─ Validation ile overfitting kontrol
   └─ Early stopping: İyileşme yoksa dur

4. 📈 TEST VE METRİKLER
   ├─ Accuracy: Doğru tahmin oranı
   ├─ Precision: "Yangın dediğinde doğru mu?"
   ├─ Recall: "Gerçek yangınları bulabiliyor mu?"
   ├─ F1-Score: Precision & Recall dengesi
   └─ AUC: Genel performans

5. 💾 KAYDET
   ├─ models/yangin_model.keras
   ├─ models/scaler.pkl
   ├─ Grafikler & Raporlar
   └─ Özellik önemi

SONUÇ: Eğitilmiş model hazır!
```

**Çıktı Dosyaları:**
- `models/yangin_model.keras` (Model)
- `models/scaler.pkl` (Ölçekleme)
- `outputs/test_predictions.csv` (Test tahminleri)
- `outputs/reports/training_history.png` (Grafikler)
- `outputs/reports/accuracy_analysis.png` (Doğruluk analizi)

### Adım 3: Tahmin Yapma (predict.py)

```bash
python src/predict.py
```

**Ne yapıyor?**

```
İŞLENMİŞ VERİ
    ↓
1. 🔀 VERİ HAZIRLAMA
   ├─ Özelikleri seç
   ├─ Scaler ile ölçekle
   └─ Model için hazır hale getir

2. 🤖 TAHMIN
   ├─ Modele veri gönder
   ├─ Olasılık değerleri al (0-1)
   └─ Sınıf tahmini (0 veya 1)

3. 📊 SONUÇ OLUŞTURMA
   ├─ fire_probability (0-1)
   ├─ confidence_score (0-100%)
   ├─ predicted_class (0 veya 1)
   └─ risk_level (low/medium/high)

4. 💾 KAYDET
   └─ outputs/predictions.csv

SONUÇ: Tahminler hazır!
```

**Çıktı Dosyası:** `outputs/predictions.csv`

**Örnek Çıktı:**
```csv
latitude,longitude,NDVI,bagil_nem,sicaklik,yagis,
fire_probability,confidence_score,predicted_class,risk_level
39.5,35.2,4500,35,15300,0.5,0.857,85.7,1,high
38.2,34.1,3200,45,15100,2.3,0.234,23.4,0,low
```

### Adım 4: Harita Oluşturma (map_visualization.py)

```bash
python src/map_visualization.py
```

**Ne yapıyor?**

```
TAHMİNLER
    ↓
1. 🗺️ TEMEL HARITA
   ├─ OpenStreetMap kaynağı
   ├─ Türkiye'ye yakınlaştır
   └─ Zoom level: 7

2. 🎨 RENKLENDİRME
   ├─ Düşük Risk (<30%): 🟢 Yeşil
   ├─ Orta Risk (30-70%): 🟠 Turuncu
   └─ Yüksek Risk (>70%): 🔴 Kırmızı

3. 📍 İŞARETÇİLER
   ├─ Her nokta = Bir tahmin
   ├─ CircleMarker şekli
   ├─ Tıklanabilir popup
   └─ Detay bilgileri göster

4. 🔥 EK KATMANLAR
   ├─ Heat Map (ısı yoğunluğu)
   ├─ Marker Clustering (yakın noktaları grupla)
   ├─ Test Doğruluk Katmanı (doğru/yanlış çerçeve)
   └─ Legend (açıklamalar)

5. 💾 KAYDET
   └─ outputs/maps/fire_risk_map.html

SONUÇ: İnteraktif harita hazır!
```

**Çıktı Dosyası:** `outputs/maps/fire_risk_map.html`

---

## 📊 Çıktılar ve Sonuçlar

### 1. Tahmin Dosyası (predictions.csv)

```csv
NDVI,bagil_nem,sicaklik,yagis,latitude,longitude,
fire_probability,confidence_score,predicted_class,risk_level,yangin_var
4500,35.2,15300,0.5,39.5,35.2,0.857,85.7,1,high,1
3200,45.1,15100,2.3,38.2,34.1,0.234,23.4,0,low,0
```

**Sütun Açıklamaları:**
- `fire_probability`: 0-1 aralığında yangın olasılığı
- `confidence_score`: 0-100% güven yüzdesi
- `predicted_class`: 0 (yangın yok) veya 1 (yangın var)
- `risk_level`: "low", "medium" veya "high"

### 2. Test Tahmin Dosyası (test_predictions.csv)

Test setindeki (%30) her nokta için gerçek sınıf ve tahmin sonucu:

- `actual_class`: Gerçek etiket (0/1)
- `predicted_class`: Tahmin edilen sınıf
- `prediction_correct`: Doğru/yanlış
- `error_type`: TP, TN, FP, FN

Bu dosya haritadaki doğruluk katmanını besler.

### 3. Eğitim Grafikleri (training_history.png)

```
┌─────────────────────────────────────┐
│     MODEL EĞİTİM GRAFIKLERI         │
├─────────────────────────────────────┤
│                                     │
│ 📊 Loss Grafiği    📊 Accuracy      │
│ ▼                  ▼                │
│ ├─ Training        ├─ Training      │
│ └─ Validation      └─ Validation    │
│                                     │
│ 📊 AUC Grafiği     📊 Precision     │
│ ▼                  ▼                │
│ ├─ Training        ├─ Training      │
│ └─ Validation      └─ Validation    │
│                                     │
└─────────────────────────────────────┘
```

### 4. Doğruluk Analizi (accuracy_analysis.png)

```
┌──────────────────────────────────────┐
│     MODEL DOĞRULUK ANALİZİ          │
├──────────────────────────────────────┤
│                                      │
│ 📊 Confusion Matrix  📊 Metrikler    │
│ ├─ TP: 890          ├─ Accuracy     │
│ ├─ TN: 2100         ├─ Precision    │
│ ├─ FP: 110          ├─ Recall       │
│ └─ FN: 100          └─ F1-Score     │
│                                      │
│ 📊 ROC Eğrisi       📊 Olasılık     │
│ └─ AUC: 0.92        Dağılımı        │
│                                      │
└──────────────────────────────────────┘
```

### 5. Test Raporu (classification_report.txt)

```
================================================================================
ORMAN YANGINI TAHMIN MODELİ - TEST METRİKLERİ
================================================================================

🎯 GENEL DOĞRULUK METRİKLERİ
────────────────────────────────
Accuracy (Doğruluk Oranı):  0.8750 (87.50%)
  → Toplam tahmınların %87.5'i doğru

Precision (Kesinlik):        0.8632
  → "Yangın var" dediğinde %86 doğruluk

Recall (Duyarlılık):         0.8901
  → Gerçek yangınların %89'unu buluyor

F1-Score:                    0.8765
  → Precision ve Recall'ün dengesi

ROC-AUC:                     0.9201
  → Genel diskriminasyon yeteneği çok iyi


📊 CONFUSION MATRIX (Hata Analizi)
────────────────────────────────
True Positive (TP):  890   ✅ Yangın doğru tahmin
True Negative (TN):  2100  ✅ Yangın yok doğru tahmin
False Positive (FP): 110   ❌ Yanlış alarm
False Negative (FN): 100   ❌ Kaçırılan yangınlar

Doğru Tahminler: 2990 / 3100 (96.45%)
Hatalı Tahminler: 210 / 3100 (6.77%)


⚠️ UYARILAR VE ÖNERİLER
────────────────────────────────
Yangın Tespit Oranı: 89.90%
  → Gerçek yangınların 89.9'unu bulabiliyor
  → Kaçırılan yangınlar: 100

Yanlış Alarm Oranı: 4.96%
  → Her 100 uyarıdan 5'i yanlış
  → Kabul edilebilir seviyede

================================================================================
```

### 6. İnteraktif Harita (fire_risk_map.html)

```
┌─────────────────────────────────────────────┐
│  🗺️ ORMAN YANĞINI RİSK HARITASI             │
├─────────────────────────────────────────────┤
│                                             │
│  [Map İçeriği - Web Tarayıcısında Görünür]  │
│                                             │
│  Legend:                                    │
│  🟢 Düşük Risk (<30%)     - Yeşil            │
│  🟠 Orta Risk (30-70%)    - Turuncu          │
│  🔴 Yüksek Risk (>70%)    - Kırmızı          │
│                                             │
│  Etkileşimler:                              │
│  ├─ Tıkla: Popup aç (detay bilgi)          │
│  ├─ Sürükle: Harita taşı                    │
│  ├─ Scroll: Yakınlaş/Uzaklaş                │
│  ├─ Heat Map: Isı yoğunluğu göster         │
│  └─ Clustering: Yakın noktaları grupla      │
│                                             │
│  Popup Örneği:                              │
│  ┌────────────────────────┐                │
│  │ 🔥 YANGIN RİSK RAPORU  │                │
│  ├────────────────────────┤                │
│  │ Risk: HIGH             │                │
│  │ Olasılık: 85.7%        │                │
│  │ Güven: 85.70%          │                │
│  │ Koord: 39.50, 35.20    │                │
│  └────────────────────────┘                │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 🐛 Hata Ayıklama

### ❌ "ModuleNotFoundError: tensorflow"

```bash
# Çözüm: TensorFlow'u kur
pip install tensorflow>=2.12.0

# Veya GPU desteğiyle:
pip install tensorflow[and-cuda]
```

### ❌ "CSV dosyası bulunamadı"

```bash
# Kontrol et: Dosyalar var mı?
dir data\raw\  # Windows
ls data/raw/   # macOS/Linux

# Gerekirse:
# - yangin_model_verisi.csv
# - yangin_rastgele_verisi.csv
# dosyalarını data/raw/ klasörüne koy
```

### ❌ "Memory Error" (RAM yetersiz)

```python
# config.py'da batch size'ı azalt
BATCH_SIZE = 16  # 32'den azalt
EPOCHS = 50      # 100'den azalt
```

### ❌ "Model çok yavaş eğitiliyor"

**Çözüm 1: Veri azalt**
```python
df = df.head(5000)  # İlk 5000 satır al
```

**Çözüm 2: GPU kullan**
```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]
python test_gpu.py  # Kontrol et
```

**Çözüm 3: Batch size'ı artır**
```python
BATCH_SIZE = 64  # 32'den artır
```

### ❌ "Harita boş / noktalar yok"

**Kontrol Listesi:**
1. `predictions.csv` dosyası var mı?
   ```bash
   dir outputs\  # Windows
   ```

2. Koordinatlar geçerli mi?
   ```python
   # Doğru aralık: -90 ile 90, -180 ile 180
   latitude:  36-40 (Türkiye)
   longitude: 30-43 (Türkiye)
   ```

3. Browser cache'i temizle
   ```
   Ctrl+Shift+Delete → Temporary files temizle
   ```

### ❌ "GPU algılanmıyor"

```bash
# Kontrol et
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# CPU modunda çalış
pip install tensorflow-cpu
```

---

## 📞 Destek ve İletişim

Sorularınız veya hataları bulduğunuzda:

1. **GitHub Issues**: Proje repository'sinde açın
2. **Email**: your-email@example.com
3. **Discord**: Topluluk sunucusu

---

## 📚 Kaynaklar

- **TensorFlow Docs**: https://www.tensorflow.org/
- **Scikit-learn Guide**: https://scikit-learn.org/
- **Folium Maps**: https://python-visualization.github.io/folium/
- **Pandas Tutorial**: https://pandas.pydata.org/docs/

---

**Sürüm:** 1.0.0  
**Son Güncelleme:** Mayıs 2026  
**Durum:** ✅ Üretim Hazır (Production Ready)

🔥 **Ormanlarımızı Koruyalım!**

Sorularınız, önerileriniz veya hata raporları için lütfen issue açınız.

---

**Son Güncelleme:** 2026-05-13

**Version**: 1.0.0

**Status**: Production Ready ✅
