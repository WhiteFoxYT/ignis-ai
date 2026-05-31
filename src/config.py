"""
Proje Konfigürasyonu
======================
Tüm sabitler ve konfigürasyon ayarları bu dosyada tanımlanmıştır.

Genel Bilgiler:
- Proje: Orman Yangını Tahmin Sistemi
- Amaç: Uydu ve meteorolojik verileri kullanarak yangın riskini tahmin et
- Version: 1.0.0
"""

# ============================================================
# TENSORFLOW UYARILARI - GİZLE (Normal başlangıç mesajları)
# ============================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Sadece ERROR ve WARNING göster
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # GPU memory dinamik büyüme
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 0'ı kullan (varsa)

from pathlib import Path

# ============================================================
# PROJE DİZİNLERİ
# ============================================================

# Base proje dizini
BASE_DIR = Path(__file__).parent.parent

# Veri dizinleri
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Model dizini
MODELS_DIR = BASE_DIR / "models"

# Çıktı dizinleri
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MAPS_DIR = OUTPUTS_DIR / "maps"

# Notebook dizini
NOTEBOOKS_DIR = BASE_DIR / "noteboks"

# ============================================================
# VERİ DOSYALARI
# ============================================================

# Giriş verileri
RAW_DATA_FILES = {
    "model_data": DATA_RAW_DIR / "yangin_model_verisi.csv",
    "random_data": DATA_RAW_DIR / "yangin_rastgele_verisi.csv",
}

# İşlenmiş veriler
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "processed_data.csv"

# Model dosyaları
MODEL_FILE = MODELS_DIR / "yangin_model.keras"
SCALER_FILE = MODELS_DIR / "scaler.pkl"

# Çıktı dosyaları
PREDICTIONS_FILE = OUTPUTS_DIR / "predictions.csv"
TEST_PREDICTIONS_FILE = OUTPUTS_DIR / "test_predictions.csv"
CLASSIFICATION_REPORT = REPORTS_DIR / "classification_report.txt"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "feature_importance.csv"
TRAINING_HISTORY_FILE = REPORTS_DIR / "training_history.png"
FIRE_RISK_MAP = MAPS_DIR / "fire_risk_map.html"
DATA_QUALITY_REPORT = REPORTS_DIR / "data_quality_report.txt"

# ============================================================
# VERİ ÖZELLİKLERİ
# ============================================================

# Giriş özellikleri (features)
FEATURES = [
    "NDVI",           # Bitki Örtüsü İndeksi
    "bagil_nem",      # Bağıl Nem (%)
    "sicaklik",       # Sıcaklık (K veya °C)
    "yagis",          # Yağış (mm)
    "latitude",       # Enlem
    "longitude",      # Boylam
]

# Hedef değişken
TARGET = "yangin_var"

# Coğrafik bilgiler
GEO_COLUMN = ".geo"
LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"

# Yangın etiketi
FIRE_LABEL_COLUMN = "etiket"

# ============================================================
# MODEL PARAMETRELERI
# ============================================================

# Random state (tekrarlanabilirlik için)
RANDOM_STATE = 42

# Train-Test split oranı
TEST_SIZE = 0.3  # %70 eğitim, %30 test
VALIDATION_SIZE = 0.1

# Veri ön işleme
SCALER_METHOD = "StandardScaler"  # StandardScaler veya MinMaxScaler

# ============================================================
# TensorFlow MODEL MİMARİSİ
# ============================================================

# Model katmanları
MODEL_CONFIG = {
    "input_shape": len(FEATURES),
    "layers": [
        {"type": "Dense", "units": 64, "activation": "relu"},
        {"type": "BatchNormalization"},
        {"type": "Dropout", "rate": 0.3},
        {"type": "Dense", "units": 32, "activation": "relu"},
        {"type": "BatchNormalization"},
        {"type": "Dropout", "rate": 0.2},
        {"type": "Dense", "units": 16, "activation": "relu"},
        {"type": "Dense", "units": 1, "activation": "sigmoid"},
    ]
}

# Optimizer ve loss
OPTIMIZER = "adam"
LEARNING_RATE = 0.001
LOSS = "binary_crossentropy"
METRICS = ["accuracy", "Precision", "Recall", "AUC"]

# ============================================================
# EĞİTİM PARAMETRELERI
# ============================================================

# Epoch ve batch size
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

# Callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# ============================================================
# RİSK SEVİYESİ KATEGORİLERİ
# ============================================================

# Risk eşikleri
RISK_THRESHOLDS = {
    "low": (0.0, 0.30),           # Düşük Risk
    "medium": (0.30, 0.70),       # Orta Risk
    "high": (0.70, 1.0),          # Yüksek Risk
}

# Risk seviyeleri için renkler (Folium haritası)
RISK_COLORS = {
    "low": "green",
    "medium": "orange",
    "high": "red",
}

# ============================================================
# HARITA AYARLARI
# ============================================================

# Türkiye merkez koordinatları
MAP_CENTER = [39.0, 35.0]
MAP_ZOOM_START = 6

# CircleMarker ayarları
CIRCLE_RADIUS = 5
CIRCLE_OPACITY = 0.7
CIRCLE_WEIGHT = 2

# Harita Optimizasyonu
MAX_MARKERS_ON_MAP = 5000          # Max nokta sayısı (performans için)
USE_MARKER_CLUSTERING = True        # Marker cluster'ı kullan
SIMPLIFIED_POPUPS = True            # Basit popup'lar (daha hızlı render)
HEATMAP_ONLY_MODE = False           # True = sadece heatmap (çok hızlı, nokta yok)

# ============================================================
# LOGGING AYARLARI
# ============================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================
# GOOGLE EARTH ENGINE AYARLARI
# ============================================================

GEE_PROJECT_ID = "ignisai-496207"
GEE_REGION_COLLECTION = "USDOS/LSIB_SIMPLE/2017"
GEE_REGION_NAME = "Turkey"

# ============================================================
# VERİ TOPLAMA PARAMETRELERI (GEE)
# ============================================================

GEE_SAMPLING_CONFIG = {
    "num_points": 1000,
    "scale": 10000,  # 10 km
    "seed": 42,
}

# Tarih aralıkları
DATE_RANGE_FULL = {
    "start": "2019-06-01",
    "end": "2025-01-12",
}

# ============================================================
# DİĞER AYARLAR
# ============================================================

# CSV encoding
CSV_ENCODING = "utf-8"

# Hata toleransı
MISSING_VALUE_THRESHOLD = 0.5  # %50'den fazla eksik veri varsa sütunu kaldır
DUPLICATE_THRESHOLD = 0.95      # %95'ten fazla benzerlik varsa duplicate say

# ============================================================
# KONTROL FONKSİYONU
# ============================================================

def ensure_directories_exist() -> None:
    """
    Tüm gerekli dizinlerin var olmasını sağla.
    Yoksa oluştur.
    """
    directories = [
        DATA_DIR,
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        REPORTS_DIR,
        MAPS_DIR,
        NOTEBOOKS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Konfigürasyonu test et
    ensure_directories_exist()
    print("✅ Tüm dizinler oluşturuldu!")
    print(f"📁 BASE_DIR: {BASE_DIR}")
    print(f"📊 Özellikler: {FEATURES}")
    print(f"🎯 Hedef: {TARGET}")
