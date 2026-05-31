"""
Tahmin Modülü
==============
Eğitilmiş modeli kullanarak yangın riskini tahmin et.

Yapılan İşlemler:
1. Modeli ve scaler'ı yükle
2. CSV'yi oku
3. .geo sütunundan koordinat çıkar
4. Ozellikleri ölçekle
5. Tahmin yap
6. Risk seviyesi ekle
7. Sonuçları kaydet
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf

from config import (
    FEATURES,
    TARGET,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    GEO_COLUMN,
    FIRE_LABEL_COLUMN,
    PROCESSED_DATA_FILE,
    MODEL_FILE,
    SCALER_FILE,
    PREDICTIONS_FILE,
    CSV_ENCODING,
)
from utils import (
    setup_logger,
    ensure_path_exists,
    parse_geojson,
    validate_coordinates,
    get_risk_level,
)
from preprocess import load_raw_data, extract_coordinates

# Logger
logger = setup_logger(__name__)


# ============================================================
# MODEL VE SCALER YÜKLEME
# ============================================================

def load_model_and_scaler() -> Tuple[tf.keras.Model, StandardScaler]:
    """
    Kaydedilmiş modeli ve scaler'ı yükle.
    
    Returns:
        (model, scaler) tuple'ı
    """
    logger.info(" Model ve scaler yükleniyor...")
    
    # Model
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {MODEL_FILE}\n"
            "Lutfen önce train.py çalıştırın!"
        )
    
    model = tf.keras.models.load_model(MODEL_FILE)
    logger.info(f" Model yüklendi: {MODEL_FILE}")
    
    # Scaler
    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"Scaler dosyası bulunamadı: {SCALER_FILE}"
        )
    
    scaler = joblib.load(SCALER_FILE)
    logger.info(f" Scaler yüklendi: {SCALER_FILE}")
    
    return model, scaler


# ============================================================
# VERİ HAZIRLAMAK
# ============================================================

def prepare_prediction_data(
    input_path: Path = None,
    use_processed: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Tahmin için veriyi hazırla.
    
    Args:
        input_path: CSV dosyası (default: PROCESSED_DATA_FILE)
        use_processed: İşlenmiş veri mi kullan?
    
    Returns:
        (X, df_with_coords, feature_names) tuple'ı
    """
    logger.info(" Tahmin verisi hazırlanıyor...")
    
    if input_path is None:
        input_path = PROCESSED_DATA_FILE if use_processed else PROCESSED_DATA_FILE
    
    if not input_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {input_path}")
    
    # CSV yükle
    df = pd.read_csv(input_path, encoding=CSV_ENCODING)
    logger.info(f" Yüklendi: {len(df):,} satır")
    
    # Ozellikleri seç (mevcut olan)
    available_features = [f for f in FEATURES if f in df.columns]
    if not available_features:
        raise ValueError(f"Hiçbir özellik bulunamadı! Verilen: {FEATURES}")
    
    if len(available_features) < len(FEATURES):
        missing = [f for f in FEATURES if f not in df.columns]
        logger.warning(f"⚠ Eksik özellikler (0 ile doldurulacak): {missing}")
    
    # X'i oluştur
    X = df[available_features].copy()
    
    # Eksik özellikleri 0 ile doldur
    for feature in FEATURES:
        if feature not in X.columns:
            X[feature] = 0.0
    
    # Özellik sırasını modele uygun hale getir
    X = X[FEATURES]
    
    logger.info(f" Ozellikler ({len(available_features)}): {available_features}")
    
    return X, df, available_features


# ============================================================
# TAHMIN YAPMA
# ============================================================

def predict_fire_probability(
    model: tf.keras.Model,
    X_scaled: np.ndarray,
) -> np.ndarray:
    """
    Yangın olasılıklarını tahmin et.
    
    Args:
        model: Eğitilmiş model
        X_scaled: Ölçeklenmiş özellikler
    
    Returns:
        Olasılık değerleri (0-1)
    """
    logger.info(" Yangın olasılıkları tahmin ediliyor...")
    
    # Tahmin yap
    probabilities = model.predict(X_scaled, verbose=0)  # (n_samples, 1)
    
    # Sıkıştır
    probabilities = probabilities.flatten()
    
    logger.info(f" {len(probabilities):,} tahmin yapıldı")
    logger.info(f"   Min: {probabilities.min():.4f}")
    logger.info(f"   Max: {probabilities.max():.4f}")
    logger.info(f"   Mean: {probabilities.mean():.4f}")
    logger.info(f"   Std: {probabilities.std():.4f}")
    
    return probabilities


# ============================================================
# SONUÇLARI POST-PROCESS
# ============================================================

def add_predictions_to_dataframe(
    df: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    DataFrame'e tahmin ve risk seviyesini ekle.
    
    Args:
        df: Orijinal DataFrame
        probabilities: Tahmin edilen olasılıklar
    
    Returns:
        Yeni sütunlarla genişletilmiş DataFrame
    """
    logger.info(" Sonuçlar DataFrame'e ekleniyor...")
    
    df = df.copy()
    
    # Olasılıkları ekle
    df['fire_probability'] = probabilities
    
    # Confidence score (0-100% arası, 0.5'ten uzaklık)
    df['confidence_score'] = (np.abs(probabilities - 0.5) * 2 * 100).round(2)
    
    # Tahmin edilen sınıf (0 veya 1, threshold=0.5)
    df['predicted_class'] = (probabilities > 0.5).astype(int)
    
    # Risk seviyesini ekle
    df['risk_level'] = df['fire_probability'].apply(get_risk_level)
    
    # İstatistikler
    risk_counts = df['risk_level'].value_counts()
    logger.info("   Risk Dağılımı:")
    for risk_level in sorted(risk_counts.index):
        count = risk_counts[risk_level]
        pct = (count / len(df)) * 100
        logger.info(f"     {risk_level.upper():.<20} {count:,} ({pct:.1f}%)")
    
    # Sınıf dağılımı
    class_counts = df['predicted_class'].value_counts()
    logger.info("   Tahmin Edilen Sınıf Dağılımı:")
    for cls in sorted(class_counts.index):
        count = class_counts[cls]
        pct = (count / len(df)) * 100
        label = "Yangın VAR" if cls == 1 else "Yangın YOK"
        logger.info(f"     {label:.<20} {count:,} ({pct:.1f}%)")
    
    return df


# ============================================================
# ANA TAHMIN FONKSİYONU
# ============================================================

def predict_fire_risk(
    input_path: Path = None,
    output_path: Path = None,
) -> pd.DataFrame:
    """
    Ana tahmin fonksiyonu.
    
    Yapılan İşlemler:
    1. Model ve scaler yükle
    2. Veri hazırla
    3. Ölçekle
    4. Tahmin yap
    5. Risk seviyesi ekle
    6. Kaydet
    
    Args:
        input_path: Giriş CSV yolu (default: PROCESSED_DATA_FILE)
        output_path: Çıkış CSV yolu (default: PREDICTIONS_FILE)
    
    Returns:
        Tahminlerle genişletilmiş DataFrame
    """
    
    if output_path is None:
        output_path = PREDICTIONS_FILE
    
    logger.info("=" * 80)
    logger.info(" YANGINI RİSK TAHMİNİ")
    logger.info("=" * 80)
    
    try:
        # 1. Model ve scaler yükle
        model, scaler = load_model_and_scaler()
        
        # 2. Veri hazırla
        X, df, available_features = prepare_prediction_data(input_path)
        
        # 3. Ölçekle
        logger.info(" Ozellikler ölçekleniyor...")
        X_scaled = scaler.transform(X)
        logger.info(" Ölçekleme tamamlandı")
        
        # 4. Tahmin yap
        probabilities = predict_fire_probability(model, X_scaled)
        
        # 5. Sonuçları ekle
        df_predictions = add_predictions_to_dataframe(df, probabilities)
        
        # 6. Kaydet
        ensure_path_exists(output_path, is_file=True)
        df_predictions.to_csv(output_path, index=False, encoding=CSV_ENCODING)
        logger.info(f" Tahminler kaydedildi: {output_path}")
        
        # İstatistikler
        logger.info(f"\n Çıktı İstatistikleri:")
        logger.info(f"   Toplam Satır: {len(df_predictions):,}")
        logger.info(f"   Sütun Sayısı: {len(df_predictions.columns)}")
        
        # Sütunları göster
        logger.info(f"\n   Sütunlar:")
        for col in df_predictions.columns:
            dtype = df_predictions[col].dtype
            logger.info(f"     - {col} ({dtype})")
        
        logger.info("\n" + "=" * 80)
        logger.info(" TAHMIN TAMAMLANDI!")
        logger.info("=" * 80)
        
        return df_predictions
    
    except Exception as e:
        logger.error(f" Tahmin hatası: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================
# BATCH TAHMIN (İsteğe bağlı)
# ============================================================

def batch_predict(
    model: tf.keras.Model,
    X_scaled: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Büyük veri setleri için batch tahmin.
    
    Args:
        model: Eğitilmiş model
        X_scaled: Ölçeklenmiş özellikler
        batch_size: Batch boyutu
    
    Returns:
        Tahmin edilen olasılıklar
    """
    logger.info(f"📦 Batch tahmin yapılıyor (batch_size={batch_size})...")
    
    n_samples = len(X_scaled)
    predictions = []
    
    for i in range(0, n_samples, batch_size):
        batch = X_scaled[i:i+batch_size]
        batch_pred = model.predict(batch, verbose=0)
        predictions.extend(batch_pred.flatten())
    
    return np.array(predictions)


# ============================================================
# MAIN (TEST)
# ============================================================

if __name__ == "__main__":
    
    try:
        # Tahmin yap
        results_df = predict_fire_risk()
        
        logger.info("\n Tahmin başarılı!")
        logger.info(f"📄 Sonuç: {PREDICTIONS_FILE}")
        
        # İlk satırları göster
        logger.info("\nİlk 5 sonuç:")
        logger.info(results_df[['fire_probability', 'risk_level']].head())
        
    except Exception as e:
        logger.error(f" Hata: {e}")
        raise
