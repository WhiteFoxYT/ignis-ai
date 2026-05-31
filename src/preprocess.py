"""
Veri Ön Isleme Modülü
=====================
CSV dosyalarını yükle, temizle ve işlenmiş hale getir.

Yapılan İşlemler:
1. CSV dosyasını oku
2. .geo sütunundan latitude/longitude çıkar
3. yangin_var = (etiket > 0) oluştur
4. Eksik değerleri temizle
5. Gerekli sütunları seç
6. Temiz veriyi kaydet
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json

from config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    RAW_DATA_FILES,
    PROCESSED_DATA_FILE,
    SCALER_FILE,
    FEATURES,
    TARGET,
    GEO_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    FIRE_LABEL_COLUMN,
    RANDOM_STATE,
    CSV_ENCODING,
    MISSING_VALUE_THRESHOLD,
    DATA_QUALITY_REPORT,
)
from utils import (
    setup_logger,
    ensure_path_exists,
    print_data_summary,
    parse_geojson,
    validate_coordinates,
)

# Logger
logger = setup_logger(__name__)


# ============================================================
# VERİ YÜKLEME
# ============================================================

def load_raw_data(input_path: Path) -> pd.DataFrame:
    """
    CSV dosyasını yükle.
    
    Args:
        input_path: CSV dosyasının yolu
    
    Returns:
        DataFrame
    """
    if not input_path.exists():
        logger.error(f" Dosya bulunamadı: {input_path}")
        raise FileNotFoundError(f"Dosya bulunamadı: {input_path}")
    
    logger.info(f" CSV yükleniyor: {input_path}")
    
    df = pd.read_csv(input_path, encoding=CSV_ENCODING)
    logger.info(f" Yüklendi: {len(df):,} satır, {len(df.columns)} sütun")
    
    return df


def load_and_label_raw_data() -> pd.DataFrame:
    """
    Ham veri dosyalarını yükle ve sınıf etiketlerini oluştur.

    Varsayım:
    - yangin_model_verisi.csv -> yangin_var = 1
    - yangin_rastgele_verisi.csv -> yangin_var = 0
    """
    model_path = RAW_DATA_FILES["model_data"]
    random_path = RAW_DATA_FILES["random_data"]

    df_fire = load_raw_data(model_path)
    df_fire[TARGET] = 1
    df_fire["data_source"] = "fire"

    if random_path.exists():
        df_random = load_raw_data(random_path)
        df_random[TARGET] = 0
        df_random["data_source"] = "random"
        df = pd.concat([df_fire, df_random], ignore_index=True)
    else:
        logger.warning(f"   ⚠ Random veri bulunamadı: {random_path}")
        df = df_fire

    logger.info(f" Birleştirilmiş veri: {len(df):,} satır")
    return df


# ============================================================
# KOORDİNAT ÇIKARMA
# ============================================================

def extract_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    .geo sütunundan lat/lon çıkar ve sütun olarak ekle.
    
    İki format destekleniyor:
    1. GeoJSON: {"type":"Point","coordinates":[lon, lat]}
    2. String: "POINT(lon lat)"
    3. Zaten lat/lon sütunları varsa, kopyala
    
    Args:
        df: GeoJSON sütunu içeren DataFrame
    
    Returns:
        Latitude ve Longitude sütunları eklenen DataFrame
    """
    logger.info(" Koordinatlar çıkarılıyor...")
    
    df = df.copy()
    
    # Eğer zaten lat/lon sütunları varsa, kopyala
    if LATITUDE_COLUMN in df.columns and LONGITUDE_COLUMN in df.columns:
        logger.info("   Lat/Lon sütunları zaten mevcut")
        return df
    
    # Eğer .geo sütunu varsa, buradan çıkar
    if GEO_COLUMN in df.columns:
        latitudes = []
        longitudes = []
        
        for idx, geo_str in enumerate(df[GEO_COLUMN]):
            if pd.isna(geo_str):
                latitudes.append(np.nan)
                longitudes.append(np.nan)
                continue
            
            try:
                # GeoJSON formatını dene
                if isinstance(geo_str, str) and (geo_str.startswith('{') or geo_str.startswith('POINT')):
                    lat, lon = parse_geojson(str(geo_str))
                    
                    if validate_coordinates(lat, lon):
                        latitudes.append(lat)
                        longitudes.append(lon)
                    else:
                        latitudes.append(np.nan)
                        longitudes.append(np.nan)
                else:
                    latitudes.append(np.nan)
                    longitudes.append(np.nan)
            
            except Exception as e:
                if idx < 5:  # İlk 5 hata için warning
                    logger.warning(f"   Koordinat parse hatası (satır {idx}): {e}")
                latitudes.append(np.nan)
                longitudes.append(np.nan)
        
        df[LATITUDE_COLUMN] = latitudes
        df[LONGITUDE_COLUMN] = longitudes
        
        valid_coords = sum(1 for lat, lon in zip(latitudes, longitudes) 
                          if lat is not None and lon is not None)
        logger.info(f"    {valid_coords:,} geçerli koordinat çıkarıldı")
    
    else:
        logger.warning("   ⚠ .geo sütunu bulunamadı!")
    
    return df


# ============================================================
# YANGNI ETİKETİ OLUŞTURMA
# ============================================================

def create_fire_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    etiket sütunundan yangin_var binary değişkenini oluştur.
    
    Kural: yangin_var = 1 eğer etiket > 0, yoksa 0
    
    Args:
        df: etiket sütunu içeren DataFrame
    
    Returns:
        yangin_var sütunu eklenen DataFrame
    """
    logger.info(" Yangın etiketi oluşturuluyor...")
    
    df = df.copy()
    
    if TARGET in df.columns:
        logger.info(" Yangın etiketi zaten mevcut, yeniden oluşturulmadı")
        return df

    if FIRE_LABEL_COLUMN not in df.columns:
        logger.error(f" {FIRE_LABEL_COLUMN} sütunu bulunamadı!")
        logger.info("   Available columns:", df.columns.tolist())
        raise ValueError(f"{FIRE_LABEL_COLUMN} sütunu bulunamadı!")
    
    # Binary etiket oluştur
    df[TARGET] = (df[FIRE_LABEL_COLUMN] > 0).astype(int)
    
    # Dağılımı göster
    fire_count = (df[TARGET] == 1).sum()
    no_fire_count = (df[TARGET] == 0).sum()
    
    logger.info(f"    Yangın: {fire_count:,} ({fire_count/len(df)*100:.1f}%)")
    logger.info(f"    Yangın Yok: {no_fire_count:,} ({no_fire_count/len(df)*100:.1f}%)")
    
    return df


# ============================================================
# EKSIK VERİ TİZLEMESİ
# ============================================================

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eksik değerleri temizle.
    
    Strategy:
    1. THRESHOLD'dan fazla eksik sütunu kaldır
    2. Kalan eksik değerleri median ile doldur (numeric)
    3. Kalan eksik değerleri ileri/geri interpolasyonla doldur
    
    Args:
        df: DataFrame
    
    Returns:
        Temiz DataFrame
    """
    logger.info(" Eksik veriler temizleniyor...")
    
    df = df.copy()
    
    # Eksik değerleri say
    missing_before = df.isnull().sum().sum()
    logger.info(f"   Toplam eksik değer: {missing_before:,}")
    
    # 1. THRESHOLD'dan fazla eksik sütunları kaldır
    missing_ratios = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratios[missing_ratios > MISSING_VALUE_THRESHOLD].index.tolist()
    
    if cols_to_drop:
        logger.info(f"   Kaldırılan sütun ({MISSING_VALUE_THRESHOLD*100}% eksik): {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # 2. Numeric sütunlardaki eksik değerleri median ile doldur
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"   {col}: Median={median_val:.2f} ile dolduruldu")
    
    # 3. Diğer sütunlardaki eksik değerleri modu ile doldur
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"   {col}: Mod ile dolduruldu")
    
    missing_after = df.isnull().sum().sum()
    logger.info(f"    Temizleme tamamlandı: {missing_before} -> {missing_after}")
    
    return df


def generate_data_quality_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_path: Path = DATA_QUALITY_REPORT,
) -> None:
    """Veri kalitesi raporu oluştur ve kaydet."""
    lines = []
    lines.append("=" * 80)
    lines.append("VERİ KALİTESİ RAPORU")
    lines.append("=" * 80)
    lines.append(f"Toplam satır (önce): {len(df_before):,}")
    lines.append(f"Toplam satır (sonra): {len(df_after):,}")
    lines.append(f"Toplam sütun (önce): {len(df_before.columns)}")
    lines.append(f"Toplam sütun (sonra): {len(df_after.columns)}")
    lines.append("")

    # Eksik değerler
    missing_before = df_before.isnull().sum().sort_values(ascending=False)
    missing_after = df_after.isnull().sum().sort_values(ascending=False)
    total_missing_before = int(missing_before.sum())
    total_missing_after = int(missing_after.sum())
    lines.append("Eksik Değerler:")
    lines.append(f"  Toplam eksik (önce): {total_missing_before:,}")
    lines.append(f"  Toplam eksik (sonra): {total_missing_after:,}")
    if total_missing_before > 0:
        lines.append("  En çok eksik içeren sütunlar (önce):")
        for col, count in missing_before.head(10).items():
            if count > 0:
                lines.append(f"    - {col}: {int(count):,}")
    lines.append("")

    # Duplicate satırlar
    duplicate_count = int(df_before.duplicated().sum())
    lines.append(f"Duplicate satır sayısı (önce): {duplicate_count:,}")
    lines.append("")

    # Koordinat geçerliliği
    if LATITUDE_COLUMN in df_before.columns and LONGITUDE_COLUMN in df_before.columns:
        valid = (
            (df_before[LATITUDE_COLUMN] >= -90) & (df_before[LATITUDE_COLUMN] <= 90) &
            (df_before[LONGITUDE_COLUMN] >= -180) & (df_before[LONGITUDE_COLUMN] <= 180) &
            df_before[LATITUDE_COLUMN].notna() & df_before[LONGITUDE_COLUMN].notna()
        )
        invalid_count = int((~valid).sum())
        lines.append(f"Geçersiz koordinat sayısı (önce): {invalid_count:,}")
        lines.append("")

    # Sınıf dağılımı
    if TARGET in df_before.columns:
        class_counts = df_before[TARGET].value_counts()
        lines.append("Sınıf Dağılımı (önce):")
        for label, count in class_counts.items():
            pct = (count / len(df_before)) * 100
            lines.append(f"  - {label}: {count:,} ({pct:.1f}%)")
        min_pct = (class_counts.min() / len(df_before)) * 100
        if min_pct < 5:
            lines.append("⚠ UYARI: Sınıf dengesizliği yüksek (min sınıf < %5)")

    ensure_path_exists(output_path, is_file=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f" Veri kalitesi raporu kaydedildi: {output_path}")


# ============================================================
# ÖZELLIK SEÇİMİ
# ============================================================

def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Gerekli özellikleri ve hedef değişkeni seç.
    
    Args:
        df: DataFrame
    
    Returns:
        (Seçilmiş DataFrame, eksik olan özellikler listesi)
    """
    logger.info(" Ozellikler seçiliyor...")
    
    df = df.copy()
    
    # Hedef sütunun varlığını kontrol et
    if TARGET not in df.columns:
        raise ValueError(f"Hedef sütun {TARGET} bulunamadı!")
    
    # Mevcut özellikleri kontrol et
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        logger.warning(f"   ⚠ Eksik özellikler: {missing_features}")
        available_features = [f for f in FEATURES if f in df.columns]
    else:
        available_features = FEATURES
    
    logger.info(f"   Seçilen özellikler: {available_features}")
    
    # Seç
    selected_cols = available_features + [TARGET]
    df = df[selected_cols]
    
    logger.info(f"    {len(available_features)} özellik seçildi")
    
    return df, missing_features


# ============================================================
# AYKIRI DEĞER TEMİZLEMESİ (İSTEĞE BAĞLI)
# ============================================================

def remove_outliers(df: pd.DataFrame, features: list, method: str = "iqr") -> pd.DataFrame:
    """
    Aykırı değerleri kaldır (IQR yöntemiyle).
    
    Args:
        df: DataFrame
        features: Hangi sütunlar kontrol edileceği
        method: "iqr" veya "zscore"
    
    Returns:
        Aykırı değerleri kaldırılmış DataFrame
    """
    logger.info("🔍 Aykırı değerler kontrol ediliyor...")
    
    df = df.copy()
    rows_before = len(df)
    
    if method == "iqr":
        for col in features:
            if col in df.columns and df[col].dtype in [np.float64, np.float32, int]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    logger.info(f"   {col}: {outliers} aykırı değer kaldırıldı")
    
    elif method == "zscore":
        from scipy import stats
        for col in features:
            if col in df.columns and df[col].dtype in [np.float64, np.float32, int]:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                threshold = 3
                outliers = (z_scores > threshold).sum()
                if outliers > 0:
                    df = df[np.abs(stats.zscore(df[col])) <= threshold]
                    logger.info(f"   {col}: {outliers} aykırı değer kaldırıldı")
    
    logger.info(f"    {rows_before - len(df)} satır kaldırıldı")
    
    return df


# ============================================================
# ANA VERİ ÖN İŞLEME FONKSİYONU
# ============================================================

def preprocess_data(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    remove_outliers_flag: bool = False,
) -> pd.DataFrame:
    """
    Veri ön işlemesinin ana fonksiyonu.
    
    Yapılan işlemler:
    1. CSV yükle
    2. Koordinat çıkar
    3. Yangın etiketi oluştur
    4. Eksik değerleri temizle
    5. Ozellikleri seç
    6. (İsteğe bağlı) Aykırı değerleri kaldır
    7. Kaydet
    
    Args:
        input_path: Giriş CSV yolu (default: ilk ham veri dosyası)
        output_path: Çıkış CSV yolu (default: PROCESSED_DATA_FILE)
        remove_outliers_flag: Aykırı değerleri kaldır mı?
    
    Returns:
        İşlenmiş DataFrame
    """
    
    # Default değerleri ayarla
    if input_path is None:
        input_path = None
    if output_path is None:
        output_path = PROCESSED_DATA_FILE
    
    logger.info("=" * 60)
    logger.info(" VERİ ÖN İŞLEME BAŞLANDI")
    logger.info("=" * 60)
    
    try:
        # 1. Yükle
        if input_path is None:
            df = load_and_label_raw_data()
        else:
            df = load_raw_data(input_path)
        print_data_summary(df)
        
        # 2. Koordinat çıkar
        df = extract_coordinates(df)
        
        # 3. Yangın etiketi oluştur (gerekirse)
        df = create_fire_label(df)
        
        # 4. Eksik değerleri temizle
        df_before_clean = df.copy()
        df = clean_missing_values(df)
        
        # 5. Ozellikleri seç
        df, missing_features = select_features(df)
        
        # 6. Aykırı değerleri kaldır
        if remove_outliers_flag:
            df = remove_outliers(df, FEATURES)
        
        # Final kontrol
        logger.info(f" Son DataFrame: {len(df):,} satır x {len(df.columns)} sütun")
        print_data_summary(df)

        # Veri kalitesi raporu (önce/sonra)
        generate_data_quality_report(df_before_clean, df)
        
        # 7. Kaydet
        ensure_path_exists(output_path, is_file=True)
        df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
        logger.info(f" İşlenmiş veri kaydedildi: {output_path}")
        
        logger.info("=" * 60)
        logger.info(" VERİ ÖN İŞLEME TAMAMLANDI")
        logger.info("=" * 60)
        
        return df
    
    except Exception as e:
        logger.error(f" Ön işleme hatası: {e}")
        raise


# ============================================================
# SCALER YÖNETIMI
# ============================================================

def create_and_save_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Scaler'ı eğitim verisine göre oluştur ve kaydet.
    
    Args:
        X_train: Eğitim verileri
    
    Returns:
        Oluşturulan ve kaydedilen scaler
    """
    logger.info(" Scaler oluşturuluyor...")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    ensure_path_exists(SCALER_FILE, is_file=True)
    joblib.dump(scaler, SCALER_FILE)
    
    logger.info(f" Scaler kaydedildi: {SCALER_FILE}")
    
    return scaler


def load_scaler() -> StandardScaler:
    """
    Kaydedilmiş scaler'ı yükle.
    
    Returns:
        StandardScaler nesnesi
    """
    if not SCALER_FILE.exists():
        raise FileNotFoundError(f"Scaler dosyası bulunamadı: {SCALER_FILE}")
    
    scaler = joblib.load(SCALER_FILE)
    logger.info(f" Scaler yüklendi: {SCALER_FILE}")
    
    return scaler


# ============================================================
# MAIN (TEST) - Doğrudan çalıştırıldığında
# ============================================================

if __name__ == "__main__":
    
    try:
        # Veri ön işleme yap
        processed_df = preprocess_data()
        
        logger.info("\n Ön işleme başarılı!")
        logger.info(f" Sonuç: {len(processed_df):,} satır")
        logger.info(f"   Sütunlar: {list(processed_df.columns)}")
        
    except Exception as e:
        logger.error(f" Hata: {e}")
        raise
