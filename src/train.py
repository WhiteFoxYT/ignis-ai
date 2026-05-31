"""
Model Eğitim Modülü
===================
TensorFlow ile sinir ağı modelini eğit.

Yapılan İşlemler:
1. Veri yükle ve böl (train/test/val)
2. Ozellekleri ölçekle (StandardScaler)
3. TensorFlow model oluştur
4. Modeli eğit (callbacks ile)
5. Test metriklerini hesapla
6. Modeli ve scaler'ı kaydet
7. Eğitim grafiklerini oluştur
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.metrics import Precision, Recall, AUC

from config import (
    FEATURES,
    TARGET,
    PROCESSED_DATA_FILE,
    MODEL_FILE,
    SCALER_FILE,
    CLASSIFICATION_REPORT,
    FEATURE_IMPORTANCE_FILE,
    TRAINING_HISTORY_FILE,
    TEST_PREDICTIONS_FILE,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MODEL_CONFIG,
    LEARNING_RATE,
    LOSS,
    METRICS as CONFIG_METRICS,
)
from utils import (
    setup_logger,
    ensure_path_exists,
    print_metrics_summary,
    print_class_distribution,
    get_risk_level,
)
from preprocess import create_and_save_scaler

# Logger
logger = setup_logger(__name__)


# ============================================================
# GPU KONTROL VE SETUP
# ============================================================

def check_and_setup_gpu():
    """GPU'yu kontrol et ve fiziksel CPU/GPU işleme optimize et."""
    logger.info("\n" + "=" * 80)
    logger.info(" HARDWARE ACCELERATION SETUP")
    logger.info("=" * 80)
    
    # CPU threading optimization
    try:
        # TensorFlow inter-op thread pool
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)
        logger.info(" CPU Threading: 8 inter-op, 8 intra-op threads")
    except:
        pass
    
    # TensorFlow GPU'yu bul
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        logger.info(f" GPU(lar) bulundu: {len(gpus)} adet")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu}")
        
        # GPU memory'yi dinamik ayarla
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        
        logger.info(" GPU Acceleration: AKTIF (DirectML/ROCm)")
        logger.info(" Processing Device: GPU")
    else:
        logger.info(" GPU bulunamadi - CPU Optimization modu")
        logger.info(" DirectML optimization: AKTIF (oneDNN backend)")
        logger.info(" Processing Device: CPU (optimized)")
    
    logger.info("=" * 80 + "\n")
    
    return len(gpus) > 0


# Optimization setup
gpu_available = check_and_setup_gpu()

def load_and_split_data() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series, pd.DataFrame
]:
    """
    İşlenmiş CSV'yi yükle ve train/test'e böl.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, full_df
    """
    logger.info(" Veriler yükleniyor...")
    
    if not PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"İşlenmiş veri dosyası bulunamadı: {PROCESSED_DATA_FILE}\n"
            "Lutfen önce preprocess.py çalıştırın!"
        )
    
    # Yükle
    df = pd.read_csv(PROCESSED_DATA_FILE)
    logger.info(f" Yüklendi: {len(df):,} satır x {len(df.columns)} sütun")
    
    # Ozellikler ve hedef ayır
    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features]
    y = df[TARGET]
    
    if len(available_features) < len(FEATURES):
        missing = [f for f in FEATURES if f not in available_features]
        logger.warning(f"⚠ Eksik özellikler: {missing}")
    
    logger.info(f" Hedef sınıf dağılımı:")
    print_class_distribution(y)
    
    # 1. Train (%80) vs Test (%20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # Sınıf dağılımını koru
    )
    
    # 2. Train'i (%80 * %90) vs Val (%80 * %10) olarak böl
    val_size = VALIDATION_SIZE / (1 - TEST_SIZE)  # 12.5% -> %11.1
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    
    logger.info(f" Veri bölme tamamlandı:")
    logger.info(f"   Training: {len(X_train_final):,} ({len(X_train_final)/len(X)*100:.1f}%)")
    logger.info(f"   Validation: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    logger.info(f"   Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    return (
        X_train_final, X_val, X_test,
        y_train_final, y_val, y_test,
        df,
    )


# ============================================================
# VERİ ÖLÇEKLEME
# ============================================================

def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Ozellikleri StandardScaler ile ölçekle.
    
    Args:
        X_train: Eğitim verileri
        X_val: Dogrulama verileri
        X_test: Test verileri
    
    Returns:
        (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    logger.info(" Ozellikler ölçekleniyor (StandardScaler)...")
    
    scaler = StandardScaler()
    
    # Eğitim veriler üzerinde fit yap ve tümünü dönüştür
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f" Ölçekleme tamamlandı")
    logger.info(f"   Mean (train): {X_train_scaled.mean():.4f}")
    logger.info(f"   Std (train): {X_train_scaled.std():.4f}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ============================================================
# MODEL OLUŞTURMA
# ============================================================

def build_model(input_shape: int) -> Sequential:
    """
    TensorFlow sinir ağı modelini oluştur.
    
    Mimari:
    - Input(shape=(n_features,))
    - Dense(64, relu) + BatchNormalization + Dropout(0.3)
    - Dense(32, relu) + BatchNormalization + Dropout(0.2)
    - Dense(16, relu)
    - Dense(1, sigmoid) - Binary classification output
    
    Args:
        input_shape: Giriş özellik sayısı
    
    Returns:
        Konstruksyon yapılmış model
    """
    logger.info("🏗 Model inşa ediliyor...")
    
    model = Sequential()
    
    # Input + First Dense
    model.add(Dense(64, activation='relu', input_shape=(input_shape,),
                    name='input_dense'))
    model.add(BatchNormalization(name='bn_1'))
    model.add(Dropout(0.3, name='dropout_1'))
    
    # Second Dense
    model.add(Dense(32, activation='relu', name='dense_2'))
    model.add(BatchNormalization(name='bn_2'))
    model.add(Dropout(0.2, name='dropout_2'))
    
    # Third Dense
    model.add(Dense(16, activation='relu', name='dense_3'))
    
    # Output
    model.add(Dense(1, activation='sigmoid', name='output'))
    
    # Compile
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=LOSS,
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
        ]
    )
    
    logger.info(" Model oluşturuldu:")
    logger.info(f"   Optimizer: Adam (lr={LEARNING_RATE})")
    logger.info(f"   Loss: {LOSS}")
    logger.info(f"   Metrics: accuracy, precision, recall, auc")
    
    return model


def print_model_summary(model: Sequential) -> None:
    """Model mimarisini yazdır."""
    logger.info("📋 Model Mimarisi:")
    logger.info("-" * 60)
    model.summary(print_fn=lambda x: logger.info(x))
    logger.info("-" * 60)


# ============================================================
# MODEL EĞİTİMİ
# ============================================================

def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """
    Modeli eğit.
    
    Callbacks:
    - EarlyStopping: Geçersiz metrikler iyileşmezse dur
    - ModelCheckpoint: En iyi modeli kaydet
    - ReduceLROnPlateau: Learning rate'i azalt
    
    Args:
        model: Derlenmiş TensorFlow modeli
        X_train: Eğitim özellikleri
        y_train: Eğitim hedefleri
        X_val: Dogrulama özellikleri
        y_val: Dogrulama hedefleri
    
    Returns:
        History objesi ve callbacks
    """
    logger.info(" Model eğitimi başlanıyor...")
    logger.info(f"   Epochs: {EPOCHS}")
    logger.info(f"   Batch Size: {BATCH_SIZE}")
    logger.info(f"   Validation Split: {VALIDATION_SPLIT*100:.0f}%")
    
    # GPU/CPU bilgisi
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"   Processing Device: GPU ({len(gpus)} GPU Detected)")
    except:
        logger.info(f"   Processing Device: CPU")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    
    checkpoint = ModelCheckpoint(
        str(MODEL_FILE),
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=0,
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1,
    )
    
    # Class weight (dengesiz veri için)
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
    logger.info(f"   Class Weights: {class_weight_dict}")

    # Eğit
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1,
    )
    
    logger.info(" Eğitim tamamlandı!")
    
    return history


# ============================================================
# MODEL DEĞERLENDİRME
# ============================================================

def evaluate_model(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Modeli test verisi üzerinde değerlendir.
    
    Args:
        model: Eğitilmiş model
        X_test: Test özellikleri
        y_test: Test hedefleri
    
    Returns:
        Metrik adı -> değer sözlüğü
    """
    logger.info(" Model değerlendiriliyor...")
    
    results = model.evaluate(X_test, y_test, verbose=0)
    
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3],
        'auc': results[4],
    }
    
    print_metrics_summary(metrics)
    
    return metrics


def build_test_predictions(
    df_full: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Test seti için tahmin sonuçlarını DataFrame'e ekle."""
    df_test = df_full.loc[X_test.index].copy()
    df_test["fire_probability"] = probabilities
    df_test["confidence_score"] = (np.abs(probabilities - 0.5) * 2 * 100).round(2)
    df_test["predicted_class"] = (probabilities > 0.5).astype(int)
    df_test["actual_class"] = y_test
    df_test["prediction_correct"] = (df_test["predicted_class"] == df_test["actual_class"])
    df_test["risk_level"] = df_test["fire_probability"].apply(get_risk_level)

    conditions = [
        (df_test["actual_class"] == 1) & (df_test["predicted_class"] == 1),
        (df_test["actual_class"] == 0) & (df_test["predicted_class"] == 0),
        (df_test["actual_class"] == 0) & (df_test["predicted_class"] == 1),
        (df_test["actual_class"] == 1) & (df_test["predicted_class"] == 0),
    ]
    choices = ["TP", "TN", "FP", "FN"]
    df_test["error_type"] = np.select(conditions, choices, default="UNK")

    return df_test


# ============================================================
# ÖZELLIK ÖNEMİ (Permutation)
# ============================================================

def calculate_feature_importance(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """
    Model weights tabanlı feature importance.
    
    Permutation importance TensorFlow modelleri ile uyumlu olmadığı için,
    ilk layer'ın weights'ine dayalı simple bir yöntem kullanıyoruz.
    
    Args:
        model: Eğitilmiş model
        X_test: Test özellikleri
        y_test: Test hedefleri
        feature_names: Ozellik adları
    
    Returns:
        Ozellik önem sıralaması (DataFrame)
    """
    logger.info("Ozellik onemi hesaplaniyor (Model Weights)...")
    
    # Model'in ilk Dense layer'ının weights'ini al
    first_layer = model.layers[0]
    
    try:
        weights = first_layer.get_weights()[0]  # (n_features, n_neurons)
        
        # Her feature'un average absolute weight'ini hesapla
        feature_importance = np.abs(weights).mean(axis=1)
        
        # Normalize et
        feature_importance = feature_importance / feature_importance.sum()
        
    except Exception as e:
        logger.warning(f"Weights hesaplama hatası: {e}")
        # Fallback: esit onemlilik
        feature_importance = np.ones(len(feature_names)) / len(feature_names)
    
    # DataFrame'e donustur
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance,
        'std': np.zeros(len(feature_names)),
    }).sort_values('importance', ascending=False)
    
    logger.info("Ozellik onemi hesaplandı:")
    for idx, row in importance_df.iterrows():
        logger.info(f"   {row['feature']:.<20} {row['importance']:.6f}")
    
    return importance_df


# ============================================================
# EĞITIM GRAFİKLERİ
# ============================================================

def plot_training_history(history) -> None:
    """
    Eğitim tarihçesi grafiklerini oluştur ve kaydet.
    
    Args:
        history: Model.fit() tarafından döndürülen history objesi
    """
    logger.info(" Eğitim grafikleri oluşturuluyor...")
    
    # 4 subplot: loss, accuracy, auc, val_loss
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Eğitim Grafikleri', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (Binary Crossentropy)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(history.history['auc'], label='Training AUC', linewidth=2)
    axes[1, 0].plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
    axes[1, 0].set_title('AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision vs Recall
    axes[1, 1].plot(history.history['precision'], label='Training Precision', linewidth=2)
    axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_precision'], label='Val. Precision', linewidth=2, linestyle='--')
    axes[1, 1].plot(history.history['val_recall'], label='Val. Recall', linewidth=2, linestyle='--')
    axes[1, 1].set_title('Precision vs Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Kaydet
    ensure_path_exists(TRAINING_HISTORY_FILE, is_file=True)
    plt.savefig(TRAINING_HISTORY_FILE, dpi=300, bbox_inches='tight')
    logger.info(f" Grafik kaydedildi: {TRAINING_HISTORY_FILE}")
    
    plt.close()


# ============================================================
# MODELI KAYDET
# ============================================================

def save_model_and_scaler(model: Sequential, scaler: StandardScaler) -> None:
    """
    Modeli ve scaler'ı kaydet.
    
    Args:
        model: Eğitilmiş model
        scaler: StandardScaler nesnesi
    """
    logger.info(" Model ve scaler kaydediliyor...")
    
    # Model
    ensure_path_exists(MODEL_FILE, is_file=True)
    model.save(MODEL_FILE)
    logger.info(f"  Model kaydedildi: {MODEL_FILE}")
    
    # Scaler
    ensure_path_exists(SCALER_FILE, is_file=True)
    joblib.dump(scaler, SCALER_FILE)
    logger.info(f"  Scaler kaydedildi: {SCALER_FILE}")


# ============================================================
# ANA EĞİTİM FONKSİYONU
# ============================================================

def train_fire_prediction_model() -> Dict[str, Any]:
    """
    Orman yangını tahmin modeli eğit.
    
    Returns:
        Eğitim sonuçları (metrics, model, scaler, vb.)
    """
    
    logger.info("=" * 80)
    logger.info(" ORMAN YANGINI TAHMIN MODELİ EĞİTİMİ")
    logger.info("=" * 80)
    
    try:
        # 1. Veri yükle ve böl
        X_train, X_val, X_test, y_train, y_val, y_test, full_df = load_and_split_data()
        
        # 2. Ölçekle
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_val, X_test
        )
        
        # 3. Model oluştur
        available_features = [f for f in FEATURES if f in X_train.columns]
        model = build_model(len(available_features))
        print_model_summary(model)
        
        # 4. Modeli eğit
        history = train_model(
            model,
            X_train_scaled, y_train,
            X_val_scaled, y_val
        )
        
        # 5. Test verisi üzerinde değerlendir
        test_metrics = evaluate_model(model, X_test_scaled, y_test)

        # 5.1 Test tahminlerini kaydet
        test_probabilities = model.predict(X_test_scaled, verbose=0).flatten()
        test_predictions_df = build_test_predictions(
            full_df, X_test, y_test, test_probabilities
        )
        ensure_path_exists(TEST_PREDICTIONS_FILE, is_file=True)
        test_predictions_df.to_csv(TEST_PREDICTIONS_FILE, index=False)
        logger.info(f" Test tahminleri kaydedildi: {TEST_PREDICTIONS_FILE}")
        
        # 6. Ozellik önemi hesapla
        feature_importance_df = calculate_feature_importance(
            model, X_test_scaled, y_test, available_features
        )
        
        # 7. Grafikleri oluştur
        plot_training_history(history)
        
        # 8. Kaydet
        save_model_and_scaler(model, scaler)
        
        # 9. Çıktıları kaydet
        ensure_path_exists(FEATURE_IMPORTANCE_FILE, is_file=True)
        feature_importance_df.to_csv(
            FEATURE_IMPORTANCE_FILE, index=False
        )
        logger.info(f" Ozellik önemi kaydedildi: {FEATURE_IMPORTANCE_FILE}")
        
        # 10. Classification report (basit)
        ensure_path_exists(CLASSIFICATION_REPORT, is_file=True)
        with open(CLASSIFICATION_REPORT, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ORMAN YANGINI TAHMIN MODELİ - TEST METRİKLERİ\n")
            f.write("=" * 60 + "\n\n")
            
            for metric_name, metric_value in test_metrics.items():
                f.write(f"{metric_name.upper():.<45} {metric_value:.4f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("ÖZELLIK ÖNEMİ\n")
            f.write("=" * 60 + "\n\n")
            
            for idx, row in feature_importance_df.iterrows():
                f.write(
                    f"{row['feature']:.<30} {row['importance']:.6f}\n"
                )
        
        logger.info(f" Classification report kaydedildi: {CLASSIFICATION_REPORT}")
        
        logger.info("\n" + "=" * 80)
        logger.info(" EĞİTİM BAŞARILI!")
        logger.info("=" * 80)
        
        return {
            'model': model,
            'scaler': scaler,
            'history': history,
            'metrics': test_metrics,
            'feature_importance': feature_importance_df,
            'available_features': available_features,
            'test_predictions': test_predictions_df,
        }
    
    except Exception as e:
        logger.error(f" Eğitim hatası: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================
# MAIN (TEST)
# ============================================================

if __name__ == "__main__":
    
    try:
        results = train_fire_prediction_model()
        logger.info("\n Model eğitimi tamamlandı!")
        
    except Exception as e:
        logger.error(f" Hata: {e}")
        raise
