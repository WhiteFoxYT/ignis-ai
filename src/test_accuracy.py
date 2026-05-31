"""
Model Test ve Doğruluk Analizi
==============================
Eğitilmiş modeli test verileri üzerinde değerlendir ve detaylı raporlar oluştur.

Yapılan İşlemler:
1. Test verileri yükle (%30)
2. Model ve scaler yükle
3. Ölçekle ve tahmin yap
4. Doğruluk metriklerini hesapla:
   - Accuracy (Doğruluk Oranı)
   - Precision (Kesinlik)
   - Recall (Duyarlılık / Hassasiyet)
   - F1-Score (Harmonik Ortalama)
   - Confusion Matrix
   - ROC-AUC
5. Detaylı rapor oluştur ve kaydet
6. Görselleştirme (hata analizi, ROC grafiği)
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
)
import joblib
import tensorflow as tf

from config import (
    FEATURES,
    TARGET,
    PROCESSED_DATA_FILE,
    MODEL_FILE,
    SCALER_FILE,
    CLASSIFICATION_REPORT,
    REPORTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    CSV_ENCODING,
)
from utils import (
    setup_logger,
    ensure_path_exists,
)

# Logger
logger = setup_logger(__name__)


# ============================================================
# TEST VERİLERİNİ HAZIRLA
# ============================================================

def load_and_split_test_data() -> Tuple[
    np.ndarray, np.ndarray, pd.DataFrame
]:
    """
    Işlenmiş verileri yükle ve %70-30 split yap.
    Test verilerini döndür.
    
    Returns:
        (X_test_scaled, y_test, scaler) tuple'ı
    """
    logger.info(" Test verileri hazırlanıyor...")
    
    if not PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"İşlenmiş veri dosyası bulunamadı: {PROCESSED_DATA_FILE}\n"
            "Lütfen önce preprocess.py çalıştırın!"
        )
    
    # Veri yükle
    df = pd.read_csv(PROCESSED_DATA_FILE)
    logger.info(f" Yüklendi: {len(df):,} satır")
    
    # Özellikler ve hedef ayır
    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features]
    y = df[TARGET]
    
    # %70 eğitim, %30 test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,  # 0.3 = %30
        random_state=RANDOM_STATE,
        stratify=y,
    )
    
    logger.info(f" Veri bölme:")
    logger.info(f"   Eğitim: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"   Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Scaler ile ölçekle
    if SCALER_FILE.exists():
        scaler = joblib.load(SCALER_FILE)
        logger.info(f" Scaler yüklendi: {SCALER_FILE}")
    else:
        scaler = StandardScaler()
        scaler.fit(X_train)  # Sadece eğitim veriler üzerinde fit
        logger.warning(" Scaler bulunamadı, yeni scaler fit edildi")
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(" Ölçekleme tamamlandı (StandardScaler)")
    
    return X_test_scaled, y_test.values, X_train_scaled, y_train.values


# ============================================================
# MODEL YÜKLE
# ============================================================

def load_model_and_scaler_for_test() -> Tuple[tf.keras.Model, StandardScaler]:
    """
    Kaydedilmiş modeli yükle.
    
    Returns:
        model objesi
    """
    logger.info(" Model yükleniyor...")
    
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {MODEL_FILE}\n"
            "Lütfen önce train.py çalıştırın!"
        )
    
    model = tf.keras.models.load_model(MODEL_FILE)
    logger.info(f" Model yüklendi: {MODEL_FILE}")
    
    return model


# ============================================================
# TAHMIN VE METRIKLER
# ============================================================

def calculate_accuracy_metrics(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Tahmin yap ve tüm doğruluk metriklerini hesapla.
    
    Args:
        model: Eğitilmiş model
        X_test: Test özellikleri
        y_test: Test hedefleri (gerçek değerler)
    
    Returns:
        Metrik adı -> değer sözlüğü
    """
    logger.info(" Tahmin yapılıyor...")
    
    # Probability tahminleri
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    
    # Sınıf tahminleri (threshold=0.5)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    logger.info(f" Tahmin tamamlandı: {len(y_pred)} örnek")
    
    # Metrikler hesapla
    logger.info("\n📊 Doğruluk Metrikleri Hesaplanıyor...")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
    }
    
    # Yazdır
    logger.info(f" Accuracy (Doğruluk Oranı): {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f" Precision (Kesinlik):      {precision:.4f}")
    logger.info(f" Recall (Duyarlılık):       {recall:.4f}")
    logger.info(f" F1-Score:                  {f1:.4f}")
    logger.info(f" ROC-AUC:                   {roc_auc:.4f}")
    
    return metrics


# ============================================================
# CONFUSION MATRIX
# ============================================================

def calculate_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, int]:
    """
    Confusion matrix hesapla ve detayları döndür.
    
    Args:
        y_test: Gerçek değerler
        y_pred: Tahmin edilen değerler
    
    Returns:
        TP, TN, FP, FN sözlüğü
    """
    logger.info("\n 🔍 Hata Analizi (Confusion Matrix)...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    logger.info(f" True Positive (TP):  {tp:,}  - Yangın doğru tahmin")
    logger.info(f" True Negative (TN):  {tn:,}  - Yangın yok doğru tahmin")
    logger.info(f" False Positive (FP): {fp:,}  - Yanlış alarm (yangın olmadığını yangın dedi)")
    logger.info(f" False Negative (FN): {fn:,}  - Kaçırılan yangınlar (yangını algılamadı)")
    
    # Hata oranları
    total = tp + tn + fp + fn
    correct = tp + tn
    incorrect = fp + fn
    
    logger.info(f"\n Hata Oranları:")
    logger.info(f" Doğru Tahminler:  {correct:,} / {total:,} ({correct/total*100:.2f}%)")
    logger.info(f" Yanlış Tahminler: {incorrect:,} / {total:,} ({incorrect/total*100:.2f}%)")
    
    # Yangın tespit oranı
    if (tp + fn) > 0:
        detection_rate = tp / (tp + fn)
        logger.info(f" Yangın Tespit Oranı: {detection_rate*100:.2f}% (FN={fn})")
    
    # False alarm oranı
    if (fp + tn) > 0:
        false_alarm_rate = fp / (fp + tn)
        logger.info(f" Yanlış Alarm Oranı: {false_alarm_rate*100:.2f}% (FP={fp})")
    
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'correct': correct, 'incorrect': incorrect, 'total': total
    }


# ============================================================
# DETAYLI RAPOR
# ============================================================

def create_detailed_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    cm_dict: Dict[str, int],
) -> str:
    """
    Detaylı metin raporu oluştur.
    
    Args:
        y_test: Gerçek değerler
        y_pred: Tahmin edilen değerler
        metrics: Metrik sözlüğü
        cm_dict: Confusion matrix sözlüğü
    
    Returns:
        Rapor metni
    """
    report_lines = [
        "=" * 80,
        "ORMAN YANĞINI RİSK TAHMİN MODELİ - TEST DOĞRULUK RAPORU",
        "=" * 80,
        "",
        "🎯 GENEL DOĞRULUK METRİKLERİ",
        "-" * 80,
        f"Accuracy (Doğruluk Oranı): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
        f"Precision (Kesinlik):       {metrics['precision']:.4f}",
        f"Recall (Duyarlılık):        {metrics['recall']:.4f}",
        f"F1-Score (Harmonik Ort.):   {metrics['f1_score']:.4f}",
        f"ROC-AUC (Eğri Altı Alan):   {metrics['roc_auc']:.4f}",
        "",
        "📊 CONFUSION MATRIX (Hata Analizi)",
        "-" * 80,
        f"True Positive (TP):         {cm_dict['TP']:,}  (Yangın doğru tahmin)",
        f"True Negative (TN):         {cm_dict['TN']:,}  (Yangın yok doğru tahmin)",
        f"False Positive (FP):        {cm_dict['FP']:,}  (Yanlış alarm)",
        f"False Negative (FN):        {cm_dict['FN']:,}  (Kaçırılan yangınlar)",
        "",
        f"Doğru Tahminler (TP+TN):    {cm_dict['correct']:,} / {cm_dict['total']:,} ({cm_dict['correct']/cm_dict['total']*100:.2f}%)",
        f"Yanlış Tahminler (FP+FN):   {cm_dict['incorrect']:,} / {cm_dict['total']:,} ({cm_dict['incorrect']/cm_dict['total']*100:.2f}%)",
        "",
        "⚠️  UYARILAR VE ÖNERİLER",
        "-" * 80,
    ]
    
    # Yangın tespit oranı
    if (cm_dict['TP'] + cm_dict['FN']) > 0:
        detection_rate = cm_dict['TP'] / (cm_dict['TP'] + cm_dict['FN'])
        report_lines.append(f"Yangın Tespit Oranı: {detection_rate*100:.2f}%")
        if detection_rate < 0.8:
            report_lines.append(f"  ⚠️  UYARI: Yangın tespit oranı düşük ({cm_dict['FN']} yangın kaçırıldı)")
    
    # False alarm oranı
    if (cm_dict['FP'] + cm_dict['TN']) > 0:
        false_alarm_rate = cm_dict['FP'] / (cm_dict['FP'] + cm_dict['TN'])
        report_lines.append(f"Yanlış Alarm Oranı: {false_alarm_rate*100:.2f}%")
        if false_alarm_rate > 0.1:
            report_lines.append(f"  ⚠️  UYARI: Yanlış alarm oranı yüksek ({cm_dict['FP']} yanlış alarm)")
    
    report_lines.extend([
        "",
        "📈 SINIFLANDIRMA RAPORU (Classification Report)",
        "-" * 80,
        classification_report(y_test, y_pred, 
                            target_names=['Yangın Yok (0)', 'Yangın Var (1)']),
        "",
        "=" * 80,
        f"Rapor Tarihi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
    ])
    
    return "\n".join(report_lines)


# ============================================================
# GÖRSELLEŞTIRME
# ============================================================

def create_accuracy_visualizations(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prob: np.ndarray,
    metrics: Dict[str, float],
    cm_dict: Dict[str, int],
) -> None:
    """
    Doğruluk görselleştirmelerini oluştur ve kaydet.
    
    Args:
        y_test: Gerçek değerler
        y_pred: Tahmin edilen sınıflar
        y_pred_prob: Tahmin olasılıkları
        metrics: Metrik sözlüğü
        cm_dict: Confusion matrix sözlüğü
    """
    logger.info("\n📊 Görselleştirmeler oluşturuluyor...")
    
    # 3x2 subplot
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Yangın Yok', 'Yangın Var'],
                yticklabels=['Yangın Yok', 'Yangın Var'])
    ax1.set_title('Confusion Matrix (Hata Matrisi)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gerçek Değer')
    ax1.set_xlabel('Tahmin Edilen Değer')
    
    # 2. Metrik Çubukları
    ax2 = plt.subplot(2, 3, 2)
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['roc_auc']
    ]
    colors = ['#2ecc71' if v >= 0.8 else '#f39c12' if v >= 0.7 else '#e74c3c' 
              for v in metric_values]
    bars = ax2.bar(metric_names, metric_values, color=colors)
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Skor')
    ax2.set_title('Doğruluk Metrikleri', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Değerleri çubukların üstüne yaz
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. ROC Eğrisi
    ax3 = plt.subplot(2, 3, 3)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc_val = auc(fpr, tpr)
    ax3.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Eğrisi (AUC = {roc_auc_val:.3f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Rastgele')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Eğrisi', fontsize=12, fontweight='bold')
    ax3.legend(loc="lower right")
    ax3.grid(alpha=0.3)
    
    # 4. Precision-Recall Eğrisi
    ax4 = plt.subplot(2, 3, 4)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
    ax4.plot(recall_vals, precision_vals, color='blue', lw=2)
    ax4.set_xlabel('Recall (Duyarlılık)')
    ax4.set_ylabel('Precision (Kesinlik)')
    ax4.set_title('Precision-Recall Eğrisi', fontsize=12, fontweight='bold')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.grid(alpha=0.3)
    
    # 5. Olasılık Dağılımı
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(y_pred_prob[y_test == 0], bins=30, alpha=0.6, label='Yangın Yok (Gerçek)', color='green')
    ax5.hist(y_pred_prob[y_test == 1], bins=30, alpha=0.6, label='Yangın Var (Gerçek)', color='red')
    ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax5.set_xlabel('Tahmin Olasılığı')
    ax5.set_ylabel('Frekans')
    ax5.set_title('Olasılık Dağılımı', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Error Analysis
    ax6 = plt.subplot(2, 3, 6)
    error_types = ['TP\n(Doğru Pos.)', 'TN\n(Doğru Neg.)', 'FP\n(Yanlış Pos.)', 'FN\n(Yanlış Neg.)']
    error_counts = [cm_dict['TP'], cm_dict['TN'], cm_dict['FP'], cm_dict['FN']]
    colors_error = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax6.bar(error_types, error_counts, color=colors_error)
    ax6.set_ylabel('Sayı')
    ax6.set_title('Hata Analizi', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # Değerleri çubukların üstüne yaz
    for bar, val in zip(bars, error_counts):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Model Test Doğruluk Analizi', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Kaydet
    viz_file = REPORTS_DIR / "accuracy_analysis.png"
    ensure_path_exists(viz_file, is_file=True)
    plt.savefig(str(viz_file), dpi=300, bbox_inches='tight')
    logger.info(f" Görselleştirme kaydedildi: {viz_file}")
    plt.close()


# ============================================================
# ANA FONKSİYON
# ============================================================

def test_model_accuracy() -> None:
    """Ana test ve doğruluk analizi fonksiyonu."""
    
    logger.info("\n" + "=" * 80)
    logger.info(" ORMAN YANĞINI RİSK TAHMİN MODELİ - TEST VE DOĞRULUK ANALİZİ")
    logger.info("=" * 80 + "\n")
    
    try:
        # 1. Test verileri yükle
        X_test, y_test, X_train, y_train = load_and_split_test_data()
        
        # 2. Model yükle
        model = load_model_and_scaler_for_test()
        
        # 3. Metrikleri hesapla
        metrics = calculate_accuracy_metrics(model, X_test, y_test)
        
        # 4. Confusion matrix
        cm_dict = calculate_confusion_matrix(y_test, metrics['y_pred'])
        
        # 5. Detaylı rapor oluştur
        report = create_detailed_report(
            y_test,
            metrics['y_pred'],
            metrics,
            cm_dict
        )
        
        # 6. Raporu kaydet
        ensure_path_exists(CLASSIFICATION_REPORT, is_file=True)
        with open(CLASSIFICATION_REPORT, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"\n📄 Rapor kaydedildi: {CLASSIFICATION_REPORT}")
        
        # 7. Raporu console'da yazdır
        logger.info("\n" + report)
        
        # 8. Görselleştirmeler oluştur
        create_accuracy_visualizations(
            y_test,
            metrics['y_pred'],
            metrics['y_pred_prob'],
            metrics,
            cm_dict
        )
        
        logger.info("\n" + "=" * 80)
        logger.info(" ✅ TEST VE DOĞRULUK ANALİZİ TAMAMLANDI")
        logger.info("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"❌ HATA: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test_model_accuracy()
