"""
Ana Pipeline
============
Tüm adımları sırasıyla çalıştır:
1. Veri ön işleme
2. Model eğitimi
3. Tahmin yapma
4. Harita oluşturma

Kullanım:
    python main.py
"""

import logging
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from config import ensure_directories_exist, FIRE_RISK_MAP
from utils import setup_logger, print_metrics_summary
from preprocess import preprocess_data
from train import train_fire_prediction_model
from predict import predict_fire_risk
from map_visualization import create_risk_map

# Logger
logger = setup_logger(__name__)


# ============================================================
# PIPELINE KONTROL
# ============================================================

def print_welcome() -> None:
    """Hoşgeldiniz mesajını yazdır."""
    print("\n" + "=" * 80)
    print("ORMAN YANGINI TAHMIN SİSTEMİ - V 1.0.0")
    print("=" * 80)
    print("\nPipeline Adımları:")
    print("  1. Veri Ön Isleme (Preprocess)")
    print("  2. Model Eğitimi (Training)")
    print("  3. Yangın Riski Tahlı (Prediction)")
    print("  4. İnteraktif Harita Oluşturma (Visualization)")
    print("\n" + "=" * 80 + "\n")


def print_step(step_number: int, step_name: str) -> None:
    """Adım başlığı yazdır."""
    print("\n")
    logger.info("=" * 80)
    logger.info(f"ADIM {step_number}/4: {step_name}")
    logger.info("=" * 80)


def print_complete() -> None:
    """Tamamlanma mesajı."""
    print("\n")
    logger.info("=" * 80)
    logger.info("TUM ADIMLAR BASARILI!")
    logger.info("=" * 80)
    logger.info("\nSonuclar:\n")
    
    from config import (
        PROCESSED_DATA_FILE,
        MODEL_FILE,
        SCALER_FILE,
        PREDICTIONS_FILE,
        FIRE_RISK_MAP,
        TRAINING_HISTORY_FILE,
        FEATURE_IMPORTANCE_FILE,
        CLASSIFICATION_REPORT,
    )
    
    files_to_check = [
        ("Islenmi Veri", PROCESSED_DATA_FILE),
        ("Egitilmis Model", MODEL_FILE),
        ("Scaler", SCALER_FILE),
        ("Tahminler", PREDICTIONS_FILE),
        ("Yangin Risk Haritasi", FIRE_RISK_MAP),
        ("Egitim Grafikleri", TRAINING_HISTORY_FILE),
        ("Ozellik Onemi", FEATURE_IMPORTANCE_FILE),
        ("Classification Report", CLASSIFICATION_REPORT),
    ]
    
    for file_name, file_path in files_to_check:
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1024**2
            logger.info(f"  {file_name:.<40} {file_path.name} ({size_mb:.2f} MB)")
        else:
            logger.info(f"  {file_name:.<40} Yok")
    
    logger.info("\n" + "=" * 80)
    print("\nTebrikler! Sistem tamamen calistirildi!\n")


# ============================================================
# PIPELINE MAIN
# ============================================================

def run_pipeline(
    skip_steps: list = None,
) -> dict:
    """
    Tüm pipeline'ı çalıştır.
    
    Args:
        skip_steps: Atlanacak adımlar (ör: [1, 2])
    
    Returns:
        Pipeline sonuçları
    """
    
    skip_steps = skip_steps or []
    results = {}
    
    print_welcome()
    
    # Dizinleri sağla
    ensure_directories_exist()
    logger.info(" Dizinler kontrol edildi/oluşturuldu\n")
    
    try:
        # ========== ADIM 1: VERİ ÖN İŞLEME ==========
        if 1 not in skip_steps:
            print_step(1, "VERİ ÖN İŞLEME (PREPROCESS)")
            
            try:
                processed_df = preprocess_data()
                results['preprocessed_data'] = processed_df
                logger.info(f" Adım 1 Tamamlandi: {len(processed_df):,} satır işlendi\n")
            
            except FileNotFoundError as e:
                logger.error(f" Adım 1 Basarisiz: {e}")
                logger.error("   💡 Lutfen veri dosyasının varlığını kontrol edin:")
                logger.error("      - data/raw/yangin_model_verisi.csv")
                raise
        else:
            logger.info("  Adım 1 Atlandi (Preprocess)\n")
        
        # ========== ADIM 2: MODEL EĞİTİMİ ==========
        if 2 not in skip_steps:
            print_step(2, "MODEL EĞİTİMİ (TRAINING)")
            
            try:
                training_results = train_fire_prediction_model()
                results['model'] = training_results['model']
                results['scaler'] = training_results['scaler']
                results['metrics'] = training_results['metrics']
                
                logger.info(f" Adım 2 Tamamlandi\n")
            
            except Exception as e:
                logger.error(f" Adım 2 Basarisiz: {e}")
                raise
        else:
            logger.info("  Adım 2 Atlandi (Training)\n")
        
        # ========== ADIM 3: TAHMIN ==========
        if 3 not in skip_steps:
            print_step(3, "YANGINI RİSK TAHMİNİ (PREDICTION)")
            
            try:
                predictions_df = predict_fire_risk()
                results['predictions'] = predictions_df
                
                logger.info(f" Adım 3 Tamamlandi: {len(predictions_df):,} tahmin yapıldı\n")
            
            except Exception as e:
                logger.error(f" Adım 3 Basarisiz: {e}")
                raise
        else:
            logger.info("  Adım 3 Atlandi (Prediction)\n")
        
        # ========== ADIM 4: HARITA OLUŞTURMA ==========
        if 4 not in skip_steps:
            print_step(4, "YANGIN RİSK HARİTASI (VISUALIZATION)")
            
            try:
                # Optimized harita: clustering + basit popup + sampling
                map_obj = create_risk_map(
                    add_heatmap=True,
                    add_clusters=True,
                    max_markers=5000,
                    simplified_popups=True
                )
                results['map'] = map_obj
                
                logger.info(f" Adım 4 Tamamlandi\n")
            
            except Exception as e:
                logger.error(f" Adım 4 Basarisiz: {e}")
                raise
        else:
            logger.info("  Adım 4 Atlandi (Visualization)\n")
        
        # Harita varsa tarayıcıda aç
        if FIRE_RISK_MAP.exists():
            try:
                logger.info("\n Harita tarayicida aciliyor...")
                map_url = FIRE_RISK_MAP.absolute().as_uri()
                webbrowser.open(map_url)
                logger.info(" Tarayici acildi!")
            except Exception as e:
                logger.warning(f" Tarayici acilmadi: {e}")
                logger.info(f" Manuel olarak acin: {FIRE_RISK_MAP.absolute()}")
        
        # Tamamlanma mesajı
        print_complete()
        
        return results
    
    except KeyboardInterrupt:
        logger.error("\n\n Pipeline kullanıcı tarafından iptal edildi!")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"\n\n PIPELINE HATASI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================
# QUICK MODE - Tahmin için sadece 3 ve 4
# ============================================================

def quick_predict() -> dict:
    """
    Hızlı tahmin modu (eğitim atla).
    
    Model ve scaler'ın var olduğunu varsayar.
    """
    logger.info(" HIZLI TAHMIN MODU")
    logger.info("   (Eğitim adımı atlanıyor)")
    
    return run_pipeline(skip_steps=[1, 2])


# ============================================================
# RETRAIN MODE - Sadece eğitimi tekrarla
# ============================================================

def retrain_model() -> dict:
    """
    Modeli yeniden eğit (tahmin atla).
    """
    logger.info("🔄 RETRAIN MODU")
    logger.info("   (Model yeniden eğitiliyor)")
    
    return run_pipeline(skip_steps=[3, 4])


# ============================================================
# CLI INTERFACE
# ============================================================

def print_usage() -> None:
    """Kullanım bilgisi yazdır."""
    print("\n" + "=" * 80)
    print("KULLANIM")
    print("=" * 80 + "\n")
    print("Tam Pipeline (Tüm adımlar):")
    print("  python main.py\n")
    
    print("Hızlı Tahmin (preprocess + training atla):")
    print("  python main.py --quick\n")
    
    print("Modeli Yeniden Eğit (prediction atla):")
    print("  python main.py --retrain\n")
    
    print("Sadece Veri Ön Isleme:")
    print("  python src/preprocess.py\n")
    
    print("Sadece Model Eğitimi:")
    print("  python src/train.py\n")
    
    print("Sadece Tahmin:")
    print("  python src/predict.py\n")
    
    print("Sadece Harita:")
    print("  python src/map_visualization.py\n")
    print("=" * 80 + "\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    # Argümanları kontrol et
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['-h', '--help']:
            print_usage()
        
        elif arg in ['-q', '--quick']:
            logger.info(" Quick mode seçildi")
            quick_predict()
        
        elif arg in ['-r', '--retrain']:
            logger.info("🔄 Retrain mode seçildi")
            retrain_model()
        
        elif arg in ['-v', '--version']:
            print("Orman Yangını Tahmin Sistemi v1.0.0")
        
        else:
            print(f" Bilinmeyen argüman: {arg}")
            print_usage()
    
    else:
        # Tam pipeline
        run_pipeline()
