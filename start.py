#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🔥 Orman Yangını Tahmin Sistemi - Ana Başlatıcı Script

Bu script tüm işleyişi otomatik olarak başlatır:
1. Veri Ön İşleme
2. Model Eğitimi  
3. Test ve Doğruluk Analizi
4. Tahmin Yapma
5. Harita Oluşturma

Kullanım:
    python start.py

Version: 1.0.0
"""

import sys
import time
import os
from pathlib import Path

# ============================================================
# TENSORFLOW UYARILARI - GİZLE (Normal başlangıç mesajları)
# ============================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Sadece ERROR ve WARNING göster
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # GPU memory dinamik büyüme

# Renkli çıktı için
class Colors:
    """Terminal renkleri"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header():
    """Ana başlık yazdır"""
    header = """
    ╔════════════════════════════════════════════════════════╗
    ║                                                        ║
    ║        🔥 ORMAN YANGINI TAHMIN SİSTEMİ 🔥            ║
    ║                                                        ║
    ║        Tüm İşleyiş Başlatıcısı (start.py)            ║
    ║                                                        ║
    ║        Version: 1.0.0                                 ║
    ║        Status: ✅ Üretim Hazır                        ║
    ║                                                        ║
    ╚════════════════════════════════════════════════════════╝
    """
    print(f"{Colors.CYAN}{Colors.BOLD}{header}{Colors.ENDC}")


def print_section(title, number=None):
    """Bölüm başlığı yazdır"""
    if number:
        prefix = f"[{number}]"
    else:
        prefix = "→"
    
    line = f"\n{Colors.BOLD}{Colors.BLUE}{prefix} {title}{Colors.ENDC}"
    print(line)
    print(f"{Colors.BLUE}{'─' * 60}{Colors.ENDC}")


def print_success(message):
    """Başarı mesajı yazdır"""
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")


def print_warning(message):
    """Uyarı mesajı yazdır"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")


def print_error(message):
    """Hata mesajı yazdır"""
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")


def print_info(message):
    """Bilgi mesajı yazdır"""
    print(f"{Colors.CYAN}ℹ️  {message}{Colors.ENDC}")


def check_environment():
    """Ortamı kontrol et"""
    print_section("ORTAM KONTROLÜ", 0)
    
    # Gerekli paketler kontrol et
    required_packages = {
        'tensorflow': 'TensorFlow/Keras',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'folium': 'Folium',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'joblib': 'Joblib',
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print_success(f"{name} kuruldu")
        except ImportError:
            print_error(f"{name} YOK")
            missing_packages.append(package)
    
    if missing_packages:
        print_warning(f"\nEksik paketler: {', '.join(missing_packages)}")
        print_info("Lütfen şu komutu çalıştırın:")
        print(f"{Colors.YELLOW}pip install {' '.join(missing_packages)}{Colors.ENDC}")
        sys.exit(1)
    
    print_success("Tüm paketler hazır!")


def check_data():
    """Veri dosyalarını kontrol et"""
    print_section("VERİ DOSYALARI KONTROLÜ", 1)
    
    base_dir = Path(__file__).parent
    data_raw_dir = base_dir / "data" / "raw"
    
    required_files = [
        "yangin_model_verisi.csv",
        "yangin_rastgele_verisi.csv",
    ]
    
    for file in required_files:
        file_path = data_raw_dir / file
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            print_success(f"{file} var ({file_size:.1f} MB)")
        else:
            print_error(f"{file} BULUNAMADI!")
            print_info(f"Dosya yolu: {file_path}")
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {file_path}")


def run_step(step_name, script_name, step_number):
    """Bir adımı çalıştır"""
    print_section(step_name, step_number)
    
    try:
        base_dir = Path(__file__).parent
        script_path = base_dir / "src" / script_name
        
        print_info(f"Çalıştırılıyor: {script_path}")
        
        start_time = time.time()
        
        # Script'i import ve çalıştır
        import importlib.util
        spec = importlib.util.spec_from_file_location("module", script_path)
        module = importlib.util.module_from_spec(spec)
        
        # sys.argv'i kontrol et (main() varsa çağır)
        sys.path.insert(0, str(base_dir / "src"))
        spec.loader.exec_module(module)
        
        # main() fonksiyonu varsa çağır
        if hasattr(module, 'main'):
            module.main()
        else:
            # Script'e özel giriş fonksiyonu çağır
            entrypoints = {
                "preprocess.py": "preprocess_data",
                "train.py": "train_fire_prediction_model",
                "test_accuracy.py": "test_model_accuracy",
                "predict.py": "predict_fire_risk",
                "map_visualization.py": "create_risk_map",
            }
            func_name = entrypoints.get(script_name)
            if func_name and hasattr(module, func_name):
                getattr(module, func_name)()
            else:
                raise RuntimeError(f"Giriş fonksiyonu bulunamadı: {script_name}")
        
        elapsed_time = time.time() - start_time
        print_success(f"{step_name} tamamlandı ({elapsed_time:.1f}s)")
        
        return True
        
    except Exception as e:
        print_error(f"{step_name} hatası: {e}")
        print_info("Hata detayları:")
        import traceback
        traceback.print_exc()
        return False


def show_summary():
    """Özet göster"""
    print_section("İŞLEYİŞ ÖZETI")
    
    base_dir = Path(__file__).parent
    output_dir = base_dir / "outputs"
    
    print_info("\n✅ Oluşturulan Dosyalar:\n")
    
    files_to_check = [
        ("Eğitilmiş Model", "models/yangin_model.keras"),
        ("Scaler Dosyası", "models/scaler.pkl"),
        ("İşlenmiş Veri", "data/processed/processed_data.csv"),
        ("Tahmin Sonuçları", "outputs/predictions.csv"),
        ("Test Tahminleri", "outputs/test_predictions.csv"),
        ("İnteraktif Harita", "outputs/maps/fire_risk_map.html"),
        ("Eğitim Grafikleri", "outputs/reports/training_history.png"),
        ("Doğruluk Analizi", "outputs/reports/accuracy_analysis.png"),
        ("Test Raporu", "outputs/reports/classification_report.txt"),
        ("Veri Kalitesi Raporu", "outputs/reports/data_quality_report.txt"),
    ]
    
    for description, file_path in files_to_check:
        full_path = base_dir / file_path
        if full_path.exists():
            print_success(f"{description}: ✅")
        else:
            print_warning(f"{description}: ⚠️ (oluşturulmamış)")
    
    print("\n" + Colors.CYAN + "─" * 60 + Colors.ENDC)
    
    print_info("\n📊 Sonraki Adımlar:\n")
    print(f"  1. Haritayı Aç:")
    print(f"     {Colors.YELLOW}start outputs/maps/fire_risk_map.html{Colors.ENDC}")
    
    print(f"\n  2. Tahminleri Kontrol Et:")
    print(f"     {Colors.YELLOW}cat outputs/predictions.csv{Colors.ENDC}")
    
    print(f"\n  3. Detaylı Raporu Oku:")
    print(f"     {Colors.YELLOW}cat outputs/reports/classification_report.txt{Colors.ENDC}")


def show_error_summary():
    """Hata özeti göster"""
    print("\n" + Colors.RED + Colors.BOLD)
    print("╔════════════════════════════════════════════════════════╗")
    print("║  ❌ İŞLEYİŞ SIRASINDA HATA OLUŞTU                    ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(Colors.ENDC)
    
    print_info("Sorun Giderme İpuçları:")
    print(f"  • {Colors.YELLOW}requirements.txt{Colors.ENDC} dosyasındaki paketleri kontrol et")
    print(f"  • {Colors.YELLOW}data/raw/{Colors.ENDC} klasöründe veri dosyaları olduğundan emin ol")
    print(f"  • {Colors.YELLOW}README.md{Colors.ENDC}'de 'Hata Ayıklama' kısmını oku")
    print(f"  • {Colors.YELLOW}GPU{Colors.ENDC} hatasıysa CPU modunda çalıştır:")
    print(f"    {Colors.YELLOW}pip install tensorflow-cpu{Colors.ENDC}")


def main():
    """Ana fonksiyon"""
    try:
        # Başlık
        print_header()
        
        # Ortamı kontrol et
        check_environment()
        
        # Veri dosyalarını kontrol et
        check_data()
        
        # İşleyiş adımları
        steps = [
            ("VERİ ÖN İŞLEME", "preprocess.py", 2),
            ("MODEL EĞİTİMİ", "train.py", 3),
            ("TEST VE DOĞRULUK ANALİZİ", "test_accuracy.py", 4),
            ("TAHMİN YAPMA", "predict.py", 5),
            ("HARITA OLUŞTURMA", "map_visualization.py", 6),
        ]
        
        failed_steps = []
        
        for step_name, script_name, step_number in steps:
            if not run_step(step_name, script_name, step_number):
                failed_steps.append(step_name)
                # Devam et başka adımlara
                print_warning(f"{step_name} başarısız oldu, sonraki adıma geçiliyor...")
                continue
            
            print()  # Boş satır
        
        # Sonuç
        if not failed_steps:
            print_section("TAMAMLANDI! ✅")
            print(f"{Colors.GREEN}{Colors.BOLD}")
            print("╔════════════════════════════════════════════════════════╗")
            print("║                                                        ║")
            print("║  ✅ TÜM ADIMLAR BAŞARIYLA TAMAMLANDI!               ║")
            print("║                                                        ║")
            print("╚════════════════════════════════════════════════════════╝")
            print(Colors.ENDC)
            
            show_summary()
            
            print("\n" + Colors.GREEN + Colors.BOLD + "🔥 Sistem hazır! Haritayı açabilirsiniz." + Colors.ENDC + "\n")
        else:
            show_error_summary()
            print(f"\nBaşarısız Adımlar: {', '.join(failed_steps)}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print_error("\n\nİşlem kullanıcı tarafından durduruldu!")
        sys.exit(0)
    
    except Exception as e:
        print_error(f"\n\nCiddi Hata: {e}")
        import traceback
        traceback.print_exc()
        show_error_summary()
        sys.exit(1)


if __name__ == "__main__":
    main()
