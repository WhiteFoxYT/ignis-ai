"""
Hızlı Başlangıç - Örnek Kullanımlar
VS Code'da GEE verilerini işlemek için örnekler
"""

from gee_data_processor import (
    initialize_gee,
    get_fire_data,
    get_ndvi_data,
    get_temperature_data,
    sample_points,
    combine_data,
    export_to_csv
)
from gee_config import GEEConfig
import os


def example_1_basic_export():
    """Örnek 1: Temel veri aktarımı"""
    print("\n" + "="*60)
    print("ÖRNEK 1: Temel Veri Aktarımı")
    print("="*60)
    
    if not initialize_gee():
        return
    
    # Tarihler
    start_date = '2022-01-01'
    end_date = '2022-12-31'
    
    # Yangın verilerini al
    fire_data, turkiye = get_fire_data(start_date, end_date)
    
    # NDVI verilerini al
    ndvi_data = get_ndvi_data(start_date, end_date, turkiye)
    
    # Örnekle
    combined = ndvi_data.addBands(fire_data)
    sampled = sample_points(combined, turkiye, num_points=500)
    
    # Kaydet
    os.makedirs('example_outputs', exist_ok=True)
    export_to_csv(sampled, 'example_outputs/example_1_fire_ndvi.csv')


def example_2_all_features():
    """Örnek 2: Tüm özelliklerle veri"""
    print("\n" + "="*60)
    print("ÖRNEK 2: Tüm Özelliklerle Veri")
    print("="*60)
    
    if not initialize_gee():
        return
    
    start_date = '2023-06-01'
    end_date = '2023-08-31'  # Sadece yaz ayları
    
    print(f"📅 Veri toplanıyor: {start_date} - {end_date}")
    
    # Ana veri koleksiyonları
    fire_data, turkiye = get_fire_data(start_date, end_date)
    ndvi_data = get_ndvi_data(start_date, end_date, turkiye)
    temp_data = get_temperature_data(start_date, end_date, turkiye)
    
    # Tümünü birleştir
    combined = ndvi_data.addBands(temp_data).addBands(fire_data)
    
    # Daha fazla nokta örnekle
    sampled = sample_points(combined, turkiye, num_points=2000)
    
    # Veriye ekle
    os.makedirs('example_outputs', exist_ok=True)
    export_to_csv(sampled, 'example_outputs/example_2_summer_data.csv')


def example_3_custom_dates():
    """Örnek 3: Özel tarih aralıkları"""
    print("\n" + "="*60)
    print("ÖRNEK 3: Özel Tarih Aralıkları")
    print("="*60)
    
    if not initialize_gee():
        return
    
    # Farklı aylar için veriler
    months = [
        ('2023-01-01', '2023-01-31', 'January'),
        ('2023-06-01', '2023-06-30', 'June'),
        ('2023-12-01', '2023-12-31', 'December'),
    ]
    
    os.makedirs('example_outputs', exist_ok=True)
    
    for start, end, month_name in months:
        print(f"\n  📅 {month_name} işleniyor...")
        
        fire_data, turkiye = get_fire_data(start, end)
        ndvi_data = get_ndvi_data(start, end, turkiye)
        combined = ndvi_data.addBands(fire_data)
        sampled = sample_points(combined, turkiye, num_points=300)
        
        output_file = f'example_outputs/example_3_{month_name.lower()}.csv'
        export_to_csv(sampled, output_file)


def example_4_config_info():
    """Örnek 4: Konfigürasyon bilgisi"""
    print("\n" + "="*60)
    print("ÖRNEK 4: Konfigürasyon Bilgisi")
    print("="*60)
    
    GEEConfig.print_config()
    
    print("\n📊 Veri Kaynakları:")
    for key, source in GEEConfig.DATA_SOURCES.items():
        print(f"  • {source['description']}")
        print(f"    Collection: {source['collection']}")
        print(f"    Band: {source['band']}\n")


if __name__ == "__main__":
    print("\n🔥 GOOGLE EARTH ENGINE - ÖRNEK KULLANIMLAR 🔥")
    print("Ayıklamak istediğiniz örneği seçin:\n")
    
    print("1. Temel Veri Aktarımı (Yangın + NDVI)")
    print("2. Tüm Özelliklerle Veri (Yaz ayları)")
    print("3. Özel Tarih Aralıkları (Aylar)")
    print("4. Konfigürasyon Bilgisi")
    print("5. Hepsini Çalıştır")
    
    choice = input("\nSeçiminiz (1-5): ").strip()
    
    try:
        if choice == '1':
            example_1_basic_export()
        elif choice == '2':
            example_2_all_features()
        elif choice == '3':
            example_3_custom_dates()
        elif choice == '4':
            example_4_config_info()
        elif choice == '5':
            example_1_basic_export()
            example_2_all_features()
            example_3_custom_dates()
            example_4_config_info()
        else:
            print("❌ Geçersiz seçim!")
    
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ İş tamamlandı!")
