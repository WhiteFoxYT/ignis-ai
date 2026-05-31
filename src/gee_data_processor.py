"""
Google Earth Engine - Fire Detection Data Processing
Colab notebook'dan VS Code'a uyarlanmış versiyon
"""

import ee
import pandas as pd
import os
from datetime import datetime

# ============================================================
# 1. GOOGLE EARTH ENGINE İNİT
# ============================================================

def initialize_gee(project_id='ignisai-496207'):
    """
    Google Earth Engine başlat
    
    Not: İlk kez çalıştırırken terminal'de şu komutu çalıştırın:
    earthengine authenticate
    """
    try:
        ee.Initialize(project=project_id)
        print("✅ Google Earth Engine başarıyla bağlandı!")
    except Exception as e:
        print(f"❌ GEE başlatma hatası: {e}")
        print("Lütfen çalıştırın: earthengine authenticate")
        return False
    return True


# ============================================================
# 2. VERİ TOPLAMA FONKSİYONLARI
# ============================================================

def get_fire_data(start_date, end_date):
    """Yangın verileri (MODIS MCD64A1)"""
    print(f"🔥 Yangın verileri toplanıyor: {start_date} - {end_date}")
    
    turkiye = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017') \
        .filter(ee.Filter.eq('country_na', 'Turkey'))
    
    fire = ee.ImageCollection('MODIS/061/MCD64A1') \
        .filterDate(start_date, end_date) \
        .select('BurnDate') \
        .max() \
        .clip(turkiye) \
        .rename('etiket')
    
    return fire, turkiye


def get_ndvi_data(start_date, end_date, turkiye):
    """NDVI verileri (MODIS MOD13Q1)"""
    print("🌿 NDVI verileri toplanıyor...")
    
    ndvi = ee.ImageCollection('MODIS/061/MOD13Q1') \
        .filterDate(start_date, end_date) \
        .select('NDVI') \
        .mean() \
        .clip(turkiye) \
        .rename('NDVI')
    
    return ndvi


def get_temperature_data(start_date, end_date, turkiye):
    """Hava sıcaklığı (ERA5)"""
    print("🌡️ Sıcaklık verileri toplanıyor...")
    
    hava_sicakligi = ee.ImageCollection('ECMWF/ERA5/DAILY') \
        .filterDate(start_date, end_date) \
        .select('mean_2m_air_temperature') \
        .mean() \
        .clip(turkiye) \
        .rename('hava_sicakligi')
    
    return hava_sicakligi - 273.15  # Kelvin to Celsius


def get_dewpoint_data(start_date, end_date, turkiye):
    """Çiğlenme noktası (ERA5)"""
    print("💧 Çiğlenme verileri toplanıyor...")
    
    cignenme = ee.ImageCollection('ECMWF/ERA5/DAILY') \
        .filterDate(start_date, end_date) \
        .select('dewpoint_2m_temperature') \
        .mean() \
        .clip(turkiye) \
        .rename('cignenme')
    
    return cignenme - 273.15  # Kelvin to Celsius


def calculate_humidity(hava_sicakligi_c, cignenme_c):
    """Bağıl nem hesapla (Magnus formülü)"""
    print("💯 Bağıl nem hesaplanıyor...")
    
    a = cignenme_c.multiply(17.625).divide(cignenme_c.add(243.04))
    b = hava_sicakligi_c.multiply(17.625).divide(hava_sicakligi_c.add(243.04))
    
    bagil_nem = a.subtract(b).exp().multiply(100).rename('bagil_nem')
    bagil_nem = bagil_nem.clamp(0, 100)
    
    return bagil_nem


def get_temperature_modis(start_date, end_date, turkiye):
    """MODIS LST (Arazi Yüzey Sıcaklığı)"""
    print("🌡️ MODIS LST sıcaklık verileri toplanıyor...")
    
    lst = ee.ImageCollection('MODIS/061/MOD11A1') \
        .filterDate(start_date, end_date) \
        .select('LST_Day_1km') \
        .mean() \
        .clip(turkiye) \
        .rename('sicaklik')
    
    return lst


def get_precipitation_data(start_date, end_date, turkiye):
    """Yağış verileri (CHIRPS)"""
    print("🌧️ Yağış verileri toplanıyor...")
    
    yagis = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start_date, end_date) \
        .select('precipitation') \
        .mean() \
        .clip(turkiye) \
        .rename('yagis')
    
    return yagis


# ============================================================
# 3. VERİ İŞLEME
# ============================================================

def combine_data(ndvi, lst, fire, yagis, humidity):
    """Tüm verileri birleştir"""
    print("🔗 Veriler birleştiriliyor...")
    
    combined_data = ndvi.addBands(lst) \
        .addBands(fire) \
        .addBands(yagis) \
        .addBands(humidity)
    
    return combined_data


def sample_points(data, turkiye, num_points=1000, scale=10000, seed=42):
    """Veriyi rastgele noktalardan örnekle"""
    print(f"🎲 {num_points} rastgele nokta oluşturuluyor...")
    
    random_points = ee.FeatureCollection.randomPoints(
        region=turkiye,
        points=num_points,
        seed=seed
    )
    
    print("📍 Noktalarda veri örnekleniyor...")
    sampled_data = data.sampleRegions(
        collection=random_points,
        scale=scale,
        geometries=True
    )
    
    return sampled_data


# ============================================================
# 4. VERİYİ CSV'YE AKTAR
# ============================================================

def export_to_csv(sampled_data, output_path):
    """EE'den veri çek ve CSV olarak kaydet"""
    print("📥 Veriler çekiliyor...")
    
    # GEE'den veri getir
    features = sampled_data.getInfo()['features']
    
    # DataFrame'e dönüştür
    data_list = []
    for feature in features:
        row = {
            'longitude': feature['geometry']['coordinates'][0],
            'latitude': feature['geometry']['coordinates'][1],
        }
        row.update(feature['properties'])
        data_list.append(row)
    
    df = pd.DataFrame(data_list)
    
    # CSV'ye kaydet
    df.to_csv(output_path, index=False)
    print(f"✅ Veri kaydedildi: {output_path}")
    print(f"   Satır sayısı: {len(df)}")
    print(f"   Sütun sayısı: {len(df.columns)}")
    
    return df


# ============================================================
# 5. MAIN FUNCTION
# ============================================================

def main():
    """Ana çalışma fonksiyonu"""
    
    print("="*60)
    print("🔥 GOOGLE EARTH ENGINE - YANGIN VERİ İŞLEYİCİ")
    print("="*60)
    
    # GEE başlat
    if not initialize_gee():
        return
    
    # Tarihler
    start_date = '2019-06-01'
    end_date = '2025-01-12'
    
    print(f"\n📅 Tarih aralığı: {start_date} - {end_date}")
    
    try:
        # 1. Temel Türkiye bağlantısını kur ve yangın verilerini al
        fire_data, turkiye = get_fire_data(start_date, end_date)
        
        # 2. Diğer verileri topla
        ndvi_data = get_ndvi_data(start_date, end_date, turkiye)
        hava_sicakligi = get_temperature_data(start_date, end_date, turkiye)
        cignenme = get_dewpoint_data(start_date, end_date, turkiye)
        humidity_data = calculate_humidity(hava_sicakligi, cignenme)
        lst_data = get_temperature_modis(start_date, end_date, turkiye)
        yagis_data = get_precipitation_data(start_date, end_date, turkiye)
        
        # 3. Verileri birleştir
        combined_data = combine_data(ndvi_data, lst_data, fire_data, yagis_data, humidity_data)
        
        # 4. Rastgele noktalardan örnekle
        sampled_data = sample_points(combined_data, turkiye, num_points=1000)
        
        # 5. CSV olarak kaydet
        output_dir = '../data'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'yangin_rastgele_verisi.csv')
        
        df = export_to_csv(sampled_data, output_path)
        
        print("\n" + "="*60)
        print("✅ İŞLEM BAŞARILI!")
        print("="*60)
        print(f"\n📊 Veri İstatistikleri:")
        print(df.describe())
        
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
