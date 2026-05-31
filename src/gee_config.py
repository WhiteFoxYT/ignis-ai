"""
Google Earth Engine Konfigürasyonu
VS Code'da GEE verilerini işlemek için gerekli ayarlar
"""

import os
import json

class GEEConfig:
    """Google Earth Engine Konfigürasyonu"""
    
    # Proje bilgileri
    PROJECT_ID = 'ignisai-496207'
    
    # Türkiye sınırları
    REGION_NAME = 'Turkey'
    REGION_COLLECTION = 'USDOS/LSIB_SIMPLE/2017'
    
    # Veri kaynakları
    DATA_SOURCES = {
        'fire': {
            'collection': 'MODIS/061/MCD64A1',
            'band': 'BurnDate',
            'description': 'Yangın Haritası (MODIS)'
        },
        'ndvi': {
            'collection': 'MODIS/061/MOD13Q1',
            'band': 'NDVI',
            'description': 'Bitki Örtüsü İndeksi'
        },
        'temperature': {
            'collection': 'ECMWF/ERA5/DAILY',
            'band': 'mean_2m_air_temperature',
            'description': 'Hava Sıcaklığı (ERA5)'
        },
        'dewpoint': {
            'collection': 'ECMWF/ERA5/DAILY',
            'band': 'dewpoint_2m_temperature',
            'description': 'Çiğlenme Noktası (ERA5)'
        },
        'lst': {
            'collection': 'MODIS/061/MOD11A1',
            'band': 'LST_Day_1km',
            'description': 'Arazi Yüzey Sıcaklığı (MODIS LST)'
        },
        'precipitation': {
            'collection': 'UCSB-CHG/CHIRPS/DAILY',
            'band': 'precipitation',
            'description': 'Yağış Verileri (CHIRPS)'
        }
    }
    
    # Örnekleme parametreleri
    SAMPLING = {
        'num_points': 1000,          # Rastgele nokta sayısı
        'scale': 10000,               # Çözünürlük (meter)
        'seed': 42                    # Tekrarlanabilirlik için
    }
    
    # Tarih aralıkları
    DATE_RANGES = {
        'full': {
            'start': '2019-06-01',
            'end': '2025-01-12',
            'description': 'Tam veri seti'
        },
        'visualization': {
            'start': '2022-07-01',
            'end': '2022-07-31',
            'description': 'Görselleştirme için örnek'
        }
    }
    
    # Dosya yolları
    PATHS = {
        'data_dir': '../data',
        'outputs_dir': '../outputs',
        'models_dir': '../models'
    }
    
    @classmethod
    def get_config(cls):
        """Konfigürasyonu döndür"""
        return {
            'project_id': cls.PROJECT_ID,
            'region_name': cls.REGION_NAME,
            'data_sources': cls.DATA_SOURCES,
            'sampling': cls.SAMPLING,
            'date_ranges': cls.DATE_RANGES,
            'paths': cls.PATHS
        }
    
    @classmethod
    def print_config(cls):
        """Konfigürasyonu konsola bas"""
        print("📋 GEE KONFİGÜRASYON:")
        print(f"  Proje ID: {cls.PROJECT_ID}")
        print(f"  Region: {cls.REGION_NAME}")
        print(f"  Örnekleme Noktaları: {cls.SAMPLING['num_points']}")
        print(f"  Çözünürlük: {cls.SAMPLING['scale']}m")


if __name__ == "__main__":
    GEEConfig.print_config()
