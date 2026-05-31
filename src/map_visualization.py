"""
Harita Gorsellestirme Modülü
=============================
Folium kullanarak yangın riski haritasını oluştur.

Yapılan İşlemler:
1. Tahmin sonuçlarını (predictions.csv) yükle
2. Merkez koordinatı hesapla
3. Folium haritası oluştur
4. Her nokta için CircleMarker ekle
5. Popup bilgisi ekle
6. Renkler: Düşük (yeşil), Orta (turuncu), Yüksek (kırmızı)
7. Haritayı kaydet
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster

from config import (
    PREDICTIONS_FILE,
    TEST_PREDICTIONS_FILE,
    FIRE_RISK_MAP,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    MAP_CENTER,
    MAP_ZOOM_START,
    CIRCLE_RADIUS,
    CIRCLE_OPACITY,
    CIRCLE_WEIGHT,
    RISK_COLORS,
    CSV_ENCODING,
)
from utils import (
    setup_logger,
    ensure_path_exists,
    get_risk_level,
    get_risk_color,
)

# Logger
logger = setup_logger(__name__)


# ============================================================
# VERİ YÜKLEME
# ============================================================

def load_predictions(input_path: Path = None) -> pd.DataFrame:
    """
    Tahmin dosyasını yükle.
    
    Args:
        input_path: Tahmin CSV yolu (default: PREDICTIONS_FILE)
    
    Returns:
        Tahmin verileri (DataFrame)
    """
    if input_path is None:
        input_path = PREDICTIONS_FILE
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Tahmin dosyası bulunamadı: {input_path}\n"
            "Lutfen önce predict.py çalıştırın!"
        )
    
    logger.info(f" Tahminler yükleniyor: {input_path}")
    df = pd.read_csv(input_path, encoding=CSV_ENCODING)
    logger.info(f" Yüklendi: {len(df):,} satır")
    
    return df


def load_test_predictions(input_path: Path = None) -> pd.DataFrame:
    """
    Test tahmin dosyasını yükle.

    Args:
        input_path: Test tahmin CSV yolu (default: TEST_PREDICTIONS_FILE)
    """
    if input_path is None:
        input_path = TEST_PREDICTIONS_FILE

    if not input_path.exists():
        raise FileNotFoundError(
            f"Test tahmin dosyası bulunamadı: {input_path}\n"
            "Lutfen önce train.py veya test_accuracy.py çalıştırın!"
        )

    logger.info(f" Test tahminleri yükleniyor: {input_path}")
    df = pd.read_csv(input_path, encoding=CSV_ENCODING)
    logger.info(f" Yüklendi: {len(df):,} satır")

    return df


# ============================================================
# KOORDİNAT İŞLEMİ
# ============================================================

def validate_coordinates(
    df: pd.DataFrame,
    lat_col: str = LATITUDE_COLUMN,
    lon_col: str = LONGITUDE_COLUMN,
) -> pd.DataFrame:
    """
    Koordinatları kontrol et ve geçersiz olanları kaldır.
    
    Args:
        df: DataFrame
        lat_col: Enlem sütun adı
        lon_col: Boylam sütun adı
    
    Returns:
        Geçerli koordinatlı DataFrame
    """
    logger.info(" Koordinatlar kontrol ediliyor...")
    
    df = df.copy()
    
    # Koordinat sütunlarının varlığını kontrol et
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(
            f"Koordinat sütunları bulunamadı: "
            f"'{lat_col}', '{lon_col}'"
        )
    
    # Geçerli koordinat aralıkları
    valid = (
        (df[lat_col] >= -90) & (df[lat_col] <= 90) &
        (df[lon_col] >= -180) & (df[lon_col] <= 180) &
        (df[lat_col].notna()) & (df[lon_col].notna())
    )
    
    invalid_count = (~valid).sum()
    if invalid_count > 0:
        logger.warning(f"   ⚠ {invalid_count} geçersiz koordinat kaldırıldı")
        df = df[valid]
    
    logger.info(f"    {len(df):,} geçerli koordinat")
    
    return df


def calculate_map_center(
    df: pd.DataFrame,
    lat_col: str = LATITUDE_COLUMN,
    lon_col: str = LONGITUDE_COLUMN,
) -> Tuple[float, float]:
    """
    Harita merkez koordinatını hesapla.
    
    Args:
        df: DataFrame
        lat_col: Enlem sütun adı
        lon_col: Boylam sütun adı
    
    Returns:
        (latitude, longitude) tuple'ı
    """
    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()
    
    logger.info(f" Harita merkezi: ({center_lat:.2f}, {center_lon:.2f})")
    
    return center_lat, center_lon


# ============================================================
# FOLIUM HARITA OLUŞTURMA
# ============================================================

def create_fire_risk_map(
    df: pd.DataFrame,
    output_path: Path = None,
    add_heatmap: bool = True,
    add_clusters: bool = True,
    max_markers: int = None,
    simplified_popups: bool = True,
) -> folium.Map:
    """
    Yangın riski haritasını oluştur (Optimize).
    
    Optimizasyonlar:
    - Nokta sayısı sampling (max_markers'a kadar)
    - Marker clustering (daha hızlı render)
    - Basitleştirilmiş popup'lar
    
    Args:
        df: Tahmin verileri
        output_path: Harita HTML yolu (default: FIRE_RISK_MAP)
        add_heatmap: Heat map katmanı ekle mi?
        add_clusters: Marker cluster ekle mi? (True = önerilen)
        max_markers: Max nokta sayısı (None = config'den, ~5000)
        simplified_popups: Basit popup'lar mı? (True = daha hızlı)
    
    Returns:
        Folium Map nesnesi
    """
    
    if output_path is None:
        output_path = FIRE_RISK_MAP
    
    if max_markers is None:
        from config import MAX_MARKERS_ON_MAP
        max_markers = MAX_MARKERS_ON_MAP
    
    logger.info(" Yangın riski haritası oluşturuluyor...")
    
    # Koordinat sütun adları
    lat_col = LATITUDE_COLUMN
    lon_col = LONGITUDE_COLUMN
    
    # Veriyi doğrula
    df = validate_coordinates(df, lat_col, lon_col)
    
    # Sampling: çok noktayı azalt (performans)
    original_len = len(df)
    if len(df) > max_markers:
        logger.info(f"    Nokta sayisi azaltiliyor: {len(df):,} -> {max_markers:,}...")
        df_display = df.sample(n=max_markers, random_state=42)
        logger.info(f"    Sampling tamamlandi (random selected)")
    else:
        df_display = df
    
    # Harita merkezi
    center = calculate_map_center(df_display, lat_col, lon_col)
    
    # Folium haritası oluştur
    map_obj = folium.Map(
        location=center,
        zoom_start=MAP_ZOOM_START,
        tiles='OpenStreetMap',
    )
    
    logger.info(f"    Harita temel katmanı oluşturuldu")
    
    # Heat map katmanı (isteğe bağlı)
    if add_heatmap and len(df) > 0:
        logger.info(f"    Heat map ekleniyor (tüm {original_len:,} veri)...")
        
        heat_data = [
            [row[lat_col], row[lon_col], row['fire_probability']]
            for idx, row in df.iterrows()
        ]
        
        HeatMap(heat_data, radius=15, blur=15, max_zoom=1, name='Heat Map').add_to(map_obj)
        logger.info(f"    Heat map eklendi")
    
    # Marker cluster katmanı (isteğe bağlı)
    if add_clusters:
        cluster_group = MarkerCluster(name='Clusters').add_to(map_obj)
        logger.info("    Marker cluster katmanı eklendi (hızlı render)")
    else:
        cluster_group = None
    
    # CircleMarker'ları ekle (sampled data)
    logger.info(f"    CircleMarker'lar ekleniyor ({len(df_display):,} nokta, popup={not simplified_popups})...")
    
    for idx, row in df_display.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        prob = row.get('fire_probability', 0)
        risk = row.get('risk_level', 'low')
        confidence = row.get('confidence_score', prob * 100)
        
        # Skip if invalid
        if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
            continue
        
        # Renk belirle
        color = get_risk_color(prob)
        
        # Popup bilgisi oluştur
        if simplified_popups:
            # Basit popup: sadece risk ve olasılık (çok hızlı)
            popup_text = f"""
            <div style='font-family: Arial, sans-serif; font-size: 12px;'>
                <b style='color: {color};'>Risk: {risk.upper()}</b><br>
                <hr style='margin: 5px 0;'>
                Olasılık: <b>{prob:.1%}</b><br>
                Güven Düzeyi: <b>{confidence:.1f}%</b><br>
                <small>Koord: {lat:.2f}, {lon:.2f}</small>
            </div>
            """
        else:
            # Detaylı popup
            popup_text = f"""
            <div style='font-family: Arial, sans-serif; font-size: 12px; width: 220px;'>
                <b style='font-size: 14px; color: {color};'>🔥 YANGIN RİSK RAPORU</b><br>
                <hr style='margin: 5px 0;'>
                <b>Risk Seviyesi:</b> {risk.upper()}<br>
                <b>Yangın Olasılığı:</b> {prob:.2%}<br>
                <b>Güven Düzeyi:</b> {confidence:.2f}%<br>
                <hr style='margin: 5px 0;'>
                <b>Konum:</b><br>
                • Enlem: {lat:.4f}<br>
                • Boylam: {lon:.4f}<br>
                <hr style='margin: 5px 0;'>
                <small><i>Tahmini bilgi - Uydu ve meteorolojik veriler temelinde</i></small>
            </div>
            """
        
        # CircleMarker ekle
        circle = folium.CircleMarker(
            location=[lat, lon],
            radius=CIRCLE_RADIUS,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=CIRCLE_OPACITY,
            weight=CIRCLE_WEIGHT,
            opacity=0.8,
        )
        
        if cluster_group is not None:
            circle.add_to(cluster_group)
        else:
            circle.add_to(map_obj)
    
    logger.info(f"    CircleMarker'lar eklendi")
    
    # Legent ekle (geliştirilmiş)
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 280px; 
                background-color: white; border:2px solid #333; z-index:9999; 
                font-size:13px; padding: 12px;
                border-radius: 5px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
                font-family: Arial, sans-serif;">
        <b style="font-size: 14px; text-decoration: underline;">🔥 YANGIN RİSK SEVİYELERİ</b><br>
        <hr style='margin: 5px 0;'>
        <span style="color: green; font-weight: bold;">● Düşük</span> Riski (< 30%)<br>
        <span style="color: orange; font-weight: bold;">● Orta</span> Riski (30% - 70%)<br>
        <span style="color: red; font-weight: bold;">● Yüksek</span> Riski (> 70%)<br>
        <hr style='margin: 5px 0;'>
        <b style="font-size: 12px;">✅ DOĞRULUK</b><br>
        <small>
            • Yeşil çerçeve: Doğru tahmin<br>
            • Kırmızı çerçeve: Yanlış tahmin<br>
        </small>
        <hr style='margin: 5px 0;'>
        <b style="font-size: 12px;">📊 MODEL BİLGİSİ</b><br>
        <small>
            • Algoritma: Sinir Ağı<br>
            • Giriş Verileri: 6 parametre<br>
            • Veri Bölümü: %70 Eğitim, %30 Test<br>
            • Detaylı Rapor: outputs/reports/<br>
        </small>
        <hr style='margin: 5px 0;'>
        <small style='color: #666;'><i>💡 Tıkla → Güven Düzeyini Gör<br>
        ⚠️  Uyarı: Tahmini bilgidir</i></small>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    logger.info(f"    Legent eklendi")
    
    # Zoom kontrolleri ekle
    map_obj.add_child(folium.LatLngPopup())
    
    # Kaydet
    ensure_path_exists(output_path, is_file=True)
    map_obj.save(str(output_path))
    logger.info(f" Harita kaydedildi: {output_path}")
    
    return map_obj


# ============================================================
# HARITA İSTATİSTİKLERİ
# ============================================================

def print_map_statistics(df: pd.DataFrame) -> None:
    """
    Harita istatistiklerini yazdır.
    
    Args:
        df: Tahmin verileri
    """
    logger.info("\n Harita İstatistikleri:")
    logger.info("-" * 50)

    if 'fire_probability' in df.columns:
        probs = df['fire_probability']
        logger.info(f"  Olasılık Minimum: {probs.min():.4f}")
        logger.info(f"  Olasılık Maksimum: {probs.max():.4f}")
        logger.info(f"  Olasılık Ortalama: {probs.mean():.4f}")
        logger.info(f"  Olasılık Std Dev: {probs.std():.4f}")

    if 'risk_level' in df.columns:
        logger.info(f"\n  Risk Dağılımı:")
        for risk in ['low', 'medium', 'high']:
            count = (df['risk_level'] == risk).sum()
            pct = (count / len(df)) * 100
            logger.info(f"    {risk.upper():.<20} {count:,} ({pct:.1f}%)")

    logger.info("-" * 50)


def add_test_accuracy_layer(
    map_obj: folium.Map,
    df_test: pd.DataFrame,
    max_markers: Optional[int] = None,
) -> None:
    """Test verileri için doğruluk katmanı ekle."""
    logger.info(" Test doğruluk katmanı ekleniyor...")
    df_test = validate_coordinates(df_test)

    if max_markers is not None and len(df_test) > max_markers:
        logger.info(f"    Test noktaları azaltılıyor: {len(df_test):,} -> {max_markers:,}")
        df_test = df_test.sample(n=max_markers, random_state=42)

    group = folium.FeatureGroup(name="Test Doğruluk", show=True)

    for _, row in df_test.iterrows():
        lat = row.get(LATITUDE_COLUMN)
        lon = row.get(LONGITUDE_COLUMN)
        prob = row.get('fire_probability', 0)
        risk = row.get('risk_level', 'low')
        correct = bool(row.get('prediction_correct', False))
        actual = row.get('actual_class', None)
        predicted = row.get('predicted_class', None)
        confidence = row.get('confidence_score', np.abs(prob - 0.5) * 2 * 100)

        if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
            continue

        fill_color = get_risk_color(prob)
        border_color = "green" if correct else "red"

        popup_text = f"""
        <div style='font-family: Arial, sans-serif; font-size: 12px;'>
            <b style='color: {fill_color};'>Risk: {risk.upper()}</b><br>
            <hr style='margin: 5px 0;'>
            Olasılık: <b>{prob:.1%}</b><br>
            Güven Düzeyi: <b>{confidence:.1f}%</b><br>
            Gerçek: <b>{'Yangın Var' if actual == 1 else 'Yangın Yok'}</b><br>
            Tahmin: <b>{'Yangın Var' if predicted == 1 else 'Yangın Yok'}</b><br>
            Doğruluk: <b>{'Doğru' if correct else 'Yanlış'}</b><br>
            <small>Koord: {lat:.2f}, {lon:.2f}</small>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=CIRCLE_RADIUS + 1,
            popup=folium.Popup(popup_text, max_width=300),
            color=border_color,
            fill=True,
            fillColor=fill_color,
            fillOpacity=0.8,
            weight=3,
            opacity=0.9,
        ).add_to(group)

    group.add_to(map_obj)
    logger.info(" Test doğruluk katmanı eklendi")


# ============================================================
# ANA HARITA OLUŞTURMA FONKSİYONU
# ============================================================

def create_risk_map(
    input_path: Path = None,
    output_path: Path = None,
    add_heatmap: bool = True,
    add_clusters: bool = True,
    max_markers: int = None,
    simplified_popups: bool = True,
    add_test_layer: bool = True,
) -> folium.Map:
    """
    Ana harita oluşturma fonksiyonu (Optimize).
    
    Args:
        input_path: Tahmin CSV yolu (default: PREDICTIONS_FILE)
        output_path: Harita HTML yolu (default: FIRE_RISK_MAP)
        add_heatmap: Heat map ekle mi?
        add_clusters: Cluster ekle mi? (True = önerilen, daha hızlı)
        max_markers: Max nokta sayısı (None = 5000 default)
        simplified_popups: Basit popup'lar mı? (True = daha hızlı)
        add_test_layer: Test doğruluk katmanını ekle mi?
    
    Returns:
        Folium Map nesnesi
    """
    
    if output_path is None:
        output_path = FIRE_RISK_MAP
    
    logger.info("=" * 80)
    logger.info(" YANGIN RİSK HARİTASI OLUŞTURMA")
    logger.info("=" * 80)
    
    try:
        # 1. Tahminleri yükle
        df = load_predictions(input_path)
        
        # 2. İstatistikleri yazdır
        print_map_statistics(df)
        
        # 3. Harita oluştur (optimized)
        map_obj = create_fire_risk_map(
            df,
            output_path=output_path,
            add_heatmap=add_heatmap,
            add_clusters=add_clusters,
            max_markers=max_markers,
            simplified_popups=simplified_popups,
        )

        # 4. Test doğruluk katmanı (varsa)
        if add_test_layer and TEST_PREDICTIONS_FILE.exists():
            try:
                df_test = load_test_predictions()
                add_test_accuracy_layer(map_obj, df_test, max_markers=max_markers)
            except Exception as e:
                logger.warning(f" Test katmanı eklenemedi: {e}")

        # 5. Layer kontrol
        folium.LayerControl().add_to(map_obj)

        # 6. Katmanlı haritayı tekrar kaydet
        ensure_path_exists(output_path, is_file=True)
        map_obj.save(str(output_path))
        logger.info(f" Harita güncellendi: {output_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info(" HARİTA OLUŞTURMA TAMAMLANDI!")
        logger.info("=" * 80)
        logger.info(f"\n Harita URL'si: {output_path}")
        logger.info(f"   Browser'da açmak için: file://{output_path.absolute()}")
        
        return map_obj
    
    except Exception as e:
        logger.error(f" Harita oluşturma hatası: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================
# ADVANCED: CLUSTER ANALİZİ
# ============================================================

def create_clustered_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    n_clusters: int = 10,
) -> None:
    """
    K-means clustering ile harita oluştur.
    
    Args:
        df: Tahmin verileri
        output_path: Çıkış dosyası
        n_clusters: Küme sayısı
    """
    from sklearn.cluster import KMeans
    
    logger.info(f" Kümeleme haritası oluşturuluyor (n_clusters={n_clusters})...")
    
    # Koordinatlar
    coords = df[[LATITUDE_COLUMN, LONGITUDE_COLUMN]].values
    
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(coords)
    
    # Harita
    center = calculate_map_center(df)
    map_obj = folium.Map(location=center, zoom_start=MAP_ZOOM_START)
    
    # Merkez noktaları
    for i, center_point in enumerate(kmeans.cluster_centers_):
        folium.CircleMarker(
            location=[center_point[0], center_point[1]],
            radius=10,
            popup=f"Küme {i}",
            color='black',
            fill=True,
            fillOpacity=0.7,
            weight=3,
        ).add_to(map_obj)
    
    # Veri noktaları
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
              'darkblue', 'darkgreen', 'cadetblue']
    
    for idx, row in df.iterrows():
        cluster_id = row['cluster']
        color = colors[cluster_id % len(colors)]
        
        folium.CircleMarker(
            location=[row[LATITUDE_COLUMN], row[LONGITUDE_COLUMN]],
            radius=5,
            color=color,
            fill=True,
            fillOpacity=0.6,
            weight=2,
        ).add_to(map_obj)
    
    map_obj.save(str(output_path))
    logger.info(f" Kümeleme haritası kaydedildi: {output_path}")


# ============================================================
# MAIN (TEST)
# ============================================================

if __name__ == "__main__":
    
    try:
        # Harita oluştur
        map_obj = create_risk_map(add_heatmap=True)
        
        logger.info("\n Harita başarıyla oluşturuldu!")
        logger.info(f" Kayıt yeri: {FIRE_RISK_MAP}")
        
    except Exception as e:
        logger.error(f" Hata: {e}")
        raise
