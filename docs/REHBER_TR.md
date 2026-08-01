<!-- Bu belge docs/GUIDE_EN.md dosyasının Türkçe muadilidir. Bölüm numaraları birebir aynıdır. -->

# IGNIS — Kapsamlı Eğitim Rehberi

**Intelligent Geospatial Neural Inference System** (Akıllı Uzamsal Sinirsel Çıkarım Sistemi)
Uydu tabanlı Yer Gözlem verilerinden Türkiye üzerinde ertesi gün orman yangını yayılımı tahmini

| | |
|---|---|
| **Bildiri numarası** | IAC-26,B1,IP,107,x110901 |
| **Sempozyum** | IAF Yer Gözlem Sempozyumu (B1), Etkileşimli Sunumlar (IP) |
| **Kongre** | 77. Uluslararası Astronotik Kongresi, 5–9 Ekim 2026, Antalya, Türkiye |
| **Yazarlar** | Muhammet Erdem Karakoyun (sorumlu yazar), Utku Doruk Kaplan, Furkan Bağıç, Mehmet İpek, Ege Kayseri, Özgür Efe Aksoy |
| **Kurum** | Antalya Yusuf Ziya Öner Fen Lisesi, Antalya, Türkiye |
| **Nihai metin son tarihi** | 14 Eylül 2026 |
| **Depo** | github.com/WhiteFoxYT/ignis-ai |

---

## İçindekiler

- [0. Bu rehber nasıl kullanılır](#0-bu-rehber-nasıl-kullanılır)
- [1. Orman yangınları ve problem](#1-orman-yangınları-ve-problem)
  - [1.1 Orman yangını aslında nedir](#11-orman-yangını-aslında-nedir)
  - [1.2 Yangın davranış üçgeni: yakıt, hava durumu, topoğrafya](#12-yangın-davranış-üçgeni-yakıt-hava-durumu-topoğrafya)
  - [1.3 Yangın neden yokuş yukarı koşar](#13-yangın-neden-yokuş-yukarı-koşar)
  - [1.4 Rüzgâr](#14-rüzgâr)
  - [1.5 Yangın rejimleri](#15-yangın-rejimleri)
  - [1.6 Türkiye neden yanıyor](#16-türkiye-neden-yanıyor)
  - [1.7 Afet yönetiminin dört evresi](#17-afet-yönetiminin-dört-evresi)
  - [1.8 Duyarlılık, yayılım değildir](#18-duyarlılık-yayılım-değildir)
- [2. Sıfırdan uzaktan algılama](#2-sıfırdan-uzaktan-algılama)
  - [2.1 Elektromanyetik tayf](#21-elektromanyetik-tayf)
  - [2.2 Bir uydu aslında neyi kaydeder](#22-bir-uydu-aslında-neyi-kaydeder)
  - [2.3 Pasif ve aktif algılama](#23-pasif-ve-aktif-algılama)
  - [2.4 Üç çözünürlük ve aralarındaki ödünleşim](#24-üç-çözünürlük-ve-aralarındaki-ödünleşim)
  - [2.5 Yörüngeler](#25-yörüngeler)
  - [2.6 MODIS, Terra ve Aqua](#26-modis-terra-ve-aqua)
  - [2.7 Bir uydu yangını nasıl tespit eder](#27-bir-uydu-yangını-nasıl-tespit-eder)
  - [2.8 FireMask güven sınıfları](#28-firemask-güven-sınıfları)
  - [2.9 NDVI](#29-ndvi)
  - [2.10 Arazi yüzey sıcaklığı](#210-arazi-yüzey-sıcaklığı)
  - [2.11 Yeniden analiz: ERA5-Land](#211-yeniden-analiz-era5-land)
  - [2.12 CHIRPS](#212-chirps)
  - [2.13 SRTM, yükseklik, eğim ve bakı](#213-srtm-yükseklik-eğim-ve-bakı)
- [3. Coğrafi temeller ve projeksiyonlar](#3-coğrafi-temeller-ve-projeksiyonlar)
  - [3.1 Dünya neden düzleştirilemez](#31-dünya-neden-düzleştirilemez)
  - [3.2 Coğrafi koordinatlara karşı projeksiyonlu koordinatlar](#32-coğrafi-koordinatlara-karşı-projeksiyonlu-koordinatlar)
  - [3.3 UTM ve 35N dilimi](#33-utm-ve-35n-dilimi)
  - [3.4 EPSG kodları](#34-epsg-kodları)
  - [3.5 Yeniden projeksiyonlama ve yeniden örnekleme](#35-yeniden-projeksiyonlama-ve-yeniden-örnekleme)
  - [3.6 Eş-kayıt ve ölçek](#36-eş-kayıt-ve-ölçek)
  - [3.7 Google Earth Engine](#37-google-earth-engine)
- [4. Sıfırdan makine öğrenmesi](#4-sıfırdan-makine-öğrenmesi)
- [5. Derin öğrenme ve evrişimli ağlar](#5-derin-öğrenme-ve-evrişimli-ağlar)
  - [5.1 Yapay nöron](#51-yapay-nöron)
  - [5.2 Aktivasyon fonksiyonları](#52-aktivasyon-fonksiyonları)
  - [5.3 Tam bağlı katmanlar görüntülerde neden başarısız olur](#53-tam-bağlı-katmanlar-görüntülerde-neden-başarısız-olur)
  - [5.4 Evrişim, elle hesaplanmış hâliyle](#54-evrişim-elle-hesaplanmış-hâliyle)
  - [5.5 Adım, dolgu ve alıcı alan](#55-adım-dolgu-ve-alıcı-alan)
  - [5.6 Havuzlama](#56-havuzlama)
  - [5.7 Kanallar](#57-kanallar)
  - [5.8 Sınıflandırmaya karşı anlamsal bölütleme](#58-sınıflandırmaya-karşı-anlamsal-bölütleme)
  - [5.9 U-Net](#59-u-net)
  - [5.10 Parametreleri saymak](#510-parametreleri-saymak)
- [6. Sınıf dengesizliği ve metrikler](#6-sınıf-dengesizliği-ve-metrikler)
  - [6.1 Karışıklık matrisi](#61-karışıklık-matrisi)
  - [6.2 Doğruluk neden yalan söyler](#62-doğruluk-neden-yalan-söyler)
  - [6.3 Kesinlik, duyarlılık, F1](#63-kesinlik-duyarlılık-f1)
  - [6.4 IoU](#64-iou)
  - [6.5 ROC eğrisi](#65-roc-eğrisi)
  - [6.6 PR eğrisi ve AUC-PR](#66-pr-eğrisi-ve-auc-pr)
  - [6.7 Kendi çelişkimiz: ROC-AUC 0.8468'e karşı AUC-PR 0.0210](#67-kendi-çelişkimiz-roc-auc-08468e-karşı-auc-pr-00210)
  - [6.8 Eşik seçimi ve kalibrasyon](#68-eşik-seçimi-ve-kalibrasyon)
  - [6.9 Temel çizgiler ve kalıcılık neden zorunludur](#69-temel-çizgiler-ve-kalıcılık-neden-zorunludur)
  - [6.10 Dengesizlikle başa çıkma teknikleri](#610-dengesizlikle-başa-çıkma-teknikleri)
- [7. IGNIS veri işleme hattı, satır satır](#7-ignis-veri-işleme-hattı-satır-satır)
  - [7.1 Sekiz kaynak ürün](#71-sekiz-kaynak-ürün)
  - [7.4 Yama çıkarımı](#74-yama-çıkarımı-stratifiedsample-ve-neighborhoodtoarray)
  - [7.5 Yeni olanlar: fire_next2, valid ve 32×32 kırpma](#75-yeni-olanlar-fire_next2-valid-ve-3232-kırpma)
  - [7.6 Normalleştirme: en önemli tek düzeltme](#76-normalleştirme-en-önemli-tek-düzeltme)
  - [7.7 Dairesel değişken problemi: bakı](#77-dairesel-değişken-problemi-bakı)
  - [7.8 Kategorik değişken problemi: arazi örtüsü](#78-kategorik-değişken-problemi-arazi-örtüsü)
  - [7.9 Yön duyarlı veri artırma](#79-yön-duyarlı-veri-artırma)
- [8. Modelin eğitimi ve GPU'lar](#8-modelin-eğitimi-ve-gpular)
- [9. Sonuçları dürüstçe okumak](#9-sonuçları-dürüstçe-okumak)
  - [9.2 Piksel düzeyindeki sonuçlar](#92-piksel-düzeyindeki-sonuçlar)
  - [9.3 Yama düzeyinde sınıflandırma ve %77 tuzağı](#93-yama-düzeyinde-sınıflandırma-ve-77-tuzağı)
  - [9.4 Kalıcılık karşılaştırması](#94-kalıcılık-karşılaştırması--en-çok-önem-taşıyan-sonuç)
  - [9.5 Teşhis edilen yedi neden](#95-teşhis-edilen-yedi-neden)
  - [9.7 Dürüstlük neden bir güçtür](#97-dürüstlük-neden-bir-güçtür)
- [10. Size sorulabilecek sorular ve nasıl cevaplanacağı](#10-size-sorulabilecek-sorular-ve-nasıl-cevaplanacağı)
- [11. Sözlük](#11-sözlük)
- [12. Kaynaklar ve ileri okuma](#12-kaynaklar-ve-ileri-okuma)

---

## 0. Bu rehber nasıl kullanılır

Bu rehber, Ekim 2026'da Antalya'da uluslararası bir dinleyici kitlesinin karşısına çıkıp IGNIS'i savunacak altı öğrenci için yazılmıştır. **Hiçbir ön bilgi varsaymaz.** Eğer "evrişim" (*convolution*) kelimesini hiç duymadıysanız, bu rehber o noktadan daha geriden başlar. U-Net'in ne olduğunu zaten biliyorsanız, doğrudan Bölüm 6'ya geçebilirsiniz; bu projenin gerçek bilimsel içeriği orada yaşar.

Rehberin üç görevi vardır.

1. **Bilimi öğretmek.** Her kavram üç katmanda açıklanır: sezgisel bir benzetme, matematiksel tanım ve o kavramın IGNIS'in içinde tam olarak nerede karşımıza çıktığı.
2. **İngilizceyi öğretmek.** Hepinizin anadili Türkçe ve sunumu İngilizce yapmak zorundasınız. Her teknik terim ilk geçtiğinde İngilizce karşılığı parantez içinde verilmiştir ve Bölüm 11, 150 terimlik iki dilli bir sözlüktür. İngilizce terimleri yüksek sesle okuyun. IAC'de "class imbalance" ifadesini kafanızın içinde çevirmeden doğrudan *söylemeniz* gerekecek.
3. **Sizi jüriye hazırlamak.** Bölüm 10, dürüst cevaplarıyla birlikte 33 zor soru içerir. Bu soruların bir kısmı size sorulacaktır. Onlara hazırlanın.

### Hangi bölüm hangi soruyu cevaplıyor

| Şunu öğrenmek istiyorsanız… | Şurayı okuyun |
|---|---|
| Yangınlar neden böyle yayılıyor? | Bölüm 1 |
| Bir uydu 700 km yukarıdan bir yangını nasıl "görür"? | Bölüm 2 |
| EPSG:32635 ne anlama geliyor? | Bölüm 3 |
| Eğitim, doğrulama, aşırı öğrenme nedir? | Bölüm 4 |
| Evrişim nedir, U-Net nedir? | Bölüm 5 |
| **Sonuçlarımız neden kötü ve "kötü" ne demek?** | **Bölüm 6 ve Bölüm 9** |
| Kodumuz veriye tam olarak ne yapıyor? | Bölüm 7 |
| Neden bir GPU'ya ihtiyacımız var ve neden AMD? | Bölüm 8 |
| Jüri bana ne soracak? | Bölüm 10 |
| Bu İngilizce kelimenin Türkçesi nedir? | Bölüm 11 |

### Başlamadan önce bir uyarı

Bu rehber **IGNIS'in mevcut durumu konusunda dürüsttür**. Model, bildiride belgelenen ön çalışma itibarıyla **iyi çalışmamaktadır**. Mümkün olan en basit tahmine karşı kaybetmektedir. Bölüm 9 bunu açıkça söyler ve nedenini tam olarak açıklar.

Bundan utanmayın. IAC gibi bir kongre en yüksek sayıyı bulma yarışması değildir. Bilimsel bir forumdur. Modelinin *neden* başarısız olduğunu ve *tam olarak neyi değiştirdiklerini* ölçülmüş kanıtlarla açıklayabilen bir lise öğrencisi ekibi, 0.997'lik doğruluğun hiçbir şey tahmin etmemekten bedavaya geldiğini bilmeden 0.99 doğruluk raporlayan bir ekipten daha iyi bilim yapıyordur. Dürüstlüğünüz elinizdeki en güçlü şeydir. Onu savunmayı öğrenin.

---

## 1. Orman yangınları ve problem

### 1.1 Orman yangını aslında nedir

Orman yangını (*wildfire*), kendi kendini sürdüren kimyasal bir tepkimedir — bitkisel maddenin hızlı oksitlenmesi — ve bir arazi boyunca yayılır. Üç bileşen gereklidir ve buna **yangın üçgeni** (*fire triangle*) denir: **yakıt** (*fuel*), **oksijen** (*oxygen*) ve **ısı** (*heat*). Bunlardan herhangi birini ortadan kaldırın, yangın durur. Su, ısıyı ortadan kaldırarak çalışır; bir yangın şeridi yakıtı ortadan kaldırarak çalışır; bir yangın battaniyesi ise oksijeni ortadan kaldırarak çalışır.

Ancak yangın üçgeni yalnızca *yanmayı* açıklar. *Yayılımı* açıklamaz. Yayılım için farklı bir modele ihtiyaç vardır ve IGNIS bu model üzerine kuruludur.

Yangın yayılır, çünkü yanan yakıt ısı yayar ve yanındaki yakıtı **ön ısıtan** sıcak gaz üretir. Ön ısıtma önce nemi, ardından bitkisel maddedeki uçucu gazları dışarı sürer — bu sürece **piroliz** (*pyrolysis*) denir. Bu gazlar tutuşma sıcaklığına ulaştığında komşu yakıt tutuşur ve süreç tekrarlanır. Yani yangın yayılımı bir ön ısıtma olayları zinciridir. Ön ısıtmayı hızlandıran veya kolaylaştıran her şey yangını daha hızlı hareket ettirir.

Ön ısıtmayı üç etken ailesi denetler. Bunlara birlikte **yangın davranış üçgeni** (*fire behaviour triangle*) denir.

### 1.2 Yangın davranış üçgeni: yakıt, hava durumu, topoğrafya

```
                      YANGIN DAVRANIŞI
                              ▲
                             / \
                            /   \
                  YAKIT    /     \    HAVA DURUMU
              (fuel)      /       \   (weather)
                         /         \
                        /___________\
                          TOPOĞRAFYA
                          (topography)
```

**Yakıt (*fuel*).** Ne kadar bitkisel madde var, düşey olarak nasıl dizilmiş ve ne kadar kuru? Yoğun, sürekli, kuru ve reçine bakımından zengin bir çam ormanı, seyrek, yeşil ve sulanan bir meyve bahçesinden yangını çok daha iyi taşır. İlgili özellikler *yük* (birim alandaki kütle), *süreklilik* (boşluklar var mı?), *dizilim* (ince dallar ve iğne yapraklar saniyeler içinde tutuşur; kalın bir gövde saatler alır) ve hepsinden önemlisi **yakıt nemidir** (*fuel moisture*). İnce ölü yakıtlar yaklaşık bir saat içinde hava ile dengeye gelir; bağıl nemin bu kadar güçlü bir yordayıcı olmasının nedeni budur.

**IGNIS'te** yakıt dört kanalla temsil edilir: `ndvi` (bitki örtüsü canlılığı, ne kadar yeşil biyokütle olduğunun vekil göstergesi), `landcover` (MODIS IGBP sınıfı, yakıt türünün vekil göstergesi), `humidity` (bağıl nem, ince yakıt neminin vekil göstergesi) ve `soil_moisture` (daha uzun vadeli kuraklığın vekil göstergesi).

**Hava durumu (*weather*).** Rüzgâr, hava sıcaklığı, bağıl nem ve yakın geçmişteki yağış. Hava durumu üçgenin en hızlı değişen ayağıdır — altı saat içinde tamamen değişebilir — ve ertesi gün tahminini bir haritalama problemi değil, bir *öngörü* problemi yapan sebep budur.

**IGNIS'te**: `air_temp`, `humidity`, `wind_speed`, `wind_u`, `wind_v`, `precip` ve `lst` (arazi yüzey sıcaklığı).

**Topoğrafya (*topography*).** Eğim, bakı ve yükseklik. Topoğrafya durağandır, ancak diğer ikisiyle güçlü biçimde etkileşir: kuzey yarımkürede güneye bakan bir yamaç daha fazla güneş ışınımı alır, dolayısıyla yakıtları daha kurudur.

**IGNIS'te**: `elevation`, `slope`, `aspect`.

### 1.3 Yangın neden yokuş yukarı koşar

Bu, jüri için yangın fiziğinin en önemli tek parçasıdır ve açıklaması kolaydır.

Alevler düşeye doğru yatar, çünkü sıcak gaz yükselir (kaldırma kuvveti, *buoyancy*). Düz zeminde alev aşağı yukarı dik durur ve önündeki yakıtı yalnızca yatay bir boşluk boyunca ışıma yoluyla ısıtır. Bir yamaçta ise zeminin kendisi aleve *doğru* eğilir, böylece yangının üzerindeki yanmamış yakıt doğrudan yükselen sıcak gaz sütununun içinde kalır. Alev artık, bir sonraki öğününün üzerine yatmış durumdadır.

```
   DÜZ ZEMİN                       YOKUŞ YUKARI
                                              🔥
      🔥                                     /  ← üstteki yakıt yükselen
     / \    yalnızca ışıma                  /     sıcak gaz sütununa gömülür
 ___/___\______yakıt___              ______/_____________
                                    /  yakıt
```

Sonuçları çarpıcıdır. Operasyonel yangın teşkilatlarında kullanılan bir pratik kurala göre, yayılma hızı her 10°'lik yokuş yukarı eğim için kabaca ikiye katlanır. Rothermel'in 1972 tarihli yayılım modeli — ki hâlâ operasyonel yangın davranışı yazılımlarının çoğunun temelidir — bunu, tepkime şiddeti terimini çarpan açık bir eğim çarpanı olarak kodlar.

**IGNIS'te**, `slope` ve `aspect` kanallarının girdi olmasının ve `elevation` kanalının önem taşımasının nedeni tam olarak budur. Ağa fizik anlatılmaz; dik eğimli ve yangını yamacın alt tarafında olan yamaların ertesi gün yokuş yukarı yangına sahip olma eğiliminde olduğunu verilerden *öğrenmesi* gerekir. Bunu öğrenmeye yetecek kadar verisi olup olmadığı ayrı bir sorudur (bkz. Bölüm 9).

### 1.4 Rüzgâr

Rüzgâr aynı anda üç şey yapar:

1. Alevi öne yatırır ve yanmamış yakıtı taşınım sütununun içine sokar — eğimle aynı mekanizma. Aslında rüzgâr ile eğim ilk yaklaşımda fiziksel olarak birbirinin yerine geçebilir; yangın davranışı modelleri bunları çoğu zaman tek bir "etkin rüzgâr" içinde birleştirir.
2. Yanma bölgesine taze oksijen sağlar ve tepkime hızını yükseltir.
3. Yanan korları cephenin önüne taşır. Buna **sıçrama** (*spotting*) denir ve bir yangının yolu, nehri veya yangın şeridini bu şekilde geçer. Sıçrama, yangın yayılımının basit ve sürekli bir cephe olmamasının nedenidir — ana yangının önünde yeni yangınlar belirir.

Rüzgâr bir sayı değil, bir **vektördür** (*vector*): büyüklüğü *ve* yönü vardır. Bu, model tasarımımız için son derece önemlidir. Rüzgârın 12 m/s olduğunu bilmek size yangının hızlı hareket edeceğini söyler; *nereye* gideceğini söylemez. IGNIS'in tek yerine üç rüzgâr kanalı taşımasının nedeni budur:

$$\text{wind\_speed} = \sqrt{u^2 + v^2}$$

burada $u$ doğuya doğru bileşendir (pozitif = rüzgâr doğuya doğru esiyor) ve $v$ kuzeye doğru bileşendir. $(u, v)$ ikilisi yönü kodlar; `wind_speed` fazlalık bir bilgidir, fakat bunu ağa açıkça vermek, ağın karekök alma işlemini öğrenmek zorunda kalmasını engeller.

Rüzgârın vektörel doğası aynı zamanda **veri artırmada** (*data augmentation*) ince bir tuzak yaratır; bunu Bölüm 7.6'da tartışıyoruz: bir yamayı soldan sağa aynalarsanız `wind_u` değerinin işaretini de ters çevirmeniz gerekir, aksi hâlde ağa yangınların rüzgâra karşı yayıldığını öğretiyor olursunuz.

### 1.5 Yangın rejimleri

**Yangın rejimi** (*fire regime*), bir ekosistemde on yıllar boyunca gözlenen karakteristik yangın örüntüsüdür: yangınların ne sıklıkta oluştuğu (frekans), ne kadar şiddetli oldukları, hangi mevsimde çıktıkları, ne kadar büyüdükleri ve hangi türde oldukları.

İngilizce olarak bilinmeye değer yangın türleri:

| Türkçe | İngilizce | Anlamı |
|---|---|---|
| Toprak altı yangını | Ground fire | Organik toprak katmanında yanar; yavaş, için için |
| Örtü yangını | Surface fire | Zemin seviyesindeki ölü örtü, ot ve çalıları yakar |
| Tepe yangını | Crown fire | Ağaç tacı boyunca yanar; en hızlı ve en yıkıcı olan |
| Sıçrama yangını | Spot fire | Cephenin önünde rüzgârın taşıdığı korlarla tutuşan yeni yangın |

Akdeniz ekosistemleri, yangın dönüş aralığı on yıllar mertebesinde olan bir **tepe yangını rejimine** sahiptir. Birçok Akdeniz bitkisi yalnızca yangına dayanıklı değil, aynı zamanda **yangına uyarlanmıştır**: *Pinus brutia* (Kızılçam, İngilizce: *Calabrian pine*) açılmak için ısı gerektiren serotin kozalaklara sahiptir, yani tür yenilenmek için düzenli aralıklı yangınlara fiilen bağımlıdır. Ekolojik sorun yangının kendisi değil, ekosistemin toparlanamayacağı kadar sık veya şiddetli yangındır.

### 1.6 Türkiye neden yanıyor

Türkiye'nin Akdeniz kıyısında — yani sizin yaşadığınız ve kongrenin düzenlendiği Antalya çevresinde — dört etken bir araya gelir.

1. **İklim.** Akdeniz iklimi (Köppen *Csa*), sıcak ve kurak yazlar ile ılıman ve yağışlı kışlar demektir. Antalya'da temmuz ve ağustos yağışı sıfıra yakındır. Bu nedenle yakıtlar her yıl üst üste üç ilâ dört ay boyunca kurur. IGNIS'in arşivini **haziran–ekim** yangın mevsimi aylarıyla sınırlamasının nedeni budur.
2. **Yakıt.** Türkiye'nin kıyı ormanlarına *Pinus brutia*, yani kızılçam hâkimdir. Reçinelidir, ölü örtü katmanında büyük miktarda kuru iğne yaprak tutar ve yangını tacına kolayca taşır. Akdeniz havzasının en yanıcı orman türlerinden biridir.
3. **Topoğrafya.** Toros Dağları (İngilizce: *Taurus Mountains*) kıyıdan dik biçimde yükselir. Dik yamaçlar Bölüm 1.3'te anlatıldığı gibi yayılımı hızlandırır ve ayrıca yangın söndürme için karadan erişimi son derece güçleştirir.
4. **Rüzgâr.** Yaz sinoptik örüntüleri kalıcı biçimde sıcak ve kuru rüzgârlar getirir. Bir sıcak hava dalgası kuvvetli rüzgârla çakıştığında yayılma hızları doğrudan müdahaleyle denetlenemez hâle gelir.

**Manavgat ve Marmaris** çevresindeki Temmuz–Ağustos 2021 yangınları referans olaydır: Türkiye'nin modern dönemdeki en ağır yangın felaketi ölçeğinde bir alanı yakmış, kitlesel tahliyelere yol açmış ve uluslararası yardım gerektirmiştir. Bölüm 9.6'nın açıkladığı gibi, mevcut arşivimizin en ciddi kısıtlarından biri **tam da bu yangınların arşivde bulunmamasıdır**.

### 1.7 Afet yönetiminin dört evresi

Afet yönetimi (*disaster management*) geleneksel olarak dört evreli bir döngü şeklinde tanımlanır. Bildiri bu çerçeveyi kullandığı için bunu İngilizce olarak ifade edebilmeniz gerekir.

| Evre | İngilizce | Cevapladığı soru | Zaman ölçeği |
|---|---|---|---|
| **Önleme / Zarar azaltma** | Prevention / Mitigation | Olayın olasılığını ve şiddetini nasıl azaltırız? | Yıllar |
| **Hazırlık** | Preparedness | Olay gerçekleşmeden önce elimizde ne hazır duruyor? | Aylar |
| **Müdahale** | Response | Olay yaşanıyor — *şimdi* ne yapacağız? | Saatler ilâ günler |
| **İyileştirme** | Recovery | Sonrasında nasıl yeniden inşa ederiz? | Aylar ilâ yıllar |

**IGNIS müdahale evresini hedefler.** Bu, bildirinin 1. bölümünde açıkça belirtilmiştir: *"The work presented here targets the response phase. Once a fire is already burning, the decisive operational question is no longer whether a fire will start but where the fire front will be tomorrow, because that determines the allocation of aircraft, ground crews and evacuation orders."* (Burada sunulan çalışma müdahale evresini hedeflemektedir. Bir yangın hâlihazırda yanıyorsa, belirleyici operasyonel soru artık bir yangının çıkıp çıkmayacağı değil, yangın cephesinin yarın nerede olacağıdır; çünkü uçakların, yer ekiplerinin ve tahliye emirlerinin dağılımını bu belirler.)

Bu cümleyi ezberleyin. Tüm projenin tek cümlelik gerekçesidir ve IGNIS'i, önleme ve hazırlık evrelerine hizmet eden çok geniş yangın *risk haritası* yazınından ayırır.

### 1.8 Duyarlılık, yayılım değildir

Bu ayrım projenin entelektüel çekirdeğidir ve ikisini asla birbirine karıştırmamalısınız.

| | **Yangın duyarlılığı** | **Yangın yayılımı** |
|---|---|---|
| İngilizce | Fire susceptibility | Fire spread |
| Soru | Yangın nerede çıkabilir? | Bir yangın yanıyor — yarın nerede olacak? |
| Zamansal mı? | Hayır — durağan harita | Evet — gün $t$ → gün $t+1$ |
| Tipik yordayıcılar | Uzun dönemli iklim ortalamaları, yollara uzaklık, nüfus yoğunluğu, eğim, yakıt türü | Bugünkü yangın maskesi + bugünkü hava durumu + yakıt + arazi |
| Hedef ne sıklıkta değişir | Yıllar içinde | Her gün |
| Yazındaki başarım | ROC-AUC çoğu zaman **0.93'ün üzerinde** | Çok daha düşük; AUC-PR genellikle küçük |
| **Bizim problemimiz bu mu?** | **Hayır** | **Evet** |

Duyarlılık neden bu kadar yüksek puan alır? Çünkü hedef neredeyse durağandır. Dik, kuru, bir yola yakın ve ıssız bir çam yamacı bu yıl da, gelecek yıl da, ondan sonraki yıl da yüksek riskli bir pikseldir. Bir model coğrafyayı ezberleyip yüksek puan alabilir. Ortada esasen hiçbir öngörü işi yoktur.

Yayılım farklıdır. Bir yamadaki yaklaşık dört bin pikselden hangisinin önümüzdeki 24 saat içinde tutuşacağını tahmin etmelisiniz. Dünkü cevap bugüne taşınmaz. Bildirinin ifadesiyle IGNIS, "must instead identify which specific pixels, out of roughly four thousand in a patch of which fewer than a dozen are typically burning, will ignite within twenty-four hours." (bunun yerine, tipik olarak bir düzineden azının yandığı, yaklaşık dört bin pikselli bir yamada hangi belirli piksellerin yirmi dört saat içinde tutuşacağını belirlemek zorundadır.)

**Bir jüri üyesi sizin 0.847'lik ROC-AUC değerinizi yayımlanmış 0.95'lik bir duyarlılık skoruyla karşılaştırırsa, cevabınız bu tablodur.** Bunlar aynı problem değildir ve sayılar karşılaştırılabilir değildir.

#### Biçimsel ifade

IGNIS, ertesi gün yayılımını **ikili anlamsal bölütleme** (*binary semantic segmentation*) olarak formüle eder.

$$X \in \mathbb{R}^{H \times W \times C}, \qquad H = W = 64, \quad C = 14$$

$$Y \in \{0,1\}^{H \times W}$$

$X$ girdi tensörüdür: gün $t$'de gözlenmiş, $H \times W$ piksellik bir yama üzerindeki $C$ adet çevresel haritanın üst üste yığılmış hâli. $Y$ hedeftir: 1 değerinin "bu piksel gün $t+1$'de yanıyor olarak tespit edildi" anlamına geldiği ikili bir maske. Model şunu üretir:

$$P(i,j) = \frac{1}{1 + e^{-z(i,j)}} \in (0,1)$$

yani $(i,j)$ pikselinin yarın yanıyor olmasına ilişkin piksel başına bir olasılık. İkili maske, bir $\tau$ değerinde eşikleme yapılarak elde edilir; ön çalışmada $\tau = 0.5$ kullanılmıştır.

---

## 2. Sıfırdan uzaktan algılama

**Uzaktan algılama** (*remote sensing*), bir nesnenin bir özelliğini ona dokunmadan ölçmek demektir. Gözünüz bir uzaktan algılayıcıdır. Uydu ise 700 kilometre uzaktan çalışan ve gözünüzün göremeyeceği dalga boylarını görebilen bir uzaktan algılayıcıdır.

### 2.1 Elektromanyetik tayf

Işık bir elektromanyetik dalgadır (*electromagnetic wave*). **Dalga boyu** (*wavelength*, simge $\lambda$), maddeyle nasıl etkileşeceğini belirler. Dalga boylarının tamamı **elektromanyetik tayfı** (*electromagnetic spectrum*) oluşturur.

```
 kısa λ ◄───────────────────────────────────────────────────► uzun λ
 
 Gama  X ışını  UV │ GÖRÜNÜR │  NIR   SWIR  │  MWIR    TIR  │ Mikrodalga  Radyo
                   │ 0.4–0.7 │ 0.7–1.3 1.3–3│  3–5     8–14 │
                   │   µm    │     µm    µm │   µm      µm  │
                        ▲        ▲            ▲         ▲
                        │        │            │         │
                  NDVI kırmızı NDVI NIR   yangın 4µm  yangın 11µm
                                          kanalı      kanalı, LST
```

`µm` bir **mikrometredir**, yani metrenin milyonda biri. Birimler jüri açısından önemlidir: "dört" değil, "dört mikrometre" deyin.

Temel fiziksel yasa **Planck yasasıdır** (*Planck's law*); bu yasa, mutlak sıfırın üzerinde sıcaklığa sahip her nesnenin ışıma yaydığını ve bu ışımanın *tayfının* sıcaklığa bağlı olduğunu söyler. Wien'in yer değiştirme yasası tepe noktasını verir:

$$\lambda_{\text{peak}} \approx \frac{2898 \ \mu\text{m}\cdot\text{K}}{T}$$

İki durumu adım adım hesaplayın, çünkü uydudan yangın tespitinin tüm temeli budur:

| Nesne | Sıcaklık | Tepe dalga boyu |
|---|---|---|
| Normal yer yüzeyi | ~300 K (27 °C) | $2898/300 \approx 9.7\ \mu$m — termal kızılötesi |
| Alevli orman yangını | ~800–1000 K | $2898/900 \approx 3.2\ \mu$m — orta dalga kızılötesi |

Bir yangın arka plandan yalnızca *daha parlak* değildir — **tayfın farklı bir bölümünde** daha parlaktır. Yörüngeden otomatik yangın tespitini mümkün kılan şey bu farktır.

### 2.2 Bir uydu aslında neyi kaydeder

Bir uydu görüntüsü fotoğraf değildir. Bir sayı ızgarasıdır.

Algılayıcı bir dedektör dizisidir. Her dedektör, küçük bir katı açıdan gelen fotonları kısa bir bütünleme süresi boyunca toplar ve bunları bir elektriksel sinyale çevirir; bu sinyal de sayısallaştırılarak bir **sayısal değere** (digital number, DN) dönüştürülür. Yer işleme aşaması ardından DN'yi fiziksel bir niceliğe çevirir:

- **Radyans / ışıma** (*radiance*) — birim alan, birim katı açı ve birim dalga boyu başına düşen enerji. Birimi: W·m⁻²·sr⁻¹·µm⁻¹.
- **Yansıtırlık** (*reflectance*) — yüzeyin yansıttığı gelen güneş ışığı oranı; [0, 1] aralığında boyutsuz bir sayı. Görünür/NIR bantlar için kullanılır.
- **Parlaklık sıcaklığı** (*brightness temperature*) — kusursuz bir kara cismin, gözlenen radyansı yayabilmesi için sahip olması gereken sıcaklık. Termal bantlar için kullanılır, simgesi $T_b$.

Ham DN'den kullanılabilir bir fiziksel değere giden düzeltme zinciri **radyometrik düzeltme** (*radiometric correction*), **atmosferik düzeltme** (*atmospheric correction*) ve her pikseli yer üzerindeki doğru konuma yerleştiren geometrik düzeltmeden oluşur.

**IGNIS'te bunların hiçbirini biz yapmıyoruz.** Kullanılan tüm ürünler NASA, ECMWF ve UCSB tarafından kalibre edilmiş, analize hazır koleksiyonlar olarak dağıtılmaktadır. Bildiri şunu belirtir: *"Because the selected products are distributed as calibrated, analysis-ready collections, no additional radiometric correction was applied."* (Seçilen ürünler kalibre edilmiş, analize hazır koleksiyonlar olarak dağıtıldığından, ek bir radyometrik düzeltme uygulanmamıştır.) Jüri atmosferik düzeltmeyi nasıl yaptığınızı sorarsa dürüst cevap şudur: veri sağlayıcıları yaptı, biz Seviye-3 ürünleri kullanıyoruz. Bu standart bir uygulamadır ve bir zayıflık değildir.

Nihai ürün bir **rasterdir** (*raster*, hücresel veri): her hücresi sabit bir yer alanını kaplayan bir **piksel** (*pixel*) olan ve her **bant** (*band*) için bir değer taşıyan iki boyutlu bir dizi.

### 2.3 Pasif ve aktif algılama

| | **Pasif algılama** | **Aktif algılama** |
|---|---|---|
| İngilizce | Passive sensing | Active sensing |
| Enerji kaynağı | Güneş ya da Dünya'nın kendi termal yayınımı | Algılayıcı kendi sinyalini yayar |
| Örnekler | MODIS, Landsat, Sentinel-2, gözünüz | Radar, LiDAR, SRTM'in radar interferometrisi |
| Geceleri çalışır mı? | Yalnızca termal bantlarda | Evet |
| Bulutun içinden görür mü? | Hayır (mikrodalga hariç) | Radar: evet |

IGNIS, değişken olan her şey için (yangın, NDVI, LST) **pasif** optik ve termal algılamayı, topoğrafya için ise bir **aktif** ürünü (radar kullanan SRTM) kullanır. Bu karışımdan söz etmeye değer, çünkü aradaki ayrımı anladığınızı gösterir.

Pasif algılamaya bağımlılık aynı zamanda gerçek bir kısıttır: **bulut ve yoğun duman optik ve termal yolu engeller**. Bir yangın geçiş anında duman örtüsünün altındaysa MODIS onu tespit edemeyebilir. Bu varsayımsal bir durum değildir — Bölüm 9.4, bunun hedef değişkenimizdeki baskın gürültü kaynaklarından biri olduğunu göstermektedir.

### 2.4 Üç çözünürlük ve aralarındaki ödünleşim

Her Yer Gözlem algılayıcısı üç tür çözünürlük arasındaki bir uzlaşmadır.

| Çözünürlük türü | İngilizce | Tanım | MODIS değeri |
|---|---|---|---|
| **Uzamsal** | Spatial resolution | Bir pikselin yer üzerindeki boyutu | 250 m – 1 km |
| **Spektral** | Spectral resolution | Dalga boyu bantlarının sayısı ve darlığı | 36 bant |
| **Zamansal** | Temporal resolution | Aynı yerin ne sıklıkta yeniden ziyaret edildiği | Günde 1–2 kez |

**Üçüne birden neden sahip olamıyoruz?** Foton bütçesi yüzünden. Bir dedektörün, elektronik gürültü tabanının üzerinde bir sinyal üretebilmesi için bütünleme süresi içinde yeterli sayıda foton toplaması gerekir. Topladığı foton sayısı (pikselin yerdeki alanı) × (spektral bandın genişliği) × (bütünleme süresi) ile orantılıdır. Daha iyi uzamsal çözünürlük için pikseli küçültürseniz bunu telafi etmek zorundasınız: spektral bandı genişletmek (daha kötü spektral çözünürlük), bütünleme süresini uzatmak (bu da daha dar bir tarama genişliği ve dolayısıyla daha kötü zamansal çözünürlük demektir) ya da daha fazla gürültüyü kabul etmek.

Bu, klasik ödünleşim tablosunu verir:

| Algılayıcı | Uzamsal | Tekrar ziyaret | Uygun olduğu iş |
|---|---|---|---|
| MODIS (Terra/Aqua) | 250 m – 1 km | 1–2×/gün | Geniş alanların günlük izlenmesi |
| VIIRS (Suomi-NPP) | 375 m (yangın) | ~1–2×/gün | Daha ince yangın tespiti |
| Landsat 8/9 | 30 m | 16 gün | Ayrıntılı haritalama, yanan alan değerlendirmesi |
| Sentinel-2 | 10–20 m | 5 gün (2 uydu) | Ayrıntılı bitki örtüsü haritalaması |
| Yer durağan (MSG/SEVIRI) | 3 km | 15 dakikada bir | Hızlı tespit, zayıf ayrıntı |

**IGNIS MODIS'i seçti, çünkü ertesi gün tahmini günlük tekrar ziyaret gerektirir.** Bir sonraki görüntü 16 gün sonra gelecekse Landsat'ın 30 m'lik ayrıntısı işe yaramaz — yangın çoktan bitmiş olur. Bu, "neden daha yüksek çözünürlüklü veri kullanmıyorsunuz?" sorusunun doğru cevabıdır ve bunu güvenle vermelisiniz. Bildirinin zaten kabul ettiği dürüst devam şudur: **375 m'lik VIIRS karşılaştırılabilir bir tekrar ziyaret süresine sahiptir ve kesinlikle daha iyi olurdu**; VIIRS'e geçiş gelecek çalışmalar arasında listelenmiştir.

### 2.5 Yörüngeler

| Yörünge türü | İngilizce | Yükseklik | Özelliği |
|---|---|---|---|
| **Alçak Dünya yörüngesi (LEO)** | Low Earth Orbit | 160–2000 km | Hızlı, yakın, yüksek ayrıntı, dar görüş |
| **Güneş eş-zamanlı (SSO)** | Sun-synchronous | ~700–800 km | Her turda ekvatoru *aynı yerel güneş saatinde* geçer |
| **Yer durağan (GEO)** | Geostationary | 35.786 km | Yörünge periyodu = 24 saat, bu nedenle tek bir boylamın üzerinde asılı kalır |

**Güneş eş-zamanlı yörünge**, düzlemi tam olarak Dünya'nın Güneş çevresindeki dolanma hızıyla (günde yaklaşık 0.986°) yalpalayan, kutuplara yakın bir yörüngedir; bu, eğimin 90°'den biraz büyük seçilmesiyle sağlanır, böylece Dünya'nın ekvatordaki şişkinliği gereken burulma momentini üretir. Sonuç olarak uydu belirli bir enlemi her zaman aynı yerel güneş saatinde gözler. Bu, görüntüleri günler arasında karşılaştırılabilir kılar — Güneş her zaman aynı açıdadır — ki bu da *değişim* tespiti için zorunludur; IGNIS'in yaptığı da tam olarak budur.

**Yer durağan yörünge**, uzamsal ayrıntıyı zamansal yoğunlukla takas eder. Meteosat, Avrupa ve Afrika'nın tamamını 15 dakikada bir görür, ama 3 km'lik piksellerle. Hızlı yangın *tespiti* için bu değerlidir. Bir yangın cephesini haritalamak için değildir.

### 2.6 MODIS, Terra ve Aqua

**MODIS** = **Mo**derate Resolution **I**maging **S**pectroradiometer (Orta Çözünürlüklü Görüntüleme Spektroradyometresi). Bu bir uydu değil, bir cihazdır ve iki farklı NASA uydusunda uçan iki kopyası vardır:

| Uydu | Fırlatılış | Yörünge | Yerel ekvator geçişi |
|---|---|---|---|
| **Terra** | Aralık 1999 | Güneş eş-zamanlı, alçalan | ~10:30 **öğleden önce** |
| **Aqua** | Mayıs 2002 | Güneş eş-zamanlı, yükselen | ~1:30 **öğleden sonra** |

Bu size sorulacak bir sorudur, o yüzden cevabını bilin: **Terra ve Aqua, aynı cihazı taşıyan, günün farklı saatlerinde gözlem yapacak şekilde bilinçli olarak yerleştirilmiş iki ayrı uydudur.** Birlikte, Dünya üzerindeki herhangi bir noktanın günde dörde kadar gözlemini sağlarlar (iki gündüz, iki gece), çünkü her uydu bir konumu yükselen/alçalan geçişinde görür.

MODIS'in 36 spektral bandı, 2330 km'lik bir tarama genişliği ve 250 m (1–2. bantlar), 500 m (3–7. bantlar) ile 1 km (8–36. bantlar) uzamsal çözünürlüğü vardır.

**IGNIS'te** dört MODIS ürünü kullanıyoruz:

| Ürün | Ne sağlar | Doğal çözünürlük |
|---|---|---|
| `MODIS/061/MOD14A1` | Terra günlük termal anomali / aktif yangın | 1 km / günlük |
| `MODIS/061/MYD14A1` | Aqua günlük termal anomali / aktif yangın | 1 km / günlük |
| `MODIS/061/MOD13Q1` | Terra 16 günlük NDVI bileşiği | 250 m / 16 gün |
| `MODIS/061/MOD11A1` | Terra günlük arazi yüzey sıcaklığı | 1 km / günlük |
| `MODIS/061/MCD12Q1` | Birleşik Terra+Aqua yıllık arazi örtüsü | 500 m / yıllık |

"061", **Koleksiyon 6.1** işleme sürümüdür. Koleksiyonlar önemlidir: Koleksiyon 5 ile Koleksiyon 6 arasındaki bir algoritma değişikliği sayıları değiştirir. Koleksiyonu her zaman belirtin.

**Terra ile Aqua'yı birleştirerek** tek bir günlük ikili yangın maskesi üretiyoruz; bir pikseli, o gün uydulardan herhangi biri orada yangın tespit ettiyse yanıyor kabul ediyoruz. Bu, bir geçişte gizlenmiş ya da fazla soğuk kalmış bir yangını yakalama şansımızı kabaca iki katına çıkarır.

### 2.7 Bir uydu yangını nasıl tespit eder

Bu, 2. Bölüm'ün teknik olarak en etkileyici kısmıdır ve 90 saniyede açıklayabilmeye değer.

**Sezgi.** Bölüm 2.1'den hatırlayın: 300 K'deki normal bir yüzeyin tepesi 10 µm civarındayken, 800–1000 K'deki alevli bir yangının tepesi 3–4 µm civarındadır. Şimdi 1 km'lik bir MODIS pikselini düşünün — bir milyon metrekare — ve içinde yalnızca 1000 m²'yi, yani pikselin %0,1'ini kaplayan bir yangın olsun. 11 µm kanalında yangın piksel ortalamasındaki parlaklık sıcaklığını neredeyse hiç değiştirmez, çünkü pikselin serin zemin olan %99,9'luk kısmı baskındır. Ancak 4 µm kanalında, sıcak kesirden yayılan radyans arka plandan muazzam ölçüde büyüktür, çünkü Planck fonksiyonu kısa dalga boylarında sıcaklıkla çok dik biçimde yükselir. Böylece 4 µm parlaklık sıcaklığı sıçrarken, 11 µm parlaklık sıcaklığı neredeyse hiç kıpırdamaz.

**Dozier'in kavrayışı.** Dozier (1981), piksel altı bir sıcak noktayı farklı dalga boylarındaki iki kanalda gözlerseniz elinizde iki denklem olacağını ve iki bilinmeyeni çözebileceğinizi gösterdi: sıcak kesrin *sıcaklığı* ve kapladığı *alan oranı*. Bu, tüm uydu tabanlı aktif yangın tespitinin kuramsal temelidir — bir pikselden çok daha küçük bir yangını tespit edip niteleyebilirsiniz.

**MODIS algoritması** (Giglio ve ark. 2016, Koleksiyon 6) bir **bağlamsal** test olarak çalışır. Aday bir piksel için:

1. $T_4$ (4 µm'deki parlaklık sıcaklığı) ile $T_{11}$ (11 µm'deki) değerlerini ve aralarındaki farkı $\Delta T = T_4 - T_{11}$ hesaplar.
2. Bariz biçimde soğuk pikselleri elemek için mutlak eşikler uygular.
3. Adayın çevresinde bir **arka plan** piksel penceresi kurar — geçerli, bulutsuz, yangın içermeyen kara pikselleri — ve bunların $T_4$ ile $\Delta T$ ortalamasını ve ortalama mutlak sapmasını hesaplar.
4. Aday, yerel arka planı istatistiksel olarak anlamlı bir payla aşıyorsa yangın ilan eder; örneğin $T_4 > \bar{T_4} + 3\delta_{T_4}$ ve $\Delta T > \overline{\Delta T} + 3\delta_{\Delta T}$.
5. **Yanlış alarmlar** için eleme testleri uygular: su üzerindeki güneş parıltısı, çöl sınırları, sıcak çıplak toprak ve kıyı pikselleri.

**Bağlamsal** kelimesi önemlidir: eşik sabit değildir, yerel arka plana uyum sağlar. Geceleyin serin bir ormandaki yangını tespit etmek, öğle vakti sıcak bir çöldekini tespit etmekten daha kolaydır ve algoritma bunu kendiliğinden hesaba katar.

**Bunun bizim için anlamı.** MODIS "yanan alan" raporlamaz. **Aktif yangın** raporlar: geçişin tam gerçekleştiği anda etkin biçimde alevlenen bir piksel. Yangın için için yanıyorsa, bulut altındaysa ya da iki geçiş arasına denk geldiyse tespit edilmez. Bu, hedef değişkenimiz üzerindeki en önemli tek çekince olup Bölüm 9.4 bunun etkisini niceliklendirir.

### 2.8 FireMask güven sınıfları

MOD14A1/MYD14A1 ürünleri, değerleri her pikselin sınıflandırmasını kodlayan `FireMask` adlı bir bant içerir:

| Değer | Anlamı |
|---|---|
| 0 | İşlenmedi (girdi verisi eksik) |
| 1 | İşlenmedi (kullanımdan kalkmış) |
| 2 | İşlenmedi (başka bir nedenle) |
| 3 | Su (yangın değil) |
| 4 | Bulut (yangın değil) |
| 5 | Yangın içermeyen kara pikseli |
| 6 | Bilinmiyor (kara pikseli, yetersiz veri) |
| **7** | **Yangın — düşük güven** |
| **8** | **Yangın — nominal güven** |
| **9** | **Yangın — yüksek güven** |

**IGNIS `FireMask >= 7` kullanır**, yani düşük güvenli tespitleri de yangın olarak kabul eder (`src/config.py` içinde `FIRE_CONFIDENCE = 7`).

Bu, açık bir ödünleşimi olan bir tasarım tercihidir ve jüri bunu kurcalayabilir:

- **Eşik 7** (bizim tercihimiz): yangın sınıfında azami **duyarlılık**. Zayıf ve sınırdaki yangınları yakalarız. Bedeli: hedef maskesinde daha fazla yanlış tespit, dolayısıyla daha fazla etiket gürültüsü.
- **Eşik 9**: azami **kesinlik**. Yalnızca kesin yangınlar. Bedeli: erken evredeki ve küçük yangınların çoğunu kaybederiz — ki operasyonel bir sistemin tahmin etmeye en çok ihtiyaç duyduğu yangınlar tam olarak bunlardır.

Pozitif sınıfımız zaten piksellerin yalnızca %0,2686'sını oluşturduğuna göre, etiket saflığını artırmak uğruna pozitiflerin üçte ikisini çöpe atmak, zaten imkânsız olan bir öğrenme problemini daha da kötüleştirirdi. Savunulabilir gerekçe budur.

### 2.9 NDVI

**NDVI** = **N**ormalized **D**ifference **V**egetation **I**ndex (Normalize Edilmiş Fark Bitki Örtüsü İndeksi).

$$\text{NDVI} = \frac{\rho_{\text{NIR}} - \rho_{\text{Red}}}{\rho_{\text{NIR}} + \rho_{\text{Red}}}$$

burada $\rho_{\text{NIR}}$ yakın kızılötesi banttaki (~0.86 µm) yansıtırlık ve $\rho_{\text{Red}}$ kırmızı banttaki (~0.65 µm) yansıtırlıktır.

**Bu neden bitki örtüsünü ölçer?** Klorofil ve yaprak yapısına dair iki olgu nedeniyle:

1. Klorofil kırmızı ışığı güçlü biçimde **soğurur** — fotosentez için kullandığı şey odur. Bu nedenle sağlıklı yaprakların kırmızı yansıtırlığı *düşüktür*, 0.03–0.05 civarında.
2. Bir yaprağın iç süngerimsi mezofil yapısı yakın kızılötesi ışığı güçlü biçimde **saçar** — yaprak NIR için esasen saydamdır ve onu içeride sektirir. Bu nedenle sağlıklı yaprakların NIR yansıtırlığı *yüksektir*, 0.40–0.50 civarında.

Dolayısıyla ikisi arasındaki karşıtlık sağlıklı bitki örtüsü için çok büyük, stres altındaki bitki örtüsü için küçük, su için ise terstir.

Adım adım örnekler:

| Yüzey | $\rho_{\text{Red}}$ | $\rho_{\text{NIR}}$ | NDVI |
|---|---|---|---|
| Yoğun, sağlıklı orman | 0.04 | 0.45 | $(0.45-0.04)/(0.49) = 0.84$ |
| Kuru ot / stres altındaki bitki örtüsü | 0.22 | 0.30 | $(0.30-0.22)/(0.52) = 0.15$ |
| Çıplak toprak | 0.25 | 0.28 | $(0.28-0.25)/(0.53) = 0.06$ |
| Su | 0.05 | 0.02 | $(0.02-0.05)/(0.07) = -0.43$ |
| Kar | 0.85 | 0.80 | $(0.80-0.85)/(1.65) = -0.03$ |

NDVI −1 ile +1 arasında değişir. Yaklaşık 0.4'ün üzerindeki değerler kayda değer bir yeşil biyokütleye işaret eder.

**Neden "normalize edilmiş fark" biçimi?** Toplama bölmek, her iki bandı da aynı ölçüde ölçekleyen çarpımsal etkileri sadeleştirir — aydınlanma açısı, topoğrafik gölgeleme, bir miktar atmosferik zayıflama. Kuzeye bakan bir yamaç her iki bantta da daha karanlıktır, ama *oransal* yapı ayakta kalır. Bu kadar çok uzaktan algılama indeksinin normalize edilmiş fark biçimini kullanmasının nedeni budur.

**IGNIS'te** NDVI, yakıt yükü vekil göstergemizdir. MOD13Q1 250 m çözünürlükte **16 günlük bir bileşiktir**, dolayısıyla her gün mevcut değildir. İşleme hattı, her gözlem tarihinden önceki 32 günlük pencere içindeki en güncel bileşiği alır ve saklanan 16-bitlik tam sayıları [−1, 1] aralığına geri çevirmek için 0.0001 MODIS ölçek çarpanını uygular.

**Dürüstçe belirtilmesi gereken kısıt:** NDVI yüksek biyokütlede **doyuma ulaşır** — taç kapandıktan sonra daha fazla yaprak eklemek NDVI'yı pek değiştirmez. Ayrıca *kuruluğu* değil, *yeşilliği* ölçer. Kısa dalga kızılötesi bandını kullanan NDWI veya NDMI gibi bir indeks, yakıt nemi için daha iyi bir vekil gösterge olurdu. Bu, gelecek çalışma olarak zikredilmeye değer, meşru bir iyileştirmedir.

### 2.10 Arazi yüzey sıcaklığı

**LST** (*land surface temperature*, arazi yüzey sıcaklığı), zeminin radyometrik yüzey sıcaklığıdır ve termal kızılötesi bantlardan (MODIS'te 11 ve 12 µm) bir **ayrık pencere algoritmasıyla** elde edilir: iki bant, atmosferdeki su buharı tarafından farklı ölçülerde zayıflatılır, dolayısıyla aralarındaki fark atmosferi düzeltmek için kullanılabilir.

**LST, hava sıcaklığı değildir.** Bu, jürinin en sevdiği sorulardan biridir ve ayrım gerçektir:

| | LST | Hava sıcaklığı (2 m) |
|---|---|---|
| İngilizce | Land surface temperature | Air temperature |
| Neyi ölçer | Zeminin/taç yüzeyinin kendi sıcaklığını | Zeminin 2 m üzerindeki havanın sıcaklığını |
| IGNIS'teki kaynağı | MODIS MOD11A1 (doğrudan gözlem) | ERA5-Land (model yeniden analizi) |
| Tipik yaz öğle değeri | Çıplak kaya 55–60 °C'ye ulaşabilir | 35 °C |
| Neye duyarlıdır | Yüzey malzemesi, nem, gölgelenme | Hava kütlesi, yatay taşınım |

Sıcak ve kuru bir günde çıplak toprak üzerindeki LST, hava sıcaklığını 20 °C aşabilir. İyi sulanan bir tarlada ise hava sıcaklığının *altında* olabilir, çünkü buharlaşma yüzeyi soğutur. İstediğimiz bilgi tam olarak bu farktır: yüzeydeki nem stresinin bir göstergesidir. **IGNIS'in `lst` ile `air_temp` kanallarını ayrı ayrı taşımasının nedeni budur — bunlar birbirinin tekrarı değildir.**

MOD11A1 bulut kaynaklı boşluklar içerdiğinden, IGNIS LST'yi **önceki üç günlük getirimin ortalaması** olarak hesaplar; bu, eksik pikselleri belirgin biçimde azaltır.

### 2.11 Yeniden analiz: ERA5-Land

Bu, IAC'de sorulma olasılığı en yüksek sorulardan biridir, o yüzden doğru anlayın.

Bir **yeniden analiz** (*reanalysis*) ne bir gözlem ne de bir öngörüdür. *Atmosferin geçmişteki durumunun bir yeniden inşasıdır*; modern, sürümü dondurulmuş bir sayısal hava tahmini modelinin tarihsel tarihler üzerinde çalıştırılması ve bu sırada mevcut her gözlemin — yer istasyonları, radyosondeler, uçaklar, gemiler, şamandıralar ve uydu radyansları — sürekli olarak özümsenmesiyle üretilir.

Bu mekanizmaya **veri özümsemesi** (*data assimilation*) denir. Her analiz adımında model kısa bir öngörü üretir; o penceredeki mevcut gözlemler öngörüyle karşılaştırılır; ve model durumu, her birinin kestirilen hatasıyla ağırlıklandırılarak ikisi arasındaki istatistiksel olarak en uygun uzlaşmaya doğru itilir. Sonuç, kimsenin hiçbir şey ölçmediği yer ve zamanlar da dâhil olmak üzere her yerde atmosferin fiziksel olarak tutarlı, boşluksuz ve ızgaralanmış bir kestirimidir.

**ERA5**, ECMWF'in beşinci nesil küresel yeniden analizidir. **ERA5-Land** ise ERA5'in yaklaşık 9 km çözünürlükte, ERA5 atmosferik zorlamasıyla sürülen, daha yüksek çözünürlüklü bir kara yüzeyi yeniden çalıştırmasıdır (Muñoz-Sabater ve ark. 2021).

**IGNIS `ECMWF/ERA5_LAND/DAILY_AGGR` kullanır** — günlük toplulaştırılmış sürüm — ve beş nicelik için: 2 m hava sıcaklığı, 2 m çiy noktası sıcaklığı, doğuya doğru rüzgâr bileşeni $u$, kuzeye doğru rüzgâr bileşeni $v$ ve hacimsel toprak suyu.

> Depo için not: kodun daha eski bir sürümü, Google Earth Engine'den kaldırılmış olan ve `ImageCollection asset ... not found` hatası üreten `ECMWF/ERA5_LAND/DAILY` kimliğini çağırıyordu. Doğru güncel kimlik `ECMWF/ERA5_LAND/DAILY_AGGR`'dir.

**Model çıktısı bir sorun mu?** Bu cevabı dürüstçe verin:

*Evet, bir kısıttır ve ödünleşim şudur. Türkiye'de sınırlı sayıda meteoroloji istasyonu vardır ve yanan bir ormanın içinde hiçbiri yoktur. İstasyon verisi nokta verisidir — 1 km'lik bir ızgarada kullanmak için onu ara değerlemeniz gerekir, ki bu da başlı başına bir modeldir ve ERA5'in veri özümseme sisteminden çok daha kaba bir modeldir. ERA5-Land fiziksel olarak tutarlı, boşluksuz, küresel olarak tekdüze ve ihtiyaç duyduğumuz tüm tarihsel dönem için serbestçe erişilebilirdir. Bilinen zayıflığı, ~9 km çözünürlükte yerel arazi kaynaklı rüzgârları temsil edememesidir — vadi kanallanması, yamaç rüzgârları ve yangının kendi taşınımla ürettiği rüzgâr alanı. Toros Dağları gibi dik arazilerde bu, gerçek bir hata kaynağıdır. Gelecekteki bir sürüm, ERA5'i WRF gibi bir mezoölçekli modelle ölçek küçültebilir.*

Özellikle son noktaya dikkat edin: **bir orman yangını kendi rüzgârını üretir**. Hiçbir yeniden analiz yangından haberdar değildir, dolayısıyla hiçbir yeniden analiz yangına doğru olan içe çekişi veya duman sütununun sürüklediği dolaşımı temsil edemez. Bu, yangın yayılımı için herhangi bir meteorolojik ürün kullanmanın temel bir kısıtıdır.

### 2.12 CHIRPS

**CHIRPS** = **C**limate **H**azards Group **I**nfra**R**ed **P**recipitation with **S**tation data (Funk ve ark. 2015).

CHIRPS *harmanlanmış* bir yağış veri kümesidir. Şunları birleştirir:
- yüksek çözünürlüklü bir iklim ortalaması (uzun dönemli yağış örüntüsü);
- bulut tepelerinin ne kadar süre çok soğuk kaldığından yağışı çıkarsayan, uydu termal-kızılötesi soğuk bulut süresi kestirimleri;
- ve uydu kestirimlerini düzeltmek için kullanılan yerinde yağış ölçer istasyon verileri.

1981'den günümüze yakın bir tarihe kadar ~5 km'lik bir ızgarada günlük yağış sağlar.

**Neden ERA5 yağışı yerine CHIRPS?** Çünkü yağış, sayısal modellerin *en az* doğru yaptığı alandır — modelin çözemediği ölçeklerdeki konveksiyona bağlıdır. Gerçek ölçüm aletlerine bağlanmış bir ürün, yağış için saf model çıktısına göre genellikle daha güvenilirdir. IGNIS'in sıcaklık/rüzgâr/toprak nemi için ERA5-Land'i, yağış için ise CHIRPS'i kullandığına dikkat edin; bu, her değişken için mevcut en iyi kaynağın bilinçli bir tercihidir ve böyle ifade edilmeye değer.

**Bir yangın modeli için çekince:** Akdeniz yangın mevsimi boyunca Antalya üzerindeki yağış temmuz ve ağustosta esasen sıfırdır. Dolayısıyla `precip` kanalı, en çok önem taşıyan durumların tam da içinde neredeyse her zaman 0'dır ve yangın mevsimi içinde çok az bilgi taşır. Son yağmurun üzerinden ne kadar geçtiğini temsil etmek açısından daha anlamlıdır — ki bunu şu anda hesaplamıyoruz. **Son yağıştan bu yana geçen gün** ya da birikimli açık kanalı, anlık günlük yağıştan çok daha bilgilendirici olurdu ve önerilmeye değer iyi bir iyileştirmedir.

### 2.13 SRTM, yükseklik, eğim ve bakı

**Mekik Radar Topografya Görevi** (Shuttle Radar Topography Mission, SRTM, Farr ve ark. 2007) Şubat 2000'de Endeavour uzay mekiğiyle uçtu. İki radar anteni taşıyordu — biri yük bölmesinde, diğeri 60 m'lik açılabilir bir direğin ucunda — ve **interferometrik yapay açıklıklı radar (InSAR)** kullanıyordu: iki antende alınan sinyaller arasındaki faz farkı ölçülerek, yer üzerindeki her noktanın yüksekliği üçgenleme ile hesaplanabilir. 11 gün içinde Dünya'nın kara yüzeyinin kabaca %80'ini haritaladı.

Sonuç bir **Sayısal Yükseklik Modelidir** (Digital Elevation Model, DEM): her piksel değerinin, referans elipsoit/jeoit üzerindeki metre cinsinden yüksekliği verdiği bir raster. `USGS/SRTMGL1_003`, 1 yay saniyelik (~30 m) küresel üründür.

Bir DEM'den iki değişken daha türetilir; her ikisi de Google Earth Engine'de `ee.Terrain.products()` ile hesaplanır.

**Eğim** (*slope*), yüzeyin dikliğidir. Sayısal olarak, dereceye çevrilmiş yükseklik gradyanının büyüklüğüdür:

$$\text{slope} = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right)$$

Uygulamada kısmi türevler, 3×3'lük bir piksel komşuluğundan sonlu farklarla kestirilir — ki bu, Bölüm 5.4'te göreceğiniz gibi, kelimenin tam anlamıyla bir evrişimdir.

**Bakı** (*aspect*), yamacın baktığı pusula yönüdür ve kuzeyden saat yönünde ölçülür:

$$\text{aspect} = \arctan2\left(\frac{\partial z}{\partial y}, \frac{\partial z}{\partial x}\right)$$

0–360° aralığına çevrilir. Bakı 0° = yamaç kuzeye bakar; 90° = doğu; 180° = güney; 270° = batı.

Bakı yangın için önemlidir, çünkü kuzey yarımkürede **güneye bakan bir yamaç** çok daha fazla doğrudan güneş ışınımı alır, dolayısıyla toprakları ve yakıtları daha sıcak ve daha kurudur. Akdeniz'de güneye bakan yamaçlar tipik olarak daha yanıcı ve daha kuru yakıt taşır.

**Kritik bir sayısal uyarı.** Bakı bir **dairesel değişkendir** (*circular variable*). 359° ile 1° neredeyse aynı yönü tarif eder — fiziksel olarak aralarında 2° vardır — ama ham sayılar olarak 358 fark ederler. Ham bakıyı bir yapay sinir ağına verirseniz, ağ kuzeyde devasa bir süreksizlik görür. **Bu, mevcut IGNIS modelinde doğrulanmış hatalardan biridir** ve Bölüm 7.7, yeni işleme hattının bunu bir sinüs/kosinüs kodlamasıyla nasıl düzelttiğini açıklar.

İkinci bir çekince: SRTM, bizim 1 km'lik ızgaramıza yeniden örneklenmiş 30 m'lik bir üründür. Bu yeniden örnekleme ince arazi yapısının çoğunu ortalama içinde yok eder — yangını kanalize eden 30 m'lik bir dere yatağı 1 km'de tamamen kaybolur. Dolayısıyla `slope` kanalımız, bir itfaiyecinin deneyimlediği yerel eğimi değil, geniş ölçekli araziyi temsil eder.
---
## 3. Coğrafi bilimlerin temelleri ve projeksiyonlar

### 3.1 Dünya neden düzleştirilemez

Bir portakal alın, kabuğunu soyun ve kabuğu yırtmadan ya da germeden bir masanın üzerine düz biçimde sermeye çalışın. Bunu yapamazsınız. Bu, pratik bir zorluk değildir; bir teoremdir.

Carl Friedrich Gauss'un *Theorema Egregium* (1827) adlı teoremi, bir yüzeyin **Gauss eğriliği**nin, germe olmaksızın bükülme altında değişmez kaldığını söyler. Bir kürenin Gauss eğriliği her yerde pozitiftir; bir düzlemin eğriliği ise sıfırdır. Dolayısıyla küreden düzleme yapılan hiçbir eşleme tüm uzaklıkları koruyamaz. Dünya'nın her düz haritası bir şeyi bozar.

**Harita projeksiyonu** (map projection), *neyin* bozulacağına dair belirli bir tercihtir. Olası özellikler şunlardır:

| Korunan özellik | Ad | İngilizce | Bedeli |
|---|---|---|---|
| Açılar / yerel biçim | **Konformal / açı koruyan** | Conformal | Alanlar bozulur |
| Alanlar | **Eşit alanlı** | Equal-area | Biçimler bozulur |
| Bir noktadan uzaklıklar | **Eşit uzaklıklı** | Equidistant | Yalnızca belirli çizgiler boyunca |
| Bir noktadan yönler | **Azimutal** | Azimuthal | Diğer her şey |

Bunlardan *birini* koruyabilirsiniz. Asla hepsini birden değil.

### 3.2 Coğrafi koordinatlara karşı projeksiyonlu koordinatlar

**Coğrafi koordinatlar** (geographic coordinates), bir elipsoidin eğri yüzeyi üzerinde **derece** cinsinden ölçülen enlem ve boylamdır.

- Antalya yaklaşık 36.9° K, 30.7° D konumundadır.
- Sorun: **bir derece sabit bir uzaklık değildir.** Bir derece enlem her zaman yaklaşık 111 km'dir. Bir derece boylam ise ekvatorda 111 km, Antalya'nın enleminde yaklaşık 89 km ($111 \times \cos 36.9° = 88.8$ km) ve kutupta 0 km'dir.

Bu durum, kare piksel ya da alan hesabı gerektiren her iş için enlem/boylamı kullanışsız kılar. Türkiye üzerindeki "0.01° × 0.01°" boyutunda bir piksel 1.11 km yüksekliğinde ama yalnızca 0.89 km genişliğindedir — yani bir kare değil, bir dikdörtgendir. Böyle bir ızgara üzerinde 3×3'lük bir çekirdeği kaydıran bir evrişimli ağ da, farkında olmadan uzamsal olarak anizotropik bir işleç uygulamış olur.

**Projeksiyonlu koordinatlar** (projected coordinates), bir projeksiyon uygulandıktan sonra düz bir düzlem üzerinde **metre** cinsinden ölçülen x ve y değerleridir. Burada 1000 m × 1000 m boyutunda bir piksel gerçekten bir kilometrekaredir.

**IGNIS projeksiyonlu koordinatlar kullanır.** Makalede ortak ızgaranın "sabit piksel boyutlarına sahip bir yamanın her zaman sabit bir yer alanına karşılık gelmesini garanti ettiği" ifadesinin nedeni budur.

### 3.3 UTM ve 35N dilimi

**UTM** = **U**niversal **T**ransverse **M**ercator (Evrensel Enlem Dilimi Merkatör).

Buradaki fikir şudur: konformal bir projeksiyonun, teğet çizgisinden uzakta ciddi biçimde bozulduğunu kabul etmek — ve sonra onu o çizgiden uzakta hiç kullanmamak. UTM, dünyayı her biri 6° boylam genişliğinde **60 dilime** böler ve her dilime, o dilimin **orta meridyeni** üzerinde merkezlenmiş kendi enlem dilimi Merkatör projeksiyonunu verir. Bir dilim içinde ölçek bozulması yaklaşık binde 1'in altında kalır.

Dilim numaralandırması 180° B'den başlar:

$$\text{dilim} = \left\lfloor \frac{\text{boylam} + 180}{6} \right\rfloor + 1$$

Bunu uygulayın ve kesin olun; çünkü bu, tam olarak bir jüri üyesinin kontrol edebileceği türden bir ayrıntıdır.

| Dilim | Boylam aralığı | Orta meridyen | Türkiye'deki şehirler |
|---|---|---|---|
| **35N** | 24° D – 30° D | 27° D | İzmir (27.1° D), Muğla / Marmaris (28.3° D), Denizli |
| 36N | 30° D – 36° D | 33° D | Antalya (30.7° D), Manavgat (31.4° D), Ankara, Mersin |
| 37N | 36° D – 42° D | 39° D | Gaziantep, Malatya |
| 38N | 42° D – 48° D | 45° D | Van, Iğdır |

Türkiye kabaca 26° D ile 45° D arasında uzanır ve bu nedenle gerçekten dört UTM dilimine yayılır.

Peki IGNIS neden tüm ülke için **EPSG:32635 = WGS 84 / UTM dilimi 35N** kullanıyor? Çünkü bir makine öğrenmesi veri kümesinin **tek bir ızgaraya** ihtiyacı vardır. Her konum kendi "öz" dilimini kullansaydı, bir dilim sınırına yakın yamalar farklı koordinat sistemlerinde yaşar ve ortak bir tensörde üst üste yığılamazdı; ayrıca 1 km'lik piksel ızgarası sınır boyunca sürekli olmazdı. Tüm ülke için tek bir dilim seçmek, doğuda daha büyük ölçek bozulmasını kabul etmek demektir — ancak yangın arşivimiz, 35. dilim içinde ya da hemen yanında yer alan Ege ve Batı Akdeniz kıyılarının ağırlığındadır.

Bunu, sorgulanmamış bir varsayılan olarak değil, bedeli açıkça belirtilmiş bilinçli bir mühendislik kararı olarak sunmaya hazır olun. Bedelin dürüst bir ifadesi: 35. dilimin orta meridyeninden kabaca 18° uzakta olan 45° D'de enlem dilimi Merkatör ölçek hatası büyür; bu nedenle IGNIS'in gelecekte Doğu Anadolu'ya genişletilmesi ya dilim değiştirmeli ya da Türkiye üzerinde merkezlenmiş bir Lambert Azimutal Eşit Alanlı projeksiyon gibi eşit alanlı bir projeksiyon benimsemelidir.

`N` harfi **kuzey yarımküre** anlamına gelir; UTM kuzey değerleri kuzeyde ekvatordan, güneyde ise yapay bir başlangıç noktasından ölçülür.

### 3.4 EPSG kodları

**EPSG kodları**, koordinat referans sistemleri için, kamuya açık bir kayıt defterinde tutulan (aslen European Petroleum Survey Group tarafından) benzersiz tam sayı tanımlayıcılarından ibarettir. Var olma nedenleri, "bu hangi koordinat sistemi?" sorusunun tek ve kesin bir yanıtının bulunmasıdır.

| EPSG | Ad | Birimler |
|---|---|---|
| 4326 | WGS 84 coğrafi (enlem/boylam) | derece |
| 3857 | WGS 84 / Sözde-Merkatör (web haritaları) | metre |
| **32635** | **WGS 84 / UTM dilimi 35N — IGNIS tarafından kullanılır** | **metre** |
| 32636 | WGS 84 / UTM dilimi 36N | metre |

UTM kuzey dilimleri için örüntü `326` + dilim numarasıdır; güney dilimleri için `327` + dilim numarası.

Bir **koordinat referans sistemi** (coordinate reference system, CRS) bir projeksiyondan fazlasıdır: bir projeksiyon *artı* bir **datum**dur; datum, referans elipsoidinin biçimini ve konumunu belirtir. Aynı koordinatları yanlış bir datumla kullanmak konumları yüzlerce metre kaydırabilir. IGNIS, GPS'in de kullandığı WGS 84 datumunu kullanır.

### 3.5 Yeniden projeksiyonlama ve yeniden örnekleme

Sekiz ürünümüz sekiz farklı ızgarada gelir: 250 m MODIS sinüzoidal, 500 m sinüzoidal, 1 km sinüzoidal, ~9 km ERA5 düzenli enlem/boylam, ~5 km CHIRPS enlem/boylam, 30 m SRTM enlem/boylam. Bunları tek bir tensöre yığabilmek için hepsinin *aynı* ızgaraya yerleştirilmesi gerekir.

**Yeniden projeksiyonlama** (reprojection), koordinat sisteminin değiştirilmesidir. **Yeniden örnekleme** (resampling) ise bunun sonucunda piksel değerlerine olan şeydir: yeni piksel merkezleri eski piksel merkezlerine denk gelmez, dolayısıyla değerlerin kestirilmesi gerekir.

Üç standart yöntem:

| Yöntem | İngilizce | Nasıl çalışır | Ne zaman kullanılır |
|---|---|---|---|
| **En yakın komşu** | Nearest neighbour | En yakın kaynak pikselin değerini alır | **Kategorik veri** — arazi örtüsü, yangın maskeleri, her türlü sınıf kodu |
| **Çift doğrusal ara değerleme** | Bilinear | Çevredeki 4 pikselin ağırlıklı ortalaması | **Sürekli veri** — sıcaklık, yükseklik, NDVI |
| **Kübik evrişim** | Cubic convolution | Çevredeki 16 pikselin ağırlıklı birleşimi | Pürüzsüzlüğün önemli olduğu sürekli veri; aşırı sapma yapabilir |

**Asla çiğnememeniz gereken kural: kategorik veride asla çift doğrusal ya da kübik yöntem kullanmayın.** Arazi örtüsü sınıfı 5 "karışık orman" ve sınıf 9 "savan" ise, bir 5 ile bir 9'un çift doğrusal ortalaması 7'dir — "açık çalılık" — ve bu, anlamlı hiçbir biçimde ikisinin arasında değildir. Orada bulunmayan bir sınıftır. Benzer biçimde, ikili bir yangın maskesini çift doğrusal ara değerlemeyle işlemek 0.37 gibi değerler üretir; bu ise dünyanın geçerli bir durumu değildir.

**IGNIS'te**: yangın maskeleri ve arazi örtüsü en yakın komşu yöntemini kullanmak zorundadır; sürekli çevresel alanlar ise 1 km'ye yeniden projeksiyonlanırken varsayılan ara değerlemeyi kullanır.

Bir de yön sorusu vardır. 30 m SRTM'den 1 km'ye **inmek**, **aşağı örnekleme / toplulaştırma**dır: 33×33 ≈ 1089 kaynak piksel tek bir hedef piksele katkı verir, dolayısıyla bilgi gerçekten ortalanarak yok edilir. ~9 km ERA5'ten 1 km'ye **çıkmak** ise **yukarı örnekleme**dir: tek bir pikselden, hepsi esasen aynı olan 81 piksel üretiyoruz. Yeni hiçbir bilgi yaratılmaz. Bunu dürüstçe söylemekte fayda var: **meteorolojimiz, 1 km'de gösterilse bile ~9 km'lik bloklar üzerinde pürüzsüzdür.** 64 km'lik bir yamayı kat eden sekiz ya da dokuz piksel, tipik olarak yalnızca bir avuç gerçekten farklı ERA5 değeri içerecektir.

### 3.6 Eş-kayıt ve ölçek

**Eş-kayıt / çakıştırma** (co-registration), her bantta $(i,j)$ pikselinin *aynı yer parçasına* karşılık gelmesini garanti etmek anlamına gelir. Bu olmadan, NDVI pikseliniz ile yangın pikseliniz 500 m kaymış olabilir ve ağ, bir yangın ile komşu vadinin bitki örtüsü arasındaki ilişkiyi öğreniyor olurdu.

IGNIS, eş-kaydı, her bandı Google Earth Engine içinde aynı CRS'ye (EPSG:32635), aynı ölçekte (1000 m) ve aynı ızgara başlangıç noktasıyla yeniden projeksiyonlayarak sağlar. GEE ızgara hizalamasını tutarlı biçimde ele aldığı için tüm bantlar piksel piksel üst üste yığılır.

**Ölçek.** GEE'de "ölçek", bir pikselin metre cinsinden nominal yer boyutu anlamına gelir. `src/config.py` içinde `SCALE = 1000`. Parmaklarınızın ucunda bulundurmaya değer bazı sonuçlar:

| Nicelik | Değer |
|---|---|
| Bir piksel | 1 km × 1 km = 1 km² = 100 hektar |
| 65 × 65'lik bir yama | 65 km × 65 km = 4,225 km² |
| 64 × 64'lük bir yama (kırpmadan sonra) | 64 km × 64 km = 4,096 km² |
| 32 × 32'lik bir yama (yeni işlem hattı) | 32 km × 32 km = 1,024 km² |
| Bugün yama başına ortalama yanan piksel sayısı | **12.3**, yani ~1,230 hektar |

Bu son satır üzerinde durmaya değer. Bir yama 4,096 km²'dir ve içindeki yangın ortalama 12.3 km² kaplar. Sinyal, görüntünün **%0.30**'udur. Bölüm 9.5, bunun modelin başarısız olma nedenlerinden biri olduğunu açıklar.

### 3.7 Google Earth Engine

**Google Earth Engine** (GEE, Gorelick vd. 2017), çok petabaytlık bir Yer Gözlem verisi kataloğunu paralel bir işleme motoruyla birlikte barındıran ve JavaScript ya da Python API'si üzerinden erişilebilen bir bulut platformudur. Altı lise öğrencisinin bir veri merkezi olmadan ülke ölçeğinde çok sensörlü bir veri kümesi kurabilmesinin nedeni budur.

Açıklayabilmeniz gereken üç GEE kavramı vardır.

**Tembel değerlendirme (lazy evaluation).** Şunu yazdığınızda

```python
img = ee.ImageCollection('MODIS/061/MOD14A1').filterDate(d0, d1).max()
```

**hiçbir hesaplama gerçekleşmez.** GEE hiçbir şey indirmez. Bir *hesaplama tarifi* — işlemlerden oluşan yönlü çevrimsiz bir çizge — kurar ve onu tutar. Somut bir sonuç isteyene kadar hiçbir şey hesaplanmaz. GEE'nin petabaytlarca veri sunabilmesinin nedeni budur: yalnızca gerçekten istediğiniz pikselleri hesaplar.

**Sunucu tarafı nesnelere karşı istemci tarafı nesneler.** Tür adı `ee.` ile başlayan her şey — `ee.Image`, `ee.Number`, `ee.List`, `ee.Feature` — bir **sunucu tarafı nesne**dir (server-side object): Google'ın sunucularında yaşayan bir hesaplamaya işaret eden bir tutamaç. Sıradan bir Python `int`, `list` ya da `float` nesnesi ise **istemci tarafı nesne**dir (client-side object): defterinizde yaşar. İkisi serbestçe karışmaz.

```python
n = ee.Number(5)          # sunucu tarafı tutamaç
if n > 3:                 # YANLIŞ — Python sunucu tarafı bir nesneyi değerlendiremez
    ...
n.getInfo() > 3           # doğru, ama aşağıya bakın
```

Klasik yeni başlayan hatası, sunucu tarafı veri üzerinde bir Python `for` döngüsü yazmaktır. Her yineleme Google'a ayrı bir gidiş-dönüş tetikler ve defterin çalışması saatler alır. Doğru yaklaşım, işlemi bir kez tanımlayan ve GEE'nin onu binlerce makineye paralelleştirmesine izin veren `.map()` kullanmaktır.

**`getInfo()` neden yavaştır.** `getInfo()` şunu söyler: *tembelliği bırak, bu hesaplama çizgesinin tamamını şimdi çalıştır ve bana yanıtı bir Python nesnesi olarak gönder.* Bu, eşzamanlı bir HTTP isteği, Google'ın kümesinde gerçek bir hesaplama ve bir bekleme demektir. Bir döngü içindeki her `getInfo()` maliyeti katlar. Üstelik `getInfo()` katı bir yük sınırına sahiptir (10 MB / 5000 eleman mertebesinde), dolayısıyla gerçek bir veri kümesini almak için kullanılamaz.

**Dışa aktarma görevleri.** Daha büyük her şey için GEE **eşzamansız dışa aktarmalar** kullanır. `Export.table.toDrive(...)` bir kuyruğa iş gönderir; iş Google'ın altyapısında — muhtemelen saatlerce — çalışır ve sonucu Google Drive'a ya da Cloud Storage'a yazar. Bu sırada defteriniz kapalı olabilir. IGNIS veri kümesini böyle üretir: her biri sıkıştırılmış bir TFRecord parçası yazan 360 günlük iş.

```
   SİZİN DEFTERİNİZ                  GOOGLE'IN SUNUCULARI
   ────────────────                  ────────────────────
   ee.Image çizgesini kur ──────►    (henüz hiçbir şey çalışmaz)

   Export.table.toDrive   ──────►    iş kuyruğa alındı
        │                                │
        │  defter kapatılabilir          │ dakikalar–saatler sürer
        ▼                                ▼
   görev durumunu denetle ◄──────    *.tfrecord.gz Drive'a yazıldı
```

---

## 4. Sıfırdan makine öğrenmesi

### 4.1 Makine öğrenmesi nedir

Klasik programlama: bir insan kuralları yazar, bilgisayar bunları veriye uygular ve cevaplar üretir.

**Makine öğrenmesi** (machine learning) bunu tersine çevirir: insan veriyi *ve* cevapları sağlar, bilgisayar da kuralları bulur.

```
  KLASİK PROGRAMLAMA                MAKİNE ÖĞRENMESİ
  ──────────────────                ────────────────
   kurallar ──┐                      veri     ──┐
              ├──► bilgisayar ──► cevaplar      ├──► bilgisayar ──► KURALLAR (model)
   veri     ──┘                      cevaplar ──┘
```

Yangın yayılımı için makine öğrenmesini kullanıyoruz çünkü kuralı kimse yazamıyor. Rothermel'inki gibi fiziksel bir model vardır, ama bu model yakıt-modeli parametrelerine, yerel rüzgâr alanlarına ve yakıt nemi ölçümlerine ihtiyaç duyar; bunlara ise tüm bir ülke genelinde 1 km çözünürlükte sahip değiliz. Makine öğrenmesi farklı bir yol sunar: algoritmaya "koşullar bunlardı ve bir sonraki adımda şu oldu" biçiminde çok sayıda örnek verin ve örüntüyü bulmasına izin verin.

### 4.2 Gözetimli öğrenme: öznitelik, etiket, örnek

**Gözetimli öğrenme** (supervised learning), her eğitim örneğinin doğru cevabı ekli olarak geldiği makine öğrenmesidir.

| Terim | İngilizce | Tanım | IGNIS'te |
|---|---|---|---|
| **Öznitelik** | Feature | Bir girdi değişkeni | 14 (yakında 21) kanaldan biri, ör. `wind_speed` |
| **Etiket / hedef** | Label / target | Tahmin etmek istediğimiz doğru cevap | `fire_next`: bu piksel $t+1$ gününde yanıyor muydu? |
| **Örnek** | Sample | Bir (öznitelikler, etiket) çifti | Bir 64×64 yama ve onun ertesi gün maskesi |
| **Model** | Model | Öznitelikleri tahmine eşleyen fonksiyon | U-Net |
| **Parametreler** | Parameters | Modelin içindeki, öğrenilen sayılar | ~1.9 milyon ağırlık |
| **Hiperparametreler** | Hyperparameters | Öğrenilmeyen, bizim seçtiğimiz ayarlar | Öğrenme oranı 1e-3, yığın boyutu 32, $\gamma = 2.0$ |

IGNIS'in tam eğitim arşivi, 2019, 2020 ve 2021 yangın mevsimlerinden **360 yangın gününe** dayanan, her biri bir yangın merkezli yamaya karşılık gelen **22,426 örnektir**.

Kurulumumuzdaki alışılmadık bir noktaya dikkat edin: etiket tek bir sayı yerine *bütün bir maske* olduğu için, her örnek 4,096 ayrı piksel düzeyinde etiket taşır. Dolayısıyla arşiv kabaca **92 milyon etiketli piksel** içerir — bu, yanıltıcı olabilecek kadar büyük bir sayıdır, çünkü o pikseller birbirinden bağımsız değildir. Aynı yamadan gelen 4,096 piksel aynı havayı, aynı araziyi ve aynı yangını paylaşır.

### 4.3 Eğitim, doğrulama ve test — ve neden üç tane

Üç ayrı veri kümesine ihtiyacınız var ve bunun nedeni açıkça anlatılmayı hak edecek kadar incedir.

| Küme | İngilizce | Amacı | Kim/ne dokunur |
|---|---|---|---|
| **Eğitim kümesi** | Training set | Model, buna uyacak biçimde parametrelerini ayarlar | Gradyan inişi |
| **Doğrulama kümesi** | Validation set | Hiperparametreleri, devir sayısını, eşiği bunun üzerinde seçeriz | *İnsan*, tekrar tekrar |
| **Test kümesi** | Test set | Performansın nihai, tek seferlik, dürüst kestirimi | Hiçbir şey — en sona kadar |

Neden yalnızca iki tane değil? Çünkü **doğrulama kümesine her baktığınızda ve bir şeyi değiştirdiğinizde, ondan modelinize biraz bilgi sızdırırsınız.** 30 farklı öğrenme oranı deneyip en iyi doğrulama skorunu vereni seçerseniz, o en iyi skor iyimser yönde yanlıdır — doğrulama kümesine fiilen 30 denemeyle uydurma yapmışsınızdır. Test kümesi, hiçbir kararın dayandırılmadığı bir sayı vermek için vardır.

**IGNIS'te bu yaşayan bir sorundur.** Ön çalıştırma, yangın gününe göre %80 / %20 bölme yapmış ve doğrulama kümesini hem erken durdurma hem de raporlama için kullanmıştır. Gerçek bir test kümesi yoktu. Yeni işlem hattı bunu **yıl temelli bir bölme** ile düzeltir:

| Bölme | Yıllar | Amaç |
|---|---|---|
| **Eğitim** | 2019 – 2023 | Parametreleri uydurmak |
| **Doğrulama** | 2024 | Erken durdurma, eşik kalibrasyonu $\tau$, normalleştirme denetimleri |
| **Test** | 2025 – 2026 | Yalnızca raporlanan nihai sayılar |

Neden rastgele değil de *yıla* göre bölmek? Bir sonraki bölümde ele alınan sızıntı yüzünden.

### 4.4 Veri sızıntısı

**Veri sızıntısı** (data leakage), değerlendirme verisinden gelen bilginin modeli etkilemesi ve raporlanan skoru gerçekte olduğundan daha iyi göstermesidir.

Uzaktan algılama veri kümelerinde sızıntının baskın biçimi **uzamsal ve zamansal öz-ilinti**dir. Naif yaklaşımı düşünün: 22,426 yamanın tümünü rastgele karıştırın ve %20'sini doğrulama için ayırın. Tek bir yangın gününde, aynı yangından 150'ye kadar yama çekilir. Bunlar aynı meteorolojik alanları (zaten ~9 km'lik ERA5 blokları üzerinde sabittir), aynı araziyi ve büyük ölçüde örtüşen 64 km'lik izdüşümleri paylaşır. Böyle iki yama neredeyse birbirinin kopyasıdır. Birini eğitime, diğerini doğrulamaya koyun; model genelleştirerek değil, ezberleyerek iyi skor alabilir.

**IGNIS bundan, bölmeyi bütün yangın günleri düzeyinde yaparak kaçınır** — bir günden gelen tüm yamalar bölmenin aynı tarafına gider. Makale bu konuda açıktır: *"Gün düzeyinde bölümleme, bu uzamsal ve zamansal sızıntı kaynağını ortadan kaldırır."* Yeni yıl temelli bölme daha da katıdır: bütün bir yangın mevsimi dışarıda tutulur, dolayısıyla aynı yangın olayı bile bölmenin iki tarafında birden görünemez.

IGNIS'in mevcut kodunun işlediği ve yeni işlem hattının düzelttiği, ikinci ve daha sessiz bir sızıntı biçimi daha vardır: **veri kümesinin tamamı üzerinden hesaplanan normalleştirme istatistikleri.** `elevation` kanalının ortalamasını ve standart sapmasını test yıllarını da içeren tüm veriyi kullanarak hesaplarsanız, modelin girdilerini biçimlendirmek için test verisini kullanmış olursunuz. Doğru yordam — yeni işlem hattında uygulanan — $\mu$ ve $\sigma$ değerlerini **yalnızca eğitim bölmesinden** hesaplamak ve aynı sabit sayıları doğrulama ile teste uygulamaktır.

### 4.5 Aşırı öğrenme ve yetersiz öğrenme

| | **Yetersiz öğrenme** | **Aşırı öğrenme / ezberleme** |
|---|---|---|
| İngilizce | Underfitting | Overfitting |
| Belirti | Hem eğitimde *hem de* doğrulamada kötü | Eğitimde iyi, doğrulamada kötü |
| Neden | Model fazla basit, çok az eğitilmiş | Model fazla karmaşık, veri fazla küçük, çok uzun eğitilmiş |
| Benzetme | Hiç çalışmamış bir öğrenci | Geçen yılın sınav cevaplarını anlamadan ezberlemiş bir öğrenci |

```
  hata
    │
    │ ╲                              ╱ doğrulama
    │  ╲                          ╱
    │   ╲___                  ╱
    │       ╲──────────────╱  ← en iyi nokta: erken durdurma
    │        ╲___
    │            ╲_________________  eğitim
    └──────────────────────────────────► devirler
      yetersiz öğr. │ iyi │  aşırı öğrenme
```

**IGNIS aşırı öğreniyor ve bunu sayılarla kanıtlayabiliriz.** Makalenin 4.2 bölümünden:

| Nicelik | Eğitim | Doğrulama |
|---|---|---|
| AUC-PR (son devir) | 0.2375 | 0.0353 |
| AUC-PR (en iyi devir, #7) | — | **0.0368** |
| Kesinlik | 0.325 | 0.09'un altında |

Eğitim AUC-PR değeri, doğrulama değerinin **6.7 katıdır**. Bu uçurum, aşırı öğrenmenin tanımıdır. Erken durdurma çalıştırmayı 25. devirde durdurdu ve 7. devrin ağırlıklarını geri yükledi.

Ancak makalenin belirttiği ve sizin de açıklamaya hazır olmanız gereken önemli inceliğe dikkat edin: *"eğitim ve doğrulama eğrilerinin ayrışması, modelin genelleştirilebilir yayılım davranışını öğrenmeden önce tek tek yangın günlerini ezberlemeye başladığını gösterir."* Tepe noktasının **7. devirde** olması ürkütücü derecede erkendir. Bu, modelin neredeyse anında ezberlemesi kolay bir şey bulduğunu düşündürür. Bölüm 9, normalleştirilmemiş girdilerin muhtemel suçlu olduğunu savunur.

### 4.6 Kayıp fonksiyonu

Bir **kayıp fonksiyonu** (loss function), sembolü $L$, bir tahminin ne kadar yanlış olduğunu ölçen tek bir sayıdır. Eğitmek, onu küçük yapan parametreleri bulmak demektir.

Piksel başına ikili bir tahmin için standart kayıp **ikili çapraz entropi**dir (binary cross-entropy, BCE):

$$L_{\text{BCE}} = -\big[y \log p + (1-y)\log(1-p)\big]$$

burada $y \in \{0,1\}$ gerçek değer ve $p \in (0,1)$ tahmin edilen olasılıktır. Davranışını denetleyin:

| Gerçek $y$ | Tahmin $p$ | Kayıp | Yorum |
|---|---|---|---|
| 1 | 0.99 | $-\log 0.99 = 0.010$ | Emin ve doğru → küçücük kayıp |
| 1 | 0.50 | $-\log 0.50 = 0.693$ | Kararsız → orta düzeyde kayıp |
| 1 | 0.01 | $-\log 0.01 = 4.605$ | Emin ve yanlış → devasa kayıp |
| 0 | 0.01 | $-\log 0.99 = 0.010$ | Emin ve doğru → küçücük kayıp |

Logaritma, emin olunarak yapılan hataları çok ağır biçimde cezalandırır ki bu da istenen davranıştır.

**Kayıp, ölçütle aynı şey değildir.** Kayıp, gradyan inişinin eniyilediği şeydir; türevlenebilir olmak zorundadır. Ölçüt (F1, IoU, AUC-PR) ise insanların önemsediği şeydir; türevlenebilir olması gerekmez. Bölüm 6.10, IGNIS'in gerçekten önemsediğimiz ölçüte daha yakın kayıpları nasıl kurduğunu açıklar.

### 4.7 Gradyan inişi

Sisli bir dağ yamacında durduğunuzu ve vadi tabanına ulaşmaya çalıştığınızı hayal edin. Vadiyi göremiyorsunuz. Ama ayaklarınızın altındaki eğimi hissedebiliyorsunuz. Bu yüzden en dik iniş yönünde bir adım atıyor, sonra yeniden hissediyor ve bunu tekrarlıyorsunuz.

İşte bu **gradyan inişidir** (gradient descent). Kayıp yüksekliktir; parametreler konumunuzdur; gradyan ise eğimdir.

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L$$

- $\theta$ — parametreler (~1.9 milyonun tamamı)
- $L$ — kayıp
- $\nabla_\theta L$ — **gradyan**: kısmi türevler $\partial L / \partial \theta_i$ vektörü; en dik *artış* yönünü gösterir
- $\eta$ — **öğrenme oranı** (learning rate), adım boyu
- Eksi işareti bizi *aşağı* götürür

**Öğrenme oranı, ayarlayacağınız en önemli hiperparametredir.**

| Öğrenme oranı | Etkisi | Benzetme |
|---|---|---|
| Çok küçük (ör. $10^{-7}$) | Eğitim aşırı yavaş; sığ bir çukurda takılabilir | Dağdan bebek adımlarıyla inmek |
| İyi (IGNIS: $10^{-3}$) | İstikrarlı, güvenilir iniş | Kendinden emin yürüyüş |
| Çok büyük (ör. $10$) | Kayıp salınır ya da patlayarak NaN olur | Vadiyi atlayıp karşı yamaca tırmanmak |

IGNIS $\eta = 10^{-3}$ kullanır ve buna bir **öğrenme oranı programı** eşlik eder: doğrulama kaybı 7 devir boyunca iyileşmediğinde $\eta$ değeri 0.5 ile çarpılır. Bu, "tabana yaklaştıkça daha küçük adımlar at" demektir.

### 4.8 Geri yayılım

Gradyan inişi, 1.9 milyon parametrenin her biri için $\partial L / \partial \theta_i$ değerine ihtiyaç duyar. Her birini sayısal olarak hesaplamak adım başına 1.9 milyon ileri geçiş gerektirirdi — imkânsız.

**Geri yayılım** (backpropagation), bunların tümünü *tek bir* geri geçişte hesaplar. Sistematik biçimde uygulanan zincir kuralından ibarettir.

Kavramsal olarak: bir yapay sinir ağı fonksiyonların bileşkesidir, $L = f_n(f_{n-1}(\dots f_1(X)))$. Zincir kuralı der ki

$$\frac{\partial L}{\partial \theta_1} = \frac{\partial L}{\partial f_n}\cdot\frac{\partial f_n}{\partial f_{n-1}}\cdots\frac{\partial f_2}{\partial f_1}\cdot\frac{\partial f_1}{\partial \theta_1}$$

Her katmanın yalnızca (a) üstteki katmandan gelen gradyanı ve (b) kendi yerel türevini bilmesi gerekir. Bunları çarpar, kendi parametreleri için gerekeni tutar ve gerisini aşağıya geçirir.

```
  İLERİ  →  X ──[katman 1]──[katman 2]──[katman 3]──► P ──► L
  GERİ   ←      ∂L/∂θ₁   ←   ∂L/∂θ₂   ←   ∂L/∂θ₃  ←──────  ∂L/∂P
```

Modern çerçeveler (TensorFlow, PyTorch) bunu otomatik olarak yapar — buna **otomatik türev alma** (automatic differentiation) denir. Siz ileri hesaplamayı tanımlarsınız; çerçeve geri hesaplamayı kurar. Elle hiçbir zaman bir geri yayılım yordamı yazmayacaksınız, ama ne yaptığını bilmeli ve sorulduğunda "zincir kuralı" sözcüklerini söyleyebilmelisiniz.

### 4.9 Devir, yığın, yineleme

Sürekli birbirine karıştırılan üç sözcük. Bunları tam olarak öğrenin, çünkü yanlış kullanırsanız bir jüri fark eder.

| Terim | İngilizce | Tanım |
|---|---|---|
| **Yığın / küme** | Batch | Tek bir ileri+geri geçişte birlikte işlenen bir örnek grubu |
| **Yineleme / adım** | Iteration / step | Bir yığın kullanılarak yapılan bir parametre güncellemesi |
| **Devir** | Epoch | Tüm eğitim kümesi üzerinde bir tam geçiş |

IGNIS sayılarıyla işlenmiş örnek. %80/%20 bölmeli 22,426 yama olduğunu varsayalım → yaklaşık 17,940 eğitim yaması, yığın boyutu 32:

$$\text{devir başına yineleme} = \left\lceil \frac{17{,}940}{32} \right\rceil = 561$$

Yani bir devir = 561 parametre güncellemesi. Eğitim en fazla 120 devir için yapılandırılmıştı, yani 67,320 güncellemeye kadar; ancak erken durdurma onu 25. devirde bitirdi.

**Neden yığınlar kullanılıyor?** İki nedenle:
1. **Bellek.** 17,940 yamanın tamamı aynı anda GPU belleğine sığmazdı (bkz. Bölüm 8.4).
2. **Gürültü yardımcı olur.** Küçük rastgele bir alt küme üzerinde güncelleme yapmak, her gradyanı gerçek gradyanın gürültülü bir kestirimi hâline getirir. Bu gürültü, eniyileyicinin sığ yerel minimumlardan kaçmasına yardım eder. Yöntemin **stokastik** gradyan inişi olarak adlandırılmasının nedeni budur.

### 4.10 Eniyileyiciler

| Eniyileyici | İngilizce | Fikir | Not |
|---|---|---|---|
| **SGD** | Stochastic gradient descent | Bölüm 4.7'deki yalın güncelleme kuralı | Basit, dikkatli ayar gerektirir |
| **Momentumlu SGD** | SGD + momentum | Bir hız biriktirerek düz bölgelerde yuvarlanmaya devam etmek | Kütlesi olan bir top gibi |
| **Adam** | — | Gradyanın birinci ve ikinci momentlerinin yürüyen kestirimlerinden parametre başına uyarlanan öğrenme oranları | **IGNIS bunu kullanır** |
| **AdamW** | — | *Ayrıştırılmış* ağırlık azaltmalı Adam; uyarlanabilir yöntemlerle L2 düzenlileştirmesini uygulamanın matematiksel olarak doğru yolu | Yeni işlem hattı için önerilir |

**Adam** (Kingma ve Ba, 2015 — makalede atıf yapılan kaynak), her parametre için şunları tutar:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(birinci moment: gradyanların ortalaması)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(ikinci moment: gradyan karelerinin ortalaması)}$$

ve şöyle günceller

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

burada $\hat m, \hat v$ yanlılık düzeltmesi yapılmış değerlerdir. Sezgi şu: gradyanı sürekli büyük olan bir parametre *daha küçük* bir etkin adım alır (çünkü $\sqrt{v}$ büyüktür); gradyanı küçük ve tutarlı olan bir parametre ise *daha büyük* bir adım alır. Her parametre fiilen kendi öğrenme oranına kavuşur. Tipik varsayılanlar $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ şeklindedir.

### 4.11 Düzenlileştirme

**Düzenlileştirme** (regularisation), aşırı öğrenmeyi azaltmak için yaptığınız her şeydir.

**Seyreltme (dropout).** Eğitim sırasında, her adımda aktivasyonların $p$ oranındaki bir kısmını rastgele sıfırlayın. IGNIS $p = 0.2$ kullanır. Benzetme: her antrenmanda rastgele iki oyuncusu eksik çalışan bir futbol takımı — her oyuncu boşluğu kapatmayı öğrenmek zorundadır, böylece takım tek bir yıldıza bağımlı olmaktan çıkar. Çıkarım anında seyreltme kapatılır ve aktivasyonlar bunu telafi edecek şekilde ölçeklenir.

**Yığın normalizasyonu (batch normalisation).** Her kanal için, aktivasyonları yığın boyunca sıfır ortalama ve birim varyansa normalleştirin, ardından öğrenilmiş bir ölçek $\gamma$ ve kaydırma $\beta$ uygulayın:

$$\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}, \qquad y = \gamma \hat{x} + \beta$$

Bu, aktivasyonları ağ boyunca uslu bir aralıkta tutar, daha büyük öğrenme oranlarına izin verir ve küçük bir düzenlileştirici gürültü ekler. **IGNIS her evrişimden sonra yığın normalizasyonu uygular.**

Bölüm 9 için önemli bir gözlem: yığın normalizasyonu ağın *içini* normalleştirir. *Girdileri* normalleştirmez. Standart sapması 515.44 olan ham `elevation` kanalımız, herhangi bir BatchNorm katmanı var olmadan önce en baştaki evrişime çarpar. Hasarın verildiği yer tam olarak o ilk katmandır.

**Ağırlık azaltma (weight decay).** Kayba $\lambda \sum \theta_i^2$ biçiminde bir ceza ekleyin; bu, veri güçlü biçimde aksini gerektirmedikçe ağırlıkları sıfıra doğru iter. L2 düzenlileştirmesine eşdeğerdir.

**Erken durdurma (early stopping).** Bir doğrulama ölçütünü izleyin; `patience` kadar devir boyunca iyileşmediğinde durun ve en iyi ağırlıkları geri yükleyin. IGNIS, doğrulama AUC-PR değerini 18 sabırla izler. Ön çalıştırmada 25. devirde durdu ve 7. devri geri yükledi.

**Veri artırma (data augmentation).** Etiketi koruyan dönüşümler uygulayarak yeni eğitim örnekleri üretin. Bölüm 7.8, IGNIS'in ihtiyaç duyduğu yön farkındalıklı veri artırmayı ele alır.

---

## 5. Derin öğrenme ve evrişimli ağlar

### 5.1 Yapay nöron

Temel birim, girdilerinin ağırlıklı toplamını hesaplar, bir yanlılık ekler ve sonucu doğrusal olmayan bir fonksiyondan geçirir.

$$y = \sigma\!\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

- $x_i$ — girdiler
- $w_i$ — ağırlıklar (öğrenilen)
- $b$ — yanlılık (öğrenilen)
- $\sigma$ — aktivasyon fonksiyonu

```
   x₁ ──w₁──┐
   x₂ ──w₂──┤
   x₃ ──w₃──┼──►  Σ  ──► +b ──► σ(·) ──► y
   ...      │
   xₙ ──wₙ──┘
```

Bir **katman**, paralel duran çok sayıda nörondur; bir **derin ağ** ise üst üste yığılmış çok sayıda katmandır. "Derin öğrenme" (deep learning), basitçe çok katmanlı yapay sinir ağları demektir.

### 5.2 Aktivasyon fonksiyonları

**Doğrusal olmayanlık neden gereklidir?** Çünkü o olmadan bütün ağ çöker. 1. katman $W_1 x$ hesaplıyorsa ve 2. katman $W_2(W_1 x)$ hesaplıyorsa, sonuç $(W_2 W_1)x = W'x$ olur — tek bir doğrusal dönüşüm. Üst üste yığılmış yüz doğrusal katman, tam olarak tek bir katman kadar ifade gücüne sahiptir. Bütün derinlik boşa gider.

| Fonksiyon | Formül | Aralık | IGNIS'te nerede kullanılır |
|---|---|---|---|
| **ReLU** | $\max(0, x)$ | $[0, \infty)$ | Kodlayıcı, darboğaz ve kod çözücüdeki her evrişimden sonra |
| **Sigmoit** | $1/(1+e^{-x})$ | $(0, 1)$ | Bir skoru olasılığa çeviren son çıktı katmanı |
| Tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ | $(-1, 1)$ | Burada kullanılmıyor |
| Sızıntılı ReLU | $\max(0.01x, x)$ | $\mathbb{R}$ | Alternatif; "ölü" nöronları önler |

**ReLU** (düzeltilmiş doğrusal birim, rectified linear unit) yaygındır; çünkü hesaplaması son derece ucuzdur ve pozitif girdiler için türevi tam olarak 1'dir; bu da derin sigmoit ağlarını felç eden "kaybolan gradyan" sorununu önler. Zayıflığı "ölmekte olan ReLU"dur: girdisi her zaman negatif olan bir nöron sonsuza dek 0 üretir ve sıfır gradyan alır, dolayısıyla asla toparlanamaz.

**Sigmoit**, herhangi bir gerçel sayıyı $(0,1)$ aralığına eşler; çıktıda ihtiyacımız olan da budur — bir olasılık.

| $z$ | $\sigma(z)$ |
|---|---|
| −5 | 0.0067 |
| −2 | 0.1192 |
| 0 | 0.5000 |
| 2 | 0.8808 |
| 5 | 0.9933 |

$\sigma(0) = 0.5$ olduğuna tam olarak dikkat edin. Varsayılan eşik $\tau = 0.5$ değerinin "ham skor $z$ pozitiftir" ifadesine karşılık gelmesinin nedeni budur. Bölüm 6.8, dengesiz bir problem için 0.5'in neden berbat bir eşik olduğunu açıklar.

### 5.3 Tam bağlı katmanlar görüntülerde neden başarısız olur

Evrişimleri göz ardı edip girdi yamamız üzerinde doğrudan bir **tam bağlı** (yoğun) katman kullandığımızı varsayalım. Girdi $64 \times 64 \times 14 = 57{,}344$ sayıdır. Bunu 1,000 nöronluk mütevazı bir gizli katmana bağlayın:

$$57{,}344 \times 1{,}000 + 1{,}000 = 57{,}345{,}000 \text{ parametre}$$

**Tek bir katmanda 57 milyon parametre** — U-Net'in tamamının otuz katı — ve ağı kurmaya daha başlamadık bile. Bu, **parametre patlaması** sorunudur.

İkinci sorun daha kötüdür. Tam bağlı bir katmanın *her girdi konumu için ayrı bir ağırlığı* vardır. "(17, 42) pikselindeki değer önemlidir" diye öğrenir — "kuru yakıtın yanındaki bir yangın önemlidir" diye değil. Yangını iki piksel sağa kaydırırsanız, artık her ağırlık yanlıştır. Katmanın, ötelenmiş örüntülerin aynı örüntü olduğuna dair bir kavrayışı yoktur. Bu, **konum duyarlılığı** — yani *öteleme eşdeğişkenliğinin* yokluğu — demektir.

Evrişim her iki sorunu birden çözer:

| Sorun | Evrişimin çözümü |
|---|---|
| Parametre patlaması | **Ağırlık paylaşımı**: aynı küçük çekirdek her konumda kullanılır |
| Konum duyarlılığı | **Öteleme eşdeğişkenliği**: girdiyi kaydırın, çıktı da aynı biçimde kayar |
| Uzamsal yapıyı yok saymak | **Yerellik**: her çıktı yalnızca küçük bir komşuluğa bağlıdır |

Karşılaştırın: 14 kanaldan 32 kanala giden 3×3'lük bir evrişimin $9 \times 14 \times 32 + 32 = 4{,}064$ parametresi vardır — o yoğun katmandan yaklaşık **14,000 kat daha az** — ve görüntüdeki her konumda çalışır.

### 5.4 Evrişimin elle işlenmiş hâli

Bir **evrişim** (convolution), küçük bir matrisi — **çekirdek** ya da **filtre** (kernel / filter) — girdinin üzerinde kaydırır ve her konumda öge bazlı çarpımların toplamını hesaplar.

$$(I * K)(i,j) = \sum_{m}\sum_{n} I(i+m,\ j+n)\, K(m,n)$$

Bunu gerçek sayılarla yapalım.

#### Örnek girdi: küçük bir yangın maskesi

`fire` kanalından 5×5'lik bir yama alın. 1 = yanıyor, 0 = yanmıyor.

```
        sütun:  0   1   2   3   4
    satır 0 [   0   0   0   0   0 ]
    satır 1 [   0   0   1   0   0 ]
    satır 2 [   0   1   1   1   0 ]
    satır 3 [   0   0   1   0   0 ]
    satır 4 [   0   0   0   0   0 ]
```

5 yanan pikselden oluşan küçük, artı biçiminde bir yangın.

#### Çekirdek A: "yanan komşularımı say"

```
    K_A =  [ 1  1  1 ]
           [ 1  0  1 ]
           [ 1  1  1 ]
```

**(2,2) konumundaki çıktı.** (2,2) üzerinde merkezlenmiş 3×3'lük pencere 1–3 satırlarını, 1–3 sütunlarını kapsar:

```
    pencere = [ 0  1  0 ]
              [ 1  1  1 ]
              [ 0  1  0 ]
```

$K_A$ ile öge öge çarpın ve toplayın:

$$(0{\cdot}1) + (1{\cdot}1) + (0{\cdot}1) + (1{\cdot}1) + (1{\cdot}0) + (1{\cdot}1) + (0{\cdot}1) + (1{\cdot}1) + (0{\cdot}1)$$
$$= 0 + 1 + 0 + 1 + 0 + 1 + 0 + 1 + 0 = \mathbf{4}$$

Merkez pikselin 4 yanan komşusu vardır.

**(1,2) konumundaki çıktı.** Pencere 0–2 satırlarını, 1–3 sütunlarını kapsar:

```
    pencere = [ 0  0  0 ]
              [ 0  1  0 ]
              [ 1  1  1 ]
```

$$0+0+0+0+(1{\cdot}0)+0+1+1+1 = \mathbf{3}$$

Bunu her geçerli konumda yapmak tam çıktıyı verir:

```
    konumlar (1,1)…(3,3):
         [ 2   2   2 ]
         [ 3   4   3 ]
         [ 2   2   2 ]
```

**Yorum:** bu tek 3×3'lük çekirdek, her piksel için çevresinde ne kadar yangın olduğunu hesaplamış oldu. Bu bir yangın yoğunluğu özniteliğidir — ve tam olarak bir pikselin yarın tutuşup tutuşmayacağını belirleyen türden bir niceliktir. **Dokuz sayıdan oluşan tek bir çekirdek, her yere uygulandığında fiziksel olarak anlamlı bir öznitelik çıkarır.** Bir CNN'in bütün fikri budur.

#### Çekirdek B: "yangın hangi tarafta?" — yönlü bir çekirdek

```
    K_B =  [ -1   0   +1 ]
           [ -2   0   +2 ]
           [ -1   0   +1 ]
```

Bu, yatay Sobel işlecidir; bir kenar bulucudur.

**(2,1) konumundaki çıktı** — pencere 1–3 satırlarını, 0–2 sütunlarını kapsar:

```
    pencere = [ 0  0  1 ]
              [ 0  1  1 ]
              [ 0  0  1 ]
```

$$(0{\cdot}{-1}) + (0{\cdot}0) + (1{\cdot}{+1}) + (0{\cdot}{-2}) + (1{\cdot}0) + (1{\cdot}{+2}) + (0{\cdot}{-1}) + (0{\cdot}0) + (1{\cdot}{+1})$$
$$= 1 + 2 + 1 = \mathbf{+4}$$

**(2,3) konumundaki çıktı** — pencere 1–3 satırlarını, 2–4 sütunlarını kapsar:

```
    pencere = [ 1  0  0 ]
              [ 1  1  0 ]
              [ 1  0  0 ]
```

$$(1{\cdot}{-1}) + 0 + 0 + (1{\cdot}{-2}) + 0 + 0 + (1{\cdot}{-1}) + 0 + 0 = -1 -2 -1 = \mathbf{-4}$$

**Yorum:** çıktının *işareti*, yangının hangi tarafta olduğunu söyler. Pozitif değer "yangın doğumda", negatif değer "yangın batımda" anlamına gelir. Böyle yönlü bir çekirdek, `wind_u` kanalıyla birleştiğinde, bir CNN'in "yangın rüzgâr yönünde yayılır" ilkesini öğrenebileceği mekanizmanın ta kendisidir. Bizimkinin bunu gerçekten öğrenip öğrenmediği açık bir sorudur — bir **kanal ablasyon çalışması** (Bölüm 10, S22) bize bunu söylerdi.

#### Can alıcı nokta

$K_A$ ya da $K_B$'yi IGNIS'e kimse yazmadı. **Çekirdek değerleri parametredir.** Rastgele sayılar olarak başlarlar ve gradyan inişi, kaybı en aza indiren öznitelikleri çıkarana kadar onları ayarlar. Yukarıdaki çekirdekler, eğitilmiş bir ağın keşfettiği şeylerin *türüne* dair örneklerdir. Uygulamada, eğitilmiş bir CNN'in ilk katmanı tipik olarak Sobel işleçlerine dikkat çekici biçimde benzeyen kenar ve gradyan bulucuları öğrenir — ağ bunları yeniden keşfeder, çünkü işe yararlar.

### 5.5 Adım, dolgu ve alıcı alan

**Adım (stride)**, çekirdeğin konumlar arasında ne kadar sıçradığıdır. Adım 1 (IGNIS'in tüm evrişimler için tercihi) her konumu değerlendirir. Adım 2 ise bir konumu atlar ve çıktı boyutunu yarıya indirir.

**Dolgu (padding).** Yukarıdaki örnekte, 5×5'lik bir girdi 3×3'lük bir çekirdekle yalnızca 3×3'lük bir çıktı verdi — kenar piksellerinin tam bir komşuluğu yoktur. Genel olarak:

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1$$

| Dolgu modu | Türkçe karşılığı | Etkisi |
|---|---|---|
| `valid` (p = 0) | Dolgusuz | Çıktı her katmanda $k-1$ kadar küçülür |
| `same` (p = (k−1)/2) | Aynı boyut | Çıktı girdiyle aynı boyutta; kenar sıfırlarla doldurulur |

**IGNIS `same` dolgusunu kullanır.** 3×3'lük bir çekirdekle $p=1$: girdinin çevresinde bir halka sıfır. Bu, bir seviye içindeki her öznitelik haritasını 64×64'te tutar; kod çözücünün atlama bağlantılarının temiz biçimde birleştirilebilmesini sağlayan da budur. Bedeli hafif bir kenar yapaylığıdır — en dıştaki pikseller yapay sıfırlar görür.

**Alıcı alan (receptive field)**, tek bir çıktı değerini etkileyen *özgün girdi* bölgesidir. Ağınızın gerçekte ne kadar coğrafi bağlam kullanabildiğini söyleyen kavram budur.

Tek bir 3×3 evrişim: alıcı alan 3×3. İki tanesini üst üste koyun: 5×5. Bir 2×2 havuzlama ekleyin, alıcı alan etkin olarak iki katına çıkar. Bunu gerçek IGNIS kodlayıcısı için hesaplayalım:

| Aşama | İşlem | Alıcı alan | Yerdeki karşılığı |
|---|---|---|---|
| Girdi | — | 1×1 | 1 km |
| Seviye 1 | iki adet 3×3 evrişim | 5×5 | 5 km |
| ↓ havuzlama /2 | | | |
| Seviye 2 | iki adet 3×3 evrişim | 13×13 | 13 km |
| ↓ havuzlama /4 | | | |
| Seviye 3 | iki adet 3×3 evrişim | 29×29 | 29 km |
| ↓ havuzlama /8 | | | |
| Darboğaz | iki adet 3×3 evrişim | **61×61** | **61 km** |

Yani IGNIS darboğazındaki tek bir nöron, **61 km × 61 km**'lik bir bölgeden gelen bilgiyi bütünleştirir — esasen 64 km'lik yamanın tamamını. Mimari açıdan bakıldığında, ağ yamanın küresel hava bağlamını *görebilir*. Bu, hazırda bulundurulacak iyi bir olgudur: 3 derinliğinin gelişigüzel seçilmediğini, yamayı kaplamaya tam olarak yettiğini gösterir.

### 5.6 Havuzlama

**Havuzlama** (pooling), her küçük bloğu özetleyerek uzamsal boyutu küçültür.

**2×2 maksimum havuzlama**, örtüşmeyen her 2×2 bloğun en büyüğünü alır:

```
   girdi 4×4            çıktı 2×2
   [ 1  3 | 2  4 ]      [ 3  4 ]
   [ 2  0 | 1  0 ]  →   [ 6  9 ]
   ─────── ───────
   [ 5  6 | 8  9 ]
   [ 1  2 | 7  3 ]
```

Üç etkisi vardır: (1) her uzamsal boyutu yarıya indirir ve hesaplamayı 4 kat azaltır; (2) kendisinden sonraki her şeyin etkin alıcı alanını iki katına çıkarır; (3) az miktarda öteleme *değişmezliği* getirir — bir piksel kayan bir öznitelik çoğu zaman aynı havuzlanmış çıktıyı üretir.

(3) numaralı etki, bölütleme için iki ucu keskin bir bıçaktır. Yangın sınırının kesin biçimde konumlandırılmasını *istiyoruz* ve havuzlama tam olarak bunu bilinçli biçimde atıyor. İşte atlama bağlantıları bu gerilimi çözmek için vardır (Bölüm 5.9).

**IGNIS'te:** her kodlayıcı bloğundan sonra 2×2 maksimum havuzlama uygulanır, böylece uzamsal boyut $64 \to 32 \to 16 \to 8$ biçiminde ilerler ve darboğaz 8×8'lik bir ızgara üzerinde çalışır.

### 5.7 Kanallar

Bir **kanal** (channel), bir yığındaki tek bir 2B haritadır. Sıradan bir fotoğrafın 3 kanalı vardır: kırmızı, yeşil, mavi. Girdi yamamızın her biri farklı bir fiziksel niceliğe karşılık gelen **14 kanalı** vardır:

```
                    64
              ┌──────────────┐
           ┌──┴───────────┐  │  ndvi
        ┌──┴───────────┐  │  │  lst
     ┌──┴───────────┐  │  │  │  air_temp
   64│              │  │  │  │  ... toplam 14 katman ...
     │  her katman  │  │  │  │  landcover
     │   64 × 64'tür│  │  ├──┘  fire
     └──────────────┘  ├──┘
                       └─────  → tensör şekli (64, 64, 14)
```

Çok kanallı bir girdi üzerinde çalışan bir evrişimin çekirdeğinin de bir kanal boyutu vardır: 14 kanallı bir girdiye uygulanan 3×3'lük bir çekirdek, gerçekte 126 ağırlıktan oluşan $3 \times 3 \times 14$'lük bir bloktur. *Bütün* kanallar boyunca eşzamanlı olarak toplar. Bir CNN'in fiziği nasıl birleştirdiğinin anahtarı budur: tek bir çekirdek, `wind_speed` kanalına pozitif ağırlık, `humidity` kanalına negatif ağırlık ve komşu `fire` piksellerine pozitif ağırlık vererek "yüksek rüzgâr VE düşük nem VE yakınlarda yangın" kuralını gerçekleyebilir.

32 çıktı kanalı elde etmek için, her biri farklı bir birleşim öğrenen 32 böyle çekirdek kullanırsınız. Bu, Bölüm 5.3'te hesaplandığı gibi tam olarak $32 \times (3 \times 3 \times 14) + 32 = 4{,}064$ parametre eder.

TensorFlow/Keras'taki tensör şekli geleneği `(yığın, yükseklik, genişlik, kanallar)` — yani "kanallar sonda"dır. PyTorch ise `(yığın, kanallar, yükseklik, genişlik)` — "kanallar başta" kullanır. **Yeni IGNIS işlem hattı PyTorch'tur, o yüzden unutmayın: (N, C, H, W).** 32×32 boyutunda, 21 kanallı, 32 yamalık bir yığın için bu `(32, 21, 32, 32)` demektir.

### 5.8 Sınıflandırmaya karşı anlamsal bölütleme

| | **Sınıflandırma** | **Anlamsal bölütleme** |
|---|---|---|
| İngilizce | Classification | Semantic segmentation |
| Girdi | Bir görüntü | Bir görüntü |
| Çıktı | Tüm görüntü için tek bir etiket | **Her piksel** için bir etiket |
| Örnek | "Bu yama büyüyen bir yangın içeriyor" | "Bu 47 piksel yarın yanacak" |
| Çıktı şekli | $K$ sayıdan oluşan bir vektör | $H \times W$ harita |

IGNIS **her ikisini de** sırayla yapar ve makale bu iki düzeyli tasarımın neden seçildiği konusunda açıktır:

1. U-Net **anlamsal bölütleme** yapar ve 64×64'lük bir olasılık haritası üretir.
2. Bir son işleme kuralı, tahmin edilen yangın piksellerini sayarak bu haritayı **yama düzeyinde bir sınıfa** — büyüyor, kararlı ya da sönüyor — indirger:

$$r = \frac{N_{t+1}}{\max(N_t,\ 1)}$$

$$r > 1.25 \Rightarrow \text{büyüyor}; \qquad 0.75 \le r \le 1.25 \Rightarrow \text{kararlı}; \qquad r < 0.75 \Rightarrow \text{sönüyor}$$

Paydadaki $\max(N_t, 1)$ ifadesi sıfıra bölmeyi önler.

Peki ikinci adım neden var? Çünkü makalenin dediği gibi, *"Operasyonel karar vericiler bir olasılık alanı yerine kategorik bir ifadeye ihtiyaç duyar."* Uçak tahsis eden bir nöbetçi amir, 4,096 elemanlı bir olasılık dizisi değil, "bu yangın büyüyor" ifadesini ister.

Neden doğrudan bir sınıflandırıcı eğitmek yerine önce bölütleme yapıldı? Makalede verilen iki neden var: *"Genişleme yönü ve yanan alanın büyüklüğü birlikte tahmin edilir ve kayıp toplulaştırılmış bir istatistik yerine her piksel üzerinde değerlendirildiği için yangın cephesinin uzamsal sürekliliği korunur."*

**Eşik değişikliğine dikkat edin.** Daha erken bir sürüm, büyüyor için $r > 1.15$ ve kararlı için alt sınır olarak $0.85$ kullanıyordu (depo README dosyasında hâlâ böyle belgeleniyor). Makale, kararlı bandını $[0.75, 1.25]$ aralığına genişletti; çünkü *"kararlı sınıf o kadar seyrekti ki fiilen öğrenilemez hâldeydi"*. Bir jüri üyesi README ile makaleyi karşılaştırırsa, açıklama budur.

### 5.9 U-Net

**U-Net**, 2015 yılında Ronneberger, Fischer ve Brox tarafından biyomedikal görüntü bölütlemesi için tanıtıldı. Bugün küçük veri kümeleri üzerinde yoğun tahmin için varsayılan mimaridir ve IGNIS'in kullandığı mimaridir.

```
 GİRDİ 64×64×14
      │
 ┌────▼─────────────┐                                 ┌───────────────────┐
 │ E1: conv3×3 ×2   │─────── atlama bağlantısı ──────►│ D1: concat + conv │──► 1×1 conv
 │     32 filtre    │           (64×64×32)            │     32 filtre     │      sigmoit
 │  64×64×32        │                                 │   64×64×32        │        │
 └────┬─────────────┘                                 └───────▲───────────┘        ▼
      │ maxpool 2×2                                           │ upconv 2×2   64×64×1
 ┌────▼─────────────┐                                 ┌───────┴───────────┐   OLASILIK
 │ E2: conv3×3 ×2   │─────── atlama bağlantısı ──────►│ D2: concat + conv │   HARİTASI
 │     64 filtre    │           (32×32×64)            │     64 filtre     │
 │  32×32×64        │                                 │   32×32×64        │
 └────┬─────────────┘                                 └───────▲───────────┘
      │ maxpool 2×2                                           │ upconv 2×2
 ┌────▼─────────────┐                                 ┌───────┴───────────┐
 │ E3: conv3×3 ×2   │─────── atlama bağlantısı ──────►│ D3: concat + conv │
 │    128 filtre    │          (16×16×128)            │    128 filtre     │
 │  16×16×128       │                                 │   16×16×128       │
 └────┬─────────────┘                                 └───────▲───────────┘
      │ maxpool 2×2                                           │ upconv 2×2
      │              ┌─────────────────────────┐              │
      └─────────────►│ DARBOĞAZ                │──────────────┘
                     │ conv3×3 ×2, 256 filtre  │
                     │        8×8×256          │
                     └─────────────────────────┘

           KODLAYICI (daraltan)           KOD ÇÖZÜCÜ (genişleten)
           "Yamada NE var?"                "TAM OLARAK NEREDE?"
```

O şeklin görünümü — solda aşağı, altta yatay, sağda yukarı — ona neden **U**-Net dendiğinin nedenidir.

#### Kodlayıcı neden sıkıştırır

Her kodlayıcı seviyesi uzamsal çözünürlüğü yarıya indirir ve kanal sayısını iki katına çıkarır. Ağ, *uzamsal* bilgiyi *anlamsal* bilgiyle takas eder. 64×64×14'te ağ, ince uzamsal ayrıntıda ham fiziksel ölçümleri tutar. 8×8×256'da ise kaba uzamsal ayrıntıda 256 son derece soyut öznitelik haritası tutar — "batıdan gelen güçlü rüzgâr ve etkin bir cephe ile birlikte dik, kuru, güneye bakan arazi" gibi şeyler.

Bu gereklidir; çünkü geniş bir alıcı alan yalnızca aşağı örnekleme yaparak (ya da çok daha fazla katmanla) elde edilebilir. Bu pikselin yanıp yanmayacağını bilmek için ağın onlarca kilometre ötede ne olduğunu bilmesi gerekir.

#### Darboğaz

**Darboğaz** (bottleneck), en derin ve en soyut temsildir: 8×8 uzamsal konum × 256 kanal = 16,384 sayı; girdideki 57,344 ile karşılaştırın. Ağın sahneye dair küresel anlayışını içerir ve modelin parametrelerinin yarısından fazlasını barındırır (~1.93 milyonun 885,248'i — bkz. Bölüm 5.10).

#### Kod çözücü neden ters evrişim kullanır

Kod çözücünün 8×8'den 64×64'e geri dönmesi gerekir. Basit yukarı örnekleme (her pikseli tekrarlamak) işe yarar, ama parametresiz, sabit bir işlemdir. Bir **ters evrişim** (transposed convolution; ayrıca ters konvolüsyon ya da yukarı evrişim de denir) *öğrenilebilir* bir yukarı örneklemedir: girdi pikselleri arasına sıfırlar yerleştirir ve sonra evriştirir; böylece ağ her özniteliği nasıl genişleteceğini öğrenir.

```
   girdi 2×2         sıfırları araya koy (adım 2)   öğrenilmiş 2×2 ile evriştir
   [ a  b ]          [ a  0  b  0 ]                 → 4×4 öğrenilmiş çıktı
   [ c  d ]     →    [ 0  0  0  0 ]        →
                     [ c  0  d  0 ]
                     [ 0  0  0  0 ]
```

IGNIS'te her kod çözücü seviyesi, adım 2 ile 2×2'lik bir ters evrişim kullanır; bu, uzamsal boyutu iki katına çıkarır ve kanal sayısını yarıya indirir.

Bilinen bir yapaylık: ters evrişimler, çekirdek boyutu adıma bölünemediğinde bir **dama tahtası deseni** üretebilir. 2×2 çekirdek ve adım 2 ile bundan kaçınılır, dolayısıyla IGNIS bu konuda güvendedir.

#### Atlama bağlantıları ne yapar — ve bir yangın cephesi için neden önemlidir

Bu, 5. bölümün en önemli kısmıdır. U-Net hakkında tek bir şey anlayacaksanız, bunu anlayın.

**Sorun.** Kodlayıcı uzamsal kesinliği atar. Üç adet 2×2 maksimum havuzlamadan sonra, her darboğaz pikseli 8×8 = 64 girdi pikselini, yani 8 km × 8 km'lik bir bloğu temsil eder. Kod çözücü 64×64'e geri yukarı örneklediğinde yalnızca *bulanık* yapılar üretebilir — sınırın o 64 pikselin hangisinde olduğunu bilemez. Bilgi basitçe yok olmuştur.

**Çözüm.** Bir **atlama bağlantısı** (skip connection), kodlayıcının öznitelik haritasını havuzlamadan *önce* kopyalar ve karşılık gelen kod çözücü öznitelik haritasına ekler. Böylece kod çözücü iki şey alır:

- aşağıdan: **anlamsal** bilgi ("burada, dik bir yamaçta batı rüzgârının sürüklediği, büyüyen bir yangın cephesi var") — *ne* konusunda isabetli, *nerede* konusunda belirsiz;
- yandan: tam çözünürlükte **uzamsal** bilgi ("sınır pikselinin tam olarak burada olduğu") — *nerede* konusunda isabetli, *ne* konusunda habersiz.

Bunu izleyen evrişim, ikisini birleştirmeyi öğrenir.

```
   Atlama bağlantıları olmadan:       Atlama bağlantıları ile:

     gerçek yangın cephesi              gerçek yangın cephesi
     ████                               ████
     ████                               ████

     tahmin (bulanık)                   tahmin (keskin)
     ▒▒▒▒▒▒                             ████
     ▒▒▒▒▒▒                             ████
     ▒▒▒▒▒▒
```

**Bu neden özellikle IGNIS için hayati.** Bir yangın cephesi *ince* bir yapıdır. Yangınlarımız 4,096 pikselin ortalama 12.3'ünü kaplar — yamanın %0.30'unu. İlgilendiğimiz niceliğin tamamı bir avuç piksel genişliğindedir. Sınırları 8 piksel kadar bulanıklaştıran bir mimari sinyalin tamamını yok ederdi. Makale bunu tam olarak şöyle ifade eder: *"Atlama bağlantıları, art arda yapılan aşağı örneklemenin aksi hâlde atacağı ince uzamsal ayrıntıyı geri kazandırır; bu da yalnızca birkaç piksel genişliğinde olabilen bir yangın sınırını belirlemek için elzemdir."*

İkinci ve daha teknik bir yarar: atlama bağlantıları, gradyanların kayıptan erken kodlayıcı katmanlarına akması için kısa bir yol sağlar; bu da derin ağların eğitilmesini kolaylaştırır — ResNet'teki artık bağlantılarla aynı ilke.

#### Biçimsel denklemler

Makale mimariyi şöyle ifade eder:

$$f_l = \text{Pool}\big(\sigma(\text{Conv}(f_{l-1}))\big), \qquad l = 1 \dots L$$

$$g_{l-1} = \sigma\big(\text{Conv}([\,\text{Up}(g_l),\ f_{l-1}\,])\big)$$

burada $f_l$, $l$ seviyesindeki kodlayıcı öznitelik haritası; $g_l$ kod çözücü öznitelik haritası; $[\cdot,\cdot]$ kanal bazlı birleştirme; $\text{Up}(\cdot)$ ters evrişim; $\sigma$ ReLU ve $f_0 = X$'tir. Çıktı ise

$$P(i,j) = \frac{1}{1+\exp(-z(i,j))} \in (0,1)$$

### 5.10 Parametreleri saymak

Parametreleri sayabilmelisiniz; çünkü bir jüri "modeliniz ne kadar büyük?" ve "neden bu kadar küçük?" diye sorabilir.

**Kurallar.**

| Katman | Parametre sayısı |
|---|---|
| $k \times k$ evrişim, $C_{\text{in}} \to C_{\text{out}}$ | $k^2 \cdot C_{\text{in}} \cdot C_{\text{out}} + C_{\text{out}}$ |
| $k \times k$ ters evrişim, $C_{\text{in}} \to C_{\text{out}}$ | $k^2 \cdot C_{\text{in}} \cdot C_{\text{out}} + C_{\text{out}}$ |
| $C$ kanal üzerinde BatchNorm | $2C$ eğitilebilir ($\gamma, \beta$) + $2C$ eğitilemez (yürüyen $\mu, \sigma$) |
| Maksimum havuzlama, ReLU, seyreltme | **0** |

**İşlenmiş örnek — en baştaki evrişim.** 3×3, 14 girdi kanalından 32 çıktı kanalına:

$$9 \times 14 \times 32 + 32 = 4{,}032 + 32 = 4{,}064$$

**IGNIS U-Net'inin tamamı:**

| Blok | İşlem | Parametreler |
|---|---|---|
| E1 | conv 14→32; conv 32→32 | 4,064 + 9,248 = **13,312** |
| E2 | conv 32→64; conv 64→64 | 18,496 + 36,928 = **55,424** |
| E3 | conv 64→128; conv 128→128 | 73,856 + 147,584 = **221,440** |
| Darboğaz | conv 128→256; conv 256→256 | 295,168 + 590,080 = **885,248** |
| D3 | upconv 256→128; conv 256→128; conv 128→128 | 131,200 + 295,040 + 147,584 = **573,824** |
| D2 | upconv 128→64; conv 128→64; conv 64→64 | 32,832 + 73,792 + 36,928 = **143,552** |
| D1 | upconv 64→32; conv 64→32; conv 32→32 | 8,224 + 18,464 + 9,248 = **35,936** |
| Çıktı | 1×1 conv 32→1 | 32 + 1 = **33** |
| BatchNorm | 14 katman, toplam 1,408 kanal | **2,816** eğitilebilir |
| | **TOPLAM** | **≈ 1,931,585** |

Bu, makalede bildirilen **≈1.9 milyon** rakamını yeniden üretir ve tablo, yüksek sesle söylenmeye değer bir şeyi gösterir: **parametrelerin %46'sı tek başına darboğazdadır.** İki adet 256 filtreli evrişim 885,248 parametreye mal olur. Modelin kapasitesinin yaşadığı yer orasıdır.

Kod çözücüdeki birleştirmeye de dikkat edin: `upconv 256→128` işlemi 128 kanal üretir; bunlar E3'ten gelen 128 atlama kanalıyla birleştirilir ve bir sonraki evrişime 256 girdi kanalı verir — dolayısıyla `conv 256→128`.

**Yeni 21 kanallı girdiyle**, yalnızca ilk evrişim değişir:

$$9 \times 21 \times 32 + 32 = 6{,}048 + 32 = 6{,}080$$

yani 2,016 parametrelik bir artış — modelin yaklaşık %0.1'i. Ders önemlidir ve bir jüriye söylenmeye değer: **iyi tasarlanmış girdi kanalları eklemek parametre açısından neredeyse bedavadır; buna karşılık darboğaza filtre eklemek çok pahalıdır.** Girdi temsiline harcanan emeğin maliyet/fayda oranı, ağı büyütmeye harcanan emeğinkinden çok daha iyidir.
---
## 6. Sınıf dengesizliği ve metrikler

**Bu bölümü iki kez okuyun.** IGNIS'in bilimsel kalbi burasıdır. IAC'de size sorulacak zor soruların neredeyse tamamı buradan gelir ve orman yangını makine öğrenmesi literatüründe yapılan hataların neredeyse tamamı bu bölümde yapılan hatalardır.

Durumu ifade etmek kolaydır. Arşivimizde piksellerin **%0.2686**'sı pozitiftir (yarın yanacak). Bu, tipik bir yamada (patch) kabaca **4.096 pikselden 11'i** demektir. Bundan sonra gelen her şey, bu tek sayının bir sonucudur.

### 6.1 Karışıklık matrisi

İkili (binary) bir problemde her tahmin dört kutudan birine düşer.

```
                          TAHMİN EDİLEN
                    Yangın        Yangın yok
                ┌───────────────┬───────────────┐
       Yangın   │      TP       │      FN       │
                │  Doğru Pozitif│ Yanlış Negatif│
                │  (true pos.)  │  (false neg.) │
GÖZLENEN        ├───────────────┼───────────────┤
    Yangın yok  │      FP       │      TN       │
                │Yanlış Pozitif │ Doğru Negatif │
                │  (false pos.) │  (true neg.)  │
                └───────────────┴───────────────┘
```

| Kutu | Terim | Anlamı | IGNIS'te operasyonel sonucu |
|---|---|---|---|
| **TP** | Doğru pozitif (true positive) | Yangın dedik, yandı | Doğru uyarı — kaynaklar doğru yere gönderildi |
| **FP** | Yanlış pozitif (false positive) | Yangın dedik, yanmadı | **Yanlış alarm** — uçaklar ve ekipler boşa harcandı |
| **FN** | Yanlış negatif (false negative) | Yangın yok dedik, yandı | **Kaçırılmış yangın** — korumasız bir köy |
| **TN** | Doğru negatif (true negative) | Yangın yok dedik, yanmadı | Doğru, ama burada tamamen bilgilendirici olmayan |

Bir orman yangını sistemi için FN ve FP'nin maliyetleri birbirinden çok farklıdır. Kaçırılan bir yangın insanları öldürebilir. Yanlış bir alarm yakıt ve uçuş saati israf eder. Bu asimetri gerçektir ve hangi metriği eniyileyeceğinizi etkilemelidir (bkz. Bölüm 6.10, Tversky kaybı).

### 6.2 Doğruluk neden yalan söyler

**Doğruluk (accuracy)**, doğru olan tahminlerin oranıdır:

$$\text{Doğruluk} = \frac{TP + TN}{TP + TN + FP + FN}$$

Şimdi bunu verimize uygulayalım. Arşiv 22.426 yama × 4.096 piksel ≈ **91.856.896 piksel** içerir. %0.2686'lık bir yaygınlıkta (prevalence):

| Nicelik | Sayı |
|---|---|
| Toplam piksel | 91.856.896 |
| Pozitif piksel (yarın yanacak) | ≈ 246.730 |
| Negatif piksel | ≈ 91.610.166 |

Şimdi **boş modeli (null model)** düşünün: tek satırdan oluşan bir program, `return 0`. Türkiye'deki her piksel için sonsuza dek "yangın yok" tahmini yapar. Karışıklık matrisi şöyledir:

| | Yangın tahmini | Yangın yok tahmini |
|---|---|---|
| **Gözlenen yangın** | TP = 0 | FN = 246.730 |
| **Gözlenen yangın yok** | FP = 0 | TN = 91.610.166 |

$$\text{Doğruluk} = \frac{0 + 91{,}610{,}166}{91{,}856{,}896} = 0.997314 = \mathbf{\%99.73}$$

**Yangın diye bir şeyi hiç duymamış olan bir program %99.73 doğruluk elde ediyor.**

Kesinliği (precision) tanımsızdır (0/0), duyarlılığı (recall) 0'dır, F1 skoru 0'dır, kesişimin birleşime oranı (IoU) 0'dır ve tamamen işe yaramazdır. Ama doğruluk sayısı muhteşem görünür. Bir makale özetinde yaygınlık belirtilmeden "%99.7 doğruluk" ifadesini görseydiniz, modelin mükemmel mi olduğunu yoksa kelimenin tam anlamıyla sabit sıfır fonksiyonu mu olduğunu anlamanın hiçbir yolu olmazdı.

**Makalenin şunu söylemesinin nedeni budur:** *"Pozitif sınıf son derece nadir olduğu için genel piksel doğruluğu bilgilendirici değildir: hiçbir yerde yangın öngörmemek zaten %99.7'den fazla doğruluk verir."*

Ezberlenmesi ve IAC'de yüksek sesle söylenmesi gereken genel kural:

> **Bir metrik yalnızca kendi temel çizgisine (baseline) göre yorumlanabilir. Hiçbir metriği, önemsiz bir tahmin edicinin ne alacağını da bildirmeden asla raporlamayın.**

Bölüm 6.7 ve Bölüm 9'da göreceğiz ki IGNIS'in kendisi, bu kuralı yama düzeyinde ihlal etmiş ve yakalanmış bir projeye örnektir — hem de kendi yazarları tarafından.

### 6.3 Kesinlik, duyarlılık, F1

Bu üç metrik TN'i tamamen göz ardı eder; burada faydalı olmalarının nedeni tam da budur.

**Kesinlik (precision)** — yanacağını *tahmin ettiğimiz* piksellerin ne kadarı gerçekten yandı?

$$\text{Kesinlik} = \frac{TP}{TP + FP}$$

Bu, "alarma güvenebilir miyim?" metriğidir. Düşük kesinlik, çok sayıda yanlış alarm anlamına gelir.

**Duyarlılık (recall)** — anma ya da hassasiyet, gerçek pozitif oranı olarak da adlandırılır — *gerçekten* yanan piksellerin ne kadarını bulduk?

$$\text{Duyarlılık} = \frac{TP}{TP + FN}$$

Bu, "bir şey kaçırdım mı?" metriğidir. Düşük duyarlılık, kaçırılan yangınlar demektir.

**İkisi birbiriyle takas edilir.** Karar eşiğini (threshold) $\tau$ düşürürseniz daha sık yangın tahmini yaparsınız: duyarlılık yükselir, kesinlik düşer. $\tau$'yu yükseltirseniz tersi olur. Uç noktalarda:

| Strateji | Kesinlik | Duyarlılık |
|---|---|---|
| Her yerde yangın tahmini ($\tau \to 0$) | yaygınlık = 0.0027 | 1.00 |
| Hiçbir yerde yangın tahmini yok ($\tau \to 1$) | tanımsız | 0.00 |

Hiçbiri işe yaramaz; bu yüzden ikisini birleştiren bir metriğe ihtiyacınız vardır.

**F1 skoru (F1-score)** — kesinlik ile duyarlılığın **harmonik ortalaması**:

$$F_1 = 2 \cdot \frac{\text{Kesinlik} \cdot \text{Duyarlılık}}{\text{Kesinlik} + \text{Duyarlılık}}$$

Neden sıradan aritmetik ortalama yerine *harmonik* ortalama? Çünkü harmonik ortalamaya küçük değer hâkim olur. Kesinliği 1.00 ve duyarlılığı 0.01 olan bir modeli karşılaştırın:

- Aritmetik ortalama: $(1.00 + 0.01)/2 = 0.505$ — kabul edilebilir görünüyor!
- Harmonik ortalama: $2(1.00 \times 0.01)/(1.01) = 0.0198$ — modelin neredeyse işe yaramaz olduğunu doğru biçimde bildiriyor.

F1, ikisinden yalnızca birinde iyi olarak başarılı olmanıza izin vermeyi reddeder.

**IGNIS değerleri (τ = 0.5, tüm arşiv):**

| Metrik | Değer | Okunuşu |
|---|---|---|
| Kesinlik | **0.0601** | Yarın yanacak diye işaretlediğimiz her 100 pikselden 6'sı gerçekten yanıyor. 94'ü yanlış alarm. |
| Duyarlılık | **0.0222** | Gerçekten yanan her 100 pikselden 2'sini buluyoruz. 98'ini kaçırıyoruz. |
| F1 | **0.0324** | Dürüst birleşik özet. |

Bu yorumları tam olarak bu sözcüklerle ifade edin. "100'de 6" ve "100'de 98'ini kaçırıyoruz", "kesinlik 0.06" ve "duyarlılık 0.02" ifadelerinden çok daha iletişimseldir ve arkasına saklanmak çok daha zordur.

### 6.4 IoU

**Kesişimin birleşime oranı** (Intersection over Union, IoU), **Jaccard indeksi (Jaccard index)** olarak da bilinir ve bölütleme (segmentation) için standart metriktir.

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$

Burada $A$ tahmin edilen maske (mask), $B$ ise gerçek maskedir.

```
      tahmin              gerçek            IoU = örtüşme / kaplanan toplam
     ┌────────┐                            
     │  ┌─────┼────┐                        ┌────────┐
     │  │/////│    │                        │  ┌─────┼────┐
     │  │/////│    │        →               │//│█████│////│
     └──┼─────┘    │                        └──┼─────┘////│
        └──────────┘                           └──────────┘
                                            █ = TP (kesişim)
                                            tüm taralı alan = birleşim
```

**Bölütleme neden doğruluk yerine IoU'yu tercih eder.** Çünkü IoU da, tıpkı kesinlik/duyarlılık/F1 gibi, TN'i hiçbir zaman saymaz. Doğru tahmin edilmiş arka plan piksellerinden oluşan devasa okyanus — 91,6 milyon adet — hiçbir katkı sağlamaz. IoU yalnızca şunu sorar: iki şekil ne kadar iyi örtüşüyor?

**F1 ile ilişkisi.** IoU ve F1 (bölütleme için Dice katsayısı ile aynı şeydir) birbirleriyle monoton biçimde ilişkilidir:

$$\text{IoU} = \frac{F_1}{2 - F_1}, \qquad F_1 = \frac{2\,\text{IoU}}{1 + \text{IoU}}$$

Kendi sayılarımızla kontrol edelim: $F_1 = 0.0324 \Rightarrow \text{IoU} = 0.0324/(2-0.0324) = 0.0324/1.9676 = 0.01647$. Raporlanan IoU **0.0165**'tir. Sayılar kendi içinde tutarlıdır; bu da değerlendirme kodunun doğru olduğuna dair iyi bir işarettir.

IoU her zaman F1'den küçük veya ona eşittir. IoU ikisinin daha sert olanıdır; bölütleme makalelerinin bunu raporlamasının nedeni de budur.

**IGNIS IoU = 0.0165.** Yorumu: tahmin edilen yanan alan ile gerçek yanan alan, birleşik ayak izlerinin %1.65'i kadar örtüşüyor. Karşılaştırma için, bir bölütleme sonucu genel olarak IoU ≈ 0.5'in üzerinde bir yerde kullanılabilir sayılır ve doğal görüntü bölütlemesinde en gelişmiş yöntemler 0.8+ değerlerine ulaşır. Kullanılabilirlikten iki büyüklük mertebesi uzağız.

### 6.5 ROC eğrisi

**ROC eğrisi (ROC curve)** (Receiver Operating Characteristic — alıcı işletim karakteristiği), eşik $\tau$ 1'den 0'a doğru tarandıkça şunları çizer:

- **y ekseni: Gerçek Pozitif Oranı (True Positive Rate)** = Duyarlılık = $TP/(TP+FN)$
- **x ekseni: Yanlış Pozitif Oranı (False Positive Rate)** = $FP/(FP+TN)$

```
   TPR
   1.0 ┤        ╭─────────────  mükemmel (AUC = 1.0)
       │      ╭─╯      ╱
       │    ╭─╯      ╱
       │  ╭─╯      ╱     ← rastgele tahmin (AUC = 0.5)
       │╭─╯      ╱
   0.0 ┼───────╱────────────────
      0.0                    1.0  FPR
```

**ROC eğrisi altındaki alan (ROC-AUC)**, bu eğrinin altında kalan alandır. Çok güzel bir olasılıksal yorumu vardır:

> **ROC-AUC, rastgele seçilmiş bir pozitif örneğin, rastgele seçilmiş bir negatif örnekten daha yüksek skor alma olasılığıdır.**

Yani ROC-AUC = 0.5 "sıralamada yazı-tura atmaktan daha iyi değil", 1.0 ise "mükemmel sıralama" demektir.

**IGNIS ROC-AUC = 0.8468.** Yorumu: yarın yanacak bir piksel ile yanmayacak bir piksel seçerseniz, modelimiz yanacak olana **zamanın %84.68'inde** daha yüksek olasılık verir. Bu, gerçekten ve kayda değer biçimde rastgeleden daha iyidir. 14 kanallı temsilin gerçek bilgi taşıdığını kanıtlar.

Peki model neden işe yaramıyor? FPR'nin paydasına bakın.

### 6.6 PR eğrisi ve AUC-PR

**Kesinlik–duyarlılık eğrisi**, $\tau$ tarandıkça şunları çizer:

- **y ekseni: Kesinlik** = $TP/(TP+FP)$
- **x ekseni: Duyarlılık** = $TP/(TP+FN)$

**AUC-PR** (kesinlik-duyarlılık eğrisi altındaki alan), Ortalama Kesinlik (Average Precision) olarak da adlandırılır ve bu eğrinin altındaki alandır.

ROC'tan belirleyici farkı:

| | ROC eğrisi | PR eğrisi |
|---|---|---|
| x ekseni paydası | $FP + TN$ — **91,6 milyon negatifi içerir** | $TP + FN$ — yalnızca 246.730 pozitif |
| y ekseni paydası | $TP + FN$ | $TP + FP$ — **hiçbir yerde TN yok** |
| TN kullanır mı? | Evet, yoğun biçimde | **Hayır** |
| Rastgele sınıflandırıcı için temel çizgi | Her zaman 0.5 | **Yaygınlığa eşittir** |

**Temel çizgi noktası kritiktir.** Rastgele bir sınıflandırıcının PR eğrisi, $y = $ yaygınlık düzeyinde yatay bir doğrudur. IGNIS için:

$$\text{AUC-PR}_{\text{rastgele}} = \text{yaygınlık} = 0.002686 \approx \mathbf{0.0027}$$

Dolayısıyla "AUC-PR = 0.0210" ifadesi, temel çizginin 0.0027 olduğunu bilmeden anlamsızdır. Bizim değerimiz

$$\frac{0.0210}{0.002686} = \mathbf{7.8\times} \text{ yaygınlık temel çizgisi}$$

Makalenin AUC-PR için *"0.00269'luk yaygınlık temel çizgisinin yaklaşık 7,8 katı"* demesinin ve hemen ardından bunun yine de *"operasyonel kullanım için çok fazla düşük"* olduğunu eklemesinin nedeni budur. Bu cümlenin her iki yarısı da birlikte söylenmelidir. Yalnızca "rastgeleden 7,8 kat daha iyi" demek çarpıtma olurdu; yalnızca "0.0210" deyip temel çizgiyi vermemek ise yorumlanamaz olurdu.

### 6.7 Kendi çelişkimiz: ROC-AUC 0.8468'e karşı AUC-PR 0.0210

İşte görünürdeki paradoks; ve bu, tüm bu rehberdeki en iyi öğretici örnektir, çünkü *bizim kendi verimizdir*.

| Metrik | Değer | Söylediği |
|---|---|---|
| ROC-AUC | **0.8468** | Model iyidir |
| AUC-PR | **0.0210** | Model neredeyse işe yaramazdır |

Her ikisi de doğru hesaplanmıştır. Her ikisi de doğrudur. Farklı şeyleri ölçüyorlar ve aşırı dengesizlik altında şiddetli biçimde ayrışıyorlar.

**Sayılarla çözümü.**

Yanlış pozitif oranı $FP/(FP+TN)$'dir. Negatif nüfusumuz yaklaşık **91,6 milyon piksel**tir. Modelin 100.000 yanlış pozitif ürettiğini varsayalım. O hâlde:

$$\text{FPR} = \frac{100{,}000}{91{,}610{,}166} = 0.0011$$

ROC eğrisinde bu esasen sıfırdır. Eğri kıpırdamaz. ROC-AUC yüksek kalır.

Ama şimdi aynı çalışma noktasında kesinliği hesaplayın. Bu 100.000 tahminin 6.000 doğru pozitif içerdiğini varsayalım:

$$\text{Kesinlik} = \frac{6{,}000}{106{,}000} = 0.057$$

**Alarmlarımızın %94'ü yanlıştır** — ve ROC eğrisi buna tamamen kördür, çünkü 100.000 yanlış pozitif, 91,6 milyon negatifin yanında bir yuvarlama hatasıdır.

Daha keskin ifade edersek:

> Aşırı dengesizlik altında, *negatiflerin sayısı o kadar büyüktür ki yanlış pozitif oranı çok küçük olabilirken yanlış pozitiflerin sayısı operasyonel olarak felakettir.* ROC-AUC oranı ölçer. Kesinlik sayıyı ölçer. Bir helikopteri nereye göndereceğine karar veren nöbetçi amir için bunlardan yalnızca biri önemlidir.

**Jüri için somut bir düşünce deneyi.** IGNIS'i Türkiye geneline yaydığınızı hayal edin. Türkiye'nin kara alanı yaklaşık 780.000 km²'dir, yani bizim 1 km'lik piksellerimizden 780.000 tanesi. Model bunların %1'ini "yarın yanacak" diye işaretlerse, bu günde 7.800 alarm demektir. 0.0601 kesinlikte bunların yaklaşık 469'u gerçek, yaklaşık 7.331'i yanlıştır. Dünyada hiçbir itfaiye teşkilatı günde 7.331 yanlış alarma göre hareket edemez. Model dağıtılabilir değildir ve ROC-AUC 0.847 bunu değiştirmez.

**Kural:**

> **Pozitif sınıf nadir olduğunda birincil metrik olarak AUC-PR kullanın, yanında her zaman yaygınlığı raporlayın ve ROC-AUC'yi yalnızca sıralama kalitesinin ikincil bir tanılayıcısı olarak ele alın.**

Makale tam olarak bu konumu benimser: AUC-PR *"çok büyük doğru negatif nüfusuna duyarsız olduğu için birincil metriktir"*. Sorgulanırsanız bunu söyleyin.

**ROC-AUC 0.847'nin bize meşru olarak *söylediği* şey.** Bu değer değersiz değildir. Modelin gerçek bir sıralama öğrendiğini söyler: yanan pikseller sistematik olarak daha yüksek skorlar alır. Bu, girdi özniteliklerinin bilgilendirici olduğu ve mimarinin bozuk olmadığı anlamına gelir. Başarısızlık, bir sıralamayı kullanılabilir bir *karara* dönüştürmededir. Bu cesaret verici bir teşhistir, çünkü zor olan kısım sıralamadır ve kalibrasyon nispeten düzeltilebilir bir şeydir.

### 6.8 Eşik seçimi ve kalibrasyon

Model sürekli bir olasılık $P(i,j) \in (0,1)$ üretir. İkili bir maske üretmek için bir **eşik (threshold)** $\tau$ seçmeniz gerekir.

**IGNIS $\tau = 0.5$ kullandı ve bu neredeyse kesinlikle yanlıştır.**

0.5 neden varsayılandır? Çünkü $\sigma(0) = 0.5$'tir, yani "ham logit pozitiftir" durumuna karşılık gelir. İki sınıf dengeli olduğunda doğal seçimdir. Bizim sınıflarımız **372'ye 1** oranında dengesizdir ($1/0.002686 = 372$). En uygun kesme noktasının 0.5'te oturmasını beklemek için hiçbir neden yoktur.

Sonuç kendi sayılarımızda görülebilir: duyarlılık 0.0222, kesinlik 0.0601'den *daha düşüktür*; bu da eşiğin çok fazla muhafazakâr olduğu anlamına gelir. Modelden "yangın" demeden önce çok emin olması isteniyor ve neredeyse hiçbir zaman emin olamıyor. $\tau$'yu düşürmek, bir miktar kesinliği çok miktarda duyarlılıkla takas eder ve büyük olasılıkla F1'i kayda değer biçimde yükseltir — hem de **hiçbir şeyi yeniden eğitmeden**.

Yeni pipeline'ın uyguladığı **doğru prosedür**:

1. Modeli eğitin.
2. **Yalnızca doğrulama kümesi (validation set) üzerinde**, çok sayıda aday eşikte kesinlik ve duyarlılığı hesaplayın (ör. $\tau = 0.01, 0.02, \dots, 0.99$).
3. F1'i maksimize eden $\tau^\star$'yı seçin (ya da operasyonel maliyet asimetrisi biliniyorsa, duyarlılığı öne çıkarmak için $\beta > 1$ ile $F_\beta$'yı maksimize eden değeri).
4. $\tau^\star$'yı dondurun ve test kümesine (test set) değiştirmeden uygulayın.

4. adım kritiktir. Eşiği test kümesi üzerinde seçerseniz, test bilgisini modelinize sızdırmış olursunuz ve raporladığınız skor şişer.

**İlgili ama ayrı bir fikir: kalibrasyon (calibration).** Bir model, 0.3 olasılık atadığı tüm pikseller arasında yaklaşık %30'u gerçekten yanıyorsa *kalibre edilmiş* sayılır. Odak kaybı (focal loss) ve pozitif ağırlıklı kayıplar, dengesizlikle savaşmak için olasılıkları kasıtlı olarak bozar; dolayısıyla **kalibre edilmemiş** çıktılar üretirler. IGNIS'in bir gün "bu köyün tehdit altında olma olasılığı %30" gibi bir çıktı vermesi gerekirse, Platt ölçekleme veya izotonik regresyon gibi bir kalibrasyon adımı gerekli olurdu. Eşiğe dayalı maske üretimi için kalibrasyon kesinlikle gerekli değildir — yalnızca sıralama önemlidir. Olasılıklar hakkında soru sorulursa yapabilmeniz gereken iyi bir ayrımdır bu.

### 6.9 Temel çizgiler ve kalıcılığın neden zorunlu olduğu

Bir **temel çizgi (baseline)**, modelinizin yenmek zorunda olduğu, kasıtlı olarak basit bir yöntemdir. Temel çizgi olmadan bir sayı hiçbir şey ifade etmez.

Ertesi gün yangın yayılımı için üç temel çizgi ilgilidir ve IGNIS'in yeni değerlendirme planı üçünü de kapsar.

| Temel çizgi | İngilizce | Kural | Neden önemli |
|---|---|---|---|
| **Kalıcılık** | Persistence | Yarının maskesi = bugünün maskesi | Mutlak taban. Maliyeti sıfır. |
| **Genişletilmiş kalıcılık** | Dilated persistence | Bugünün maskesi, her yöne 1 piksel genişletilmiş | Mümkün olan en kaba büyüme kavramını ekler |
| **Rüzgâr yönlü büyüme** | Wind-directed growth | Bugünün maskesi, ERA5 rüzgâr vektörü boyunca kaydırılmış/genişletilmiş | Hiçbir öğrenme olmadan baskın fiziksel sürücüyü ekler |

**Kalıcılık en önemlisidir ve pazarlık konusu değildir.** Yangınlar günlerce yanar. Dünkü yanan pikselin bugün de yanıyor olması çok muhtemeldir. "Yarın tam olarak bugüne benziyor" tahmini hiçbir veri, hiçbir model, hiçbir GPU ve hiçbir eğitim gerektirmez. **Sinir ağınız bunu yenemiyorsa, sinir ağınız hiçbir katkı sunmamıştır.**

Bunu ölçtük. 1.054 yama içeren 45 dosyalık bir örneklem üzerinde:

| Yöntem | Kesinlik | Duyarlılık | F1 | IoU |
|---|---|---|---|---|
| **Kalıcılık** ("yarın = bugün") | 0.0430 | **0.0963** | **0.0595** | **0.0306** |
| **IGNIS U-Net** (τ = 0.5) | 0.0601 | 0.0222 | 0.0324 | 0.0165 |
| **Oran (kalıcılık / model)** | 0.72× | **4.34×** | **1.84×** | **1.85×** |

**Model kaybediyor.** Kalıcılık, 92 milyon etiketli piksel üzerinde eğitilmiş 1,9 milyon parametreli bir evrişimli sinir ağının F1'inin 1,84 katına ve IoU'sunun 1,85 katına ulaşıyor.

Modelin daha iyi yaptığı tek şey kesinliktir (0.0601'e karşı 0.0430) — bir alarm verdiğinde biraz daha sık haklıdır. Ama o kadar az alarm verir ki duyarlılığı 4,3 kat daha kötüdür ve F1 bunu doğru biçimde cezalandırır.

Bölüm 9 bu konuda ne yapılacağını tartışıyor. Şimdilik, bu karşılaştırmayı yapmak için gereken entelektüel dürüstlüğe dikkat edin. Bunu hesaplamamak kolay olurdu. **Hesaplamak, yayımlamak ve teşhis etmek bilimsel katkının kendisidir.**

### 6.10 Dengesizlikle başa çıkma teknikleri

Beş yaklaşım ailesi vardır. IGNIS bunlardan birini kullandı ve bir kombinasyona geçiyor.

#### (a) Yeniden örnekleme

**Aşırı örnekleme (oversampling)** azınlık sınıfı örneklerini çoğaltır; **eksik örnekleme (undersampling)** çoğunluk sınıfı örneklerini atar.

*Piksel düzeyinde* bölütleme için bu tuhaftır, çünkü bir pikseli, tüm yamasını çoğaltmadan çoğaltamazsınız. *Yapabileceğiniz* şey **yama** düzeyinde yeniden örneklemedir — örneğin çok sayıda pozitif piksel içeren yamaları aşırı örneklemek. IGNIS ilişkili ama daha zarif bir şey yapıyor: **en baştan yalnızca aktif yangın pikselleri merkezli yamalar çıkarıyor.** Her eğitim örneğinin yangın içermesi garantidir. Türkiye üzerinde rastgele bir yama neredeyse tamamen boş olurdu.

Yeni pipeline **32×32 merkez kırpması** (Bölüm 7.5) ile daha da ileri gider; bu, arka planın bir tür uzamsal eksik örneklemesidir: yamanın orantısız biçimde çok sayıda negatif içeren dış halkasını atar ve pozitif yoğunluğunu kabaca dört katına çıkarır.

#### (b) Sınıf ağırlıkları / pozitif ağırlıklı BCE

Nadir sınıfın, yaygın sınıf kadar toplam gradyan katkısı yapması için pozitif piksellerin kaybını $w > 1$ bir çarpanla çarpın.

$$L = -\big[\,w \cdot y\log p + (1-y)\log(1-p)\,\big]$$

PyTorch'ta bu `BCEWithLogitsLoss(pos_weight=w)` şeklindedir. Teorik olarak dengeli değer şudur:

$$w = \frac{\text{negatif sayısı}}{\text{pozitif sayısı}} = \frac{1 - 0.002686}{0.002686} \approx 371$$

Pratikte $w = 371$ çoğu zaman eğitimi kararsızlaştırır ve neredeyse her yerde yangın tahmin eden bir model üretir; bu yüzden 10–50 aralığındaki değerler yaygın başlangıç noktalarıdır. Deponun `SPREAD_POS_WEIGHT` varsayılanı 10'dur — ki 372:1'lik bir dengesizlik göz önüne alındığında bu tartışmalı biçimde çok fazla küçüktür. Bu, modelin neden az tahmin ürettiğine dair somut ve test edilebilir bir hipotezdir ve planlanmış belirli bir deney olarak zikredilmeye değerdir.

#### (c) Odak kaybı

Yoğun nesne tespiti için Lin ve arkadaşları (2017) tarafından tanıtılmıştır — tam olarak bizim yapımıza sahip bir problem: birkaç nesne, bir arka plan okyanusu.

$$L_{\text{odak}} = -\alpha\,(1 - p_t)^{\gamma}\,\log(p_t)$$

Burada $p_t$, modelin *gerçek sınıf için* tahmin ettiği olasılıktır (yani $y=1$ ise $p_t = p$, $y=0$ ise $p_t = 1-p$), $\alpha$ sınıfları dengeler ve $\gamma$ **odaklanma parametresidir**.

**$(1-p_t)^\gamma$ aslında ne yapar?** Kolay örnekler için bir ağırlık düşürme çarpanıdır. IGNIS'in kullandığı değer olan $\gamma = 2$ ile adım adım inceleyelim:

| Durum | $p_t$ | $(1-p_t)^2$ | O pikselin kaybına etkisi |
|---|---|---|---|
| Kolay arka plan, güvenle doğru | 0.99 | $0.01^2 = 0.0001$ | Kayıp **10.000×** azaltıldı |
| Kolay arka plan, doğru | 0.90 | $0.10^2 = 0.01$ | Kayıp **100×** azaltıldı |
| Belirsiz | 0.50 | $0.50^2 = 0.25$ | Kayıp 4× azaltıldı |
| Zor, neredeyse yanlış | 0.10 | $0.90^2 = 0.81$ | Kayıp neredeyse hiç azalmadı |
| Yanlış | 0.01 | $0.99^2 = 0.98$ | Kayıp esasen değişmedi |

Önemsiz biçimde doğru olan devasa arka plan piksel nüfusu dört büyüklük mertebesi bastırılır; böylece toplam gradyana seyrek ve zor yangın pikselleri hâkim olur. Makale bunu tam olarak şöyle ifade eder: *"Bir arka plan pikseli zaten yüksek güvenle sınıflandırıldığında, $p_t$ bire yaklaşır ve $(1-p_t)^\gamma$ çarpanı onun katkısını bastırır; böylece gradyana seyrek ve zor yangın pikselleri hâkim olur."*

$\gamma = 0$'ın sıradan ağırlıklı çapraz entropiyi geri verdiğine dikkat edin. Daha büyük $\gamma$ daha sert odaklanır.

IGNIS $\gamma = 2.0$ ve $\alpha = 0.80$ kullanır. Makalenin kendi teşhisi bunun yetersiz olduğu yönündedir: *"%0.27 yaygınlıkta, odak kaybı tek başına yetersizdir."*

#### (d) Dice kaybı

Dice kaybı problemi tamamen farklı bir yönden ele alır: piksel başına bir kaybı düzeltmek yerine, **kaybın kendisini bir örtüşme metriği hâline getirir.**

Dice katsayısı F1 ile aynı niceliktir:

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$$

Bunu bir kayıp olarak kullanmak için türevlenebilir olması gerekir; bu yüzden sert sayımları olasılık toplamlarıyla değiştirerek **yumuşak Dice**'ı kullanırız:

$$L_{\text{Dice}} = 1 - \frac{2\sum_i p_i y_i + \epsilon}{\sum_i p_i + \sum_i y_i + \epsilon}$$

Burada toplamlar tüm pikseller üzerinden alınır, $p_i$ tahmin edilen olasılık, $y_i \in \{0,1\}$ etikettir ve $\epsilon$ boş bir yamada sıfıra bölmeyi önleyen küçük bir sabittir (tipik olarak 1).

**Bu neden dengesizliğe karşı bağışıktır?** Çünkü formülde TN hiç yer almaz. 11 pozitif piksel içeren bir yama ile 500 pozitif piksel içeren bir yama karşılaştırılabilir katkı yapar, çünkü payda maske boyutuna göre normalleştirir. Dice kaybı, doğrudan gerçekten ölçtüğümüz şeyi eniyiler.

Zayıflığı kararsızlıktır: **sıfır** pozitif piksel içeren bir yamada — ve unutmayın, **yamalarımızın %58.9'unda yarın sıfır yangın pikseli vardır** — pay, tahmin ne olursa olsun 0'dır ve gradyan kötü davranır. Dice'ın neredeyse hiçbir zaman tek başına kullanılmamasının nedeni budur.

**Yeni IGNIS kaybı bir melezdir:**

$$L = 0.5 \cdot L_{\text{BCE}}(\text{pos\_weight}) + 0.5 \cdot L_{\text{YumuşakDice}}$$

BCE, boş yamalar dâhil her yerde kararlı ve iyi koşullanmış bir piksel başına gradyan sağlar; Dice ise eniyilemeyi iyi bölge örtüşmesine doğru çeker. Bu kombinasyon, tam da bu nedenle, artık tıbbi ve uzaktan algılama bölütlemesinde standart uygulamadır.

#### (e) Tversky ve Odak Tversky kaybı

Tversky indeksi, FP ve FN'i *farklı* ağırlıklandırarak Dice'ı genelleştirir:

$$T = \frac{TP}{TP + \alpha\,FP + \beta\,FN}, \qquad L_{\text{Tversky}} = 1 - T$$

| $\alpha$ | $\beta$ | Sonuç |
|---|---|---|
| 0.5 | 0.5 | Tam olarak Dice katsayısı |
| 1.0 | 1.0 | Tam olarak IoU / Jaccard |
| **0.3** | **0.7** | **FN'i FP'den fazla cezalandırır → duyarlılığı yükseltir** |
| 0.7 | 0.3 | FP'yi daha fazla cezalandırır → kesinliği yükseltir |

**IGNIS'in alternatif amaç fonksiyonu $\alpha = 0.3$, $\beta = 0.7$ kullanır.** Gerekçe operasyoneldir ve bunu tam bu terimlerle ifade etmelisiniz: *bir orman yangını uyarı sistemi için kaçırılmış bir yangın (FN), yanlış bir alarmdan (FP) çok daha maliyetlidir; bu yüzden FN'i kasıtlı olarak 2,33 kat daha ağır ağırlıklandırıyoruz.* Bu istatistiksel bir hile değildir; gerçek dünyadaki maliyet asimetrisini amaç fonksiyonuna kodlamaktır. Bir jüriye söylenecek güçlü bir noktadır.

**Odak Tversky** ayrıca zor durumlara odaklanmak için her şeyi bir kuvvete yükseltir:

$$L_{\text{OdakTversky}} = (1 - T)^{1/\gamma_{\!T}}$$

#### Kayıp stratejisinin özeti

| Sürüm | Kayıp | Gerekçe |
|---|---|---|
| Ön çalışma (makale) | Odak, $\gamma=2.0$, $\alpha=0.80$ | Yoğun dengesizlik için standart tercih; **yetersiz olduğu kanıtlandı** |
| Yeni pipeline (birincil) | $0.5\,$BCE(pos_weight) $+\ 0.5\,$YumuşakDice | BCE'den kararlılık + Dice'tan bölge örtüşmesi |
| Yeni pipeline (alternatif) | Odak Tversky, $\alpha=0.3$, $\beta=0.7$ | Operasyonel güvenlik için açık duyarlılık tercihi |

Bölüm 9'daki teşhisin kaçınılmaz kıldığı son ve önemli bir uyarı: **hiçbir kayıp fonksiyonu bozuk bir girdi temsilini düzeltemez.** Standart sapması 515 olan yükseklik ilk evrişim katmanına hâkim olurken standart sapması 0.07 olan toprak nemi görünmez durumdaysa, kaybı değiştirmek işe yaramayacaktır. Önce normalleştirme gelmelidir. Yeni pipeline'ın *amaç fonksiyonunu* değiştirmeden önce *veriyi* değiştirmesinin nedeni budur.
---

## 7. IGNIS veri pipeline'ı, satır satır

Bu bölüm, bir fotonun bir sinir ağının içinde bir sayıya dönüşmeden önce başına gelenleri adım adım anlatıyor. TFRecord dosyasına kadar olan her şey Google Earth Engine'de, ondan sonraki her şey yerel olarak gerçekleşir.

```
  ┌──────────────────────── GOOGLE EARTH ENGINE ──────────────────────────┐
  │                                                                       │
  │  8 koleksiyon  →  tarih ve Türkiye'ye göre süz  →  EPSG:32635 @       │
  │                     1000 m'ye yeniden projeksiyonla                   │
  │                                              →  zamansal kompozitleme │
  │                                              →  birim dönüşümü        │
  │                                              →  wind_speed, humidity, │
  │                                                 slope, aspect türetimi│
  │                                              →  tek bir Image'da yığ  │
  │                    stratifiedSample (≤150 yangın pikseli/gün)         │
  │                    neighborhoodToArray (65×65 çekirdek)               │
  │                    Export.table.toDrive(format='TFRecord')            │
  └───────────────────────────────┬───────────────────────────────────────┘
                                  │  *.tfrecord.gz  (yangın günü başına bir parça)
                                  ▼
  ┌───────────────────────── YEREL EĞİTİM ────────────────────────────────┐
  │  TFRecord ayrıştır  →  uzunluk kontrolü  →  kırp  →  normalleştir     │
  │  →  kodla  →  artır  →  yığınla  →  U-Net  →  kayıp  →  Adam  →       │
  │     kontrol noktası                                                   │
  └───────────────────────────────────────────────────────────────────────┘
```

### 7.1 Sekiz kaynak ürün

**Hızlı başvuru: hangi ürün hangi uydudan geliyor?**

Bir jüriden gelmesi en muhtemel olgusal soru budur; bu yüzden tam olarak
cevaplayabilmek önemlidir — özellikle de uydu *olmayan* iki kalem için.

| Ürün kimliği | Platform / uydu | Cihaz | Bize verdiği |
|---|---|---|---|
| `MODIS/061/MOD14A1` | **Terra** (NASA EOS AM-1, Aralık 1999) | MODIS | Aktif yangın maskesi, gün t |
| `MODIS/061/MYD14A1` | **Aqua** (NASA EOS PM-1, Mayıs 2002) | MODIS | Aktif yangın maskesi, gün t |
| `MODIS/061/MOD13Q1` | **Terra** | MODIS | NDVI (bitki örtüsü) |
| `MODIS/061/MOD11A1` | **Terra** | MODIS | Arazi yüzey sıcaklığı |
| `MODIS/061/MCD12Q1` | **Terra + Aqua** birleşik | MODIS | IGBP arazi örtüsü / yakıt sınıfı |
| `USGS/SRTMGL1_003` | **Space Shuttle Endeavour**, STS-99 görevi (Şubat 2000) | SRTM radar interferometresi | Yükseklik, eğim, bakı |
| `UCSB-CHG/CHIRPS/DAILY` | *Harmanlanmış*: jeostasyoner uydular (GOES, Meteosat) **artı yer yağış istasyonları** | termal kızılötesi + istasyon | Yağış |
| `ECMWF/ERA5_LAND/DAILY_AGGR` | **Uydu değil** — ECMWF yeniden analiz modeli | çok sayıda gözlemi özümser | Hava sıcaklığı, çiğ, rüzgâr u/v, toprak nemi |

Yani doğrudan bağlı olduğumuz uydular **Terra** ve **Aqua**'dır; ikisi de MODIS
cihazını taşır. Buna ek olarak, Şubat 2000'de Space Shuttle Endeavour ile uçurulan
tarihsel **SRTM** görevi vardır (arazi statik olduğu için 2000 yılı ölçümü hâlâ
geçerlidir).

İki kalem sıkça yanlış tarif edilir ve bunu sunumda yanlış söylemek, bir uzmanın
kolayca yakalayacağı türden bir hatadır:

- **ERA5-Land bir uydu değildir.** Bir *yeniden analizdir* (reanalysis): geçmişe
  doğru çalıştırılan, o dönemde mevcut olan her gözlemle (uydular, radyosondalar,
  uçaklar, yer istasyonları) kısıtlanan bir hava modeli. Bir ölçüm değil, fiziksel
  olarak tutarlı bir atmosfer tahmini verir.
- **CHIRPS de saf uydu ürünü değildir.** Jeostasyoner uydulardan gelen bulut tepesi
  sıcaklığını (termal kızılötesi) yerdeki yağış istasyonu kayıtlarıyla harmanlar.

Tek cümlelik dürüst cevap: *"Yangın ve bitki örtüsüyle ilgili her şey için Terra ve
Aqua, arazi için SRTM, hava için de bir yeniden analiz ve harmanlanmış bir yağış
ürünü."*


| Değişken | GEE koleksiyonu | Doğal çözünürlük | Fizikteki rolü |
|---|---|---|---|
| Aktif yangın (Terra) | `MODIS/061/MOD14A1` | 1 km / günlük | Bugünün yangın maskesi; hedefin kaynağı |
| Aktif yangın (Aqua) | `MODIS/061/MYD14A1` | 1 km / günlük | Günde ikinci bir bakış için Terra ile birleştirilir |
| NDVI | `MODIS/061/MOD13Q1` | 250 m / 16 gün | Yakıt yükü |
| Arazi yüzey sıcaklığı | `MODIS/061/MOD11A1` | 1 km / günlük | Yüzey enerji durumu, kuraklık stresi |
| Meteoroloji | `ECMWF/ERA5_LAND/DAILY_AGGR` | ~9 km / günlük | Hava sıcaklığı, çiy noktası, rüzgâr $u$/$v$, toprak nemi |
| Yağış | `UCSB-CHG/CHIRPS/DAILY` | ~5 km / günlük | Yağmur |
| Topografya | `USGS/SRTMGL1_003` | 30 m / statik | Yükseklik, eğim, bakı |
| Arazi örtüsü | `MODIS/061/MCD12Q1` | 500 m / yıllık | Yakıt türü |

Çalışma alanı: Türkiye, `country_na = 'Turkey'` ile `USDOS/LSIB_SIMPLE/2017` sınırı tarafından tanımlanır. Dönem: yangın mevsimi ayları olan Haziran–Ekim; mevcut arşivde 2019–2021 yılları, yenisinde 2024'e ve ötesine uzanıyor.

### 7.2 GEE içinde uyumlaştırma

**Yeniden projeksiyonlama (reprojection).** Her bant 1000 m'de EPSG:32635 üzerine yerleştirilir. `fire` ve `landcover` için neden en yakın komşu ile yeniden örneklemenin (nearest neighbour resampling), sürekli alanlar için ise neden ara değerlemenin kullanılması gerektiğine dair Bölüm 3.5'e bakın.

**Kalite maskeleme.** MODIS QA bantları, bulut kirliliğine uğramış ve geçersiz gözlemleri kaldırmak için kullanılır. Su, bulut ve işlenmemiş pikselleri temsil eden FireMask sınıfları yangın dışı olarak ele alınır.

**Zamansal kompozitleme.** Üç farklı tekrar ziyaret oranı için üç farklı strateji:

| Bant | Kompozitleme kuralı | Nedeni |
|---|---|---|
| `ndvi` | Önceki 32 gün içindeki en güncel 16 günlük kompozit | MOD13Q1 tekrar ziyareti 16 gündür |
| `lst` | Önceki 3 günlük ölçümün ortalaması | MOD11A1'deki bulut boşluklarını doldurur |
| Meteoroloji, `fire` | Gözlem tarihinin kendisine ait değer | Her ikisi de gerçekten günlüktür |

**Birim dönüşümü.** LST ve hava sıcaklığı için Kelvin → Celsius; ERA5 yağışı için metre → milimetre; NDVI'ye standart MODIS ölçek çarpanı 0.0001 uygulanır, böylece [−1, 1] aralığında kalır.

### 7.3 Türetilen kanallar

On dört kanaldan üçü okunmak yerine hesaplanır.

**Rüzgâr hızı:**
$$\text{wind\_speed} = \sqrt{u^2 + v^2} = \texttt{hypot}(u, v)$$

**Magnus formülü ile bağıl nem.** ERA5-Land, °C cinsinden hava sıcaklığı $T$ ve çiy noktası sıcaklığı $T_d$ sağlar, ama bağıl nem sağlamaz. Magnus yaklaşımı bunları dönüştürür:

$$\text{RH} = 100 \cdot \exp(A - B), \qquad A = \frac{17.625\,T_d}{243.04 + T_d}, \qquad B = \frac{17.625\,T}{243.04 + T}$$

[0, 100] % aralığına kırpılır.

Fiziği: çiy noktası, havanın doyuma ulaşması için soğutulması gereken sıcaklıktır. $T_d = T$ ise hava doymuştur (RH = %100). $T$ ile $T_d$ arasındaki fark ne kadar büyükse hava o kadar kurudur. Üstel terimler, doyma buhar basıncı eğrisine Magnus yaklaşımıdır ve RH, gerçek buhar basıncının doyma buhar basıncına oranıdır.

Çözümlü örnek: $T = 35$ °C, $T_d = 10$ °C (tipik bir sıcak ve kuru Antalya öğleden sonrası).
$A = 17.625 \times 10 / 253.04 = 0.6965$; $B = 17.625 \times 35 / 278.04 = 2.2183$;
$\text{RH} = 100\,e^{0.6965 - 2.2183} = 100\,e^{-1.5218} = \%21.8$.
Bu tehlikeli biçimde kurudur — ince ölü yakıtlar en kuru hâllerinde olacaktır.

**Topografya:** SRTM sayısal yükseklik modeline uygulanan `ee.Terrain.products()`, Bölüm 2.13'te açıklanan sonlu fark yöntemini kullanarak `elevation`, `slope` ve `aspect`'i tek çağrıda verir.

### 7.4 Yama çıkarımı: `stratifiedSample` ve `neighborhoodToArray`

**Adım 1 — nereden örnekleneceğinin seçimi.** Her yangın günü için `stratifiedSample`, aktif yangın piksellerinden en çok `MAX_POINTS_PER_DAY = 150` konum seçer.

**Tabakalı örnekleme (stratified sampling)**, düzgün rastgele örnekleme yerine her sınıf içinde ayrı ayrı örnekleme yapmak demektir. Burada iki amaca hizmet eder: örneklerin ülke genelinden değil yangın piksellerinden çekilmesini garanti eder (rastgele bir Türkiye pikseli neredeyse hiçbir zaman yanmıyordur) ve tek bir devasa yangın olayının arşivi boğmasını önler. 150 üst sınırı olmasaydı, tek başına 2021 Manavgat yangını on binlerce neredeyse aynı yamayı katabilir ve kayba hâkim olabilirdi.

Gizli bir koşul vardır: `MIN_FIRE_PIXELS = 5`. Ülkenin tamamında 5'ten az aktif yangın pikseli olan bir gün atlanır, çünkü büyük olasılıkla yanlış bir tespittir ya da öğrenilemeyecek kadar küçüktür.

**Adım 2 — komşuluğun çıkarılması.** `neighborhoodToArray`, bir rasteri piksel başına dizilere dönüştüren GEE işlemidir. Bir çekirdek verildiğinde — burada `ee.Kernel.square(radius=32)` — her pikselin skaler değerini **komşuluğundaki tüm değerler dizisiyle** değiştirir.

```
  Yarıçapı 32 olan kare bir çekirdek kenar başına (2 × 32 + 1) = 65 piksel kapsar
  → örneklenen her nokta, bant başına 65 × 65'lik bir dizi olur
  → 14 bant + fire_next + fire_next2 + valid → nokta başına bir kayıt
```

Yani örneklenen tek bir nokta eksiksiz bir çok kanallı görüntü yaması üretir ve `Export.table.toDrive(..., 'TFRecord')` günün tüm nokta koleksiyonunu tek bir sıkıştırılmış parça olarak yazar.

**Adım 3 — kırpma.** 65 tek sayıdır ve U-Net'in üç adet 2×2 havuzlaması 8'e bölünebilen bir boyut gerektirir. Bu nedenle yamalar eğitim zamanında **64×64**'e kırpılır. Yeni pipeline çok daha agresif kırpar — bkz. Bölüm 7.5.

**Adım 4 — geçerlilik kontrolü.** Sahne kenarlarında maskeleme nedeniyle kesilmiş kayıtlar bir uzunluk kontrolüyle tespit edilip atılır. Tüm prosedürü 360 yangın gününe uygulamak **22.426 geçerli yama** üretti.

> **Bilinmeye değer tarihsel bir hata.** Daha eski bir defter, her örnek noktaya `fire` adında skaler bir özellik ekliyordu. Bu ad, 65×65'lik `fire` bandıyla çakışıyor ve onu sessizce 1×1'lik bir değere çöktürüyordu; bu da `Can't parse serialized Example ... Key: fire` hatalarına ve eğitim zamanında sıfır geçerli kareye yol açıyordu. Düzeltme, dışa aktarmadan önce nokta özelliklerini kaldırmak ve `unmask(0, False)` kullanmaktı. Size hiç "en zor hata neydi" diye sorulursa, bu gerçek ve öğretici bir cevaptır.

### 7.5 Yeni olan ne: `fire_next2`, `valid` ve 32×32 kırpma

Dışa aktarılan veriye, her biri ölçülmüş bir soruna yanıt veren üç değişiklik.

#### `fire_next2` — ±1 günlük hedef

**Ölçülmüş sorun.** Örneklenen arşivde **yamaların %58.9'unda $t+1$ gününde sıfır yangın pikseli vardır**, buna karşılık $t$ gününde ortalama **12.3 piksel** yanmaktadır. Bir yangın genellikle bir gecede kendiliğinden sönmez. Genellikle olan şey, uydunun onu görememesidir: bulut, yoğun duman, elverişsiz bir geçiş zamanı ya da Terra veya Aqua tepeden geçtiği anda alevli değil için için yanan bir yangın.

Başka bir deyişle, **etiket, yangın davranışını ölçtüğü kadar uydu geçiş şansını da ölçmektedir.** Ağdan kısmen rastgele bir şeyi tahmin etmesi isteniyor ve hiçbir mimari rastgele bir hedefi öğrenemez.

**Düzeltme.** $t+2$ gününün yangın maskesini tutan ikinci bir hedef bandı, `fire_next2`, dışa aktarın ve eğitim hedefini bunların birleşimi olarak tanımlayın:

$$Y = \max\big(\text{fire\_next}(t{+}1),\ \text{fire\_next2}(t{+}2)\big)$$

Modelin artık yanıtladığı soru, "MODIS onu yarın yanarken yakalayacak mı?" yerine **"bu piksel önümüzdeki 24–48 saat içinde yangın etkinliği gösterecek mi?"** hâline gelir.

Bu değişiklik hakkında yapılması gereken iki dürüst tespit:

- **Bu bilimsel olarak savunulabilir bir yeniden formülasyondur, bir hile değil.** Operasyonel planlama açısından 48 saatlik bir pencere, tartışmasız 24 saatlikten *daha* faydalıdır. Ve ölçülmüş bir etiket gürültüsü kaynağını doğrudan hedef alır.
- **Problemi değiştirir, dolayısıyla öncesi ve sonrası sonuçlar karşılaştırılabilir değildir.** Bunu söylemek zorundasınız. Bu değişiklikten sonra F1'de görülen herhangi bir iyileşme kısmen daha kolay bir hedeften kaynaklanır ve temel çizgilere karşı karşılaştırma aynı tanım altında yeniden hesaplanmalıdır.

#### `valid` — uydurma sıfır maskesi

**Ölçülmüş sorun.** Her yamadaki piksellerin yaklaşık **%15'i sıfırdır — ve aynı ~%15 oranı her bir çevresel bantta ortaya çıkar.** Bu rastlantı ele veren şeydir: gerçek fiziksel alanlar sıfırlarının nerede olduğu konusunda anlaşmazlar. Nemin, yüksekliğin ve NDVI'nin aynı yerlerde sıfır olması için hiçbir neden yoktur.

Nedeni GEE kodundadır: `clip(REGION)` ardından `unmask(0)`. Türkiye ulusal sınırına kırpmak, sınır dışındaki her pikseli *maskeli* yapar; `unmask(0)` ise maskeli her pikseli 0 sayısıyla değiştirir. Yamalarımız 65 km genişliğinde olduğu ve Türkiye'deki yangınların çoğu kıyıya yakın olduğu için, her kıyı yamasının büyük bir kısmı denizdir ya da Yunanistan veya Suriye toprağıdır — hepsi sıfırla doldurulmuştur.

**Bu neden yıkıcıdır.** Modelin şu iki durumu ayırt etmesinin hiçbir yolu yoktur:

| Piksel anlamı | `humidity` değeri | `elevation` değeri | `ndvi` değeri |
|---|---|---|---|
| Gerçek: deniz seviyesinde yanmış bir ova üzerinde kupkuru hava | 0 | 0 | 0 |
| Uydurma: Türkiye dışı, hiç veri yok | 0 | 0 | 0 |

Sıfır bağıl nem fiziksel olarak neredeyse imkânsızdır. Sıfır yükseklik deniz seviyesi demektir. Sıfır NDVI çıplak kaya demektir. Ağa, her yamanın geniş ve sistematik olarak konumlanmış bir bölgesinin bu tuhaf özelliklere sahip olduğu öğretiliyor — daha da kötüsü, o bölgenin *şekli* kıyı şeridiyle ilişkilidir ve kıyı şeridi de yangınların nerede olduğuyla ilişkilidir. Bu ders kitabı örneği bir **sahte korelasyon (spurious correlation)** vakasıdır ve doğrulama AUC-PR'nin neden 7. devirde (epoch) zirve yaptığına dair makul bir açıklamadır: ağ kıyı şeridi şeklini neredeyse anında buldu ve onu ezberlemeye başladı.

**Düzeltme.** Açık bir `valid` bandı dışa aktarın: gerçek verinin bulunduğu yerde 1, dolgu yapılan yerde 0. Sonra:

1. `valid`'i 21. girdi kanalı olarak ekleyin, böylece ağa hangi piksellerin uydurma olduğu *söylenmiş* olur;
2. Kaybı `valid` ile çarpın, böylece uydurma bir pikselden asla gradyan hesaplanmaz;
3. Geçersiz pikselleri normalleştirme istatistiklerinin dışında tutun, böylece uydurma sıfırlar $\mu$ ve $\sigma$'yı bozmaz.

3. madde göründüğünden daha önemlidir. `humidity` değerlerinin %15'i uydurma bir 0 ise, hesaplanan ortalama aşağı çekilir ve hesaplanan standart sapma, doğada var olmayan iki modlu bir dağılım tarafından şişirilir.

#### 32×32 merkez kırpması

**Ölçülmüş sorun.** 65×65'lik bir yama 4.225 pikseldir; 64×64'lük bir kırpma 4.096'dır. Yangınlarımız en çok 65 piksel, ortalama ise **12.3** piksel kaplar. Yani sinyal görüntünün **%1.5'inin altındadır** ve tipik olarak yaklaşık %0.3'tür. Ağ, kapasitesinin neredeyse tamamını yangından 30 km uzaktaki boş araziye bakmakla harcar; ki bu arazi yarınki cepheyi hiçbir şekilde etkileyemez.

**Düzeltme.** Merkezdeki **32×32** pencereye kırpın, yani 32 km × 32 km'yi kaplayan 1.024 piksel. Yangın yapı gereği merkezdedir, dolayısıyla kırpma yangını korur ve uzak çevreyi atar.

$$\frac{4{,}096}{1{,}024} = 4.0 \Rightarrow \text{pozitif piksel yoğunluğunda kabaca } 4\times$$

İkincil faydalar: örnek başına 4× daha az hesaplama, dolayısıyla aynı bellek için 4× daha büyük yığınlar; ve artık yamayı rahatça aşan bir alıcı alan (receptive field), böylece her çıktı pikseli tüm sahneyi görür.

Dürüstçe ifade edilen bedel: uzun menzilli bağlamı kaybediyoruz. Yarınki yayılım 40 km uzaktaki bir şey tarafından yönlendiriliyorsa, artık onu göremeyiz. 24 saatlik yangın yayılım hızlarının günde birkaç kilometre mertebesinde olduğu göz önüne alındığında, 32 km'lik bağlam neredeyse kesinlikle yeterlidir — ama bu varsayılmamalı, deneysel olarak doğrulanmalıdır.

### 7.6 Normalleştirme: en önemli tek düzeltme

**Ölçülmüş sorun.** Mevcut pipeline **hiçbir girdi normalleştirmesi yapmıyor**. Ham kanalların ölçülmüş istatistikleri şöyle:

| Kanal | En küçük | En büyük | Standart sapma |
|---|---|---|---|
| `elevation` | −4 | 4.978 | **515.44** |
| `aspect` | 0 | 359 | **107.87** |
| `landcover` | 0 | 17 | **4.25** |
| `ndvi` | — | — | **0.20** |
| `soil_moisture` | — | — | **0.07** |

Orana bakın: $515.44 / 0.07 = 7{,}363$. Ham birimlerde yükseklik, toprak neminden **yedi bin kat daha fazla** değişiyor.

**Bu ilk katmanı neden yok eder.** Bölüm 5.7'den hatırlayın ki ilk evrişim şunu hesaplar:

$$z = \sum_{c=1}^{14}\sum_{m,n} w_{c,m,n}\, x_{c,m,n} + b$$

Tüm ağırlıklar aynı küçük rastgele dağılımdan ilklendirilir (Glorot/He ilklendirmesi), dolayısıyla eğitimin başında her kanalın ağırlıkları karşılaştırılabilirdir. Ama *girdiler* dört büyüklük mertebesi farklıdır. Yükseklik terimi ±500 × w civarında değerler katkılar; toprak nemi terimi ±0.07 × w civarında değerler katkılar. **Yükseklik ve bakı kanalları toplama tamamen hâkim olur; diğer on iki kanal sayısal olarak görünmezdir.**

$w_c$'ye göre gradyan $x_c$ ile orantılıdır, dolayısıyla yükseklik ağırlıkları da toprak nemi ağırlıklarından binlerce kat daha büyük gradyanlar alır. Bu nedenle gradyan inişi ilk devirlerini yükseklik ve bakı yolağını ayarlamakla geçirir ve nemi, rüzgârı ve NDVI'yi fiilen göz ardı eder.

Bu, büyük olasılıkla başarısızlığın baskın nedenidir. **Çoğunlukla yükseklik ve bakı gören bir yangın yayılım modeli, aslında statik bir arazi modelidir — ki bu, çözmediğimizi açıkça söylediğimiz duyarlılık problemidir.**

**Düzeltme: z-skoru normalleştirme (z-score normalisation)**, standardizasyon olarak da adlandırılır.

$$x' = \frac{x - \mu}{\sigma}$$

Bu dönüşümden sonra her kanalın ortalaması 0 ve standart sapması 1 olur; böylece on dördü de — yakında on bir sürekli olanı — ilk katmana eşit koşullarda girer.

**İstatistiklerin nereden gelmesi gerektiği.** $\mu$ ve $\sigma$ **yalnızca eğitim ayrımından** hesaplanmalı, sonra doğrulama ve teste değiştirilmeden uygulanmalıdır.

Neden? Çünkü $\mu$ ve $\sigma$, gradyan inişiyle öğrenilmeseler bile, öğrenilmiş parametrelerdir. Bunları tüm veri kümesi üzerinde hesaplarsanız, test yılları hakkındaki bilgi — ne kadar sıcak, ne kadar kuru oldukları, yangınların ne kadar dağlık yerlerde olduğu — her eğitim örneğine uygulanan dönüşüme akar. Bu **veri sızıntısıdır (data leakage)** (Bölüm 4.4) ve test skorunuz iyimser hâle gelir. Yamaları ayrım boyunca karıştırmaya kıyasla küçük bir sızıntıdır, ama gerçek bir sızıntıdır ve ciddi bir yayın ortamının hakemleri bunu soracaktır.

Yeni pipeline'da: $\mu, \sigma$'yı 2019–2023 eğitim yılları üzerinde, `valid = 0` olan pikselleri hariç tutarak hesaplayın ve çıkarımın (inference) tam olarak aynı sayıları kullanması için model kontrol noktasının yanında saklayın.

### 7.7 Dairesel değişken problemi: bakı

**Sorun.** Bakı (aspect), kuzeyden saat yönünde derece cinsinden ölçülür, 0–359. İki yamacı düşünün:

| Yamaç | Bakı | Fiziksel yön |
|---|---|---|
| A | 359° | Neredeyse tam kuzey |
| B | 1° | Neredeyse tam kuzey |

Fiziksel olarak aralarında **2°** vardır. Sayısal olarak aralarında **358** vardır — kuzey (0°) ile güney (180°) arasındakinden bile büyük bir fark. Yalnızca sayıları gören bir sinir ağı için, kuzeye bakan yamaçlar girdi aralığının zıt uçlarında iki ayrı nüfusa bölünür.

Daha kötüsü: 107.87'lik ham standart sapma, kanalın devasa bir sayısal yayılıma sahip olduğunu söyler; dolayısıyla yukarıda anlatılan normalleştirme *eksikliğinden* sonra bakı, ilk katmandaki ikinci en baskın kanaldır — ve sayısal yapısı anlamsızdır.

**Düzeltme: sinüs/kosinüs kodlaması.** Tek `aspect` kanalını iki kanalla değiştirin:

$$\text{aspect}_{\sin} = \sin\!\left(\frac{\pi \cdot \text{aspect}}{180}\right), \qquad \text{aspect}_{\cos} = \cos\!\left(\frac{\pi \cdot \text{aspect}}{180}\right)$$

Her ikisi de [−1, 1] aralığındadır, ki bu zaten iyi ölçeklenmiştir ve süreksizlik ortadan kalkar:

| Bakı | Yön | sin | cos |
|---|---|---|---|
| 0° | Kuzey | 0.000 | **1.000** |
| 1° | Kuzey | 0.017 | 1.000 |
| 90° | Doğu | **1.000** | 0.000 |
| 180° | Güney | 0.000 | **−1.000** |
| 270° | Batı | **−1.000** | 0.000 |
| 359° | Kuzey | −0.017 | 1.000 |

359° ile 1°'yi karşılaştırın: $(-0.017, 1.000)$ ile $(0.017, 1.000)$. Aralarındaki Öklid uzaklığı 0.035'tir — gerçekten küçük, tam da fiziğin gerektirdiği gibi. 0° ile 180°'yi karşılaştırın: $(0, 1)$ ile $(0, -1)$, uzaklık 2 — maksimum. Kodlama artık doğru geometriye sahiptir.

Ek bir fayda: iki bileşenin doğrudan fiziksel anlamı vardır. `aspect_cos` bir **kuzeylik** indeksidir (+1 = tamamen kuzeye bakan, −1 = tamamen güneye bakan) ve kuzey yarım kürede bu, ağın ihtiyaç duyduğu güneş maruziyeti / yakıt kuruluğu ekseninin neredeyse tam olarak kendisidir. `aspect_sin` bir doğuluk indeksidir ve sabah ile öğleden sonra ısınmasıyla etkileşir.

Aynı teknik her dairesel değişkene uygulanır: rüzgâr yönü, yılın günü, günün saati. IGNIS'te rüzgâr yönü zaten doğru ele alınmaktadır, çünkü bir yön açısı yerine $u$ ve $v$ taşıyoruz — ve $(u,v)$ zaten hızla ölçeklenmiş bir sinüs/kosinüs kodlamasıdır.

### 7.8 Kategorik değişken problemi: arazi örtüsü

**Sorun.** `landcover`, 1'den 17'ye kadar bir tam sayı olan MODIS IGBP sınıf kodunu tutar:

| Kod | IGBP sınıfı | Kod | IGBP sınıfı |
|---|---|---|---|
| 1 | Herdem yeşil iğne yapraklı orman | 10 | Otlaklar |
| 2 | Herdem yeşil geniş yapraklı orman | 11 | Kalıcı sulak alanlar |
| 3 | Yaprak döken iğne yapraklı orman | 12 | Ekili alanlar |
| 4 | Yaprak döken geniş yapraklı orman | 13 | Kentsel ve yapılaşmış |
| 5 | Karışık orman | 14 | Ekili alan / doğal bitki örtüsü mozaiği |
| 6 | Kapalı çalılıklar | 15 | Kalıcı kar ve buz |
| 7 | Açık çalılıklar | 16 | Çıplak alan |
| 8 | Odunsu savanlar | 17 | Su kütleleri |
| 9 | Savanlar | | |

Bunlar **etiketlerdir**, nicelik değil. Ama 17 tam sayısını bir evrişime beslediğinizde, aritmetik onu bir büyüklük olarak ele alır. Ağa örtük olarak şunlar söylenir:

- Su (17), herdem yeşil geniş yapraklı ormanın (2) "8,5 katıdır";
- Çıplak alan (16), kar (15) ile su (17) "arasındadır";
- Otlak (10) ile ekili alanların (12) ortalaması kalıcı sulak alanlardır (11).

**Bu ifadelerin her biri saçmalıktır.** Arazi örtüsü sınıfları üzerinde bir sıralama yoktur. Numaralandırma keyfîdir; NASA kodları herhangi bir sırada atayabilirdi ve fiziksel dünya değişmezdi.

**Düzeltme: bire-bir / tek-sıcak kodlama (one-hot encoding).** Tek tam sayı kanalını, tam olarak biri 1 olan $K$ ikili kanalla değiştirin:

```
   sınıf 5 (karışık orman), K = 6 grup:

   grup:       orman   çalı  otlak  tarım  yakıtsız  sulak
   kanal:    [   1  ,    0  ,   0  ,   0  ,    0   ,    0   ]
```

Artık sınıflar arasında hiçbir aritmetik ilişki ima edilmez. Her sınıf ilk katmanda kendi bağımsız ağırlığını alır; böylece ağ, çam ormanının çok yanıcı ve suyun yanıcı olmadığını, onları bir doğru üzerine yerleştirmeye zorlanmadan öğrenebilir.

**Neden 17 kanal yerine 6 grup?** İki neden. Birincisi, tutumluluk: bir yamanın çoğunda sabit kalan bir değişken için 17 ek kanal savurganlıktır ve nadir sınıfların (kalıcı kar, yaprak döken iğne yapraklı) Türkiye'de neredeyse hiç eğitim örneği olmazdı. İkincisi, fiziksel ilgililik: yangın için önemli olan **yakıt davranışıdır** ve birkaç IGBP sınıfı yakıt olarak aynı biçimde davranır. Bu nedenle yeni pipeline, 17 IGBP sınıfını **6 yakıt grubuna** indirger — kabaca kapalı orman, çalılık/maki, otlak ve savan, tarım arazisi, sulak alan ve su, kentsel ve çıplak alan gibi yakıtsız yüzeyler.

> *Tam IGBP-kodundan-yakıt-grubuna eşlemesi yeni pipeline'ın yapılandırmasında tanımlıdır ve makale yazılırken hafızadan değil koddan alıntılanmalıdır.*

**Yeni 21 kanallı girdi için kanal muhasebesi:**

| Grup | Kanallar | İşlem |
|---|---|---|
| Sürekli | `ndvi`, `lst`, `air_temp`, `humidity`, `wind_speed`, `wind_u`, `wind_v`, `precip`, `soil_moisture`, `elevation`, `slope` | 11 kanal, eğitim ayrımının $\mu, \sigma$ değerleriyle z-skoru normalleştirilmiş |
| Dairesel | `aspect` | 2 kanal: $\sin$, $\cos$ |
| Kategorik | `landcover` | 6 kanal: yakıt grubu tek-sıcak |
| İkili | `fire` (bugünün maskesi) | 1 kanal, zaten 0/1, normalleştirme yok |
| İkili | `valid` | 1 kanal, gerçek veri maskesi |
| | **Toplam** | **21** |

Eski girdiyle karşılaştırın: doğrusal muamele gören bir dairesel değişken ve sıralı muamele gören bir kategorik değişken dâhil, 14 ham ve normalleştirilmemiş kanal. Kanal sayısı %50 arttı; temsilin *kalitesi* çok daha fazla arttı ve parametre maliyeti 2.016 ek ağırlıktı (Bölüm 5.10).

### 7.9 Yön farkındalıklı veri artırma

**Veri artırma (data augmentation)**, doğru cevabı değiştirmeyen dönüşümler uygulayarak mevcut örneklerden yeni eğitim örnekleri üretir. 360 günden gelen yalnızca 22.426 yama ile veri artırma değerlidir.

Bariz dönüşümler yatay ve dikey aynalamalar ile 90° döndürmelerdir. **Ama verimiz vektörler içerir ve vektörler bir aynalamadan değişmeden çıkmaz.**

Yatay bir aynalamayı düşünün (sol–sağ, yani doğu–batı aynalaması):

```
   Aynalama ÖNCESİ                    Naif aynalama SONRASI (YANLIŞ)
   
   rüzgâr →→→  🔥░░░                   ░░░🔥  →→→ rüzgâr
               yangın doğuya yayılır    yangın sağda,
               (doğru fizik)            rüzgâr hâlâ doğuya esiyor
                                        → yangın rüzgâra KARŞI yayılıyor
                                        → fiziksel olarak imkânsız
```

Rüzgârı aynalamadan görüntüyü aynalamak, ağa yangınların rüzgâra karşı yayıldığını öğretir. Bu, hiç veri artırma yapmamaktan daha kötüdür: aktif olarak yanlış fizik enjekte eder.

**Doğru dönüşüm kuralları:**

| Dönüşüm | Görüntü | `wind_u` | `wind_v` | `aspect_sin` | `aspect_cos` |
|---|---|---|---|---|---|
| Yatay aynalama (D–B aynası) | sütunları çevir | **× (−1)** | değişmez | **× (−1)** | değişmez |
| Dikey aynalama (K–G aynası) | satırları çevir | değişmez | **× (−1)** | değişmez | **× (−1)** |

Gerekçesi:
- `wind_u` doğuya doğru olan bileşendir. Doğu ile batıyı aynalamak işaretini tersine çevirir.
- `wind_v` kuzeye doğru olan bileşendir. Kuzey ile güneyi aynalamak işaretini tersine çevirir.
- `aspect_sin` yamacın doğuluğudur. Bir D–B aynası onu tersine çevirir.
- `aspect_cos` yamacın kuzeyliğidir. Bir K–G aynası onu tersine çevirir.
- Skalerler — `elevation`, `slope`, `ndvi`, `humidity`, `wind_speed` — değişmez, çünkü yönleri yoktur.

**90° döndürmelerin daha zor olduğuna dikkat edin** ve çok dikkatli uygulanmadıkça bunlardan kaçınmak en iyisidir; çünkü 90°'lik bir döndürme $u$ ile $v$'yi uygun işaretlerle takas etmek *ve* bakı kodlamasını 90° döndürmek zorundadır, *ve* — ince bir biçimde — bakı ile güneş arasındaki ilişkiyi değiştirir; ki bu ilişki kuzey yarım kürede döndürme altında simetrik değildir. Aynalamalar güvenli seçimdir. Aynı güneş asimetrisi nedeniyle K–G aynalaması bile tartışmalı biçimde sorgulanabilir ve yalnızca D–B aynalamasını test eden bir ablasyon çalışması makul bir deney olurdu.

Bu, IAC'de dile getirilecek mükemmel bir ayrıntıdır. Verinizi standart bir bilgisayarlı görü tarifi uygulamak yerine fiziksel olarak anladığınızı gösterir. Yayımlanmış orman yangını derin öğrenme makalelerinin çoğu, rüzgâr kanallarına dokunmadan yamalarını aynalar.

### 7.10 TFRecord

**TFRecord**, TensorFlow'un ikili serileştirme biçimidir: her biri serileştirilmiş bir protokol tamponu tutan, uzunluk önekli ve CRC ile denetlenen kayıtlar dizisi. GEE doğrudan bu biçime dışa aktarabilir; IGNIS'in bunu kullanmasının nedeni budur.

Avantajları: kompakt, akışa uygun (veri kümesinin tamamını hiçbir zaman RAM'e yüklemezsiniz) ve sıkıştırılabilir — parçalarımız `.tfrecord.gz` biçimindedir.

Bizim bağlamımızdaki dezavantajı: bu bir TensorFlow biçimidir ve yeni pipeline PyTorch'tur. İki seçenek vardır — TFRecord'ları PyTorch'ta hafif bir okuyucu kütüphanesi aracılığıyla okumak ya da bir kereliğine `.npz`/`.npy` bellek eşlemeli diziler veya WebDataset parçaları gibi doğal bir biçime dönüştürmek. Veri kümesinin yalnızca 22.426 yama olduğu göz önüne alındığında, bellek eşlemeli NumPy dizilerine bir kerelik dönüşüm muhtemelen en basit ve yüklenmesi en hızlı seçenektir.

---
## 8. Modelin eğitimi ve GPU'lar

### 8.1 CPU'ya karşı GPU

| | **CPU** | **GPU** |
|---|---|---|
| Türkçesi | Merkezi işlem birimi | Grafik işlem birimi |
| Çekirdek sayısı | 8–32, çok karmaşık | binlerce, çok basit |
| Neye göre eniyilenmiş | Gecikme — tek bir işi hızlı bitirmek | İş hacmi — pek çok işi aynı anda bitirmek |
| Benzetme | Birkaç profesör | Aritmetik işlem yapan on bin öğrenci |

Bir CPU çekirdeğinde dal tahmini, sırasız yürütme ve büyük önbellekler bulunur; karmaşık ve öngörülemeyen kodu hızlı çalıştırmak için tasarlanmıştır. Bir GPU çekirdeğinde bunların neredeyse hiçbiri yoktur, ama binlercesi vardır ve hepsi aynı komutu farklı veriler üzerinde eşzamanlı olarak yürütür (SIMD/SIMT).

**Sinir ağı eğitimi neden GPU'ya bu kadar iyi uyuyor?** Çünkü özünde bir sinir ağının yaptığı her şey matris çarpımıdır ve matris çarpımı *utanç verici derecede paralel* bir işlemdir.

$$C_{ij} = \sum_k A_{ik} B_{kj}$$

Her bir çıktı elemanı $C_{ij}$, yalnızca $A$'nın bir satırına ve $B$'nin bir sütununa bağlıdır. Hiçbir çıktı bir başka çıktıya bağlı değildir. Dolayısıyla $M \times N$ elemanın tamamı, farklı çekirdekler tarafından, hiçbir eşgüdüme gerek kalmadan **eşzamanlı olarak** hesaplanabilir.

Evrişim (convolution) de bir matris çarpımıdır. Standart gerçekleştirim (`im2col`), her 3×3×14 alıcı alanı (receptive field) büyük bir matrisin bir sütununa açar; böylece evrişimin tamamı tek bir yoğun matris çarpımına dönüşür. Üçgen çizmek için üretilmiş donanımın derin öğrenme (deep learning) için ideal makine çıkmasının sebebi tam olarak budur.

### 8.2 CUDA'ya karşı ROCm

| | **CUDA** | **ROCm** |
|---|---|---|
| Üretici | NVIDIA | AMD |
| Açılımı | Compute Unified Device Architecture | Radeon Open Compute |
| Durumu | Tescilli, olgun, baskın | Açık kaynak, daha yeni, hızla gelişiyor |
| Çekirdek dili | CUDA C++ | HIP (CUDA C++ ile neredeyse aynı) |
| Derin öğrenme kütüphaneleri | cuDNN, cuBLAS | MIOpen, rocBLAS |

CUDA on yıldır fiilî standarttır; okuyacağınız hemen her öğreticinin NVIDIA donanımı varsaymasının sebebi budur. **ROCm**, AMD'nin açık kaynaklı eşdeğeridir. PyTorch'ta arayüz bilinçli olarak birebir aynıdır — bir ROCm derlemesinde de yine `device = 'cuda'` ve `tensor.cuda()` yazarsınız, çünkü PyTorch'un ROCm derlemesi bu çağrıları HIP üzerine eşler. Bu durum insanları şaşırttığı için bilinmeye değer.

### 8.3 Donanım: RX 9070 XT, gfx1201, RDNA 4

| Özellik | Değer |
|---|---|
| GPU | AMD Radeon RX 9070 XT |
| Yonga | **Navi 48** |
| Mimari | **RDNA 4** |
| Komut kümesi hedefi (ISA) | **gfx1201** |
| Bellek | 16 GB GDDR6 |
| İşletim sistemi | **Arch Linux** |
| Yazılım yığını | **ROCm + PyTorch (ROCm derlemesi)** |

**`gfx1201` nedir?** GPU'nun komut kümesi mimarisi için kullanılan LLVM hedef tanımlayıcısıdır. ROCm, çekirdekleri belirli gfx hedefleri için önceden derler ve sizin hedefinizi içermeyen bir derleme basitçe çalışmaz — bir şey ters gittiğinde sorulacak ilk sorunun "bu ROCm sürümü gfx1201'i destekliyor mu?" olmasının sebebi budur. RDNA 4 desteği görece yeni ROCm sürümlerinde geldi; bu nedenle **ROCm ve PyTorch sürümlerini `requirements.txt` içinde sabitlemek ve makalede kayda geçirmek, isteğe bağlı bir ayrıntı değil, yeniden üretilebilirliğin parçasıdır**.

**Proje neden TensorFlow'dan PyTorch'a geçti.** Belirtmeye değer üç pratik sebep:

1. **Projenin geliştirme ortamı Arch Linux'a taşındı ve tüm yığın PyTorch + ROCm üzerine kuruldu.** TensorFlow'un yerel Windows üzerinde 2.11 sürümünden itibaren — NVIDIA dâhil hiçbir üretici için — GPU desteği bulunmuyordu; ekibin gördüğü uyarı mesajı bir hata değil, doğru bir bilgiydi. Arch Linux üzerinde PyTorch'un ROCm derlemesi GPU'ya doğrudan erişim sağlar. <!-- güncellendi: PyTorch+ROCm -->
2. **`tensorflow-rocm`, ROCm sürümlerinin gerisinde kalır** ve RDNA 4 gibi çok yeni mimarilere desteği tarihsel olarak yavaş olmuştur. PyTorch'un ROCm derlemeleri yeni donanımı çok daha hızlı takip eder. <!-- güncellendi: PyTorch+ROCm -->
3. PyTorch'un anlık (eager) yürütmesi, özel bir kayıp fonksiyonunun (loss function), özel bir veri artırmanın (augmentation) ve özel bir örnekleyicinin ayıklanmasını çok daha kolaylaştırır — IGNIS'in üçüne de ihtiyacı var.

TensorFlow yeni işlem hattından tamamen çıkarılmıştır. <!-- güncellendi: PyTorch+ROCm -->

**Hazırda tutulması gereken dürüst bir kayıt:** U-Net küçüktür (≈1,9 M parametre, 32×32 yamalar). Bu proje için CPU üzerinde eğitim tamamen mümkündür — devir (epoch) başına birkaç dakika meselesi. GPU, günde daha çok deney yapmayı sağlayan bir kolaylıktır; bilimsel bir zorunluluk değildir. Bunu abartmayın.

### 8.4 Sayısal hassasiyet ve karma hassasiyet

Bir kayan noktalı sayı; bir işaret biti, bir üs (aralığı belirler) ve bir mantis (hassasiyeti belirler) olarak saklanır.

| Biçim | Bit | İşaret | Üs | Mantis | En büyük büyüklük | Bağıl hassasiyet |
|---|---|---|---|---|---|---|
| **float32** (fp32) | 32 | 1 | 8 | 23 | ~3,4 × 10³⁸ | ~7 ondalık basamak |
| **float16** (fp16) | 16 | 1 | 5 | 10 | ~6,5 × 10⁴ | ~3 ondalık basamak |
| **bfloat16** (bf16) | 16 | 1 | **8** | 7 | ~3,4 × 10³⁸ | ~2 ondalık basamak |

Üs sütununu dikkatle okuyun — bütün hikâye orada.

**bfloat16, float32'nin 8 üs bitini korur.** Dolayısıyla float32 ile tam olarak aynı *dinamik aralığa* sahiptir; sadece daha az anlamlı basamak saklar. **float16'nın yalnızca 5 üs biti vardır**, bu yüzden en büyük değeri yaklaşık 65.504, en küçük normal sayısı ise yaklaşık $6 \times 10^{-5}$'tir.

**Bu neden bf16'yı eğitim için sayısal olarak daha kararlı kılar.** Derin bir ağdaki gradyanlar çoğu zaman aşırı küçüktür — $10^{-7}$ ve altı sıradandır. float16'da böyle bir gradyan **alt taşma ile sıfırlanır** ve ilgili parametre öğrenmeyi durdurur. Tersine, talihsiz bir biçimde büyük bir aktivasyon **sonsuza taşabilir** ve bir kez NaN belirdiğinde tek bir adım içinde bütün ağa yayılır. fp16 için standart geçici çözüm **kayıp ölçekleme**'dir: geri yayılımdan (backpropagation) önce kaybı büyük bir sabitle çarpar, sonrasında gradyanları bölersiniz; böylece her şey fp16'nın dar penceresi içinde kalır. İşe yarar, ama kendisi de bozulabilecek fazladan bir hareketli parçadır.

bfloat16'nın bunların hiçbirine ihtiyacı yoktur. float32'de temsil edilebilen her sayı, daha az hassas biçimde de olsa bfloat16'da temsil edilebilir. Sinir ağı eğitiminde aralık, hassasiyetten çok daha önemlidir — bf16'nın benimsenmesinin ardındaki temel deneysel bulgu budur.

**Karma hassasiyet (mixed precision)**, hesaplama yükü ağır işlemleri (evrişimler, matris çarpımları) 16 bit ile çalıştırırken **ağırlıkların (weight) ve eniyileyici (optimiser) durumunun ana kopyasını float32'de tutmak** demektir. Böylece aritmetik iş hacmini kabaca 2 katına çıkarır, aktivasyon belleğini yarıya indirir ve yine de float32 kalitesinde parametre güncellemeleri elde edersiniz. Ana ağırlıkların fp32 kalması zorunludur, çünkü bir ağırlık güncellemesi ağırlığın kendisinden çok daha küçük mertebelerde olabilir; bf16'da $1{,}0$ değerine $10^{-6}$ eklemek hiçbir şey yapmaz.

**IGNIS için öneri:** RDNA 4 üzerinde bf16 karma hassasiyet kullanın; herhangi bir tuhaflık gözlerseniz başka hiçbir şeyi ayıklamadan önce tam fp32'ye geri dönün — model, hız maliyetini karşılayabilecek kadar küçüktür.

### 8.5 VRAM ve yığın boyutu

**VRAM** (video belleği), GPU'nun kendi belleğidir. Her şeyin oraya sığması gerekir: model parametreleri, eniyileyici durumu, aktivasyonlar ve güncel yığın (batch).

| Tüketici | Boyut | Not |
|---|---|---|
| Parametreler | 1,93 M × 4 B ≈ **7,7 MB** | fp32 |
| Adam eniyileyici durumu | 2 × parametre ≈ **15,4 MB** | Parametre başına $m$ ve $v$ |
| Gradyanlar | ≈ **7,7 MB** | |
| Girdi yığını (32 × 21 × 32 × 32, fp32) | ≈ **2,8 MB** | |
| Aktivasyonlar (tüm katmanlar, yığın 32) | onlarca MB | daha büyük modellerde baskın kalem |

Toplam: mevcut 16 GB'a karşılık 100 MB'ın çok altında. **VRAM, IGNIS için uzaktan yakından bir kısıt değildir.** Sorulursa bunu açıkça söyleyin — GPU'nun bir zorunluluk değil kolaylık olmasının bir başka sebebi de budur.

Yığın boyutuyla ölçeklenen kalem aktivasyon belleğidir ve bu ölçeklenme kabaca doğrusaldır:

$$\text{aktivasyon belleği} \propto \text{yığın} \times H \times W \times \text{katmanlar boyunca toplam kanal} \times \text{bayt}$$

32×32'lik kırpma, bunu 64×64'e kıyasla 4 kat azaltır; yani yığın boyutunu bedelsiz biçimde dört katına çıkarabilirsiniz — belirtmeye değer, işe yarar bir yan etki.

**Yığın boyutu yalnızca belleği değil, öğrenmeyi nasıl etkiler?**

| Yığın boyutu | Gradyan gürültüsü | Etkisi |
|---|---|---|
| Küçük (8–32) | Yüksek | Daha gürültülü güncellemeler, kötü minimumlardan daha iyi kaçış, çoğu zaman daha iyi genelleme |
| Büyük (256–1024) | Düşük | Daha pürüzsüz, devir başına daha hızlı, daha kötü genelleyen keskin minimumlara yakınsayabilir |

Yaygın bir kural, **doğrusal ölçekleme kuralı**'dır: yığın boyutunu $k$ ile çarparsanız öğrenme oranını (learning rate) da $k$ ile çarpın, çünkü her güncelleme artık $k$ kat daha çok veri üzerinden ortalama alıyordur. IGNIS, yığın 32 ve öğrenme oranı $10^{-3}$ kullanır; yığın 128'e geçerseniz öğrenme oranı olarak $4\times10^{-3}$ deneyin — kısa bir ısınma dönemiyle birlikte, çünkü büyük öğrenme oranları ilk birkaç yüz adımda kararsızdır.

### 8.6 Veri yükleme darboğazı

Pratikte asıl karşılaşacağınız sorun budur.

```
   ┌─────────┐  parça okuma  ┌──────────┐   sıkıştırma açma  ┌─────────┐  artırma  ┌─────┐
   │  DİSK   │──────────────►│  CPU     │───────────────────►│  CPU    │──────────►│ GPU │
   └─────────┘               └──────────┘                    └─────────┘           └─────┘
      YAVAŞ                     YAVAŞ                            YAVAŞ              HIZLI
```

CPU, yığınları GPU'nun tükettiği hızda hazırlayamıyorsa GPU boşta bekler. Pahalı bir GPU alıp, darboğaz tek bir CPU iş parçacığındaki gzip açma işlemi olduğu için %15 kullanım görmek tamamen mümkündür. Verimiz `.tfrecord.gz` olarak saklanıyor, dolayısıyla her bir yığın sıkıştırma açma gerektiriyor.

Burada da geçerli olan standart çareler:

| Teknik | Ne yapar |
|---|---|
| **Paralel işçiler** (`num_workers > 0`) | Birkaç CPU süreci yığınları eşzamanlı hazırlar |
| **Ön getirme** | GPU $n$. yığını hesaplarken $n+1$. yığın hazırlanır |
| **Sabitlenmiş bellek** (`pin_memory=True`) | Sayfa kilitli ana bellek, daha hızlı ve eşzamansız ana bilgisayar→cihaz aktarımı sağlar |
| **Önbellekleme / tek seferlik dönüştürme** | Bir kez açıp belleğe eşlenmiş bir `.npy` dizisine yazın; sonraki okumalar neredeyse anlıktır |
| **Veri artırmayı GPU'da yapmak** | Çevirmeler ve işaret değişimleri önemsiz tensör işlemleridir |

22.426 yamalık, 32×32×21 boyutunda ve float32 türünde bir veri kümesi için tüm arşiv yaklaşık $22{,}426 \times 21 \times 1024 \times 4 \approx 1,9$ GB'tır — **tamamını sistem belleğine, hatta 16 GB'lık VRAM'e yüklemeye yetecek kadar küçük.** Tek başına bu değişiklik veri yükleme darboğazını tümüyle ortadan kaldırır ve muhtemelen elde edilebilecek en yüksek değerli mühendislik iyileştirmesidir. Biri eğitim verimliliğini sorarsa bundan söz edin.

---

## 9. Sonuçları dürüstçe okumak

Bu bölümdeki her şey ölçülmüştür. Hiçbir şey tahmin değildir. Bir sayı henüz mevcut değilse öyle işaretlenmiştir.

### 9.1 Arşiv

| Özellik | Değer |
|---|---|
| Yangın günü (günlük parçalar) | 360 |
| Geçerli yama sayısı | 22.426 |
| Yama boyutu (dışa aktarılan / model girdisi) | 65 × 65 / 64 × 64 |
| Etiketli piksel | ≈ 91,9 milyon |
| Kapsanan yıllar | 2019, 2020, 2021 (2019 ve 2020 için Haziran–Ekim; 2021 için Haziran–Temmuz sonu) |
| Pozitif piksel yaygınlığı | **%0,2686** |
| $t$ gününde yama başına ortalama yangın pikseli | **12,3** |
| $t+1$ gününde sıfır yangın pikseli olan yamalar | **%58,9** |

### 9.2 Piksel düzeyindeki sonuçlar

Model: U-Net, ≈1,9 M parametre, odak kaybı (focal loss) ($\gamma = 2.0$, $\alpha = 0.80$), Adam öğrenme oranı $10^{-3}$, yığın 32; 22.426 yamanın tamamı üzerinde $\tau = 0.5$ eşiğiyle (threshold) değerlendirilmiştir.

| Ölçüt | Değer | Sade sözcüklerle anlamı |
|---|---|---|
| Pozitif yaygınlık (prevalence) | %0.2686 | Her 4.096 pikselden yaklaşık 11'i yarın yanıyor |
| **ROC-AUC** (ROC eğrisi altındaki alan) | **0.8468** | Yanan bir piksel ile yanmayan bir piksel verildiğinde, %84,7 oranında doğru sıralıyoruz |
| **AUC-PR** (kesinlik-duyarlılık eğrisi altındaki alan) | **0.0210** | 0.00269'luk rastgele temel çizginin (baseline) 7,8 katı — gerçek bilgi, kullanılamaz büyüklük |
| **Kesinlik (precision)** | **0.0601** | İşaretlediğimiz 100 pikselin 6'sı yanıyor; 94'ü yanlış alarm |
| **Duyarlılık (recall)** | **0.0222** | Yanan 100 pikselin 2'sini buluyoruz; 98'ini kaçırıyoruz |
| **F1** | **0.0324** | Dürüst birleşik skor |
| **IoU** (kesişimin birleşime oranı) | **0.0165** | Tahmin edilen ve gerçek yanık alanlar, birleşimlerinin %1,65'i kadar örtüşüyor |

İç tutarlılık denetimi: $\text{IoU} = F_1/(2-F_1) = 0.0324/1.9676 = 0.0165$. ✓ Değerlendirme kodu aritmetik olarak sağlam; sayılar sadece düşük.

### 9.3 Yama düzeyinde sınıflandırma ve %77 tuzağı

Karışıklık matrisi (confusion matrix); satırlar = gözlenen sınıf, sütunlar = tahmin edilen sınıf:

| Gözlenen \ Tahmin | Sönümlenen | Kararlı | Büyüyen | **Satır toplamı** |
|---|---|---|---|---|
| **Sönümlenen** | **17.071** | 62 | 300 | 17.433 |
| **Kararlı** | 2.099 | **64** | 119 | 2.282 |
| **Büyüyen** | 2.519 | 27 | **165** | 2.711 |
| **Sütun toplamı** | 21.689 | 153 | 584 | **22.426** |

Bunlardan türetilen sınıf başına skorlar:

| Sınıf | Kesinlik | Duyarlılık | F1 | Arşivdeki payı |
|---|---|---|---|---|
| Sönümlenen | 17.071 / 21.689 = 0.787 | 17.071 / 17.433 = **0.979** | 0.873 | **%77,74** |
| Kararlı | 64 / 153 = 0.418 | 64 / 2.282 = **0.028** | 0.053 | %10,18 |
| Büyüyen | 165 / 584 = 0.283 | 165 / 2.711 = **0.061** | 0.100 | %12,09 |
| **Genel doğruluk (accuracy)** | | | **0.7714** | |
| **Makro-F1** | | | **0.3418** | |

Şimdi kritik karşılaştırma.

| Sınıflandırıcı | Doğruluk |
|---|---|
| **Basit kural: her zaman "sönümlenen" de** | **0.7774** |
| **IGNIS U-Net + büyüme kuralı** | **0.7714** |
| **Fark** | **−0.0060 — model DAHA KÖTÜ** |

**Bu, "%77 doğruluk" tuzağıdır ve bunu otuz saniyede açıklayabilmelisiniz.**

> Yama sınıflandırıcımız %77,14 doğruluk alıyor. Kulağa saygın geliyor. Ama arşivimizdeki yamaların %77,74'ü sönümlenen sınıfına ait. Tek satırlık bir kod — `return "extinguishing"` — %77,74 alıyor. Sinir ağımız bunun **0,6 puan altında** kalıyor. Doğruluk rakamı, modelin herhangi bir becerisini değil, arşivin sınıf dağılımını yansıtıyor. **Makro-F1'i, yani 0.3418'i** de bu yüzden raporluyoruz ve bakılması gereken sayı makro-F1'dir: üç sınıfın F1 değerini eşit ağırlıkla ortalar, dolayısıyla model çoğunluk sınıfının (majority class) arkasına saklanamaz.

Karışıklık matrisi başarısızlığın tam olarak nasıl olduğunu gösteriyor. 22.426 tahminin **21.689'u — yani %96,7'si — "sönümlenen"**. Model esasen çoğunluk sınıfını öğrenmiş, neredeyse başka bir şey öğrenmemiştir. En çok tespit etmemiz gereken sınıf olan *büyüyen* için duyarlılığı **0.061**'dir: gerçekten büyüyen 2.711 yangından 165'ini bulmuştur.

**Burada neden makro-F1 doğru ölçüttür.** Makro-F1, sınıf başına F1 skorlarını sınıf büyüklüğüne göre ağırlıklandırmadan ortalar:

$$\text{makro-}F_1 = \frac{0.873 + 0.053 + 0.100}{3} = 0.3418$$

Her zaman "sönümlenen" diyen basit bir sınıflandırıcı $(0.875 + 0 + 0)/3 \approx 0.29$ alırdı. Yani 0.3418'lik makro-F1'imiz bu tabanın üzerindedir — gerçek ama çok küçük miktarda bir beceri. Bu karşılaştırmanın "%77"den ne kadar daha bilgilendirici olduğuna dikkat edin.

### 9.4 Kalıcılık karşılaştırması — en çok önem taşıyan sonuç

1.054 yama içeren 45 parçalık bir örneklem üzerinde ölçülmüştür:

| Yöntem | Kesinlik | Duyarlılık | F1 | IoU | Maliyet |
|---|---|---|---|---|---|
| **Kalıcılık (persistence)** — "yarın = bugün" | 0.0430 | **0.0963** | **0.0595** | **0.0306** | sıfır |
| **IGNIS U-Net**, $\tau = 0.5$ | **0.0601** | 0.0222 | 0.0324 | 0.0165 | 1,9 M parametre, GPU, 92 M etiketli piksel |

```
   F1 karşılaştırması

   Kalıcılık    ████████████████████████  0.0595
   IGNIS U-Net  █████████████             0.0324
                └──────────────────────────────
                0                          0.06
```

**Model, F1'de 1,84; IoU'da 1,85 kat farkla kalıcılığa yenilmektedir.**

Bu, değerlendirmenin tamamındaki en önemli tek cümledir ve nihai makalede yer almalıdır. "Yarın bugüne benzer" ifadesini yenemeyen bir model, tanımı gereği, hiçbir şey yapmamaya kıyasla hiçbir değer katmamıştır.

Yanına eklenmeye değer iki nüans var; çünkü ikisi de doğrudur ve ikisi de birer mazeret değildir:

1. **Model kesinlikte kazanıyor** (0.0601'e karşı 0.0430). Alarmları kalıcılığınkinden %40 daha güvenilir. Sorun, çok az alarm üretmesi. Bu, bir eğitim sorununun yanı sıra bir eşik sorununa da (Bölüm 6.8) işaret ediyor.
2. **Eşik hiçbir zaman kalibre edilmedi.** $\tau = 0.5$ bir varsayılandı, bir tercih değil. Duyarlılığın (0.0222) kesinlikten (0.0601) düşük olması, çok yüksek ayarlanmış bir eşiğin imzasıdır. F1'i, doğrulama kümesinde eniyi olan $\tau^\star$ değerinde yeniden hesaplamak sıfır maliyetli bir deneydir; yeniden eğitimden önce yapılmalıdır ve bu farkın bir kısmını pekâlâ kapatabilir.

### 9.5 Teşhis edilen yedi neden

Buradaki her madde tahmin edilmemiş, **ölçülmüştür**. Bu bölümü bir özür değil bilimsel bir katkı hâline getiren şey budur.

| # | Bulgu | Kanıt | Ciddiyet |
|---|---|---|---|
| **1** | **Model kalıcılığa yeniliyor** | F1 0.0324'e karşı 0.0595; IoU 0.0165'e karşı 0.0306 (45 dosya, 1.054 yama) | Kritik — yararlılık iddiasını geçersiz kılıyor |
| **2** | **Girdi normalleştirmesi yok** | `elevation` σ = 515.44 (aralık −4 ile 4.978), `aspect` σ = 107.87, `landcover` σ = 4.25 iken `soil_moisture` σ = 0.07, `ndvi` σ = 0.20. Oran 7.363:1 | Kritik — ilk evrişim katmanı fiilen yalnızca yükseklik ve bakıyı görüyor |
| **3** | **Her yamanın ≈%15'i sahte sıfır** | *Bütün* çevresel bantlarda aynı ~%15 sıfır oranı; sebebi, Türkiye sınırı dışındaki pikseller için `clip(REGION)` + `unmask(0)` | Yüksek — model "bağıl nem = %0" ile "veri yok" durumunu ayırt edemiyor; kıyı şeridi biçiminde sahte bir öznitelik (feature) yaratıyor |
| **4** | **Hedef, kısmen uydu geçişi şansını ölçüyor** | $t$ gününde ortalama 12,3 piksel yanarken yamaların %58,9'unda $t+1$ gününde sıfır yangın pikseli var | Yüksek — bulut, duman ve yörünge zamanlaması etiketin büyük kısmını belirliyor |
| **5** | **Yama, yangına göre çok büyük** | Yangınlar en fazla 65 piksel, ortalama 12,3 piksel kaplıyor; yama ise 4.225 piksel → sinyal %1,5'in altında | Orta — ağın kapasitesinin çoğu ilgisiz araziye harcanıyor |
| **6** | **2021 mega yangını eksik** | Arşiv 26 Temmuz 2021'de bitiyor; Manavgat ve Marmaris yangınları 28 Temmuz 2021'de başladı | Yüksek — yakın Türkiye tarihinin en önemli yayılım olayı eğitim verisinde yok |
| **7** | **Değerlendirme örneklem içiydi** | 360 parçanın tamamı, eğitim günleri dâhil, değerlendirildi | Kritik — yukarıdaki her sayı **iyimser** bir üst sınırdır |

7. bulgu vurguyu hak ediyor, çünkü tablonun tamamının okunuşunu tersine çeviriyor. Makale bunu zaten kabul ediyor: *"Değerlendirme, eğitim için kullanılan günleri de içeren 360 parçalık arşivin tamamı üzerinde yürütülmüştür… Bildirilen rakamlar bu nedenle örneklem içi tanılardır ve iyimser bir üst sınır olarak okunmalıdır."* Nitekim eğitim sırasında kaydedilen ayrık (held-out) sayılar **daha kötüydü**: en iyi devirde AUC-PR 0.0368, kesinlik 0.076, duyarlılık 0.072.

Dolayısıyla dürüst ifade şudur: **gerçek örneklem dışı başarım 0.0324'lük F1'den daha kötüdür ve kalıcılığa kıyasla daha da kötüdür.**

Bir jüriyi en çok şaşırtacak olan 6. bulgudur. Türkiye'nin modern dönemdeki en belirleyici orman yangını olayı — Temmuz–Ağustos 2021 Manavgat ve Marmaris yangınları — eğitim verisinde yoktur. Arşiv, 2021'de Haziran'dan *Temmuz sonuna* kadar uzanır ve 26 Temmuz'da durur; bu yangınlar 28 Temmuz'da başlamıştır. İki gün. Ülke kayıtlarındaki en uç yayılım davranışı ve arşiv ondan iki gün önce kesiliyor. Bu nedenle arşivi genişletmek yalnızca "daha çok veri" meselesi değildir — modelin en çok öğrenmesi gereken davranış rejimini dâhil etme meselesidir.

### 9.6 Yeni mimaride ne değişiyor ve her değişiklik neyi hedefliyor

| Değişiklik | Hedeflediği bulgu | Beklenen mekanizma |
|---|---|---|
| 11 sürekli kanalın z-skoru normalleştirmesi; istatistikler yalnızca eğitim bölünmesinden | **#2** | Tüm kanallar ilk katmana karşılaştırılabilir ölçeklerde giriyor |
| `aspect` → $\sin$/$\cos$ (2 kanal) | **#2** | Kuzeydeki sahte 358 birimlik süreksizliği kaldırıyor |
| `landcover` → 6 yakıt (fuel) grubuna tek-sıcak kodlama | **#2** | Sınıf kodları arasındaki sahte sıralamayı kaldırıyor |
| `valid` maske kanalı + kayıp maskeleme | **#3** | Ağa hangi piksellerin uydurma olduğu bildiriliyor ve onlardan gradyan gelmiyor |
| ±1 gün hedefi, $Y = \max(\text{fire\_next}, \text{fire\_next2})$ | **#4** | Kaçırılan geçişlerden kaynaklanan etiket gürültüsünü azaltıyor; problemi 24–48 saatlik etkinlik olarak yeniden çerçeveliyor |
| 32 × 32 merkez kırpma | **#5** | ≈4 kat pozitif yoğunluk, 4 kat daha az hesaplama, alıcı alan yamayı aşıyor |
| $0.5\,$BCE(pos_weight) $+ 0.5\,$SoftDice; alternatif FocalTversky($\alpha{=}0.3,\beta{=}0.7$) | sınıf dengesizliği | Kararlı gradyanlar artı bölge örtüşmesinin doğrudan eniyilenmesi; açık duyarlılık tercihi |
| Yıl temelli bölünme: eğitim 2019–2023, doğrulama 2024, test 2025–2026 | **#6, #7** | Gerçek bir ayrık test kümesi; arşiv Temmuz 2021'in ötesine genişletiliyor |
| Yön bilinçli veri artırma (çevirme ⇒ `wind_u`/`wind_v` işaretini ters çevir, bakı $\sin$/$\cos$ değerlerini ayarla) | küçük arşiv | Yanlış fizik enjekte etmeden daha çok örnek |
| Temel çizgiler: kalıcılık, genişletilmiş kalıcılık, rüzgâr yönlü büyüme | **#1** | Bildirilen her sayı, yenmesi gereken şeyin karşısına konuyor |
| Eşik $\tau$, F1'i en büyükleyecek biçimde doğrulama kümesinde kalibre ediliyor | eşik | Keyfî $\tau = 0.5$ değerini ortadan kaldırıyor |
| Arch Linux üzerinde PyTorch + ROCm, TensorFlow kaldırıldı | araç zinciri | RDNA 4'te GPU erişimi; özel kayıplar, örnekleyiciler ve veri artırma çok daha kolay <!-- güncellendi: PyTorch+ROCm --> |

**Yeni işlem hattının sonuçları: eğitim tamamlandığında doldurulacaktır.** Yeni model için hiçbir sayıyı, 2025–2026 test bölünmesi üzerinde, doğrulamayla kalibre edilmiş bir eşikle ölçülmeden ve üç temel çizginin yanında raporlanmadan alıntılamayın.

### 9.7 Dürüstlük neden bir güçtür

Kalıcılığa yenilen bir model sunmanın bir zayıflık olduğunu düşünebilirsiniz. Dört sebeple tam tersidir.

**1. Bir jüri şişirilmiş bir sayıyı fark edebilir; dürüst bir sayıyı fark edemez.** IAF Yeryüzü Gözlem sempozyumunda bölütleme (segmentation) üzerine çalışan herkes, %0,27 yaygınlıklı bir problemde 0.997 doğruluğun hiçbir şey ifade etmediğini bilir. Bunu manşet sonuç olarak raporlayan bir ekip tek bir slaytta bütün güvenilirliğini kaybeder. "Doğruluk %99,7 ve bu anlamsız; işte sebebi, işte AUC-PR ve onun yaygınlık temel çizgisi" diyen bir ekip ise güven kazanır.

**2. Ölçülmüş teşhisin kendisi katkıdır.** Herkes bir U-Net eğitebilir. Ama çok az ekip her girdi kanalının standart sapmasını ölçer, 7.363:1'lik bir ölçek oranı keşfeder, birbiriyle ilgisiz bantlarda aynı olan %15 sıfır oranını tek bir `unmask(0)` çağrısına kadar geri izler, hedeflerinin %58,9'unun boş olduğunu niceler ve bunların hepsini belirli bir başarısızlık kipine bağlar. Bu gerçek bir tanı çalışmasıdır ve bilimsel bir izleyicinin saygı duyduğu şey budur.

**3. Makalenin zaten taahhüt ettiği tutum budur.** Makale, çerçevenin *"henüz çoğunluk sınıfı temel çizgisini aşmadığını"* söylüyor ve durum notu şunu belirtiyor: *"Bu sonuçlar olduğu gibi, herhangi bir iyileştirme yapılmadan sunulmuştur."* Bu tercihi zaten yaptınız. Özür diler gibi değil, kendinden emin biçimde savunun.

**4. Bilim gerçekte böyle işler.** Doğru teşhis konmuş bir olumsuz sonuç bir alanı ilerletir; yeniden üretilemeyen olumlu bir sonuç ilerletmez. Makalenin kendi kapanış formülasyonu doğru olanıdır ve ezberlenmeye değer: *"bir sistemin henüz yapamadıklarının doğru bir dökümü, onu iyileştirmenin ön koşuludur."*

### 9.8 Hazırda tutulacak üç cümle

Bir jüri üyesi size otuz saniye verirse şunu söyleyin:

> *"IGNIS eksiksiz, yeniden üretilebilir, uçtan uca bir işlem hattıdır: sekiz uydu ve yeniden analiz ürünü, Google Earth Engine'de 1 km'lik ulusal bir ızgara üzerinde uyumlulaştırılmış; 22.426 yangın merkezli eğitim yaması ve piksel bazında ertesi gün yayılımını tahmin eden bir U-Net üretilmiştir.*
>
> *Ön modelimiz henüz çalışmıyor. Piksel F1'i 0.0324; kalıcılık temel çizgisi ise 0.0595. Yama doğruluğu %77,1 ve bu, %77,7'lik çoğunluk sınıfı tabanının altında. Bu sayıları raporluyoruz çünkü ölçtüğümüz şey bunlar.*
>
> *Arşivin kendisinden yedi belirli neden teşhis ettik — 7.363:1'lik kanal ölçek oranıyla girdi normalleştirmesinin yokluğu, bir maskeleme hatasından gelen %15 uydurma sıfır, kaçırılan uydu geçişlerinden kaynaklanan %58,9 boş hedef ve örneklem içi bir değerlendirme — ve yeniden inşa edilen işlem hattı bunların her birini doğrudan ele alıyor."*

Bu güçlü, savunulabilir ve dürüst bir konumdur; temel çizgisiz büyük bir sayıdan çok daha iyi bir cevaptır.

---

## 10. Size sorulabilecek sorular ve nasıl cevaplayacağınız

Bir jürinin muhtemelen ulaşacağı kabaca sıraya göre otuz soru. Her biri için: soruyu soranın gerçekte neyi sınadığı ve savunulabilir bir cevap. Bunları İngilizce yüksek sesle söyleyerek çalışın.

---

### S: Kalıcılık temel çizgisiyle karşılaştırma yaptınız mı?

*Sınanan: Bir yangın yayılım modelinin "yarın = bugün" kuralını yenmesi gerektiğini biliyor musunuz?*

Evet ve kaybediyoruz. Kalıcılık; kesinlik 0.0430, duyarlılık 0.0963, F1 0.0595 ve IoU 0.0306 veriyor. Bizim U-Net'imiz $\tau = 0.5$ eşiğinde 0.0601, 0.0222, 0.0324 ve 0.0165 veriyor. Kalıcılık F1'de 1,84 kat daha iyi. Bunu raporluyoruz, çünkü basit temel çizgiyi yenemeyen bir model değer kanıtlamamıştır. Yedi belirli neden teşhis ettik ve yeniden inşa edilen işlem hattı her birini ele alıyor; kalıcılık, genişletilmiş kalıcılık ve rüzgâr yönlü büyüme artık standart değerlendirme tablomuzun parçası.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "Yes, we did, and we lose. Persistence reaches an F1 of 0.0595 and an IoU of 0.0306, while our U-Net reaches only 0.0324 and 0.0165 — persistence is 1.84 times better in F1. We report this openly, because a model that cannot beat the trivial baseline has not yet demonstrated any value, and all three baselines are now part of our standard evaluation table."

---

### S: Yama doğruluğunuz %77. Bu iyi mi?

*Sınanan: Çoğunluk sınıfı tuzağını anlıyor musunuz?*

Hayır — yararsızdan da kötü ve bunu makalede de söylüyoruz. Sönümlenen sınıfı arşivin %77,74'ü; dolayısıyla her zaman "sönümlenen" diyen tek satırlık bir sınıflandırıcı %77,74 alır. Biz %77,14 alıyoruz, yani basit tabanın 0,6 puan *altında*. Makro-F1'i, yani 0.3418'i bu yüzden raporluyoruz ve karışıklık matrisi tahminlerimizin %96,7'sinin "sönümlenen" olduğunu bu yüzden gösteriyor. Doğruluk sayısı beceriyi değil, sınıf yaygınlığını yansıtıyor.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "No — it is actually worse than useless, and we say so in the paper. The extinguishing class makes up 77.74 % of our archive, so a one-line classifier that always predicts 'extinguishing' scores 77.74 %, while we score 77.14 % — six tenths of a point below that floor. That is why we report macro-F1, which is 0.3418, and why 96.7 % of our predictions turn out to be the majority class."

---

### S: ROC-AUC'niz 0.847 ama AUC-PR'niz 0.021. Bu bir çelişki değil mi?

*Sınanan: Aşırı dengesizlik altında ölçütleri anlıyor musunuz?*

İkisi de doğru; farklı şeyleri ölçüyorlar. ROC-AUC'nin yanlış pozitif oranının paydasında 91,6 milyon doğru negatif var; dolayısıyla 100.000 yanlış pozitif 0.0011'lik bir yanlış pozitif oranı olarak kaydediliyor — ROC eğrisinde görünmez. Kesinliğin içinde ise hiç doğru negatif yok, bu yüzden aynı 100.000 yanlış pozitif yıkıcı oluyor. ROC-AUC 0.8468 bize sıralamanın gerçekten bilgilendirici olduğunu söylüyor — yanan bir piksel, yanmayan bir pikselin %84,7 oranında üstünde sıralanıyor. AUC-PR 0.0210 ise bu sıralamayı kullanılabilir bir maskeye dönüştürmenin başarısız olduğunu söylüyor. %0,27 yaygınlıkta AUC-PR birincil ölçüttür ve daima kendi temel çizgisine, yani yaygınlığa eşit olan 0.00269'a karşı alıntılanmalıdır. Bizim 0.0210'umuz bu temel çizginin 7,8 katı — gerçek bilgi, kullanılamaz büyüklük.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "There is no contradiction; the two metrics answer different questions. ROC-AUC has 91.6 million true negatives in the denominator of its false positive rate, so a hundred thousand false alarms are almost invisible, whereas precision contains no true negatives at all and is destroyed by them. So ROC-AUC 0.8468 tells us the ranking carries real information, while AUC-PR 0.0210 — 7.8 times the prevalence baseline of 0.00269 — tells us that turning that ranking into a usable mask still fails."

---

### S: VIIRS 375 m sunuyorken neden 1 km çözünürlük?

*Sınanan: Çözünürlük tercihi bilinçli miydi?*

Çünkü 1 km, hem yordayıcımız hem de hedefimiz olan MODIS aktif yangın ürününün doğal çözünürlüğüdür ve yukarı örneklemeyle (upsampling) sahte ayrıntı üretmek istemedik. Ayrıca 1 km'deki MOD11A1 arazi yüzey sıcaklığıyla da uyuşuyor. Bununla birlikte haklısınız: 375 m'deki VIIRS benzer bir günlük tekrar ziyaret süresine sahip ve kesinlikle daha iyi olurdu — göremediğimiz yangın cephelerini çözebiliyor. 1 km'lik ızgaramız ulaşılabilir IoU'ya sert bir tavan koyuyor, çünkü bir yangın cephesinin gerçek bir piksellik ilerlemesi 1 km'lik bir hedef maskede çoğu zaman basitçe görünmez. VIIRS (`NOAA/VIIRS/001/VNP14A1`) makalede gelecek çalışma olarak açıkça adlandırılıyor.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "We chose 1 km because it is the native resolution of the MODIS active fire product, which is both our predictor and our target, and we did not want to fabricate detail through upsampling. You are right that VIIRS at 375 m has a comparable daily revisit and would be strictly better — our 1 km grid puts a hard ceiling on achievable IoU, because a genuine one-pixel advance of a fire front is often invisible at that scale. VIIRS is named explicitly in the paper as future work."

---

### S: ERA5-Land bir model çıktısı, gözlem değil. Bu bir sorun değil mi?

*Sınanan: Yeniden analizin (reanalysis) ne olduğunu anlıyor musunuz?*

Evet, bu gerçek bir sınırlılık ve bunu bilinçli olarak seçtik. ERA5-Land bir yeniden analizdir: sürümü dondurulmuş bir sayısal hava tahmini modelinin, mevcut tüm gözlemleri özümseyerek geçmiş tarihler üzerinde yeniden çalıştırılmasıdır. Bir tahmin değildir ve doğrudan bir ölçüm de değildir. Alternatif — seyrek yer istasyonlarını ara değerlemek — de bir modeldir ve çok daha kabadır; ayrıca yanan bir ormanın içinde istasyon yoktur. ERA5-Land bize ~9 km'de boşluksuz, fiziksel olarak tutarlı ve küresel ölçekte türdeş alanlar veriyor. Bilinen zayıflığı, 9 km'nin Toros Dağları'ndaki araziye bağlı rüzgârları temsil edememesi ve yangının kendi konvektif olarak indüklenmiş rüzgâr alanını hiçbir yeniden analizin bilememesidir. WRF gibi bir orta ölçek modeliyle dinamik ölçek küçültme doğru bir sonraki adım olurdu.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "Yes, and it is a limitation we accepted deliberately. ERA5-Land is a reanalysis — a frozen numerical weather model rerun over historical dates while assimilating all available observations — so it is neither a forecast nor a direct measurement. The alternative, interpolating sparse ground stations, is also a model and a far cruder one, and there are no weather stations inside a burning forest. The real weakness is that 9 km cannot resolve terrain-driven winds in the Taurus Mountains, and dynamical downscaling with a mesoscale model such as WRF would be the correct next step."

---

### S: Sınıf dengesizliğini nasıl ele aldınız?

*Sınanan: Bir tekniğin adını saymak değil, anlayışın derinliği.*

Ön model, $\gamma = 2.0$ ve $\alpha = 0.80$ ile odak kaybı kullandı. $(1-p_t)^\gamma$ çarpanı, kendinden emin biçimde doğru bilinen arka plan piksellerinin kaybını $p_t = 0.99$'da 10.000 kata kadar bastırır; böylece gradyan seyrek yangın pikselleri tarafından baskın hâle gelir. Dengesizliği yapısal olarak da ele alıyoruz: her yama etkin bir yangın pikseli merkez alınarak kesildiği için hiçbir örnek tamamen boş değil. Kendi vardığımız sonuç, %0,27 yaygınlıkta bunun yetersiz kaldığıdır. Yeniden inşa edilen işlem hattı melez bir kayıp kullanıyor: $0.5\,\text{BCE}(\text{pos\_weight}) + 0.5\,\text{SoftDice}$; alternatif olarak, kaçırılan yangınları yanlış alarmlardan bilinçli biçimde 2,33 kat daha ağır cezalandıran Focal Tversky ($\alpha = 0.3$, $\beta = 0.7$). Buna ek olarak pozitif yoğunluğu dört katına çıkaran 32×32 merkez kırpma ve $\tau = 0.5$ yerine doğrulamayla kalibre edilmiş bir eşik kullanılıyor.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "The preliminary model used focal loss with gamma 2.0 and alpha 0.80, which suppresses the loss of confidently-correct background pixels by up to ten thousand times, and we also centre every patch on an active fire pixel so that no sample is completely empty. Our measured conclusion is that this was still insufficient at 0.27 % prevalence. The rebuilt pipeline therefore uses a hybrid loss — half BCE with a positive weight, half soft Dice — with Focal Tversky as an alternative that penalises missed fires 2.33 times more than false alarms, plus a 32 by 32 centre crop that quadruples positive density."

---

### S: Veri sızıntısını nasıl önlediniz?

*Sınanan: Titizlik.*

Yama düzeyinde bölünmeyi asla tek tek yamalara göre değil, **bütün yangın gününe** göre yapıyoruz. Bir güne ait 150 yamanın tamamı aynı meteorolojiyi, aynı araziyi ve büyük ölçüde örtüşen ayak izlerini paylaşır; bunları bölünme boyunca karıştırmak, neredeyse aynı olan örnekleri her iki tarafa koyar ve doğrulama skorlarını şişirir. Yeni işlem hattı daha da katı: **yıl temelli bir bölünme** — eğitim 2019–2023, doğrulama 2024, test 2025–2026 — böylece bütün bir yangın mevsimi ayrı tutuluyor ve aynı yangın olayı iki tarafta birden görünemiyor. Ayrıca normalleştirme istatistikleri $\mu$ ve $\sigma$'yı yalnızca eğitim bölünmesinden hesaplıyoruz, çünkü bunları veri kümesinin tamamı üzerinden hesaplamak test kümesi bilgisini girdi dönüşümüne sızdırırdı.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "We split by whole fire day, never by individual patch, because all one hundred and fifty patches from a single day share the same meteorology, the same terrain and heavily overlapping footprints — shuffling them would put near-duplicates on both sides of the split. The new pipeline is stricter still: a year-based split with 2019 to 2023 for training, 2024 for validation and 2025 to 2026 for testing, so an entire fire season is held out. We also compute the normalisation statistics from the training split only, so that no test-set information leaks into the input transformation."

---

### S: Değerlendirmeniz örneklem içiydi. Bu sonuçlarınızı geçersiz kılmıyor mu?

*Sınanan: Kendi en zayıf noktanızı zaten bilip bilmediğiniz.*

Sonuçları iyimser bir üst sınır hâline getiriyor ve bunu makalenin 4.3 bölümünde açıkça belirtiyoruz. Bildirilen rakamlar, eğitim günleri dâhil 360 parçanın tamamı üzerinde hesaplandı. Eğitim sırasında kaydedilen ayrık sayılar daha kötüydü: en iyi devirde AUC-PR 0.0368, kesinlik 0.076, duyarlılık 0.072. Yani gerçek örneklem dışı başarım, zaten kötü olan raporladığımız sayıların altındadır. Bunu tam anlamıyla ayrık bir protokole düzeltmek yol haritamızdaki ilk maddedir ve yıl temelli bölünmeyle uygulanmaktadır.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "It makes them an optimistic upper bound, and we state that explicitly in Section 4.3 of the paper. The reported figures were computed over all 360 shards, including the training days, whereas the held-out numbers we logged during training were worse — AUC-PR 0.0368 at the best epoch, precision 0.076, recall 0.072. So the true out-of-sample performance is below the already poor numbers we report, and fixing this with a strictly held-out, year-based protocol is the first item on our roadmap."

---

### S: Neden ConvLSTM değil de U-Net?

*Sınanan: Mimari muhakeme.*

U-Net üç sebeple doğru ilk tercihti. Küçük veri kümelerinden yoğun tahmin için standart mimaridir; atlama bağlantıları (skip connection) bir yangın cephesinin ince sınır yapısını korur ki yangın 4.096 pikselin 12'sini kapladığında bu önemlidir; ve problemin kurgulanış biçimine uyar — tek bir günlük çevresel durumun ertesi günün maskesine eşlenmesi. Bir ConvLSTM ise günlerden oluşan bir *dizi* tüketir ve şu anda attığımız, gerçekten değerli bir fiziksel bilgi olan yayılım momentumunu öğrenebilirdi. Oradan başlamadık, çünkü yinelemeli bir modelin daha çok parametresi olur, daha çok veri gerektirir ve ayıklanması çok daha zordur — ve dürüst olmak gerekirse, tek kareli model kalıcılığı yenemiyorsa yinelemeyi eklemek yalnızca altta yatan veri sorunlarını gizlerdi. Uzamsal-zamansal mimariler makalede gelecek çalışma olarak adlandırılıyor.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "U-Net was the right first choice: it is the standard architecture for dense prediction from small datasets, its skip connections preserve the thin boundary of a fire front when the fire occupies only twelve pixels out of four thousand, and it matches the problem as we formulated it. A ConvLSTM would consume a sequence of days and could learn spread momentum, which is genuinely valuable information we currently discard. We did not start there because a recurrent model needs more data and is much harder to debug — and if the single-frame model cannot beat persistence, adding recurrence would only hide the underlying data problems."

---

### S: Yama sınıflandırıcınız çoğunluk sınıfını yeniyor mu?

*Sınanan: 2. sorudaki tuzağın aynısı, "evet" demenin cazip geleceği biçimde sorulmuş.*

Hayır. Çoğunluk sınıfı = %77,74, bizim modelimiz = %77,14. 0,6 puan altındayız. Makro-F1 0.3418; basit sınıflandırıcı için bu değer kabaca 0.29 olur, yani küçük miktarda gerçek bir beceri var; ama manşet doğruluk rakamı bunun kanıtı değil.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "No. The majority class is 77.74 % and our model reaches 77.14 %, so we are six tenths of a point below it. Our macro-F1 of 0.3418 against roughly 0.29 for the trivial classifier does indicate a small amount of genuine skill, but the headline accuracy figure is certainly not evidence of it."

---

### S: Bu sistem operasyonel olarak kullanılabilir mi?

*Sınanan: Dürüstlük ve muhakeme.*

Hayır, mevcut hâliyle kullanılamaz ve aksini iddia etmiyoruz — makale bunu "tamamlanmış bir operasyonel tahmin sistemi değil, metodolojik bir temel ve belgelenmiş bir temel çizgi" olarak tanımlıyor. Aritmetiği düşünün: Türkiye kabaca 780.000 km². Model piksellerin yalnızca %1'ini işaretlese bile bu günde 7.800 alarm demek; 0.0601 kesinlikte bunların yaklaşık 7.300'ü yanlış olurdu. Hiçbir itfaiye teşkilatı buna göre hareket edemez. Operasyonel olarak hazır olan şey işlem hattıdır: herhangi bir tarih aralığı, herhangi bir bölge, ham görüntüleri yerelde saklamaya gerek kalmadan Google Earth Engine'deki açık koleksiyonlardan yeniden üretilebilir.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "No, not in its current state, and we do not claim otherwise — the paper describes it as a methodological foundation and a documented baseline, not a finished operational system. Consider the arithmetic: Türkiye is about 780,000 square kilometres, so flagging even one percent of pixels would mean 7,800 alarms a day, and at a precision of 0.0601 roughly 7,300 of them would be false. What is operationally ready is the pipeline itself, which can regenerate any date range and any region from public collections in Google Earth Engine."

---

### S: Yaygınlığınız %0,27. Neden bugünkü yangının yumuşatılmış bir versiyonunu tahmin etmiyorsunuz?

*Sınanan: Bunun tam olarak genişletilmiş kalıcılık temel çizgisi olduğunu bilip bilmediğiniz.*

Bu tam olarak genişletilmiş kalıcılık temel çizgisidir ve artık değerlendirme planımızdaki üç temel çizgiden biridir. Ciddi bir rakiptir: düz kalıcılık zaten modelimizi yeniyor. Genişletilmiş kalıcılık onu daha da fazla yenerse, bu ulaşılabilir becerinin ne kadarının çevresel zorlama değil saf geometri olduğuna dair önemli bir bilgidir — ve öğrenmeye dayalı herhangi bir modelin bunun üzerine değer kattığı gösterilmelidir.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "That is exactly the dilated persistence baseline, and it is now one of the three baselines in our evaluation plan. It is a serious competitor, because plain persistence already beats our model. If dilated persistence beats it by a further margin, that tells us how much of the achievable skill is pure geometry rather than environmental forcing — and any learned model must then be shown to add value on top of it."

---

### S: Her bantta piksellerinizin yüzde on beşi sıfır. Açıklayın.

*Sınanan: Kendi verinizi inceleyip incelemediğiniz.*

Bu, kanal başına istatistikleri ölçerek bulduğumuz bir hataydı ve ipucu, ~%15'lik sıfır oranının birbiriyle ilgisiz bantlarda *tıpatıp aynı* olmasıydı — nemin, yüksekliğin ve NDVI'ın aynı yerlerde sıfır olması için fiziksel hiçbir sebep yok. Sebep, Earth Engine kodundaki `clip(REGION)` çağrısının ardından gelen `unmask(0)`: Türkiye ulusal sınırı dışındaki pikseller maskeleniyor ve ardından sıfır sayısıyla dolduruluyor. Yamalar 65 km genişliğinde olduğu ve Türkiye'deki yangınlar kıyıda kümelendiği için her kıyı yamasının büyük bir kısmı sıfırla doldurulmuş deniz veya yabancı ülke toprağı. Model, "bağıl nem %0" — fiziksel olarak neredeyse imkânsız — durumu ile "veri yok" durumunu ayırt edemiyor. Daha kötüsü, uydurulmuş bölgenin yangınların meydana geldiği yerlerle ilişkili bir kıyı şeridi biçimi var; bu, ağın ezberleyebileceği sahte bir özniteliktir. Çözüm, 21. girdi kanalı olarak eklenen, kaybı maskelemekte kullanılan ve normalleştirme istatistiklerinin dışında tutulan açık bir `valid` bandıdır.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "That was a bug we found by measuring per-channel statistics, and the giveaway was that the fifteen percent zero rate was identical across completely unrelated bands — humidity, elevation and NDVI have no physical reason to be zero in the same places. The cause is a clip to the national boundary followed by unmask with the value zero, so every coastal patch is partly sea or foreign territory filled with fabricated zeros. The model cannot distinguish 'relative humidity is zero percent' from 'no data', and the fabricated region even has a coastline shape that correlates with where fires occur. Our fix is an explicit valid band used as a 21st input channel and to mask the loss."

---

### S: Girdi verileriniz normalleştirildi mi?

*Sınanan: Uygulamalı makine öğrenmesindeki en temel soru.*

Ön modelde hayır — ve bu, teşhis ettiğimiz yedi nedenden biri. Ham kanal istatistiklerini ölçtük: `elevation` −4 ile 4.978 m aralığında 515.44 standart sapmaya, `aspect` 107.87'ye, `landcover` 4.25'e sahipken `soil_moisture` 0.07, `ndvi` ise 0.20'ye sahip. Bu 7.363:1'lik bir oran. Tüm ilk katman ağırlıkları karşılaştırılabilir büyüklüklerde ilklendiği için ilk evrişim fiilen yalnızca yüksekliği ve bakıyı görüyor ve bunların gradyanları binlerce kat daha büyük. Çoğunlukla araziyi gören bir yangın *yayılım* modeli, durağan bir *duyarlılık* modeline dejenere olmuştur — ki bu, çözmediğimizi söylediğimiz problemin ta kendisi. Yeniden inşa edilen işlem hattı, on bir sürekli kanalın tamamını yalnızca eğitim bölünmesinden alınan istatistiklerle z-skoru ile normalleştiriyor.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "In the preliminary model it was not, and that is one of our seven diagnosed causes. We measured the raw statistics: elevation has a standard deviation of 515.44 and aspect 107.87, while soil moisture has 0.07 and NDVI 0.20 — a ratio of 7,363 to 1. Since all first-layer weights are initialised at comparable magnitudes, the first convolution effectively sees only elevation and aspect, which means a fire spread model had quietly degenerated into a static susceptibility model. The rebuilt pipeline z-score normalises all eleven continuous channels using statistics from the training split only."

---

### S: Bakı 0–360 derece. Kuzeydeki başa sarma sorununu nasıl ele alıyorsunuz?

*Sınanan: Değişken türleri üzerine düşünüp düşünmediğiniz.*

Ön model bunu ele almadı ve bu, normalleştirme probleminin ikinci bileşeni. 359° ve 1° bakı, fiziksel olarak 2° uzaklıktadır ama sayısal olarak 358 birim uzaktadır; dolayısıyla kuzeye bakan yamaçlar girdi aralığının iki zıt ucunda iki topluluğa bölünmüştü. Çözüm bir sinüs/kosinüs kodlamasıdır: her ikisi de [−1, 1] aralığında olan $\sin(\pi a/180)$ ve $\cos(\pi a/180)$ olmak üzere iki kanal. Bu kodlamada 359° ile 1° arasındaki Öklit uzaklığı 0.035 iken 0° ile 180° arasındaki uzaklık 2'dir — doğru geometri budur. Ek kazanç olarak $\cos$, doğrudan bir "kuzeylik" indeksidir; kuzey yarım kürede bu, esasen güneşe maruz kalma ve yakıt kuruluğu eksenidir.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "The preliminary model did not handle it, and that is the second component of our normalisation problem: an aspect of 359 degrees and one of 1 degree are two degrees apart physically but 358 units apart numerically, so north-facing slopes were split into two populations at opposite ends of the input range. The fix is a sine-cosine encoding into two channels, under which 359 and 1 degrees sit at a Euclidean distance of 0.035 while 0 and 180 degrees sit at a distance of 2 — the correct geometry. As a bonus, the cosine channel is a direct northness index, which in the northern hemisphere is essentially the solar-exposure and fuel-dryness axis."

---

### S: Arazi örtüsü 1'den 17'ye kadar bir sınıf kodu. Bunu bir sayı olarak mı verdiniz?

*Sınanan: Kategorik değişkenlerin anlaşılması.*

Ön modelde evet — ve bu saçmalık ima ediyor: suyun (17) her dem yeşil geniş yapraklı ormanın (2) 8,5 katı olduğunu ve otlak (10) ile ekili alanın (12) ortalamasının kalıcı sulak alan (11) olduğunu. IGBP sınıf kodları üzerinde bir sıralama yoktur; NASA onları herhangi bir biçimde numaralandırabilirdi. Yeniden inşa edilen işlem hattı 17 IGBP sınıfını fiziksel olarak anlamlı 6 yakıt grubuna indiriyor ve bunları tek-sıcak kodluyor; böylece her grup bağımsız bir ağırlık alıyor ve sınıflar arasında hiçbir aritmetik ilişki ima edilmiyor.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "In the preliminary model, yes — and that implies nonsense: that water, class 17, is eight and a half times evergreen broadleaf forest, class 2, and that the average of grassland and cropland is permanent wetland. There is no ordering on IGBP class codes at all; NASA could have numbered them any way they liked. The rebuilt pipeline collapses the seventeen IGBP classes into six physically meaningful fuel groups and one-hot encodes them, so each group receives an independent weight."

---

### S: Yamalarınızın yüzde elli dokuzunda yarın yangın yok. Bir yangın bir gecede nasıl yok olur?

*Sınanan: Kendi etiket gürültünüzü anlamanız.*

Genellikle yok olmaz. Genellikle olan şey, uydunun onu görememesidir. MODIS *aktif yangın* raporlar — geçiş anında etkin biçimde alev alan bir piksel — yanmış alanı değil. Bulut, yoğun duman, elverişsiz bir geçiş saati veya alevlenmek yerine için için yanan bir yangın; hepsi sıfır üretir. Dolayısıyla hedefimizin kayda değer bir kısmı yangın davranışını değil, uydu geçiş şansını ölçüyor ve kısmen rastgele bir hedefi hiçbir mimari öğrenemez. Yanıtımız, ikinci bir hedef bandı dışa aktarmak ve etiketi $Y = \max(\text{fire}_{t+1}, \text{fire}_{t+2})$, yani "sonraki 24–48 saatlik yangın etkinliği" olarak tanımlamaktır. Bu, etiket gürültüsünü azaltıyor ve muhtemelen operasyonel olarak daha yararlı; problemi değiştirdiğini açıkça söylüyoruz, dolayısıyla öncesi ve sonrası doğrudan karşılaştırılabilir değil ve tüm temel çizgiler yeni tanım altında yeniden hesaplanmalı.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "Usually it does not vanish — usually the satellite simply failed to see it. MODIS reports active fire, meaning a pixel that is flaming at the exact moment of overpass, so cloud, thick smoke, an unfavourable overpass time or a smouldering fire all produce a zero. That means a substantial part of our target measures satellite overpass luck rather than fire behaviour, and no architecture can learn a partly random target. Our response is to redefine the label as the maximum of day t plus one and day t plus two — next 24 to 48 hour fire activity — and to recompute every baseline under that new definition."

---

### S: Neden düşük güvenilirlikli tespitleri de içeren FireMask ≥ 7 kullandınız?

*Sınanan: Eşiğin bir karar mı yoksa bir kaza mı olduğu.*

Yangın sınıfındaki duyarlılığı en büyüklemek için bilinçli olarak. FireMask 7, 8 ve 9 sırasıyla düşük, olağan ve yüksek güvenilirliktir. 9 şartı koşmak etiketleri arındırırdı ama zayıf ve erken evredeki yangınların çoğunu — yani operasyonel bir sistemin en çok ihtiyaç duyduğu yangınları — dışarıda bırakırdı. %0,27 yaygınlıkta pozitiflerimizin üçte ikisini atmayı göze alamayız. Bedeli, yanlış tespitlerden gelen ek etiket gürültüsüdür; bunu kabul ediyor ve açıkça belirtiyoruz.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "That was a deliberate decision to maximise recall on the fire class. FireMask values 7, 8 and 9 correspond to low, nominal and high confidence, and requiring 9 would purify the labels but discard most weak and early-stage fires — exactly the ones an operational system most needs to catch. At 0.27 % prevalence we simply cannot afford to throw away two thirds of our positives, so we accept the extra label noise and state it openly."

---

### S: Yamanız 65 km genişliğinde ama yangın 12 piksel. Yama fazlasıyla büyük değil mi?

*Sınanan: Yama boyutunun seçilmiş mi yoksa devralınmış mı olduğu.*

Evet — bu, teşhis ettiğimiz yedi nedenden biri. 64×64'lük bir yama 4.096 pikseldir; yangın bunların ortalama 12,3'ünü kaplar, dolayısıyla sinyal yaklaşık %0,3'tür ve hiçbir zaman yaklaşık %1,5'i aşmaz. Ağın kapasitesinin çoğu, yarının cephesini etkileyemeyecek olan, yangından 30 km uzaktaki araziye harcanır. Yeniden inşa edilen işlem hattı merkezdeki 32×32'ye, yani 1.024 piksele kırpıyor; bu, pozitif yoğunluğu dört katına çıkarıyor ve hesaplamayı 4 kat azaltıyor. Bedeli uzun menzilli bağlamın kaybıdır, ama 24 saatlik yayılım birkaç kilometre mertebesindedir, dolayısıyla 32 km'lik bağlam fazlasıyla yeterli olmalıdır — bunu varsaymak yerine doğrulayacağız.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "Yes, and it is one of our seven diagnosed causes. A 64 by 64 patch contains 4,096 pixels while the fire occupies on average only 12.3 of them, so the signal is about 0.3 percent and never more than one and a half percent — most of the network's capacity is spent on land thirty kilometres away that cannot influence tomorrow's fire front. The rebuilt pipeline crops to the central 32 by 32, which quadruples positive density and cuts computation fourfold, and since 24-hour spread is on the order of a few kilometres, that context should be ample — we will verify that rather than assume it."

---

### S: Arşiviniz Temmuz 2021'de bitiyor. 2021 Manavgat yangını eksik değil mi?

*Sınanan: Verinizin kapsamını bilip bilmediğiniz.*

Eksik ve bu, projedeki en can yakıcı boşluk. Arşivimiz 26 Temmuz 2021'de bitiyor; Manavgat ve Marmaris yangınları 28 Temmuz 2021'de başladı. İki gün. Yakın Türkiye tarihindeki en uç ve en ağır sonuçlu yayılım davranışı eğitim verisinde yok. Bu, "daha az verimiz var" demekten daha önemli: model, en çok öğrenmesi gereken rejimi hiç görmedi. Arşivi 2022–2026'ya genişletmek birinci veri önceliğimizdir ve yıl temelli bölünmeye çoktan yansıtılmıştır.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "It is, and it is the most painful gap in the project. Our archive ends on the 26th of July 2021 and the Manavgat and Marmaris fires began on the 28th — we miss them by two days. This matters far more than simply having less data, because the model has never seen the very behaviour regime it most needs to learn. Extending the archive to 2022 through 2026 is our first data priority and is already reflected in the year-based split."

---

### S: Modelinizin gerçek alıcı alanı nedir? Yeterince bağlam görebiliyor mu?

*Sınanan: Mimari anlayış.*

Üç kodlayıcı (encoder) düzeyi, düzey başına iki adet 3×3 evrişim ve aralarında 2×2 maksimum havuzlama ile alıcı alan 5 → 13 → 29 → 61 piksel olarak büyür. 1 km çözünürlükte bir darboğaz nöronu bu nedenle 61 km × 61 km'lik bir bölgeyi bütünler — esasen 64 km'lik yamanın tamamını. Mimari olarak ağ tüm bağlamı görebiliyor; 3 derinliği yamaya göre boyutlandırılmıştı. Yeni 32×32 kırpmayla alıcı alan yamayı rahatlıkla aşıyor ki küresel hava bağlamının önemli olduğu bir bölütleme görevinde istenen tam olarak budur.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "With three encoder levels, two 3 by 3 convolutions per level and 2 by 2 max-pooling between them, the receptive field grows from 5 to 13 to 29 to 61 pixels. At one kilometre resolution that means a bottleneck neuron integrates a 61 by 61 kilometre region — essentially the entire patch — so architecturally the network can already see the full context. With the new 32 by 32 crop the receptive field comfortably exceeds the patch, which is exactly what you want for a segmentation task where global weather context matters."

---

### S: Hangi girdi kanallarının gerçekten katkı sağladığını nereden biliyorsunuz?

*Sınanan: Ablasyon çalışması yapıp yapmadığınız.*

Henüz yapmadık ve bu yol haritasında — makale kanal ablasyonu deneylerini açıkça adlandırıyor. Yöntem, bir kanalı veya kanal grubunu çıkararak yeniden eğitmek ve ayrık bölünmedeki AUC-PR düşüşünü ölçmektir. Bizim için iki ablasyon özellikle bilgilendirici: rüzgâr kanallarını çıkarmak — model yönlü yayılımı öğrendiyse bu ciddi zarar vermelidir; ve yükseklik ile bakıyı çıkarmak — 7.363:1'lik ölçek oranı göz önüne alındığında mevcut modelin neredeyse tamamen bunlara dayandığından şüpheleniyoruz. Ablasyonu normalleştirme düzeltildikten sonra çalıştırmayı tercih ederiz, çünkü normalleştirilmemiş bir modeli ablasyona sokmak çoğunlukla normalleştirme hatasını ölçmek olurdu.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "We have not run the ablations yet, and the paper names channel-ablation experiments explicitly as future work. The method is to retrain with one channel or group removed and measure the drop in AUC-PR on the held-out split. Two ablations are especially informative for us: removing the wind channels, which should hurt badly if the model has learned directional spread, and removing elevation and aspect, which — given the 7,363 to 1 scale ratio — we suspect the current model relies on almost exclusively. We would rather run them after normalisation is fixed, since ablating an unnormalised model would mostly measure the normalisation bug."

---

### S: $\gamma = 2$ ve $\alpha = 0.8$ ile odak kaybı — neden bu değerler?

*Sınanan: Hiperparametrelerin ayarlanmış mı yoksa kopyalanmış mı olduğu.*

Bunlar, yoğun nesne tespiti için odak kaybını tanıtan makale olan Lin ve arkadaşlarının (2017) önerdiği değerlere yakındır — bizimkiyle aynı yapıya sahip bir problem. Bunları sistematik olarak ayarlamadık ve bu bir sınırlılıktır. Ölçülmüş sonucumuz, %0,27 yaygınlıkta tek başına odak kaybının yetersiz olduğudur; bu yüzden $\gamma$ ve $\alpha$'yı daha fazla aramak yerine kayıp ailesini değiştirdik — alternatifi Focal Tversky olan bir BCE+Dice melezine — ve önce veri temsilini değiştirdik, çünkü hiçbir kayıp fonksiyonu normalleştirilmemiş girdileri telafi edemez.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "They are close to the values recommended by Lin and colleagues in 2017, the paper that introduced focal loss for dense object detection — a problem with the same structure as ours. We did not tune them systematically, which is a limitation we acknowledge. Our measured conclusion was that focal loss alone is insufficient at 0.27 % prevalence, so instead of searching gamma and alpha further we changed loss family and, more importantly, changed the data representation first — because no loss function can compensate for unnormalised inputs."

---

### S: Eşiğiniz neden $\tau = 0.5$?

*Sınanan: 0.5'in keyfî olduğunu bilip bilmediğiniz.*

O bir karar değil, bir varsayılandı ve bu, saptadığımız bir hata. 0.5 yalnızca sınıflar dengeli olduğunda doğal kesme noktasıdır; bizimkiler 372'ye 1 dengesiz. Yanlış olduğunun kanıtı kendi sayılarımızda: duyarlılık (0.0222), kesinlikten (0.0601) düşük ki bu, çok yüksek ayarlanmış bir eşiğin imzasıdır. Doğru yordam — yeniden inşa edilen işlem hattında uygulanıyor — $\tau$'yu doğrulama kümesinde taramak, F1'i en büyükleyen değeri seçmek, onu dondurmak ve test kümesine değiştirmeden uygulamaktır. Bu, herhangi bir yeniden eğitimden önce çalıştırılması gereken sıfır maliyetli bir deneydir.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "It was a default, not a decision, and we identify it as a mistake. A threshold of 0.5 is the natural cut only when the classes are balanced, whereas ours are imbalanced 372 to 1, and the evidence is in our own numbers: recall of 0.0222 below precision of 0.0601 is the signature of a threshold set far too high. The correct procedure, now implemented, is to sweep the threshold on the validation set, pick the value that maximises F1, freeze it, and apply it unchanged to the test set — a zero-cost experiment that should be run before any retraining."

---

### S: Modelinizin 1,9 milyon parametresi ve 22.426 örneği var. Umutsuzca aşırı parametreli değil mi?

*Sınanan: Kapasiteye karşı veri anlayışı.*

Naif oran ürkütücü görünüyor, ama bölütleme için yanlış karşılaştırma bu: her örnek 4.096 piksel düzeyinde etiket taşıyor, yani arşiv kabaca 92 milyon etiketli çıktı barındırıyor. Asıl sorun, bu piksellerin bağımsız olmaması — bir yamadan gelen 4.096 piksel tek bir hava alanını, tek bir araziyi ve tek bir yangını paylaşıyor ve 150'ye kadar yama tek bir günü paylaşıyor. Etkin örneklem büyüklüğü 92 milyon piksele değil, 360 yangın gününe daha yakın. Ve aşırı öğrenmeyi (overfitting) gerçekten gözlüyoruz: eğitim AUC-PR'si 0.2375'e ulaşırken doğrulamada 0.0353 kaldı, yani 6,7 katlık bir uçurum; doğrulama tepe noktası 7. devirdeydi. Yanıtlarımız: daha çok veri (2022–2026), yön bilinçli veri artırma ve — her şeyden önce — girdi temsilini düzeltmek; çünkü kıyı şeridi biçimli sahte sıfır örüntüsünü ezberleyen bir ağ, onu ne kadar küçültürseniz küçültün ezberlemeye devam eder.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "The naive ratio looks alarming, but it is the wrong comparison for segmentation: each sample carries 4,096 pixel-level labels, so the archive holds roughly 92 million labelled outputs. The real problem is that those pixels are not independent — one patch shares a single weather field, a single terrain and a single fire, so our effective sample size is closer to 360 fire days than to 92 million pixels. And we do observe overfitting: training AUC-PR reached 0.2375 against 0.0353 on validation, a 6.7-fold gap, with the validation peak at epoch 7. Our answer is more data, direction-aware augmentation, and first of all fixing the input representation."

---

### S: Modeliniz neden duyarlılıkta kalıcılıktan kötü ama kesinlikte daha iyi?

*Sınanan: Kendi karışıklık matrisinizi okuyabilip okuyamadığınız.*

Çünkü model çok az alarm üretiyor. Kalıcılık, o an yanan her pikseli işaretler; bu ona 0.0963 duyarlılık kazandırır. Bizim modelimiz $\tau = 0.5$ eşiğinde o kadar az piksel işaretliyor ki duyarlılığı 0.0222 — 4,3 kat daha kötü — buna karşılık ürettiği alarmlar bir miktar daha güvenilir: kesinlik 0.0430'a karşı 0.0601. F1 bu ikisi arasındaki dengesizliği doğru biçimde cezalandırıyor. Bu örüntü tanısaldır: çok yüksek bir eşiğe ve pozitif sınıfı yetersiz ağırlıklandıran bir kayba işaret eder; yeniden inşa edilen işlem hattı her ikisini de ele alıyor.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "Because the model issues very few alarms at all. Persistence flags every currently burning pixel, which gives it a recall of 0.0963, whereas our model at a threshold of 0.5 flags so few pixels that its recall is only 0.0222 — 4.3 times worse — while the alarms it does issue are somewhat more reliable, with precision 0.0601 against 0.0430. F1 correctly punishes that imbalance, and the pattern is diagnostic: it points at a threshold that is too high and a loss that under-weights the positive class."

---

### S: Dört dilime yayılan bir ülke için neden UTM 35N dilimi?

*Sınanan: Coğrafi bilgi yetkinliği.*

Çünkü bir makine öğrenmesi veri kümesinin tek ve sürekli bir ızgaraya ihtiyacı vardır. 35N dilimi 24°–30° D'yi kapsar; Türkiye kabaca 26°–45° D arasında uzanır ve gerçekten 35'ten 38'e kadar dilimleri keser. Her yangın "kendi" dilimini kullansaydı, sınıra yakın yamalar farklı koordinat sistemlerinde olur ve ortak bir tensöre yığılamazdı. Tek bir ulusal ızgara seçtik ve doğuda daha büyük ölçek bozulmasını kabul ettik; arşivimize Ege ve batı Akdeniz yangınları hâkim olduğu için bu kabul edilebilir. IGNIS Doğu Anadolu'ya genişletilirse doğru çözüm ya bölge başına dilim ya da Türkiye merkezli eşit alanlı bir projeksiyondur.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "Because a machine-learning dataset needs one single continuous grid. Zone 35N covers 24 to 30 degrees east, while Türkiye spans roughly 26 to 45 and genuinely crosses zones 35 to 38 — if each fire used its own zone, patches near a boundary would sit in different coordinate systems and could not be stacked into a common tensor. We chose one national grid and accepted larger scale distortion in the east, which is acceptable because our archive is dominated by Aegean and western Mediterranean fires. If IGNIS is extended to eastern Anatolia, the correct fix is either a per-region zone or an equal-area projection centred on Türkiye."

---

### S: Yama sınıflandırıcınız 0.75 ve 1.25 eşiklerini kullanıyor. Bunlar nereden geliyor?

*Sınanan: Etiketleme kuralının ilkeli olup olmadığı.*

Bunlar büyüme oranı bantlarını tanımlar: $r = N_{t+1}/\max(N_t,1)$ olmak üzere $r > 1.25$ büyüyen, $0.75 \le r \le 1.25$ kararlı, $r < 0.75$ sönümlenen. Bunlar türetilmiş değil, seçilmiştir. Daha önceki bir sürüm 1.15 ve 0.85 kullanıyordu; bandı genişlettik, çünkü o ayarlarda kararlı sınıfı o kadar seyrekti ki fiilen öğrenilemez hâldeydi. Aynı kural gözlenen ve tahmin edilen maskelere birebir aynı biçimde uygulanıyor, dolayısıyla iki sınıf dizisi doğrudan karşılaştırılabilir. Bunları fiziksel sabitler olarak değil, operasyonel uzlaşımlar olarak sunarız; eşikler üzerinde bir duyarlılık analizi makaleyi güçlendirecektir.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "They define the growth-ratio bands: above 1.25 is growing, between 0.75 and 1.25 is stable, and below 0.75 is extinguishing. They were chosen rather than derived — an earlier version used 0.85 and 1.15, and we widened the band because at those settings the stable class was so rare that it was effectively unlearnable. The same rule is applied identically to the observed and the predicted masks, so the two class sequences remain directly comparable, and we present these as operational conventions rather than physical constants."

---

### S: Neden TensorFlow'dan PyTorch'a geçtiniz?

*Sınanan: Mühendislik muhakemesi.*

Üç sebep. Birincisi, geliştirme ortamımızı Arch Linux'a taşıdık ve yığını PyTorch + ROCm üzerine kurduk; TensorFlow'un yerel Windows üzerinde 2.11 sürümünden bu yana hiçbir üretici için GPU desteği yoktu. <!-- güncellendi: PyTorch+ROCm --> İkincisi, `tensorflow-rocm` ROCm sürümlerinin gerisinde kalıyor ve donanımımız çok yeni bir kart: AMD RX 9070 XT — Navi 48, RDNA 4, gfx1201; PyTorch'un ROCm derlemeleri yeni mimarileri çok daha hızlı takip ediyor. Üçüncüsü, PyTorch'un anlık yürütmesi özel bir maskelenmiş kaybı, yön bilinçli bir veri artırmayı ve tabakalı bir örnekleyiciyi gerçekleştirmeyi ve ayıklamayı çok daha kolaylaştırıyor; IGNIS'in üçüne de ihtiyacı var. Şunu da eklemeliyiz: model, CPU üzerinde eğitimi tamamen uygulanabilir kılacak kadar küçük; GPU bize başka türlü elde edemeyeceğimiz bir sonuç değil, günde daha çok deney kazandırıyor.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "There were three reasons. We moved our development environment to Arch Linux and built the whole stack on PyTorch with ROCm, since TensorFlow has had no GPU support on native Windows since version 2.11 for any vendor, and tensorflow-rocm consistently lags behind ROCm releases — which matters because our hardware is an AMD RX 9070 XT, a Navi 48 RDNA 4 card with the gfx1201 target, and PyTorch's ROCm builds track new architectures much more quickly. PyTorch's eager execution also makes it far easier to implement and debug a custom masked loss, a direction-aware augmentation and a stratified sampler, all three of which IGNIS needs. We should add that the model is small enough that CPU training is entirely viable — the GPU buys us more experiments per day, not a result we could not otherwise obtain."

---

### S: Değiştireceğiniz tek bir en önemli şey nedir?

*Sınanan: Muhakeme ve öncelikler.*

Girdi normalleştirmesi ve onunla birlikte `valid` maskesi. Bu ikisi en ucuz değişikliklerdir ve en temel soruna dokunurlar: şu anda ilk evrişim katmanı fiilen yalnızca yüksekliği ve bakıyı görüyor ve gördüğünün yaklaşık %15'i uydurma. Diğer her iyileştirme — daha iyi bir kayıp, daha çok veri, yinelemeli bir mimari — temsilin üzerine inşa edilir ve temeli düzeltmeden çatıyı iyileştirmek bize hiçbir şey söylemez. Ondan sonra sırasıyla: tam anlamıyla ayrık bir değerlendirme, eşik kalibrasyonu, 2021 mega yangınını da içeren 2022–2026 arşiv genişletmesi ve ancak ondan sonra mimari değişiklikler.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "Input normalisation, together with the valid mask. Those two are the cheapest changes and they address the most fundamental problem, because right now the first convolutional layer effectively sees only elevation and aspect, and about fifteen percent of what it sees is fabricated. Every other improvement — a better loss, more data, a recurrent architecture — is built on top of that representation, and improving the roof before fixing the foundation would tell us nothing. After that, in order: a strictly held-out evaluation, threshold calibration, extending the archive to 2022 through 2026 including the 2021 mega-fire, and only then architectural changes."

---

### Rahat bir cevabı olmayan, hazırlıklı olunması gereken üç soru

---

### S: Modeliniz çalışmıyorsa, gerçekte ne katkı sağladınız?

Sekiz açık Yeryüzü Gözlem ürününden eğitilmiş bir bölütleme modeline uzanan, herhangi bir tarih aralığı veya bölge için ham görüntüleri yerelde saklamadan yeniden üretilebilen, eksiksiz ve açık bir işlem hattı; Türkiye için özel olarak inşa edilmiş ilk böylesi veri kümesi; bir olasılık alanını operasyonel bir ifadeye dönüştüren iki düzeyli bir çıktı tasarımı; ve standart bir mimarinin bu probleme naif biçimde uygulanmasının neden başarısız olduğuna dair ölçülmüş, nicel bir teşhis. En hararetle savunacağımız madde sonuncusudur. Yayımlanmış pek çok orman yangını derin öğrenme makalesi, yaygınlığı belirtmeden veya kalıcılıkla karşılaştırmadan manşet doğruluk değerleri raporluyor. Biz her ikisini de ölçtük, kaybettiğimizi gördük ve bunu yayımladık.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "A complete, reproducible, open pipeline from eight public Earth Observation products to a trained segmentation model, regenerable for any date range or region without storing raw imagery locally — and the first such dataset built specifically for Türkiye. Alongside that, a two-level output design that converts a probability field into an operational statement, and a measured, quantitative diagnosis of why the naive application of a standard architecture to this problem fails. That last item is the one we would defend hardest: many published wildfire deep-learning papers report headline accuracies without stating prevalence or comparing to persistence. We measured both, found that we lose, and published it."

---

### S: Uluslararası bir kongrede olumsuz bir sonuç sunmaktan utanmıyor musunuz?

Hayır. Doğru ve ölçülmüş bir teşhise sahip olumsuz bir sonuç bir katkıdır; yeniden üretilemeyen olumlu bir sonuç değildir. Raporladığımız her sayı tam olarak ölçtüğümüz şeydir — değerlendirmemizin örneklem içi ve dolayısıyla iyimser olduğu gerçeği dâhil. Daha büyük sayıya sahip ekip olmaktansa, salondaki bir uzmanın güvendiği ekip olmayı tercih ederiz.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "Not at all. A negative result with a correct, measured diagnosis is a contribution; an unreproducible positive result is not. Every number we report is exactly what we measured, including the fact that our evaluation was in-sample and therefore optimistic. We would rather be the team that a specialist in the audience trusts than the team with the larger number."

---

### S: Siz lise öğrencisisiniz. Bunun ne kadarını gerçekten siz yaptınız?

Veri kümesi tasarımını, Earth Engine işlem hattını, modeli, değerlendirmeyi, teşhisi ve makaleyi. Özellikle teşhis — kanal başına standart sapmaları ölçmek, birbirinin aynı olan %15'lik sıfır oranını tek bir `unmask(0)` çağrısına kadar geri izlemek, hedeflerin %58,9'unun boş olduğunu nicelemek ve kalıcılık temel çizgisini hesaplamak — kendi arşivimiz üzerinde kendi başımıza yaptığımız bir çalışmadır ve sayıların herhangi birini adım adım sizinle birlikte gözden geçirebiliriz.

> **Sunumda söyleyebileceğiniz kalıp (EN):**
> "The dataset design, the Earth Engine pipeline, the model, the evaluation, the diagnosis and the manuscript. The diagnosis in particular — measuring per-channel standard deviations, tracing an identical 15 % zero rate back to a single unmask call, quantifying that 58.9 % of our targets are empty, and computing the persistence baseline — is work we did ourselves on our own archive. We would be happy to walk you through any of those numbers in detail."

---
## 11. Sözlük

Türkçe terim | İngilizce terim | Açıklama. Türk alfabesi sırasına göre, Türkçe terime göre alfabetik olarak sıralanmıştır.

| Türkçe | İngilizce | Açıklama |
|---|---|---|
| **Ablasyon çalışması (bileşen çıkarma analizi)** | Ablation study | Bir bileşen ya da girdi kanalı çıkarılarak modelin yeniden eğitilmesi; böylece o bileşenin ne kadar katkı sağladığı ölçülür. IGNIS, rüzgârın ağ için gerçekten önemli olup olmadığını sınamak amacıyla kanal ablasyonları planlamaktadır. |
| **Adam** | Adam | Gradyan ortalaması ve varyansının hareketli tahminlerinden yararlanarak her parametre için uyarlanabilir öğrenme oranları tutan bir eniyileyici. IGNIS, Adam'ı $10^{-3}$ öğrenme oranıyla kullanır. |
| **AdamW** | AdamW | Ayrıştırılmış ağırlık azaltmasına sahip Adam; L2 düzenlileştirmesini uyarlanabilir eniyileyicilerle birleştirmenin matematiksel olarak doğru yolu. |
| **Adım** | Stride | Bir çekirdeğin konumlar arasında ne kadar ilerlediği. IGNIS evrişimlerde 1, havuzlama ve ters evrişimlerde 2 adım kullanır. |
| **Ağırlık** | Weight | Bir nöron ya da çekirdek içinde bir girdiye uygulanan, öğrenilmiş çarpan. |
| **Ağırlık azaltma** | Weight decay | Aşırı öğrenmeyi azaltmak için büyük ağırlıkların cezalandırılması; L2 düzenlileştirmesine eşdeğerdir. |
| **Ağırlık paylaşımı** | Weight sharing | Aynı çekirdeğin her uzamsal konumda kullanılması. Yoğun bir katmanın 57 milyon parametreye ihtiyaç duyacağı yerde bir evrişimin 4,064 parametreyle yetinmesinin nedeni budur. |
| **Aktif algılama** | Active sensing | Cihazın kendi sinyalini yaydığı uzaktan algılama (radar, LiDAR). SRTM aktif radar kullanmıştır. |
| **Aktif yangın** | Active fire | Uydu geçişi anında etkin biçimde alev almış olarak tespit edilen piksel. Yanmış alanla aynı şey değildir. |
| **Aktivasyon fonksiyonu** | Activation function | Bir katmanın ağırlıklı toplamından sonra uygulanan doğrusal olmayan fonksiyon. Bu olmadan üst üste yığılmış katmanlar tek bir doğrusal dönüşüme indirgenir. |
| **Alçak Dünya yörüngesi** | Low Earth Orbit (LEO) | Kabaca 160 ile 2,000 km arasındaki yörüngeler. Terra ve Aqua bu yörüngede uçar. |
| **Alıcı alan** | Receptive field | Özgün girdinin, tek bir çıktı değerini etkileyen bölgesi. IGNIS'in darboğazı: 61×61 piksel = 61 km. |
| **Anlamsal bölütleme** | Semantic segmentation | Bir görüntünün her pikseline bir sınıf etiketi atanması. IGNIS'in temel problem kurgusu. |
| **Arazi örtüsü** | Land cover | Yeryüzünü kaplayan fiziksel malzeme. IGNIS, yakıt türü vekili olarak MODIS MCD12Q1 IGBP sınıflarını kullanır. |
| **Arazi yüzey sıcaklığı** | Land surface temperature (LST) | Termal kızılötesinden elde edilen, yerin radyometrik yüzey sıcaklığı. 2 m hava sıcaklığından farklıdır; bu fark kimi zaman 20 °C'ye ulaşır. |
| **Aşırı öğrenme (ezberleme)** | Overfitting | Eğitim verisinde iyi, görülmemiş veride kötü başarım. IGNIS, eğitim ile doğrulama AUC-PR değerleri arasında 6.7× fark göstermektedir. |
| **Aşırı örnekleme** | Oversampling | Sınıfları dengelemek için azınlık sınıfı örneklerinin çoğaltılması. |
| **Atlama bağlantısı** | Skip connection | Bir kodlayıcı öznitelik haritasının doğrudan eşleşen kod çözücü düzeyine kopyalanması. Havuzlamanın yok ettiği ince uzamsal ayrıntıyı geri getirir — ince bir yangın cephesi için hayati önemdedir. |
| **Atmosferik düzeltme** | Atmospheric correction | Yüzey yansıtırlığını geri elde etmek için uydu ölçümünden atmosferin etkisinin çıkarılması. IGNIS'in kullandığı ürünlerde hâlihazırda uygulanmıştır. |
| **Bakı** | Aspect | Bir yamacın baktığı pusula yönü; kuzeyden saat yönünde 0–360°. Dairesel bir değişkendir; IGNIS'in onu sinüs ve kosinüs olarak kodlamasının nedeni budur. |
| **Bant** | Band | Bir rasterdaki tek bir ölçülmüş nicelik; örneğin bir dalga boyu aralığı ya da türetilmiş bir değişken. |
| **bfloat16** | bfloat16 | float32'nin 8 üs bitine ve yalnızca 7 mantis bitine sahip 16 bitlik kayan noktalı sayı. Dinamik aralığı float32 ile aynıdır, bu nedenle kayıp ölçeklemesine gerek duymaz — eğitimde float16'dan sayısal olarak daha kararlıdır. |
| **Bire-bir kodlama (tek-sıcak kodlama)** | One-hot encoding | Kategorik bir değişkenin, tam olarak biri 1 olan $K$ ikili kanalla temsil edilmesi. Sınıf kodları arasındaki yanlış sıralamayı ortadan kaldırır. |
| **Boş model** | Null model | Önemsiz kestirici; örneğin daima "yangın yok" demek. IGNIS'te %99.73 piksel doğruluğu ve %77.74 yama doğruluğu elde eder. |
| **CHIRPS** | CHIRPS | Climate Hazards Group InfraRed Precipitation with Station data: uydu ve yer istasyonu ölçümlerini harmanlayan, ~5 km çözünürlüklü günlük yağış veri kümesi. |
| **CUDA** | CUDA | NVIDIA'nın tescilli GPU hesaplama platformu. Fiilî standarttır; öğreticilerin çoğunun NVIDIA donanımı varsaymasının nedeni budur. |
| **Çekirdek (filtre)** | Kernel / filter | Bir evrişimde girdi üzerinde kaydırılan, öğrenilmiş ağırlıklardan oluşan küçük matris. IGNIS baştan sona 3×3 çekirdek kullanır. |
| **Çıkarım** | Inference | Eğitilmiş bir modelin yeni veri üzerinde çalıştırılarak kestirim üretilmesi. |
| **Çift doğrusal ara değerleme** | Bilinear interpolation | En yakın dört kaynak pikselin ağırlıklı ortalaması alınarak yeniden örnekleme. Yalnızca sürekli veriler içindir; sınıf kodları için asla kullanılmaz. |
| **Çoğunluk sınıfı** | Majority class | En sık görülen sınıf. IGNIS'in yama görevinde %77.74 ile "sönme". |
| **Dairesel değişken** | Circular variable | En büyük ve en küçük değerlerin fiziksel olarak komşu olduğu değişken; örneğin 359° ve 1°'deki bakı. Sinüs/kosinüs kodlaması gerektirir. |
| **Darboğaz** | Bottleneck | Bir U-Net'in en derin, en sıkıştırılmış katmanı. IGNIS'te 8×8×256 boyutundadır ve parametrelerin %46'sını barındırır. |
| **Datum** | Datum | Referans elipsoidi ve onun konumlandırılması; bir koordinat referans sisteminin parçasıdır. IGNIS, GPS ile aynı olan WGS 84'ü kullanır. |
| **Derin öğrenme** | Deep learning | Çok katmanlı yapay sinir ağlarıyla yapılan makine öğrenmesi. |
| **Devir** | Epoch | Eğitim kümesi üzerinden bir tam geçiş. 17,940 eğitim yaması ve 32'lik yığınla bir devir 561 yinelemedir. |
| **Dışa aktarma görevi** | Export task | Büyük bir sonucu hesaplayıp Drive'a ya da Cloud Storage'a yazan eşzamansız Google Earth Engine işi. |
| **Dice katsayısı / Dice kaybı** | Dice coefficient / Dice loss | Sayısal olarak F1 ile özdeş bir örtüşme ölçüsü. Kayıp olarak kullanıldığında doğrudan bölge örtüşmesini eniyiler ve doğru negatifleri bütünüyle yok sayar. |
| **Doğru negatif** | True negative (TN) | Yangın yok denildi ve gerçekten olmadı. IGNIS'te bunlardan ≈91.6 milyon adet vardır; doğru negatifleri sayan ölçütlerin burada işe yaramamasının nedeni budur. |
| **Doğru pozitif** | True positive (TP) | Yangın kestirildi ve gerçekten yandı. |
| **Doğrulama kümesi** | Validation set | Hiperparametreleri seçmek, eğitimi durdurmak ve eşiği kalibre etmek için kullanılan veri. IGNIS'in yeni doğrulama bölümü 2024'tür. |
| **Doğruluk** | Accuracy | Tüm kestirimler içinde doğru olanların oranı. Sınıf dengesizliği altında tehlikeli biçimde yanıltıcıdır: her yerde "yangın yok" demek IGNIS'te %99.73 doğruluk verir. |
| **Dolgu** | Padding | Evrişim çıktısının girdi boyutunu koruması için kenara sıfırlardan bir çerçeve eklenmesi. IGNIS `same` dolgusu kullanır. |
| **Duyarlılık (anma)** | Recall | $TP/(TP+FN)$: yanan pikseller içinde bulabildiklerimizin oranı. IGNIS: 0.0222. |
| **Düzeltilmiş doğrusal birim** | ReLU | $\max(0,x)$. Ucuzdur, kaybolan gradyanları önler ve IGNIS'te her evrişimden sonra kullanılır. |
| **Düzenlileştirme** | Regularisation | Aşırı öğrenmeyi azaltmak için yapılan her şey: seyreltme, ağırlık azaltma, yığın normalizasyonu, erken durdurma, veri artırma. |
| **Eğim** | Slope | Arazinin derece cinsinden dikliği. Yangın yayılımı, yokuş yukarı her 10°'de kabaca ikiye katlanır. |
| **Eğitim kümesi** | Training set | Modelin parametrelerini uydurduğu veri. IGNIS'in yeni eğitim bölümü 2019–2023'tür. |
| **Eksik örnekleme** | Undersampling | Sınıfları dengelemek için çoğunluk sınıfı örneklerinin atılması. |
| **Elektromanyetik tayf** | Electromagnetic spectrum | Gama ışınlarından radyo dalgalarına, elektromanyetik ışımanın tüm dalga boyu aralığı. |
| **En yakın komşu ile yeniden örnekleme** | Nearest neighbour resampling | En yakın kaynak pikselin değerinin alınması. Sınıf kodları ve ikili maskeler için tek doğru yöntemdir. |
| **Eniyileyici** | Optimiser | Gradyanları parametrelere uygulayan algoritma: SGD, momentum, Adam, AdamW. |
| **EPSG kodu** | EPSG code | Bir koordinat referans sistemini benzersiz biçimde tanımlayan tam sayı. 32635 = WGS 84 / UTM dilim 35N. |
| **ERA5-Land** | ERA5-Land | ECMWF'nin ~9 km çözünürlüklü kara yüzeyi yeniden analizi. Doğrudan ölçüm değil, gözlemlerle kısıtlanmış model çıktısıdır. |
| **Erken durdurma** | Early stopping | Bir doğrulama ölçütü iyileşmeyi bıraktığında eğitimin durdurulması ve en iyi ağırlıkların geri yüklenmesi. IGNIS 25. devirde durmuş ve 7. devri geri yüklemiştir. |
| **Eş-kayıt (çakıştırma)** | Co-registration | Her bantta $(i,j)$ pikselinin yeryüzünde aynı parçaya karşılık gelmesinin güvence altına alınması. |
| **Eşik** | Threshold | Bir olasılık haritasını ikili maskeye dönüştüren olasılık kesimi $\tau$. IGNIS 0.5 kullanmıştır; yeni işlem hattı bu değeri doğrulama kümesi üzerinde kalibre etmektedir. |
| **Etiket** | Label | Bir eğitim örneğine iliştirilmiş doğru yanıt. IGNIS'te ertesi günün yangın maskesi. |
| **Evrensel Enlem Dilimi Merkatör** | UTM | Dünyayı, her biri kendi enine Merkatör projeksiyonuna sahip 60 adet altı derecelik dilime bölen projeksiyon sistemi. |
| **Evrişim** | Convolution | Küçük bir çekirdeğin bir girdi üzerinde kaydırılması ve her konumda eleman bazlı çarpımların toplamının hesaplanması. Evrişimli sinir ağının (CNN) çekirdek işlemidir. |
| **F1 skoru** | F1-score | Kesinlik ile duyarlılığın harmonik ortalaması. Hangisi küçükse ona baskın biçimde bağlıdır; bu nedenle bir model yalnızca birinde iyi olarak başarılı olamaz. |
| **FireMask** | FireMask | Piksel sınıflandırmasını kodlayan MODIS bandı; 7 = düşük, 8 = normal, 9 = yüksek güvenilirlikli yangın. IGNIS ≥ 7 değerlerini kabul eder. |
| **float16 (fp16)** | float16 (fp16) | Yalnızca 5 üs bitine sahip 16 bitlik kayan noktalı sayı; en büyük değer ≈ 65,504. Küçük gradyanlarda alt taşmaya yatkındır; kayıp ölçeklemesi gerektirir. |
| **float32 (fp32)** | float32 (fp32) | Standart 32 bitlik kayan noktalı sayı: 8 üs biti, 23 mantis biti, ~7 ondalık basamaklık hassasiyet. |
| **Geçiş** | Overpass | Bir uydunun bir konumun üzerinden geçmesi. Terra ekvatoru ~10:30'da, Aqua ~13:30'da geçer. |
| **Genişletilmiş kalıcılık** | Dilated persistence | Bir temel çizgi: bugünün yangın maskesinin her yönde bir piksel genişletilmesi. |
| **Geri yayılım** | Backpropagation | Zincir kuralını kullanarak, kaybın her parametreye göre gradyanını tek bir geri geçişte hesaplayan algoritma. |
| **gfx1201** | gfx1201 | IGNIS'te kullanılan AMD RDNA 4 GPU'suna ait LLVM komut kümesi hedef tanımlayıcısı. ROCm'nin bu hedef için derlenmiş olması gerekir. |
| **Google Earth Engine (GEE)** | Google Earth Engine (GEE) | Petabaytlarca Yer Gözlem verisini paralel bir işleme motoruyla barındıran bulut platformu. IGNIS'in tüm ön işleme zinciri burada çalışır. |
| **Gözetimli öğrenme** | Supervised learning | Doğru yanıtın verildiği örneklerden öğrenme. |
| **Gradyan inişi** | Gradient descent | Parametrelerin, kaybı azaltan yönde yinelemeli olarak hareket ettirilmesi: $\theta \leftarrow \theta - \eta\nabla L$. |
| **Güneş eş-zamanlı yörünge** | Sun-synchronous orbit | Her enlemi her gün aynı yerel güneş saatinde geçen, kutuplara yakın yörünge; böylece görüntüler günler arasında karşılaştırılabilir olur. |
| **Harita projeksiyonu** | Map projection | Eğri Dünya yüzeyinin bir düzleme yayılmasına ilişkin kural. Daima bir şeyi bozar; seçim, neyin bozulacağıyla ilgilidir. |
| **Havuzlama** | Pooling | Her bloğun özetlenmesiyle uzamsal boyutun küçültülmesi. |
| **Hiperparametre** | Hyperparameter | Gradyan inişiyle öğrenilmek yerine insan tarafından seçilen ayar: öğrenme oranı, yığın boyutu, $\gamma$, $\tau$. |
| **IGBP** | IGBP | International Geosphere–Biosphere Programme; MODIS MCD12Q1'in kullandığı 17 sınıflı arazi örtüsü şeması bu programa aittir. |
| **İkili çapraz entropi** | Binary cross-entropy (BCE) | İkili kestirim için standart kayıp, $-[y\log p + (1-y)\log(1-p)]$. Kendinden emin hataları ağır biçimde cezalandırır. |
| **İstemci tarafı nesne** | Client-side object | Google Earth Engine'de, sunucu tarafı bir `ee.*` tutamacının aksine, kendi not defterinizde yaşayan sıradan bir Python nesnesi. |
| **Jaccard indeksi** | Jaccard index | IoU'nun bir başka adı. |
| **Kalıcılık temel çizgisi** | Persistence baseline | Yarının yangın maskesinin bugünkü ile aynı olacağını kestirmek. Hiçbir maliyeti yoktur ve şu anda IGNIS modelini geçmektedir. |
| **Kalibrasyon** | Calibration | Bir model, 0.3 olasılık atadığı pikseller arasında yaklaşık %30'u gerçekten yanıyorsa kalibre edilmiştir. Odak kaybı ve ağırlıklı kayıplar kalibrasyonu bilerek bozar. |
| **Kanal** | Channel | Bir harita yığını içindeki tek bir 2B harita. IGNIS girdisinde 14 kanal vardır; yeni işlem hattında bu sayı 21 olmaktadır. |
| **Karışıklık matrisi (hata matrisi)** | Confusion matrix | Kestirilen sınıfa karşı gözlenen sınıfın tablosu. IGNIS'in üç yama sınıfı için 3×3 boyutundadır. |
| **Karma hassasiyet** | Mixed precision | Ana ağırlıklar ve eniyileyici durumu float32'de tutulurken hesaplamanın 16 bitte yapılması. Kabaca 2× iş hacmi sağlar. |
| **Kategorik değişken** | Categorical variable | Değerleri sıralaması olmayan etiketlerden oluşan değişken; örneğin arazi örtüsü sınıfı. Tam sayı olarak verilmemeli, bire-bir kodlanmalıdır. |
| **Kayıp fonksiyonu** | Loss function | Bir kestirimin ne kadar yanlış olduğunu ölçen, türevlenebilir tek bir sayı; gradyan inişinin küçülttüğü niceliktir. |
| **Kesinlik** | Precision | $TP/(TP+FP)$: yangın olarak işaretlenen pikseller içinde gerçekten yananların oranı. IGNIS: 0.0601. |
| **Kesinlik-duyarlılık eğrisi altındaki alan** | AUC-PR | Kesinlik–duyarlılık eğrisinin altında kalan alan. Nadir sınıf problemleri için birincil ölçüt. Rastgele temel çizgisi pozitif yaygınlığa eşittir; IGNIS için 0.00269. |
| **Kesişimin birleşime oranı** | Intersection over Union (IoU) | $TP/(TP+FP+FN)$; kestirilen ve gerçek maskelerin örtüşmesinin, birleşimlerine oranı. Standart bölütleme ölçütü. IGNIS: 0.0165. |
| **Kod çözücü** | Decoder | Bir U-Net'in genişleyen yarısı; ters evrişimler ve atlama bağlantıları kullanarak uzamsal çözünürlüğü geri kazandırır. |
| **Kodlayıcı** | Encoder | Bir U-Net'in daralan yarısı; uzamsal çözünürlüğü anlamsal soyutlama karşılığında feda eder. |
| **Koordinat referans sistemi** | Coordinate reference system (CRS) | Bir projeksiyon ile bir datumun birleşimi; koordinatların Dünya üzerindeki konumlara nasıl eşleneceğini tanımlar. IGNIS EPSG:32635 kullanır. |
| **Magnus formülü** | Magnus formula | Hava sıcaklığı ile çiy noktasını bağıl neme dönüştüren bağıntı. IGNIS'in `humidity` kanalını türetmek için kullanılır. |
| **Makro-F1** | Macro-F1 | Sınıf bazlı F1 skorlarının ağırlıksız ortalaması. Çoğunluk sınıfı tarafından şişirilemez. IGNIS: 0.3418. |
| **Maksimum havuzlama** | Max pooling | Her 2×2 bloğun en büyük değerinin alınması; uzamsal boyutu yarıya indirir ve sonraki katmanların alıcı alanını ikiye katlar. |
| **Maske** | Mask | Hangi piksellerin bir koşulu sağladığını işaretleyen ikili raster. IGNIS yangın maskeleri ile bir `valid` veri maskesi kullanır. |
| **Maskeyi kaldırma** | Unmask | Maskelenmiş pikselleri sabit bir değerle değiştiren GEE işlemi. `unmask(0)`, IGNIS'teki %15'lik sahte sıfırların kaynağıdır. |
| **MODIS** | MODIS | Moderate Resolution Imaging Spectroradiometer; 36 bantlı, hem Terra hem Aqua üzerinde taşınan alıcı. Beş IGNIS ürününün kaynağıdır. |
| **Normalize edilmiş fark bitki örtüsü indeksi** | NDVI | $(\rho_{NIR}-\rho_{Red})/(\rho_{NIR}+\rho_{Red})$. Sağlıklı bitki örtüsünde yüksektir; çünkü klorofil kırmızıyı soğurur, yaprak yapısı ise yakın kızılötesini saçar. |
| **Normalleştirme** | Normalisation | Tüm kanalların karşılaştırılabilir büyüklükte olması için girdilerin yeniden ölçeklenmesi. Ön çalışma niteliğindeki IGNIS modelinde yoktur — teşhis edilen yedi kusurdan biridir. |
| **Nöron** | Neuron | Temel birim: girdilerin ağırlıklı toplamı artı bir yanlılık teriminin bir aktivasyon fonksiyonundan geçirilmesi. |
| **Odak kaybı** | Focal loss | $-\alpha(1-p_t)^\gamma \log p_t$. $(1-p_t)^\gamma$ çarpanı kolay örnekleri bastırır; böylece zor ve nadir örnekler gradyana egemen olur. |
| **Öğrenme oranı** | Learning rate | Gradyan inişindeki adım büyüklüğü. En belirleyici hiperparametre. IGNIS, plato durumunda yarıya indirilen $10^{-3}$ değerini kullanır. |
| **Ölçek** | Scale | GEE'de bir pikselin metre cinsinden nominal yer boyutu. IGNIS 1000 kullanır. |
| **Örnek** | Sample | Bir (öznitelikler, etiket) çifti. IGNIS'te 22,426 adet vardır. |
| **Öznitelik** | Feature | Bir girdi değişkeni. IGNIS'te çevresel kanallardan biri. |
| **Parametre** | Parameter | Modelin içinde yer alan ve öğrenilen sayı. IGNIS'te ≈1,931,585 adet bulunur. |
| **Parlaklık sıcaklığı** | Brightness temperature | Kusursuz bir kara cismin, gözlenen radyansı yayabilmesi için sahip olması gereken sıcaklık. MODIS yangın tespitinin üzerinde çalıştığı nicelik. |
| **Pasif algılama** | Passive sensing | Enerji kaynağı olarak Güneş'i ya da Dünya'nın kendi yayınımını kullanan uzaktan algılama. Bulut ve yoğun duman tarafından engellenir. |
| **Piksel** | Pixel | Bir rasterin tek bir hücresi. IGNIS'te 1 km × 1 km = 100 hektar. |
| **Piroliz** | Pyrolysis | Bitki malzemesinin, tutuşmadan önce ısıl bozunmayla yanıcı gazlara ayrışması. Ön ısınmanın yangın yayılımını sürüklediği mekanizma. |
| **Planck yasası** | Planck's law | Belirli bir sıcaklıktaki bir cismin yaydığı termal ışımanın tayfını betimler. Uyduyla yangın tespitinin fiziksel temeli. |
| **Radyans (ışıma)** | Radiance | Birim alan, katı açı ve dalga boyu başına düşen enerji; W·m⁻²·sr⁻¹·µm⁻¹ cinsinden. Bir algılayıcının fiziksel olarak ölçtüğü niceliktir. |
| **Radyometrik düzeltme** | Radiometric correction | Ham algılayıcı sayımlarının fiziksel radyansa dönüştürülmesi. IGNIS'in kullandığı ürünlerde hâlihazırda uygulanmıştır. |
| **Raster (hücresel veri)** | Raster | Her biri bant başına bir değer tutan piksellerden oluşan ızgara. |
| **ROC eğrisi** | ROC curve | Eşik süpürülürken doğru pozitif oranının yanlış pozitif oranına karşı çizilmesi. |
| **ROC eğrisi altındaki alan** | ROC-AUC | Rastgele seçilmiş bir pozitifin, rastgele seçilmiş bir negatiften daha yüksek puan alma olasılığı. IGNIS: 0.8468 — sıralaması iyidir, ancak dengesizlik altında yanlış alarm sayısına kördür. |
| **ROCm** | ROCm | AMD'nin açık kaynaklı GPU hesaplama platformu; CUDA'nın muadili. |
| **Rüzgâr bileşenleri** | Wind components (u, v) | $u$ = doğuya doğru, $v$ = kuzeye doğru, m/s cinsinden. Birlikte yönü kodlarlar; normları rüzgâr hızıdır. |
| **Sayısal yükseklik modeli** | Digital elevation model (DEM) | Yer yüksekliğinin rasteri. IGNIS, 30 m çözünürlüklü SRTM'yi 1 km'ye yeniden örnekleyerek kullanır. |
| **Seyreltme** | Dropout | Ortak uyarlanmayı önlemek için eğitim sırasında aktivasyonların bir kısmının rastgele sıfırlanması. IGNIS 0.2 oranını kullanır. |
| **Sıçrama** | Spotting | Ana cephenin önünde, rüzgârla taşınan korların tutuşturduğu yeni yangınlar. Yangınların yangın şeritlerini aşma biçimi. |
| **Sınıf dengesizliği** | Class imbalance | Bir sınıfın diğerinden çok daha kalabalık olması. IGNIS'te pozitif sınıf, piksellerin %0.2686'sıdır; oran 372:1'dir. |
| **Sınıflandırma** | Classification | Piksel başına bir etiket yerine, tüm görüntüye tek bir etiket atanması. |
| **Sigmoit** | Sigmoid | $1/(1+e^{-z})$; herhangi bir gerçel sayıyı $(0,1)$ aralığına eşler. IGNIS'in çıkış aktivasyonudur ve puanı olasılığa dönüştürür. |
| **Spektral çözünürlük** | Spectral resolution | Bir algılayıcının ölçtüğü dalga boyu bantlarının sayısı ve darlığı. |
| **SRTM** | SRTM | Shuttle Radar Topography Mission, 2000; radar interferometrisi kullanarak neredeyse küresel kapsamda 30 m çözünürlüklü bir sayısal yükseklik modeli üretmiştir. |
| **Sunucu tarafı nesne** | Server-side object | GEE'de, not defterinizde değil Google'ın sunucularında yaşayan bir hesaplamaya ait `ee.*` tutamacı. |
| **Tabakalı örnekleme** | Stratified sampling | Örneklemenin tekdüze değil, her sınıf içinde ayrı ayrı yapılması. Tek bir büyük yangının arşive egemen olmasını önler. |
| **Tam bağlı katman** | Fully connected layer | Her girdinin her çıktıya bağlandığı katman. 64×64×14 boyutlu bir yamada 1,000 nöron için 57 milyon parametre gerekirdi. |
| **Tarama genişliği** | Swath | Bir algılayıcının tek geçişte görüntülediği şeridin genişliği. MODIS: 2,330 km. |
| **Tembel değerlendirme** | Lazy evaluation | Bir hesaplamanın yürütülmeden yalnızca tanımının kurulması. GEE'nin, gereksinim duymadığı petabaytları hesaplamaktan kaçınma yöntemi. |
| **Temel çizgi (baz çizgisi)** | Baseline | Bir modelin başarımının anlam taşıyabilmesi için geçmesi gereken, bilerek basit tutulmuş yöntem. IGNIS kalıcılık, genişletilmiş kalıcılık ve rüzgâr yönlü büyüme temel çizgilerini kullanır. |
| **Tensör** | Tensor | Çok boyutlu dizi. IGNIS'in girdisi örnek başına 3. dereceden, yığın başına 4. dereceden bir tensördür. |
| **Tepe yangını** | Crown fire | Ağaç tepe tacı boyunca ilerleyen yangın. En hızlı ve en yıkıcı tür; Akdeniz çam ormanlarının belirgin özelliği. |
| **Ters evrişim (transpoze evrişim)** | Transposed convolution | U-Net kod çözücüsünde uzamsal çözünürlüğü ikiye katlamak için kullanılan, öğrenilebilir bir yukarı örnekleme işlemi. |
| **Test kümesi** | Test set | Yansız bir başarım tahmini için, en sonda tam olarak bir kez kullanılan veri. IGNIS'in yeni test bölümü 2025–2026'dır. |
| **TFRecord** | TFRecord | TensorFlow'un ikili kayıt biçimi. Google Earth Engine doğrudan bu biçime dışa aktarım yapabilir. |
| **Toprak nemi** | Soil moisture | Toprağın hacimsel su içeriği, m³/m³. IGNIS'te bir kuraklık göstergesi. |
| **Tversky kaybı** | Tversky loss | Yanlış pozitifleri ve yanlış negatifleri farklı ağırlıklandıran, Dice'ın genelleştirilmiş hâli. IGNIS, duyarlılığı önceliklendirmek için $\alpha=0.3$, $\beta=0.7$ kullanır. |
| **U-Net** | U-Net | Atlama bağlantılarına sahip, kodlayıcı–kod çözücü mimarisinde bir bölütleme ağı; Ronneberger ve ark. (2015) tarafından tanıtılmıştır. IGNIS'in mimarisi. |
| **Uzamsal çözünürlük** | Spatial resolution | Tek bir pikselin yer üzerindeki boyutu. |
| **Vektör** | Vector | Rüzgâr gibi hem büyüklüğü hem yönü olan nicelik. Görüntü çevirmeleri altında tutarlı biçimde dönüştürülmelidir. |
| **Veri artırma** | Augmentation (data) | Etiketi koruyan dönüşümlerle yeni eğitim örnekleri üretilmesi. IGNIS'te çevirmelerin ayrıca rüzgâr bileşenlerinin işaretini de değiştirmesi gerekir. |
| **Veri özümsemesi** | Data assimilation | Bir yeniden analizin, atmosferin geçmiş durumunu kestirmek üzere model tahminlerini gözlemlerle harmanlama süreci. |
| **Veri sızıntısı** | Data leakage | Değerlendirme verisindeki bilginin modeli etkilemesine ve raporlanan skorun şişmesine yol açan her yol. |
| **Video belleği** | VRAM | GPU'nun ayrılmış belleği. Tüm modeli ve eniyileyici durumu 100 MB'ın altına sığan IGNIS için bir kısıt değildir. |
| **VIIRS** | VIIRS | Visible Infrared Imaging Radiometer Suite; 375 m çözünürlükte aktif yangın tespiti sağlar. Makalede, MODIS'in 1 km çözünürlüğünden gelecekteki bir yükseltme olarak anılmaktadır. |
| **Yakıt** | Fuel | Yanan bitkisel malzeme. Yük, süreklilik, dizilim ve nem ile nitelenir. |
| **Yakıt nemi** | Fuel moisture | Yakıtın su içeriği. İnce ölü yakıtlar hava nemiyle yaklaşık bir saat içinde dengeye gelir; bağıl nemin bu denli güçlü bir kestirici olmasının nedeni budur. |
| **Yama** | Patch | Daha büyük bir rasterdan çıkarılan küçük görüntü penceresi. IGNIS yamaları 65×65 boyutundadır, 64×64'e kırpılır; yeni işlem hattında 32×32'dir. |
| **Yangın cephesi** | Fire front | Yanan alanın ilerleyen sınırı. İnce bir yapıdır; atlama bağlantılarının önemli olmasının nedeni budur. |
| **Yangın davranış üçgeni** | Fire behaviour triangle | Yakıt, hava durumu ve topoğrafya — bir yangının nasıl yayılacağını belirleyen üç etken ailesi. |
| **Yangın duyarlılığı (riski)** | Fire susceptibility | Bir yangının nerede başlamasının olası olduğu. Durağandır, zamansal bileşeni yoktur. IGNIS'in problemi **değildir**. |
| **Yangın rejimi** | Fire regime | Bir ekosistemdeki yangının belirleyici örüntüsü: sıklık, şiddet, mevsim, büyüklük ve tür. |
| **Yangın üçgeni** | Fire triangle | Yakıt, oksijen ve ısı — yanma için gereken üç koşul. Yangın *davranış* üçgeninden farklıdır. |
| **Yangın yayılımı** | Fire spread | Halihazırda yanmakta olan bir yangının yarın nerede olacağı. IGNIS'in problemi. Doğası gereği zamansaldır. |
| **Yanlış negatif** | False negative (FN) | Yangın yok denildi, ama yandı. İşletimsel olarak en tehlikeli hata. |
| **Yanlış pozitif** | False positive (FP) | Yangın kestirildi, ama yanmadı. Yanlış alarm. |
| **Yansıtırlık** | Reflectance | Bir yüzeyin, gelen güneş ışığından yansıttığı kesir; [0,1] aralığında boyutsuzdur. |
| **Yaygınlık (görülme oranı)** | Prevalence | Pozitif sınıfa ait örneklerin oranı. IGNIS: piksel düzeyinde %0.2686. |
| **Yeniden analiz** | Reanalysis | Dondurulmuş bir sayısal modelin tarihsel gözlemleri özümsemesiyle geçmiş atmosfer durumlarının yeniden kurulması. ERA5-Land bunlardan biridir. |
| **Yeniden örnekleme** | Resampling | Yeniden projeksiyonlama ya da ölçek değişiminin ardından piksel değerlerinin yeni bir ızgarada kestirilmesi. |
| **Yeniden projeksiyonlama** | Reprojection | Bir rasterin bir koordinat referans sisteminden diğerine dönüştürülmesi. |
| **Yer durağan yörünge** | Geostationary orbit | 35,786 km yükseklikte, periyodu 24 saate eşit olan ve bu nedenle uydunun tek bir boylam üzerinde asılı kalmasını sağlayan yörünge. Zamansal çözünürlüğü yüksek, uzamsal çözünürlüğü düşüktür. |
| **Yetersiz öğrenme** | Underfitting | Hem eğitim hem doğrulama verisinde kötü başarım; model fazla basittir ya da yeterince eğitilmemiştir. |
| **Yığın (küme)** | Batch | Tek bir ileri ve geri geçişte birlikte işlenen örnek grubu. IGNIS 32 kullanır. |
| **Yığın normalizasyonu** | Batch normalisation | Aktivasyonların yığın boyunca sıfır ortalama ve birim varyansa normalleştirilmesi, ardından öğrenilmiş bir ölçek ve kaydırmanın uygulanması. IGNIS'te her evrişimden sonra uygulanır. Ham girdileri normalleştir*mez*. |
| **Yineleme (adım)** | Iteration / step | Bir yığın kullanılarak yapılan tek bir parametre güncellemesi. |
| **Yörünge** | Orbit | Bir uydunun Dünya çevresinde izlediği yol. |
| **Yukarı örnekleme** | Upsampling | Raster çözünürlüğünün artırılması. Yeni bilgi üretmez — ERA5'in 9 km'lik alanları, 1 km'ye yukarı örneklendikten sonra da 9 km'lik alanlar olarak kalır. |
| **Z-skoru normalleştirme** | z-score normalisation | $x' = (x-\mu)/\sigma$; her kanala sıfır ortalama ve birim varyans kazandırır. İstatistikler yalnızca eğitim bölümünden gelmelidir. |
| **Zamansal çözünürlük** | Temporal resolution | Bir algılayıcının aynı yeri ne sıklıkta yeniden ziyaret ettiği. MODIS: günde 1–2 kez. |

---

## 12. Kaynakça ve ileri okuma

Numaralandırma, iki belge arasında karışıklık olmadan geçiş yapabilmeniz için makaledeki sırayı izlemektedir.

### Temel yöntem kaynakları

**[10] Huot, F., Hu, R.L., Goyal, N., Sankar, T., Ihme, M., Chen, Y.-F. (2022). *Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading from Remote-Sensing Data.* IEEE Transactions on Geoscience and Remote Sensing 60, 1–13.**
*Neden okumalı:* IGNIS'in örnek aldığı çalışma budur. Kıta Amerikası için tam olarak bizim problemimizi — günlük uzaktan algılama kestiricilerinin ertesi günün yangın maskesiyle eşleştirilmesi — tanımlar ve görevin öğrenilebilir ama zor olduğunu ortaya koyar. Veri kümesi tasarımı, kanal seçimi ve her şeyden önce aşırı dengesizlik altında ölçütleri nasıl raporladıkları için okuyun. Bir jüri üyesi "bunu başka kim yaptı?" diye sorarsa, ilk anılacak isim budur.

**[17] Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015, LNCS 9351, Springer, 234–241.**
*Neden okumalı:* Mimarimizin çıkış noktası. Kısa ve açıktır; U biçimli ağın şeması, sonraki her makalenin yeniden çizdiği şekildir. Atlama bağlantısı gerekçesi için Bölüm 2'yi okuyun — bizimkiyle aynı yapıya sahip bir problem için yazılmıştır: az sayıda etiketli görüntü ve tam olarak sınırlandırılması gereken ince yapılar.

**[18] Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017, 2980–2988.**
*Neden okumalı:* Kayıp fonksiyonumuz. Makalenin açılış savı — yoğun kestirimde arka planın amaç fonksiyonunu bastırdığı — farklı bir alanda ifade edilmiş hâliyle tam olarak bizim problemimizdir. Bölüm 3, $(1-p_t)^\gamma$ türetimini ve kullandığımız değerlerin dayandığı $\gamma$ ile $\alpha$ üzerine deneysel incelemeyi verir.

**[19] Kingma, D.P., Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR 2015.**
*Neden okumalı:* Eniyileyicimiz. Birinci ve ikinci moment tahminlerinin ne yaptığını ve ilk birkaç adımda neden yanlılık düzeltmesine gerek duyulduğunu anlamak için Algoritma 1'i ve Bölüm 2'yi okuyun. $\beta_1$ ile $\beta_2$'nin ne anlama geldiğini söyleyebilmek, yetkinlik gösteren küçük bir ayrıntıdır.

### Veri ve platform kaynakları

**[12] Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., Moore, R. (2017). *Google Earth Engine: Planetary-Scale Geospatial Analysis for Everyone.* Remote Sensing of Environment 202, 18–27.**
*Neden okumalı:* IGNIS'in bütün ön işleme zincirinin üzerinde çalıştığı platform. Mimari bölümü için okuyun — tembel değerlendirme, dağıtık yürütme modeli ve istemci/sunucu ayrımının neden var olduğu. Bir döngü içinde `getInfo()` çağırmanın neden hata olduğunu açıklar.

**[13] Giglio, L., Schroeder, W., Justice, C.O. (2016). *The Collection 6 MODIS Active Fire Detection Algorithm and Fire Products.* Remote Sensing of Environment 178, 31–41.**
*Neden okumalı:* Hedef değişkenimizin geldiği yer burasıdır. Bağlamsal test tanımını ve yanlış alarm eleme testlerini okuyun. Ayrıca MODIS'in neleri tespit *edemediğine* dair dürüst kaynaktır; etiket gürültüsü teşhisimizin dayanağı da budur: hedeflerimizin %58.9'u boştur ve bu makale nedenini açıklar.

**[14] Muñoz-Sabater, J., Dutra, E., Agustí-Panareda, A., et al. (2021). *ERA5-Land: A State-of-the-Art Global Reanalysis Dataset for Land Applications.* Earth System Science Data 13(9), 4349–4383.**
*Neden okumalı:* Meteorolojimiz. "Yeniden analiz"in ne anlama geldiğini kavramak için üretim bölümünü, bilinen yanlılıklar için değerlendirme bölümünü okuyun — bu konu size mutlaka sorulacaktır. S5'i yetkiyle yanıtlamanızı sağlayan kaynaktır.

**[15] Funk, C., Peterson, P., Landsfeld, M., et al. (2015). *The Climate Hazards Infrared Precipitation with Stations — A New Environmental Record for Monitoring Extremes.* Scientific Data 2, 150066.**
*Neden okumalı:* Yağışımız. Uydu kaynaklı soğuk bulut süresi kestirimlerinin yer istasyonu verileriyle nasıl harmanlandığını görmek için okuyun; ERA5 yağışı yerine CHIRPS'i yeğlememizin gerekçesi budur.

**[16] Farr, T.G., Rosen, P.A., Caro, E., et al. (2007). *The Shuttle Radar Topography Mission.* Reviews of Geophysics 45(2), RG2004.**
*Neden okumalı:* Topoğrafyamız. Radar interferometrisinin yüksekliği nasıl ürettiğini görmek için görev tanımını okuyun — gerçekten zarif bir mühendislik örneğidir ve tüm pasif ürünlerimizle karşıtlık kurmak için iyi bir *aktif* uzaktan algılama örneğidir.

### Orman yangını bilimi ve derleme kaynakları

**[6] Jain, P., Coogan, S.C.P., Subramanian, S.G., Crowley, M., Taylor, S., Flannigan, M.D. (2020). *A Review of Machine Learning Applications in Wildfire Science and Management.* Environmental Reviews 28(4), 478–505.**
*Neden okumalı:* Alana giriş için tek başına en iyi kaynak. Birkaç yüz çalışmayı tarar ve bunları problem türüne göre düzenler; IGNIS'i tanımlayan duyarlılık-yayılım ayrımı da buradan gelir. Yalnızca tek bir derleme okuyacaksanız önce bunu okuyun.

**[8] Andrianarivony, H.S., Akhloufi, M.A. (2024). *Machine Learning and Deep Learning for Wildfire Spread Prediction: A Review.* Fire 7(12), 482.**
*Neden okumalı:* Doğrudan en ilgili derleme ve ölçüt seçimlerinizi savunurken atıf yapılacak kaynak. Tam olarak bizim savımızı öne sürer: ölçüt seçimi ve sınıf dengesizliği raporlanan başarıma egemendir ve yaygınlık belirtilmedikçe sonuçlar karşılaştırılabilir değildir. Biri sizin sayılarınızı başka bir makalenin sayılarıyla karşılaştırırsa dayanağınız bu kaynaktır.

**Rothermel, R.C. (1972). *A Mathematical Model for Predicting Fire Spread in Wildland Fuels.* USDA Forest Service Research Paper INT-115.**
*Neden okumalı:* Makine öğrenmesinin klasik fiziksel alternatifi ve dünya genelinde işletimsel yangın davranışı yazılımlarının hâlâ temeli. Eğim ve rüzgâr çarpanları için okuyun — bu rehberin 1.3 ve 1.4 bölümlerindeki fizik buradan gelir. "Veri odaklı yaklaşımımız Rothermel türü fiziksel modelleri tamamlar ve fizikle bilgilendirilmiş melez yaklaşımlar etkin bir araştırma yönüdür" diyebilmek güçlü bir yanıttır.

**[9] Pham, B.T., Jaafari, A., Avand, M., et al. (2020). *Performance Evaluation of Machine Learning Methods for Forest Fire Modeling and Prediction.* Symmetry 12(6), 1022.**
*Neden okumalı:* *Duyarlılık* için verilen "0.93 üzeri ROC-AUC" rakamının kaynağı budur. Bu sayıların neden bizimkilerle karşılaştırılabilir olmadığını atıf göstererek açıklayabilmek için özellikle okuyun.

**[11] Shadrin, D., Illarionova, S., Gubanov, F., et al. (2024). *Wildfire Spreading Prediction Using Multimodal Data and Deep Neural Network Approach.* Scientific Reports 14, 2606.**
*Neden okumalı:* Yayılım probleminin güncel, çok kipli bir derin öğrenme ele alışı; IGNIS'i mevcut yazın içinde konumlandırmak ve kanal seçimlerini karşılaştırmak için yararlıdır.

**[7] Alkhatib, R., Sahwan, W., Alkhatieb, A., Schütt, B. (2023). *A Brief Review of Machine Learning Algorithms in Forest Fires Science.* Applied Sciences 13(14), 8275.**
*Neden okumalı:* Algoritma ailelerinin derli toplu bir taraması. Makaledeki İlgili Çalışmalar tablosu için yararlıdır.

### Bağlam ve gerekçe kaynakları

**[1] Reid, C.E., Brauer, M., Johnston, F.H., Jerrett, M., Balmes, J.R., Elliott, C.T. (2016). *Critical Review of Health Impacts of Wildfire Smoke Exposure.* Environmental Health Perspectives 124(9), 1334–1343.**
*Neden okumalı:* Girişteki insan sağlığı gerekçesi. Orman yangını dumanının ölçülebilir solunumsal, kalp-damar ve perinatal zarara yol açtığı savı için somut kanıt sunar.

**[2] Gill, A.M., Stephens, S.L., Cary, G.J. (2013). *The Worldwide "Wildfire" Problem.* Ecological Applications 23(2), 438–454.**
*Neden okumalı:* Problemin bölgesel değil küresel olduğunu ortaya koyar — uluslararası bir dinleyici kitlesi için yararlı bir çerçeveleme.

**[3] Elhami-Khorasani, N., Ebrahimian, H., Buja, L., et al. (2022). *Conceptualizing a Probabilistic Risk and Loss Assessment Framework for Wildfires.* Natural Hazards 114, 1153–1169.**
**[4] Carvalho, A., Monteiro, A., Flannigan, M., Solman, S., Miranda, A.I., Borrego, C. (2011). *Forest Fires in a Changing Climate and Their Impacts on Air Quality.* Atmospheric Environment 45(31), 5545–5553.**
*Neden okumalı:* İklim değişikliği eğilimi savı: uzayan yangın mevsimleri, daha şiddetli olaylar ve bunların hava kalitesine yansıyan sonuçları.

**[5] Bailon-Ruiz, R., Bit-Monnot, A., Lacroix, S. (2022). *Real-Time Wildfire Monitoring with a Fleet of UAVs.* Robotics and Autonomous Systems 152, 104071.**
*Neden okumalı:* Tamamlayıcı gözlem stratejisi. Uydu tabanlı yer gözlemi ile İHA'ların neden farklı işletimsel sorulara yanıt verdiğini — dakikalar-saatler ölçeğinde taktik düzey ile günlük ulusal ölçek — açıklamak için yararlıdır.

### Sıfırdan başlayan bir öğrenci için önerilen okuma sırası

1. Jain ve ark. 2020 [6] — alanda yön bulun.
2. Huot ve ark. 2022 [10] — IGNIS'in çözdüğü problemi tam olarak kavrayın.
3. Ronneberger ve ark. 2015 [17] — mimariyi kavrayın.
4. Lin ve ark. 2017 [18] — sınıf dengesizliğini ve kayıp fonksiyonunu kavrayın.
5. Giglio ve ark. 2016 [13] — etiketlerin nereden geldiğini ve neleri kaçırdığını kavrayın.
6. Andrianarivony ve Akhloufi 2024 [8] — sonuçların dürüstçe nasıl raporlanacağını kavrayın.

---

*Bu rehber, IAC-26,B1,IP,107,x110901 numaralı makalede raporlanan ön temel çalışma itibarıyla IGNIS'in durumunu belgelemektedir. Yeniden inşa edilen PyTorch işlem hattının — 21 normalleştirilmiş kanal, ±1 gün hedef, 32×32 kırpma, melez BCE+Dice kaybı, yıl temelli bölümleme ve kalibre edilmiş eşik ile kalıcılık, genişletilmiş kalıcılık ve rüzgâr yönlü büyüme temel çizgilerine karşı değerlendirilmiş hâlinin — sonuçları **eğitim tamamlandığında doldurulacaktır**. Yeni model için hiçbir sayıyı, ayrı tutulan 2025–2026 test bölümü üzerinde ölçülmeden önce alıntılamayın.*
