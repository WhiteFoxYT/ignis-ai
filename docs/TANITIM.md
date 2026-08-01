# IGNIS — Tanıtım ve İletişim Stratejisi

**Bu belge ekip içindir.** Kime, ne zaman, ne söyleyeceğimizi ve —daha
önemlisi— henüz ne söylemeyeceğimizi belirler.

Son güncelleme: 1 Ağustos 2026
Karar sahibi: altı kişilik IGNIS ekibi (Antalya Yusuf Ziya Öner Fen Lisesi)

---

## 0. Tek cümlelik strateji

**Önce doğrula, sonra duyur.**

Model kendi temel çizgisini geçene kadar IGNIS'i hiçbir kuruma veya basına
"çalışan bir yangın tahmin sistemi" olarak sunmuyoruz.

---

## 1. Neden bu strateji

Bugün elimizde ölçülmüş şu tablo var (v1 arşivi, 45 parça / 1054 yama):

| | Model | Kalıcılık (persistence) temel çizgisi |
|---|---|---|
| IoU | 0.0165 | **0.0306** |
| F1 | 0.0324 | **0.0595** |

"Yarın = bugün" diyen tek satırlık, parametresiz, eğitimsiz bir kural modelimizi
yeniyor. Ayrıca yama düzeyi doğruluğumuz (0.7714) çoğunluk sınıfı payının
(0.7774) altında — yani her seferinde "sönüyor" demek daha yüksek skor verirdi.

Bu tablo değişmeden önce bir kuruma gidersek üç şey olur:

1. **Teknik ekipleri bunu on dakikada fark eder.** OGM ve AFAD'da uzaktan
   algılama bilen insanlar var. Temel çizgi karşılaştırması, bu alanda ilk
   sorulan sorudur. Hazırlıksız yakalanmak, sonraki her konuşmayı zorlaştırır.
2. **Bir kez "abartan lise ekibi" damgası yerse geri alınmaz.** Güvenilirlik
   asimetriktir: kazanmak yavaş, kaybetmek anlıktır.
3. **Yangın tahmini can güvenliği alanıdır.** Çalışmayan bir tahmin sistemini
   çalışıyor gibi sunmak, tahliye ve ekip yerleştirme kararlarını etkileyebilir.
   Bu, akademik bir nezaket meselesi değil, sorumluluk meselesidir.

Buna karşılık, **bugün bile dürüstçe anlatabileceğimiz gerçek bir hikâye var**
(bkz. Bölüm 5).

---

## 2. Bugün savunulabilir olan çerçeve

Şunu söyleyebiliriz:

> "Türkiye için ertesi gün yangın yayılımını tahmin etmeye çalışan, uçtan uca
> yeniden üretilebilir bir veri hattı ve dürüstçe raporlanmış bir temel çizgi
> kurduk. Modelimiz şu anda kalıcılık temel çizgisini geçmiyor ve bunun
> nedenlerini ölçtük."

Şunu **söyleyemeyiz**:

> ~~"Yangınların ertesi gün nereye yayılacağını %77 doğrulukla tahmin
> ediyoruz."~~

O %77, sınıf dengesizliğinin bir yan ürünü. Bu cümleyi kurmak, yukarıdaki
tablonun varlığında dürüst değil.

### Yangın riski ≠ yangın yayılımı

Bu ayrımı her konuşmada net tutuyoruz. Literatürde "yangın duyarlılığı"
(susceptibility — yangın nerede *çıkabilir*) için ROC-AUC 0.93 üstü sonuçlar
var. O statik bir problem. Bizimki zamansal problem ve çok daha zor. İkisini
karıştırmak, sayılarımızı olduğundan iyi gösterir.

---

## 3. OGM ve AFAD'a yaklaşım: **çözüm değil, veri talebi**

Kuruma gittiğimizde konumumuz şu **olmayacak**:

> ~~"Size yangın yayılımını tahmin eden bir sistem getirdik."~~

Konumumuz şu **olacak**:

> "Bir araştırma yürütüyoruz. Doğrulama yapabilmek için yangın çevre
> (perimeter) kayıtlarınıza ihtiyacımız var."

### Neden veri talebi doğru çerçeve

**Teknik olarak asıl ihtiyacımız bu.** Şu anda hedefimiz MODIS termal anomali
maskesi: 1 km çözünürlük, günde iki geçiş. Bulut altındaki yangını, eşiğin
altında kalan yangını ve geçişler arasındaki yayılımı göremiyor. Yamaların
%58.9'unda t+1 gününde hiç yangın pikseli yok — oysa t gününde ortalama 12.3
piksel yanıyor. Bu, yangının söndüğü anlamına gelmiyor; uydunun o an göremediği
anlamına geliyor.

**Hiçbir mimari, etiketinin tarif edebildiğinden iyi tahmin edemez.** OGM'nin
yangın çevre kayıtları *ölçülmüş*, çıkarsanmış değil. O veri, doğruluğumuzun
önündeki en büyük tavanı kaldırır — ağ mimarisinde yapacağımız hiçbir
değişiklikten daha fazla.

**Ayrıca ilişki açısından da doğru.** Veri talebi, karşı tarafı uzman
konumuna koyar; çözüm sunumu ise rakip ya da satıcı konumuna. Lise ekibi olarak
birincisi hem daha dürüst hem daha etkili.

### Talep neyi içerecek

- Belirli yangınlar için tarihli yangın çevre poligonları (2019–2026)
- Mümkünse müdahale başlangıç/bitiş zamanları
- Veri kullanım koşulları: yalnızca akademik doğrulama, yayında kaynak gösterimi

### Zamanlama

**IAC sunumundan sonra.** Kabul edilmiş bir IAF bildirisiyle gitmek, talebi
ciddiye alınır kılar. Öncesinde gitmek için elimizde yeterli dayanak yok.

---

## 4. Basın: IAC'den sonra

### Şu an basına gitmiyoruz

Yerel basın "Antalyalı lise öğrencileri yangınları tahmin eden yapay zekâ
geliştirdi" başlığını atar. Bu başlık yanlış olur ve düzeltilmesi mümkün olmaz.
Bir kez çıktıktan sonra her teknik açıklamamız savunma gibi görünür.

### IAC sonrası anlatılabilecek gerçek hikâye

Doğruluk iddiası **gerektirmeyen**, tamamen doğru ve gerçekten haber değeri olan
bir hikâyemiz var:

> **Antalyalı bir lise ekibinin bildirisi, Antalya'da düzenlenen IAC 2026'ya
> kabul edildi.**

77. Uluslararası Astronotik Kongresi, 5–9 Ekim 2026'da Antalya'da toplanıyor.
Altı lise öğrencisinin IAF Dünya Gözlem Sempozyumu'na kabul edilmesi başlı başına
bir haber. Buna ek olarak:

- Yangın mevsiminin ortasında, yangınlardan en çok etkilenen ilde yapılan bir çalışma
- Tamamen açık kaynak, yeniden üretilebilir bir veri hattı
- Sonuçları dürüstçe raporlayan, temel çizgisini gizlemeyen bir bilimsel tutum

Bu üçüncü madde aslında en güçlü olanı. **Kendi modelinin kaybettiğini söyleyen
bir lise ekibi, kazandığını söyleyen bir lise ekibinden daha inandırıcıdır.**

### Basınla konuşurken kural

Gazeteci mutlaka "peki ne kadar doğru tahmin ediyor?" diye soracak. Hazır cevap:

> "Şu anda modelimiz, 'yarın bugünle aynı olacak' diyen basit bir kuralı
> geçemiyor. Bunu ölçtük ve bildiride açıkça yazdık. Şu an yaptığımız, sistemin
> neden çalışmadığını anlamak — ve bunun cevabını da bulduk: uydu etiketlerinin
> çözünürlüğü. Bir sonraki adımımız Orman Genel Müdürlüğü'nden gerçek yangın
> sınırı verisi talep etmek."

Bu cevap dürüst, anlaşılır ve aslında hikâyeyi güçlendiriyor.

---

## 5. Şu anda ne yapabiliriz

Doğrulama beklerken duran bir proje değiliz. Bugün yapılabilecekler:

| Yapılabilir | Yapılamaz |
|---|---|
| IAC bildirisini tamamlamak (son tarih 14 Eylül 2026) | Doğruluk iddiası içeren duyuru |
| Okul içi ve fen lisesi çevresinde sunum | OGM/AFAD'a çözüm sunumu |
| Açık kaynak depoyu ve rehberleri yayınlamak | Basın bülteni |
| Yöntem ve veri hattı üzerine anlatım | "Sistem çalışıyor" mesajı |
| Sponsor/burs başvurularında "IAC'ye kabul edilmiş araştırma" ifadesi | "Yangın tahmin sistemi geliştirdik" ifadesi |

---

## 6. Duyuru için eşik

Aşağıdakilerin **hepsi** sağlandığında Bölüm 3 ve 4'teki adımlara geçilir:

- [ ] Model, ayrılmış test bölmesinde (2025–2026) kalıcılık temel çizgisini
      **hem IoU hem F1'de** geçiyor (yani IoU > 0.0306 ve F1 > 0.0595)
- [ ] Eşik, yalnızca doğrulama bölmesinde kalibre edilmiş; test'te ayar yapılmamış
- [ ] Genişletilmiş kalıcılık ve rüzgâr yönlü büyüme temel çizgileri de geçilmiş
- [ ] Sonuç, `src/evaluate.py` çıktısıyla yeniden üretilebiliyor
- [ ] Yama düzeyi doğruluğu çoğunluk sınıfı payının üstünde

Bu kutulardan biri bile boşsa duyuru yok. **Sayıyı güzelleştirmek yerine
durup nedenini araştırıyoruz.**

---

## 7. Kim ne söylüyor

Tek ağızdan konuşmak önemli. Herhangi birimize teknik bir soru geldiğinde:

- Cevabı biliyorsak, ölçülmüş sayılarla cevaplarız
- Bilmiyorsak, **"bunu ölçmedik"** deriz — tahmin yürütmeyiz
- Sonuçlar hakkındaki sorularda mutlaka temel çizgiyi de söyleriz

Sunumda kullanılacak İngilizce kalıplar `docs/REHBER_TR.md` Bölüm 10'da.

---

## 8. Özet

1. Model temel çizgiyi geçene kadar kurum ve basın yok.
2. Kuruma gidince **veri talebiyle** gidiyoruz, çözüm sunumuyla değil.
3. Basına IAC'den sonra ve **kabul hikâyesiyle** gidiyoruz, doğruluk iddiasıyla değil.
4. Dürüstlük burada bir kısıt değil, sahip olduğumuz en güçlü kart.
