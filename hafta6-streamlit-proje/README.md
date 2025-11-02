# Market Sepeti Analizi — Öğrenci Dostu Uygulama 💜🛒

Bu proje, **Market Basket Analysis (Market Sepeti Analizi)** için hem komut satırı tabanlı (CLI) hem de etkileşimli Streamlit arayüzlü örnekler içerir. Öğrenciler için anlaşılır, adım adım bir yaklaşım sunar: veri yükleme → temel istatistikler → birliktelik analizi → kural (association rule) oluşturma → ürün önerileri.

---

## İçerik (Dosyalar)

- `basit_market_analizi.py`  
  - CLI (komut satırı) odaklı sınıf `BasitMarketAnalizi`. CSV okur, temel istatistikleri yazdırır, birliktelik ve kural analizleri yapar ve grafikler üretir.
- `streamlit_app.py` veya `basit_streamlit_denemesi.py`  
  - Streamlit ile hazırlanmış etkileşimli web arayüzü. Mor tema (grafik renkleri mor tonları) ve panellerde yerel kontroller (her panelde ilgili slider/select) içerir.
- `data/basket_analysis.csv`  
  - (Örnek/varsayılan) giriş verisi. Uygulamalar bu dosyayı okumaya çalışır. Projeye veri eklemezsen uygulama uyarı verecektir.

---

## Öne Çıkan Özellikler

- Öğrenci dostu, sade ve açıklayıcı tasarım.
- Streamlit arayüzünde:
  - Üst menü (sayfa seçimi) — `Ana Sayfa`, `Veri`, `Popüler Ürünler`, `Birliktelik`, `Kurallar`, `Öneriler`.
  - Her panelin içinde ilgili kontrol (Min Support, Min Confidence, grafik için gösterilecek üst ürün sayısı).
  - Mor temalı bar grafikler (tüm bar grafikleri mor tonlarında).
  - Veri sekmesinde statik `st.table` ile “titreşim” (yeniden boyutlanma) sorunu önlendi.
- CLI sınıfı `BasitMarketAnalizi`:
  - Veri yükleme, temel istatistikler, popuplar, matplotlib grafikleri (görselleştirme), birliktelik ve kural analizleri.

---

## Gereksinimler

Python 3.9 veya üzeri önerilir.

Örnek `requirements.txt`:

streamlit>=1.18

pandas>=1.5

numpy>=1.22

plotly>=5.0

matplotlib>=3.5

seaborn>=0.12

## Kurulum

1. Repo klonla veya dosyaları proje klasörüne koy:

git clone <repo-url>  # opsiyonel
cd <repo-folder>


2. (Tercihen) sanal ortam oluştur:

python -m venv .venv
###  Windows
.venv\\Scripts\\activate
###  macOS / Linux
source .venv/bin/activate


3. Gereksinimleri yükle:

pip install -r requirements.txt


4. Eğer requirements.txt yoksa doğrudan:

pip install streamlit pandas numpy plotly matplotlib seaborn

Veri Formatı (Beklenen CSV)


5. Uygulamalar data/basket_analysis.csv yolundan okumaya çalışır. Beklenen yapı:

Satırlar = ayrı sepetler (örnek: işlem/bir müşteri)

Sütunlar = ürün isimleri

Hücre değerleri = True/False veya 1/0 veya 'True' gibi ürünün o sepette bulunup bulunmadığını gösteren işaretler


## Kullanım
Streamlit arayüzü (önerilen, etkileşimli)

Projede basit_streamlit_denemesi.py varsa çalıştır:

streamlit run basit_streamlit_denemesi.py

### Tarayıcıda açılan arayüzde:

- Üst menüden sayfa seç.

- Her panelde ilgili parametreleri ayarla (Support / Confidence / Gösterilecek ürün sayısı).

- Analizleri çalıştır ve görselleştirmeleri incele.

### CLI (terminal)

basit_market_analizi.py içindeki main() fonksiyonunu çalıştır:

python basit_market_analizi.py

Kod örneği, data/basket_analysis.csv dosyasını yükler ve konsola sonuçları yazdırır; ayrıca matplotlib penceresinde grafikler gösterir.


#### Nasıl çalışır (kısa teknik özet)

- Veri hazırlama: CSV okunur; True/1 olan hücreler sepet listeleri (sepetler) haline çevrilir.

- Ürün popülerliği: Her ürün kaç sepette göründüğü hesaplanır.

- Birliktelik analizi: İkili ürün kombinasyonları sayılır; seçilen min_support eşiği kullanılarak önemli birliktelikler filtrelenir.

- Kural analizi: Bulunan birliktelikler üzerinden confidence ve lift hesaplanarak antecedent → consequent şeklinde kurallar çıkarılır.

- Öneriler: Seçilen bir ürün için en iyi consequent önerileri listelenir (güvene göre sıralı).





