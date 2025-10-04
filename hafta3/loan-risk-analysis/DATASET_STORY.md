
# 📖 Veri Seti Hikayesi: Loan Risk Dataset

## 🏦 Senaryo

Bir fintech girişimi, kısa vadeli kredi (loan) başvurularının **geri ödenip ödenmeyeceğini** tahmin eden bir sistem geliştirmek istiyor. Amaç, kredi riskini baştan öngörerek:

* ⚡ Karar sürecini hızlandırmak
* 📊 Daha objektif kredi değerlendirmesi yapmak
* 💰 Kredi kayıplarını azaltmak
* 🎯 Daha doğru müşteri segmentasyonu sağlamak

---

## 📊 Veri Seti: "Loan Risk Data"

**Kaynak**: Kaggle – *Loan Prediction / Loan Risk* datasetleri
**Yapı**: Her satır bir kredi başvurusunu temsil eder.

### 👥 Demografik Bilgiler

* **age**: Başvuru sahibinin yaşı
* **education**: Eğitim seviyesi (`High School`, `College`, `Bachelor`, `Master or Above`)
* **Gender**: Cinsiyet (`male`, `female`)

### 💰 Finansal Bilgiler

* **Principal**: Çekilen kredi tutarı
* **terms**: Vade süresi (gün cinsinden; 7, 15 veya 30 gün)
* **principal_per_term**: Bir dönem başına düşen kredi tutarı (türetilmiş değişken)

### 📅 Tarih Bilgileri

* **effective_date**: Kredinin başladığı tarih
* **due_date**: Kredinin geri ödenmesi gereken tarih
* **planned_term_days**: Planlanan süre (due - effective, gün cinsinden; türetilmiş değişken)

### ⚠️ Hedef Değişken

* **paid** / **default**: Kredinin ödenip ödenmediği

  * `1`: Kredi zamanında ödenmiş (**iyi müşteri**)
  * `0`: Kredi ödenmemiş (**riskli müşteri**)

---

## 🎭 Gerçek Dünya Senaryosu

### 📈 İş Problemi

1. **Dengesiz Veri**: Çoğu müşteri krediyi ödüyor, az sayıda müşteri ödeyemiyor.
2. **Yanlış Pozitif Maliyeti**: İyi müşteriyi reddetmek → gelir kaybı.
3. **Yanlış Negatif Maliyeti**: Riskli müşteriye kredi vermek → direkt zarar.
4. **Hızlı Karar Gereksinimi**: Özellikle kısa vadeli kredilerde anlık skor çok önemli.

---

## 🔍 Analiz Hedefleri

1. **EDA (Keşifsel Veri Analizi)**

   * Yaş, eğitim, cinsiyet gibi faktörlerin riskle ilişkisini görmek.
   * Tarih ve vade uzunluğunun ödeme davranışına etkisini incelemek.

2. **Dengesiz Veri Çözümü**

   * **SMOTE**: Azınlık sınıfı (default) için sentetik örnek üretmek.
   * **Undersampling**: Çoğunluk sınıfını azaltmak.
   * **Class Weights**: Algoritmalara sınıf ağırlığı tanımlamak.

3. **Modelleme**

   * **Logistic Regression**: Basit ve yorumlanabilir.
   * **XGBoost**: Güçlü ensemble metodu, yüksek performanslı.

---

## 🏆 Başarı Kriterleri

### 📊 Teknik Metrikler

* **ROC AUC**: Ayırma gücü (0.75+ hedef)
* **Precision / Recall**: Riskli müşterileri doğru yakalama başarısı
* **F1-Score**: Denge metriği

### 💼 İş Metrikleri

* **False Positive Rate**: İyi müşteri kaybı düşük olmalı
* **False Negative Rate**: Riskli müşteri kabul oranı çok düşük olmalı
* **Model Yorumlanabilirliği**: Kararların neden verildiği açıklanabilmeli

---

## 🎯 Proje Değeri

### 📚 Eğitim Açısından

* Dengesiz veri setlerinde strateji geliştirmeyi öğretir
* Farklı modelleme yaklaşımlarını karşılaştırmayı sağlar
* End-to-end ML pipeline kurma deneyimi kazandırır
* Streamlit ile canlı dashboard ve scoring uygulaması geliştirmeyi gösterir

### 🏢 İş Açısından

* Kredi değerlendirme süresini dakikalar → saniyelere indirir
* Kredi portföyündeki riskleri azaltır
* Müşteri deneyimini iyileştirir (hızlı onay/red)

---

## ⚠️ Etik Hususlar

### 🛡️ Bias Önleme

* Eğitim veya cinsiyet gibi değişkenlerin karar mekanizmasındaki etkisi şeffaf takip edilir.
* Düzenli fairness testleri yapılır.

### 🔒 Veri Güvenliği

* Tarih ve kimlik bilgileri anonimleştirilir.
* GDPR/KVKK uyumlu süreçler uygulanır.

---

## 🎯 Proje Değeri

### 📚 **Eğitim Açısından**
Bu proje öğrencilere şunları öğretir:
- Gerçek dünya veri problemleri
- Dengesiz veri çözümleri
- Model karşılaştırma teknikleri
- End-to-end ML pipeline
- Streamlit ile dashboard geliştirme
- Cloud deployment



## 🚀 **Sonuç**

Bu proje, öğrencilere **gerçek dünya makine öğrenmesi problemlerini** deneyimleme fırsatı veriyor. Sadece teknik beceriler değil, aynı zamanda:

- İş problemi anlama
- Veri hikayesi çıkarma  
- Etik ML uygulamaları
- Production deployment

konularında da deneyim kazandırıyor.

---



✨ *Bu hikaye, projenin sadece “bir ML alıştırması” değil, gerçek hayattaki kredi risk yönetiminin bir prototipi olduğunu vurgulamak için hazırlanmıştır.*
