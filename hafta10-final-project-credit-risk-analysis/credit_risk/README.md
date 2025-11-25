# 🏦 Kredi Risk Analizi ve Tahmin Sistemi

Bu proje, makine öğrenmesi algoritmalarını kullanarak kredi başvurularının risk durumunu (temerrüt/geri ödememe riski) analiz eden ve tahminleyen uçtan uca bir veri bilimi projesidir.

Proje, **veri işleme pipeline'ları**, dengesiz veri seti yönetimi (SMOTE/Weighted), model eğitimi ve **Streamlit** tabanlı interaktif bir kullanıcı arayüzü içerir.

## 🚀 Özellikler

* **Çoklu Model Eğitimi:** XGBoost, LightGBM ve Logistic Regression modelleri eğitilir ve kıyaslanır.
* **Dengesiz Veri Yönetimi:** `Class Weighting` ve `SMOTE` teknikleri ile dengeli öğrenme sağlanır.
* **Pipeline Yapısı:** Veri ön işleme ve modelleme adımları `Scikit-learn Pipeline` ile modüler hale getirilmiştir.
* **Overfitting Analizi:** Train ve Test setleri arasındaki performans farkını analiz eden özel modül (`check.py`).
* **İnteraktif Arayüz:** Kullanıcıların kredi parametrelerini girip anlık risk skoru alabileceği modern bir **Streamlit** arayüzü.

## 📈 Veri Seti Hakkında

Bu proje, kredi başvurularının finansal ve demografik özelliklerini içeren kapsamlı bir veri seti kullanmaktadır. Veri seti; başvuru sahibinin yaşı, geliri, mülkiyet durumu ve kredi geçmişi gibi risk analizinde kritik öneme sahip faktörleri barındırır. Amacımız, bu tarihsel verileri analiz ederek, gelecekteki kredi başvurularında olası temerrüt (geri ödememe) riskini önceden tahmin etmek ve finansal karar alma süreçlerini desteklemektir.

Veri setine [Kaggle üzerinden ulaşabilirsiniz.](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

**Sütun Açıklamaları:**

* `person_age`: Başvuru sahibinin yaşı.
* `person_income`: Yıllık gelir.
* `person_home_ownership`: Ev mülkiyet durumu (Kira, Kendine Ait, İpotek, Other).
* `person_emp_length`: Yıl cinsinden iş deneyimi süresi.
* `loan_intent`: Kredinin kullanım amacı (Eğitim, Sağlık, Girişim, vb.).
* `loan_grade`: Kredi derecelendirme notu.
* `loan_amnt`: Talep edilen kredi tutarı.
* `loan_int_rate`: Kredi faiz oranı.
* `loan_status`: **Hedef Değişken** (0: Borcunu ödedi, 1: Temerrüde düştü/Ödemedi).
* `loan_percent_income`: Kredi tutarının yıllık gelire oranı.
* `cb_person_default_on_file`: Geçmişte temerrüt kaydı var mı? (Y: Evet, N: Hayır).
* `cb_person_cred_hist_length`: Kredi geçmişi uzunluğu (Yıl).


## 📂 Proje Yapısı

```text
├── artifacts/             # Eğitilen modeller (.pkl) ve özellik listeleri (.json)
├── data/
│   ├── raw/               # Ham veri seti
│   └── processed/         # İşlenmiş veri seti
├── src/
│   ├── data_loader.py     # Veri yükleme ve klasör yönetimi
│   ├── eda.py             # Keşifçi Veri Analizi (EDA)
│   ├── preprocessing.py   # Veri temizleme ve özellik mühendisliği
│   ├── train.py           # Modellerin eğitimi ve kaydedilmesi
│   ├── check.py           # Overfitting kontrolü
│   └── streamlit_app.py   # Web arayüzü
├── requirements.txt       # Gerekli kütüphaneler
└── README.md              # Proje dokümantasyonu
```

## 🛠️ Kurulum

#### 1. Projeyi klonlayın: 
```text
git clone https://github.com/....

cd credit-risk-project
```

#### 2. Sanal ortam oluşturun :
```text
python -m venv venv
```
##### - Windows için:
```text
source venv/Scripts/Activate
```
##### - Mac/Linux için:
```text
source venv/bin/activate
```
#### 3. Gereksinimleri yükleyin:
```text
pip install -r requirements.txt
```
## 💻 Çalıştırma Adımları
Projeyi sıfırdan çalıştırmak için aşağıdaki adımları sırasıyla terminalde (src klasörü içindeyken) uygulayın:

#### 1. Veriyi Yükleme: Ham veriyi proje içine dahil eder.
```text
python data_loader.py
```
#### 2. Veri Analizi (İsteğe Bağlı): Veri setinin istatistiksel özetini görürsünüz.
```text
python eda.py
```
#### 3. Model Eğitimi: XGBoost, LightGBM ve Logistic Regression modellerini eğitir ve artifacts klasörüne kaydeder.
```text
python train.py
```
#### 4. Model Kontrolü: Modellerin aşırı öğrenip öğrenmediğini (Overfitting) test eder.
```text
python check.py
```
#### 5. Uygulamayı Başlatma: Arayüzü açmak için:
```text
streamlit run streamlit_app.py
```
## 📊 Model Performansı

Eğitilen modeller arasında XGBoost, hem doğruluk hem de riskli sınıfı yakalama (Recall/F1) başarısı nedeniyle canlı sistemde kullanılmak üzere seçilmiştir.

## 📊 Model Performansı ve Karşılaştırma

Eğitilen modeller arasında **XGBoost**, yüksek doğruluk oranı ve riskli sınıfı (1) yakalamadaki başarısı nedeniyle canlı sistemde kullanılmak üzere seçilmiştir.

| Model | Accuracy | Precision (Riskli Sınıf) | Recall (Riskli Sınıf) | F1-Score (Riskli Sınıf) | ROC AUC | Durum |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost (Weighted)** | **%92** | **0.82** | **0.80** | **0.81** | **0.949** | ✅ **Seçilen Model** |
| LightGBM (Weighted) | %92 | 0.82 | 0.79 | 0.81 | 0.949 | Başarılı (Alternatif) |
| Logistic Reg. (SMOTE) | %81 | 0.54 | 0.79 | 0.64 | 0.872 | Yetersiz |

### 🏆 Neden XGBoost Seçildi?
* **Dengeli Performans:** Hem **Accuracy (%92)** hem de **ROC AUC (0.949)** skorlarında en yüksek performansı göstermiştir.
* **Kritik Başarı (Recall):** Kredi risk tahminlemesinde hayati önem taşıyan "Riskli Müşteriyi Yakalama" (Recall) oranında **0.80** ile en yakın rakibi LightGBM'in (0.79) önündedir.
* **Düşük Yanlış Alarm:** Logistic Regression'a kıyasla çok daha yüksek bir **Precision (0.82)** değerine sahiptir, bu da yanlış pozitiflerin (riskli olmayan müşteriye riskli denmesi) minimize edildiğini gösterir.

## 🖼️ Uygulama Görüntüsü
(Not: Kendi ekran görüntünüzü buraya ekleyebilirsiniz)

## 🤝 Katkıda Bulunma

- Bu projeyi forklayın.

- Yeni bir özellik dalı (feature branch) oluşturun (git checkout -b feature/YeniOzellik).

- Değişikliklerinizi commit edin (git commit -m 'Yeni özellik eklendi').

- Dalınızı pushlayın (git push origin feature/YeniOzellik).

- Bir Pull Request oluşturun.

----------------------------------------------------
Geliştirici: Rüveyda Ekiz
