# 💳 Uçtan Uca Dolandırıcılık Tespiti (Fraud Detection) Pipeline

Bu proje, bir makine öğrenimi modelini eğitmek, değerlendirmek ve çalıştırmak için tasarlanmış uçtan uca bir MLOps/Data Science pipeline'ını (**`FraudDetectionPipeline`** sınıfı) uygulamaktadır. Pipeline, veri hazırlığından başlayıp model seçimine, performans izlemeye ve açıklanabilirliğe kadar tüm adımları kapsamaktadır.

---

## 🚀 Proje Yapısı ve Ana Özellikler

| Özellik | Kullanılan Araçlar/Yöntemler | Açıklama |
| :--- | :--- | :--- |
| **Veri Yönetimi** | `pandas`, `numpy`, Sentetik Veri Üretimi | Eğer harici bir veri seti sağlanmazsa, **sentetik dolandırıcılık verisi** oluşturulur ve `train`/`test` setlerine ayrılır. |
| **Ön İşleme** | `FeaturePreprocessor` (**Robust Scaling**, **One-Hot Encoding**) | Eksik değerler doldurulur, sayısal özellikler ölçeklenir ve kategorik özellikler kodlanır. |
| **Sınıf Dengeleme** | `ImbalanceHandler` (**SMOTE/ADASYN**) | Dolandırıcılık verilerindeki ciddi dengesizliği gidermek için eğitim verisine sentetik örnekleme uygulanır. |
| **Model Eğitimi** | `RandomForestClassifier`, `LogisticRegression`, `IsolationForest`, `LocalOutlierFactor` (LOF) | Hem **supervised** hem de **unsupervised** yöntemler eğitilerek performansları karşılaştırılır. |
| **Model İzleme (MLOps)** | **MLflow** | Tüm model eğitimleri, hiperparametreleri, metrikleri ve artefaktları MLflow kullanılarak izlenir ve loglanır. |
| **Model Değerlendirme** | `FraudEvaluator` (**ROC-AUC**, **PR-AUC**, **F1-Score**, Confusion Matrix) | Modeller, dolandırıcılık tespiti için kritik olan metriklerle test setinde değerlendirilir. **Performans eşikleri** (`min_roc_auc`, `min_pr_auc`) kontrol edilir. |
| **Açıklanabilirlik** | `ModelExplainer` (**SHAP**) | En iyi modelin (Random Forest/Logistic Regression) global ve yerel tahminlerinin nedenleri SHAP kütüphanesi ile analiz edilir ve dolandırıcılık örüntüleri incelenir. |
| **Kalıcılık** | `joblib` | Eğitilmiş modeller ve ön işleyici nesneleri diske kaydedilir ve gerektiğinde yüklenir. |

---

## 🛠️ Pipeline Çalışma Akışı (`run_full_pipeline`)

Pipeline'ın uçtan uca çalıştırılması aşağıdaki sırayı takip eder:

1.  **Veri Yükleme:** Sentetik veri oluşturulur veya harici veri yüklenir ve `train`/`test` setlerine ayrılır.
2.  **Ön İşleme & Dengeleme:** Veri temizlenir, dönüştürülür ve sınıf dengesizliği giderilir (**SMOTE/ADASYN**).
3.  **Model Eğitimi:** Tanımlanan tüm modeller eğitilir.
    * **Unsupervised Modeller (IF, LOF):** Supervised modellerden ayrı bir mantıkla (SMOTE'suz data ile) eğitilir ve MLflow'a izole loglanır.
4.  **Model Değerlendirme:** Modeller test verisi üzerinde değerlendirilir ve metrikler **MLflow**'a loglanır. Performans eşiklerinin altında kalan modeller uyarılır.
5.  **En İyi Model Seçimi:** En yüksek ROC-AUC skoruna sahip **supervised** model seçilir.
6.  **Açıklanabilirlik (`explain_models`):** Seçilen en iyi model için **SHAP** analizi yapılır (ModelExplainer'ın mevcut olması şartıyla).
7.  **Kaydetme (`save_models`):** Preprocessor ve eğitilen tüm modeller diske kaydedilir.

---

## 💻 Komut Satırı Arayüzü (CLI) Kullanımı

Pipeline, `python -m src.pipeline` komutuyla farklı modlarda çalıştırılabilir.

### 1. Eğitim Modu (`--mode train`)

Uçtan uca tüm pipeline'ı çalıştırır (Veri yükleme, ön işleme, eğitim, değerlendirme, açıklama, kaydetme).

#### Sentetik veri kullanarak tüm pipeline'ı çalıştır ve modelleri kaydet

python -m src.pipeline --mode train --save_models

#### Harici veri (data.csv) kullanarak pipeline'ı çalıştır

python -m src.pipeline --mode train --data path/to/data.csv


### 2. Tahmin Modu (--mode predict)

Yüklenmiş modelleri kullanarak yeni veriler üzerinde tahmin yapar.

#### Önceden kaydedilmiş Random Forest modelini yükle ve tahmin yap

python -m src.pipeline --mode predict --load_models --model random_forest

#### Not: Tahmin için kullanılacak veri, --data argümanı ile belirtilebilir.


#### 3. Açıklama Modu (--mode explain)
Önceden eğitilmiş bir modelin karar mekanizmasını (SHAP) analiz eder.

#### Önceden kaydedilmiş Random Forest modelini yükle ve açıkla

python -m src.pipeline --mode explain --load_models --model random_forest

### ⚠️ Kritik Uyarılar

- Açıklanabilirlik Modülü: Pipeline, explainability_clean adlı bir modül bekler. Bu modül projede bulunmazsa (veya gerekli kütüphaneler yüklü değilse), açıklanabilirlik adımı atlanır ve bir uyarı verilir.

- MLflow Autologging: mlflow.sklearn.autolog() kullanıldığı için supervised modellerin temel parametreleri ve metrikleri otomatik loglanır. Unsupervised modeller (IF, LOF) için loglama manuel olarak yönetilir ve izole edilir.