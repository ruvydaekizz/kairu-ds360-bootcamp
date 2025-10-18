# Hafta 4 - Fraud Detection Eğitim Modülleri

Bu klasör Hafta 4 finans sprinti fraud detection konularını öğretmek için hazırlanmış 5 adet eğitim modülü içerir.

## 📁 İçerik

### Eğitim Scriptleri
- `1_outlier_detection_with_save.py` - Isolation Forest ve LOF
- `2_feature_scaling_encoding_with_save.py` - Feature scaling ve encoding yöntemleri
- `3_roc_auc_pr_auc_with_save.py` - ROC-AUC ve PR-AUC metrikleri
- `4_shap_lime_explanation_with_save.py` - SHAP/LIME model açıklaması
- `5_cicd_pipeline_deployment_with_save.py` - CI/CD pipeline ve deployment

### Görsel Çıktılar
- `visualizations/` - Her script için ayrı klasörlerde kaydedilen görseller
  - `script1_outlier_detection/` - Outlier detection görselleri
  - `script2_feature_scaling/` - Feature scaling/encoding görselleri
  - `script3_roc_pr_auc/` - ROC-AUC/PR-AUC görselleri
  - `script4_shap_lime/` - SHAP/LIME açıklama görselleri
  - `script5_cicd_deployment/` - CI/CD pipeline görselleri

## 🚀 Kullanım

### Scriptleri Çalıştırma
```bash
# Script 1: Outlier Detection
python 1_outlier_detection_with_save.py

# Script 2: Feature Scaling & Encoding
python 2_feature_scaling_encoding_with_save.py

# Script 3: ROC-AUC & PR-AUC Metrics
python 3_roc_auc_pr_auc_with_save.py

# Script 4: SHAP/LIME Explainability
python 4_shap_lime_explanation_with_save.py

# Script 5: CI/CD Pipeline & Deployment
python 5_cicd_pipeline_deployment_with_save.py
```

### Gereksinimler
```bash
# Temel kütüphaneler (tüm scriptler için)
pip install numpy pandas matplotlib seaborn scikit-learn

# Script 4 için ek kütüphaneler (opsiyonel)
pip install shap lime

# Script 5 için ek kütüphaneler (opsiyonel)
pip install pyyaml
```

## 📊 Her Script'te Neler Var?

### 1. Outlier Detection (Isolation Forest ve LOF)
- ✅ Fraud benzeri veri seti oluşturma
- ✅ Isolation Forest algoritması
- ✅ Local Outlier Factor (LOF)
- ✅ Performans karşılaştırması
- ✅ Hiperparametre analizi
- ✅ Gerçek dünya uygulamaları

**Çıkan Görseller:** 4 adet PNG dosyası

### 2. Feature Scaling ve Encoding
- ✅ StandardScaler, MinMaxScaler, RobustScaler
- ✅ PowerTransformer
- ✅ Label Encoding, One-Hot Encoding, Ordinal Encoding
- ✅ Model performans karşılaştırması
- ✅ Pipeline örneği

**Çıkan Görseller:** 6 adet PNG dosyası

### 3. ROC-AUC ve PR-AUC Metrikleri
- ✅ İmbalanced dataset analizi
- ✅ Confusion matrix analizi
- ✅ ROC curve ve ROC-AUC
- ✅ Precision-Recall curve ve PR-AUC
- ✅ Threshold optimizasyonu
- ✅ Business impact analizi

**Çıkan Görseller:** 7 adet PNG dosyası

### 4. SHAP/LIME Model Açıklaması
- ✅ Model explainability konseptleri
- ✅ SHAP (TreeExplainer, summary plots)
- ✅ LIME (local explanations)
- ✅ Individual prediction açıklamaları
- ✅ Business case örneği
- ✅ Safe imports (kütüphane yoksa da çalışır)

**Çıkan Görseller:** 8 adet PNG dosyası

### 5. CI/CD Pipeline ve Deployment
- ✅ Project structure visualization
- ✅ CI/CD pipeline flow
- ✅ Model lifecycle management
- ✅ Deployment strategies (Blue-Green, Canary, etc.)
- ✅ Monitoring dashboard
- ✅ Cost analysis
- ✅ Security & compliance

**Çıkan Görseller:** 9 adet PNG dosyası

## 🎯 Eğitim Hedefleri

Her script:
- **Teorik bilgi** - Konseptlerin açıklanması
- **Pratik uygulama** - Kod örnekleri ve implementasyon
- **Görselleştirme** - Anlayışı kolaylaştıran grafikler
- **Gerçek dünya bağlantıları** - Business impact ve uygulamalar
- **Best practices** - Endüstri standartları

## 📈 Çıktı Özeti

**Toplam görsel sayısı:** 34 adet PNG dosyası
- Script 1: 4 görsel
- Script 2: 6 görsel  
- Script 3: 7 görsel
- Script 4: 8 görsel
- Script 5: 9 görsel

Her görsel yüksek çözünürlükte (300 DPI) kaydedilir ve eğitim materyallerinde kullanılabilir.

## 🔧 Özellikler

- **Fonksiyon içermeyen design** - Doğrudan çalıştırılabilir
- **Eğitim odaklı** - Adım adım açıklamalar
- **Otomatik görsel kaydetme** - Sıralı dosya isimleri
- **Error handling** - Kütüphane yoksa bile çalışır
- **Fraud detection context** - Gerçek business case'ler

## 📝 Notlar

- Scriptler bağımsız çalışır, sıra önemli değil
- Görsel klasörleri otomatik oluşturulur
- SHAP/LIME kütüphaneleri opsiyonel (script 4)
- Her script sonunda toplam görsel sayısı rapor edilir

## 🎓 Kullanım Senaryoları

1. **Classroom teaching** - Projeksiyonla gösterim
2. **Self-study** - Bireysel öğrenme
3. **Workshop** - Hands-on training
4. **Documentation** - Görsel materyaller için
5. **Presentation** - Business stakeholder'lara sunum