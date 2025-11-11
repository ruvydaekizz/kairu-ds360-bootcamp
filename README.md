# 🎓 Data Science 360 Bootcamp

Kapsamlı Data Science ve MLOps eğitimi - 10 haftalık program

## 📅 Bootcamp Programı

### ✅ Hafta 2 - MLOps ve Deployment Temelleri
**Konular:** DVC, MLflow, FastAPI, Docker, GitHub Actions
- 📁 [01-dvc-versioning](hafta2/01-dvc-versioning/) - Veri versiyonlama
- 📁 [02-mlflow-tracking](hafta2/02-mlflow-tracking/) - Deney takibi
- 📁 [03-fastapi-ml-api](hafta2/03-fastapi-ml-api/) - ML API geliştirme
- 📁 [04-docker-deployment](hafta2/04-docker-deployment/) - Containerization
- 📁 [05-github-actions-ci](hafta2/05-github-actions-ci/) - CI/CD pipeline
- 🚢 [titanic-mlops](hafta2/titanic-mlops/) - **Kapsamlı proje örneği**

### ✅ Hafta 3 - Dengesiz Veri ve Model Karşılaştırması
**Konular:** SMOTE, Undersampling, Class Weights, Logistic Regression vs XGBoost, Streamlit Dashboard
- 💰 [loan-risk-analysis](hafta3/loan-risk-analysis/) - **Kredi risk analizi projesi**
  - Kaggle veri seti entegrasyonu
  - SMOTE ve undersampling teknikleri
  - Model karşılaştırması (LR vs XGBoost)
  - Streamlit dashboard geliştirme
  - 🚀 [Render deployment ready](RENDER_DEPLOYMENT_GUIDE.md)

### ✅ Hafta 4 - Finans Sprinti: Fraud Detection 🏦
**Konular:** Outlier Detection, Feature Engineering, ROC-AUC/PR-AUC, SHAP/LIME, CI/CD Pipeline
- 📁 [modul](hafta4/modul/) - **Eğitim modülleri (5 script)**
  - Isolation Forest ve LOF outlier detection
  - Feature scaling ve encoding yöntemleri  
  - ROC-AUC ve PR-AUC metrikleri analizi
  - SHAP/LIME model açıklanabilirlik
  - CI/CD pipeline ve deployment
- 🏦 [fraud-detection](hafta4/fraud-detection/) - **Kapsamlı fraud detection projesi**
  - Credit Card Fraud Detection dataset
  - Gerçek dünya imbalanced learning
  - Production-ready MLOps pipeline
  - Interactive demo ve comprehensive evaluation

### ✅ Hafta 5 - Time Series Forecasting: M5 Competition 🏪
**Konular:** ARIMA, Prophet, LightGBM, Time Series CV, Prefect Automation, Docker Deployment
- 🏪 [hafta5](hafta5/) - **M5 Forecasting Modular Pipeline**
  - 📖 **M5 Dataset Story**: Walmart'ın 5 yıllık satış verisi (2011-2016)
  - 🏗️ **Modular Architecture**: 7 specialized modules (P1-P7)
  - 📊 **Model Comparison**: ARIMA (~46% sMAPE) vs Prophet (~28% sMAPE) vs LightGBM (~33% sMAPE)
  - ⚙️ **Feature Engineering**: Lag, rolling, seasonal features
  - ✅ **Time Series CV**: Rolling-origin cross-validation (proper temporal splitting)
  - 🔄 **Prefect Automation**: Daily forecasting pipeline (09:00 Europe/Istanbul)
  - 🐳 **Docker Ready**: Production containerization with volume mounting
  - 📚 **Comprehensive Docs**: 1,500+ line documentation with dataset story

### ✅ Hafta 6 - Market Basket Analysis 🛒
**Konular:** Association Rules, Support-Confidence-Lift, Cross-selling, Interactive Analysis
- 🛒 [hafta6](hafta6/) - **Market Sepeti Analizi Projesi**
  - Market Basket Analysis temelleri
  - Support, Confidence, Lift kavramları
  - Association Rules ve ürün birliktelikleri
  - Cross-selling ve mağaza optimizasyonu
  - Interactive Streamlit dashboard
  - 📊 999 sepet x 16 ürün veri seti analizi

### ✅ Hafta 7 - Metin İşleme ve NLP Optimizasyonu 🤖
**Konular:** Text Cleaning, Tokenization, TF-IDF, BERT, PII Masking, Model Optimization, FastAPI
- 📝 [hafta7](hafta7/) - **Türkçe NLP ve Model Optimizasyonu Projesi**
  - Sentetik sağlık verisi (500 kayıt) ile çalışma
  - Türkçe metin temizleme ve tokenization (NLTK + SpaCy)
  - TF-IDF analizi ve doküman benzerliği
  - BERT tabanlı semantik analiz (multilingual)
  - PII maskeleme (Regex + Presidio) ile güvenlik
  - Model optimizasyonu (DistilBERT, quantization)
  - Production-ready FastAPI servisi
  - 🏥 Sağlık verilerinde gizlilik koruması

### 🔜 Hafta 8
- *İçerik belirlenecek*

### 🔜 Hafta 9
- *İçerik belirlenecek*

### 🎯 Hafta 10 - Sunumlar ve Değerlendirme
- *Proje sunumları ve değerlendirme*

## 🚀 Bu Hafta İçin Hızlı Başlangıç

### Hafta 7 - Metin İşleme ve NLP Optimizasyonu
```bash
# Proje klasörüne git
cd hafta7/

# Virtual environment aktifleştir
source venv/bin/activate

# SpaCy modelini yükle
python -m spacy download en_core_web_sm

# Tüm örnekleri çalıştır (demo)
python run_examples.py

# API servisi başlat
python api/main.py  # Terminal 1
python api/test_api.py  # Terminal 2 - API testleri

# PII maskeleme örneği
python src/pii_masking.py
```

### Hafta 6 - Market Basket Analysis
```bash
# Proje klasörüne git
cd hafta6/

# Virtual environment aktifleştir
source venv/bin/activate

# Konsol uygulaması çalıştır
python basit_market_analizi.py

# Web dashboard başlat
streamlit run basit_streamlit_app.py
```

### Hafta 4 - Fraud Detection
```bash
# Proje klasörüne git
cd hafta4/fraud-detection/

# Virtual environment oluştur
python -m venv venv && source venv/bin/activate

# Dependencies kur
pip install -r requirements.txt

# Interactive demo başlat
python run_demo.py

# Eğitim modülleri çalıştır
cd ../modul/
python 1_outlier_detection_with_save.py
```

### Hafta 3 - Loan Risk Analysis
```bash
# Proje klasörüne git
cd hafta3/loan-risk-analysis/

# Kurulumu başlat
./start.sh

# EDA analizi yap
cd src && python eda.py

# Model eğitimi
python models.py

# Streamlit dashboard
streamlit run streamlit_app/app.py
```

### Hafta 2 - MLOps Temelleri
```bash
# Sanal ortamı aktif et
source ds360/bin/activate

# İkinci hafta projelerine git
cd hafta2/

# Titanic MLOps projesini incele
cd titanic-mlops/
uvicorn src.api:app --reload
```

## 📚 Öğrenme Yolu

### Hafta 2 - MLOps Foundation
1. **DVC** → Veri versiyonlama temelleri
2. **MLflow** → Model eğitimi ve takip  
3. **FastAPI** → API geliştirme
4. **Docker** → Containerization
5. **GitHub Actions** → CI/CD automation
6. **Titanic MLOps** → Tüm teknolojilerin entegrasyonu

### Hafta 3 - Imbalanced Data & Modeling
1. **EDA** → Veri keşfi ve analizi
2. **SMOTE** → Sentetik örnekleme
3. **Undersampling** → Çoğunluk sınıf azaltma
4. **Class Weights** → Sınıf ağırlıklandırma
5. **Model Comparison** → LR vs XGBoost
6. **Streamlit** → Dashboard ve deployment

### Hafta 4 - Fraud Detection & MLOps
1. **Outlier Detection** → Isolation Forest & LOF
2. **Feature Engineering** → Scaling, encoding, imbalance handling
3. **Advanced Metrics** → ROC-AUC vs PR-AUC analysis
4. **Explainability** → SHAP & LIME model interpretation
5. **CI/CD Pipeline** → Production deployment strategies
6. **Business Impact** → Cost-benefit analysis & threshold optimization

### Hafta 6 - Market Basket Analysis
1. **Data Understanding** → Transaction data and product relationships
2. **Support Calculation** → Item frequency analysis
3. **Association Rules** → Confidence and lift metrics
4. **Interactive Analysis** → Streamlit dashboard development
5. **Business Applications** → Cross-selling and store optimization
6. **Advanced Patterns** → Multi-item combinations and recommendations

### Hafta 7 - NLP & Text Processing
1. **Text Preprocessing** → Turkish text cleaning and tokenization
2. **TF-IDF Analysis** → Term frequency and document similarity
3. **BERT Integration** → Semantic analysis with multilingual models
4. **PII Protection** → Privacy preservation with regex and AI
5. **Model Optimization** → DistilBERT and quantization techniques
6. **API Development** → Production FastAPI services

## 📖 Ek Kaynaklar

- [KAPSAMLI_REHBER.md](hafta2/KAPSAMLI_REHBER.md) - Detaylı teknik rehber
- 🚀 [RENDER_DEPLOYMENT_GUIDE.md](RENDER_DEPLOYMENT_GUIDE.md) - **Multi-project Render deployment**
- Her proje klasöründeki README.md dosyalarını okuyun
- Practical örnekler ve hands-on projeler

---
*Bu bootcamp MLOps dünyasında profesyonel olmak için gerekli tüm becerileri kapsar*
