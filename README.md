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

### 🔜 Hafta 4
- *İçerik belirlenecek*

### 🔜 Hafta 5
- *İçerik belirlenecek*

### 🔜 Hafta 6
- *İçerik belirlenecek*

### 🔜 Hafta 7
- *İçerik belirlenecek*

### 🔜 Hafta 8
- *İçerik belirlenecek*

### 🔜 Hafta 9
- *İçerik belirlenecek*

### 🎯 Hafta 10 - Sunumlar ve Değerlendirme
- *Proje sunumları ve değerlendirme*

## 🚀 Bu Hafta İçin Hızlı Başlangıç

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

## 📖 Ek Kaynaklar

- [KAPSAMLI_REHBER.md](hafta2/KAPSAMLI_REHBER.md) - Detaylı teknik rehber
- 🚀 [RENDER_DEPLOYMENT_GUIDE.md](RENDER_DEPLOYMENT_GUIDE.md) - **Multi-project Render deployment**
- Her proje klasöründeki README.md dosyalarını okuyun
- Practical örnekler ve hands-on projeler

---
*Bu bootcamp MLOps dünyasında profesyonel olmak için gerekli tüm becerileri kapsar*