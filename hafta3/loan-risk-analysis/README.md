# 💰 Loan Risk Analysis Project

Bu proje, kredi başvurularının risk analizini yapan bir makine öğrenmesi projesidir. SMOTE, undersampling ve class weights teknikleri kullanılarak dengesiz veri problemi çözülmüş, Logistic Regression ve XGBoost modelleri ile karşılaştırma yapılmıştır.

## 📁 Proje Yapısı

```
loan-risk-analysis/
├── data/                    # Veri dosyaları
│   ├── loan_data.csv       # Ana veri seti
│   └── processed/          # İşlenmiş veriler
├── notebooks/               # Jupyter notebook'lar
│   └── 01_eda.ipynb        # Keşifsel veri analizi
├── src/                     # Python source kodları
│   ├── data_loader.py      # Veri yükleme
│   ├── preprocessing.py    # Veri ön işleme
│   ├── eda.py              # Keşifsel veri analizi
│   └── train.py            # Model eğitimi
├── streamlit_app/          # Streamlit uygulaması
│   └── app.py              # Ana dashboard
├── artifacts/              # Eğitilmiş modeller ve metadata
│   ├── model_*.pkl         # Trained models
│   ├── preprocessor_*.pkl  # Preprocessing pipelines
│   └── feature_schema*.json # Feature schemas
├── models/                 # Model çıktıları (opsiyonel)
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version (Render)
├── Procfile               # Deployment config (Render)
├── .gitignore             # Git ignore dosyası
├── start.sh               # Başlatma scripti
└── README.md              # Bu dosya
```

## 🚀 Hızlı Başlangıç

### 1. Kurulum
```bash
# Projeyi klonlayın
git clone <repository-url>
cd loan-risk-analysis

# Başlatma scriptini çalıştırın
./start.sh
```

### 2. Veri Analizi
```bash
# Jupyter notebook ile EDA
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Model Eğitimi
```bash
# Modelleri eğitin
cd src
python train.py
```

### 4. Streamlit Dashboard
```bash
# Dashboard'u çalıştırın
streamlit run streamlit_app/app.py
```

## 🎯 Özellikler

### Veri İşleme Teknikleri
- **SMOTE**: Sentetik azınlık örnekleri oluşturma
- **Random Undersampling**: Çoğunluk sınıfını azaltma
- **Class Weights**: Sınıf ağırlıklarını dengeleme

### Makine Öğrenmesi Modelleri
- **Logistic Regression**: Basit ve yorumlanabilir
- **XGBoost**: Güçlü gradient boosting

### Streamlit Dashboard
- 🔍 Risk tahmini arayüzü
- 📊 Model performans metrikleri
- 📈 Veri analizi görselleştirmeleri

## 🌐 Deployment (Render/Railway)

### 🚀 Render.com (ÖNERİLEN)
**Multi-project repo için hazır!**

1. **Root Directory Build:**
   - Root Directory: `hafta3/loan-risk-analysis`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

2. **Detaylı rehber:** 📋 [RENDER_DEPLOYMENT_GUIDE.md](../../RENDER_DEPLOYMENT_GUIDE.md)

### Railway.app
1. GitHub repo'yu Railway'e bağlayın
2. Start Command: `streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0`

## 📊 Model Performansı

Modeller AUC score ile değerlendirilir:
- Logistic Regression (Original): ~0.75
- Logistic Regression (SMOTE): ~0.78
- XGBoost (Original): ~0.82
- XGBoost (Class Weights): ~0.80

## 🔧 Teknik Detaylar

### Dependencies
- pandas: Veri manipülasyonu
- scikit-learn: Makine öğrenmesi
- imbalanced-learn: Dengesiz veri teknikleri
- xgboost: Gradient boosting
- streamlit: Web dashboard
- plotly: İnteraktif görselleştirme

### Veri Seti
Kaggle'dan "zhijinzhai/loandata" veri seti kullanılmaktadır.

## 📝 Kullanım Notları

Bu proje eğitim amaçlıdır ve aşağıdaki konuları öğretir:
- Dengesiz veri problemleri
- SMOTE ve undersampling teknikleri
- Model karşılaştırması
- Streamlit ile dashboard geliştirme
- Cloud deployment

⚠️ **Önemli**: Bu dashboard gerçek kredi kararları için kullanılmamalıdır.

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun
3. Commit yapın
4. Push edin
5. Pull request oluşturun

## 📄 Lisans

Bu proje eğitim amaçlıdır ve açık kaynaklıdır.