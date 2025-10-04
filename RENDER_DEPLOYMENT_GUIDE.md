# 🚀 Render.com Deployment Rehberi

Bu rehber, **multi-project repo yapısında** bulunan DS360 Bootcamp projelerini Render.com'da deploy etmek için hazırlanmıştır.

## 📁 Repo Yapısı

```
ds360_ikincihafta/
├── hafta2/           # MLOps projeleri
│   ├── titanic-mlops/
│   └── diğer projeler...
├── hafta3/           # Loan Risk Analysis
│   └── loan-risk-analysis/
├── hafta4/           # Gelecek projeler
└── RENDER_DEPLOYMENT_GUIDE.md
```

## 🎯 Render Deployment Seçenekleri

### Option 1: Root Directory Build (ÖNERİLEN)
Root directory'den belirli bir projeyi deploy etmek için.

### Option 2: Subtree Deployment
Sadece proje klasörünü ayrı repo'da deploy etmek.

---

## 🔥 Option 1: Root Directory Build

Bu yaklaşımda ana repo'yu Render'a bağlayıp build komutlarıyla istediğiniz projeyi seçersiniz.

### Step 1: GitHub Repository Hazırlığı

1. Reponuzun public olduğundan emin olun
2. Main branch'de tüm değişiklikleri commit edin

### Step 2: Render Service Oluşturma

1. [Render.com](https://render.com)'a giriş yapın
2. **New +** → **Web Service** seçin
3. GitHub reponuzu seçin: `ds360_ikincihafta`

### Step 3: Build & Deploy Ayarları

#### 🏗️ Build Settings

| Setting | Value |
|---------|-------|
| **Name** | `ds360-loan-risk-analysis` |
| **Region** | `Oregon (US West)` |
| **Branch** | `main` |
| **Root Directory** | `hafta3/loan-risk-analysis` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true` |

#### 📋 Detaylı Konfigürasyon

```yaml
# Render Build Configuration
root_directory: hafta3/loan-risk-analysis
build_command: pip install -r requirements.txt
start_command: streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

### Step 4: Environment Variables

Gerekirse environment variables ekleyin:

| Variable | Value | Description |
|----------|-------|-------------|
| `PYTHON_VERSION` | `3.12.7` | Python versiyonu |
| `PORT` | `10000` | Render port (otomatik) |

### Step 5: Deploy & Monitor

1. **Create Web Service** butonuna tıklayın
2. Build loglarını izleyin
3. Deploy tamamlandığında URL'nizi alın

---

## 🌟 Hafta 3 - Loan Risk Analysis Özel Ayarları

### Gerekli Dosyalar ✅

Hafta 3 projesi için aşağıdaki dosyalar hazırlanmıştır:

```
hafta3/loan-risk-analysis/
├── Procfile                    # ✅ Start komutu
├── runtime.txt                 # ✅ Python 3.12.7
├── requirements.txt            # ✅ Dependencies
├── streamlit_app/
│   └── app.py                 # ✅ Optimized app
├── artifacts/                 # ✅ Model files
│   ├── model_xgb_smote.pkl
│   ├── preprocessor_smote.pkl
│   └── feature_schema_smote.json
└── data/                      # ✅ Dataset
```

### Build Command İçin Özel Notlar

```bash
# Render build command
pip install -r requirements.txt

# Start command (Procfile'dan otomatik alınır)
streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

---

## 🔄 Diğer Haftalar İçin Template

### Hafta 2 - Titanic MLOps (FastAPI)

```yaml
root_directory: hafta2/titanic-mlops
build_command: pip install -r requirements.txt
start_command: uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

### Hafta 4+ - Gelecek Projeler

Yeni projeler için aynı pattern'i kullanın:

```yaml
root_directory: hafta4/[proje-adı]
build_command: pip install -r requirements.txt
start_command: [proje-spesifik-komut]
```

---

## 🛠️ Troubleshooting

### Common Issues

1. **Build Failed - Requirements**
   ```bash
   # Solution: requirements.txt kontrolü
   pip install -r requirements.txt
   ```

2. **Port Binding Error**
   ```bash
   # Solution: --server.port=$PORT kullanın
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **File Not Found - Artifacts**
   ```bash
   # Solution: Model dosyalarının repo'da olduğunu kontrol edin
   ls hafta3/loan-risk-analysis/artifacts/
   ```

4. **Python Version Mismatch**
   ```bash
   # Solution: runtime.txt ekleyin
   echo "python-3.12.7" > runtime.txt
   ```

### Debug Commands

```bash
# Local test
cd hafta3/loan-risk-analysis
pip install -r requirements.txt
streamlit run streamlit_app/app.py

# File existence check
ls -la artifacts/
ls -la streamlit_app/
```

---

## 🎓 Öğrenci Rehberi

### 1. Kendi Fork'unuzu Oluşturun

```bash
# 1. Bu repo'yu fork edin
# 2. Kendi fork'unuzu clone edin
git clone https://github.com/[USERNAME]/ds360_ikincihafta.git
cd ds360_ikincihafta
```

### 2. Proje Değişiklikleri Yapın

```bash
# Hafta 3 projesinde değişiklik yapmak için
cd hafta3/loan-risk-analysis/

# Kendi özelleştirmelerinizi ekleyin
# Model parametrelerini değiştirin
# Dashboard'u geliştirin
```

### 3. Render'a Deploy Edin

1. Fork'unuzu Render'a bağlayın
2. Root directory: `hafta3/loan-risk-analysis` 
3. Build & deploy ayarlarını yukarıdaki gibi yapın

### 4. URL'nizi Paylaşın

Deploy tamamlandığında örnek URL:
```
https://ds360-loan-risk-analysis-abc123.onrender.com
```

---

## 📚 Faydalı Linkler

- [Render Python Documentation](https://render.com/docs/deploy-python)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [Multi-Project Repository Best Practices](https://render.com/docs/monorepos)

---

## 💡 Pro Tips

1. **Free Tier Limitations**: Free tier 750 saat/ay limit
2. **Cold Starts**: Bedava servislerde 15dk sonra uyku modu
3. **Build Time**: İlk build 5-10dk sürebilir
4. **Auto-Deploy**: GitHub'a push → otomatik deploy
5. **Custom Domain**: Paid plan ile özel domain kullanabilirsiniz

---

**🎯 Bu rehber ile DS360 Bootcamp projelerinizi kolayca deploy edebilirsiniz!**

> Sorularınız için: GitHub Issues veya bootcamp Slack kanalı