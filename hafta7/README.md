# Hafta 7: Metin İşleme ve NLP Optimizasyonu

Bu proje Türkçe sağlık verilerinde metin işleme, doğal dil işleme (NLP) ve model optimizasyonu tekniklerini gösterir.

## 📁 Proje Yapısı

```
hafta7/
├── data/                          # Veri dosyaları
│   ├── synthetic_health_data.py   # Sentetik sağlık verisi üreteci
│   └── synthetic_health_data.csv  # Üretilen veri seti
├── examples/                      # Örnek scriptler
│   ├── 01_text_cleaning_tokenization.py  # Metin temizleme ve tokenization
│   ├── 02_tfidf_analysis.py       # TF-IDF analizi
│   ├── 03_bert_analysis.py        # BERT tabanlı analiz
│   └── 04_model_optimization.py   # Model optimizasyonu
├── src/                          # Ana kaynak kodları
│   └── pii_masking.py            # PII maskeleme
├── api/                          # FastAPI servisi
│   ├── main.py                   # API ana dosyası
│   └── test_api.py              # API test scriptleri
├── requirements.txt              # Python bağımlılıkları
└── README.md                    # Bu dosya
```

## 🚀 Kurulum

### 1. Sanal Ortam Oluşturma ve Aktivasyon

```bash
# Hafta 7 dizinine git
cd hafta7

# Sanal ortam oluştur
python -m venv venv

# Sanal ortamı aktif et
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. NLTK Verilerini İndirme (İlk Kullanım)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 3. SpaCy İngilizce Modeli (PII Maskeleme için gerekli)

```bash
python -m spacy download en_core_web_sm
```

### 4. SpaCy Türkçe Modeli (Opsiyonel)

```bash
python -m spacy download tr_core_news_sm
```

## 📊 Veri Üretimi

Sentetik sağlık verisi oluşturmak için:

```bash
python data/synthetic_health_data.py
```

Bu script `data/synthetic_health_data.csv` dosyasını oluşturur ve aşağıdaki bilgileri içerir:
- Hasta bilgileri (ad, soyad, TC kimlik, telefon, email)
- Sağlık verileri (tanı, doktor, hastane, kan grubu)
- Adres ve notlar

## 🔧 Örneklerin Çalıştırılması

### 1. Metin Temizleme ve Tokenization

```bash
python examples/01_text_cleaning_tokenization.py
```

**Özellikler:**
- Türkçe karakterleri koruyarak metin temizleme
- NLTK ve SpaCy ile tokenization
- Stop word filtreleme
- Cümle bölütleme

### 2. TF-IDF Analizi

```bash
python examples/02_tfidf_analysis.py
```

**Özellikler:**
- Terim-frekans analizi
- Doküman benzerliği hesaplama
- K-means kümeleme
- En önemli terimlerin görselleştirilmesi

### 3. BERT Tabanlı Analiz

```bash
python examples/03_bert_analysis.py
```

**Özellikler:**
- Türkçe BERT embeddings
- Semantik benzerlik analizi
- PCA ile boyut azaltma
- Duygu analizi (opsiyonel)

### 4. Model Optimizasyonu

```bash
python examples/04_model_optimization.py
```

**Özellikler:**
- Model boyutu karşılaştırması
- DistilBERT ile hızlandırma
- Quantization teknikleri
- Embeddings sıkıştırma

## 🔐 PII Maskeleme

Kişisel bilgileri maskelemek için:

```bash
python src/pii_masking.py
```

**Maskelenen Bilgiler:**
- TC Kimlik numaraları → `[TC_KIMLIK]`
- Telefon numaraları → `[TELEFON]` / `<PHONE_NUMBER>`
- E-mail adresleri → `[EMAIL]` / `<EMAIL_ADDRESS>`
- Doktor/hasta isimleri → `[DOKTOR_ADI]` / `<PERSON>`
- Adres bilgileri → `[ADRES]`
- IBAN numaraları → `[IBAN]`

**İki Farklı Yöntem:**
- **Regex**: Türkçe odaklı pattern matching
- **Presidio**: AI tabanlı gelişmiş PII tanıma

## 🌐 FastAPI Servisi

### API Başlatma

```bash
# Ana dizinden çalıştır
python api/main.py

# Veya uvicorn ile
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### API Testleri

```bash
# API'nin çalıştığından emin olduktan sonra
python api/test_api.py
```

### API Endpoints

- `GET /` - Ana sayfa
- `POST /clean-text` - Metin temizleme
- `POST /tfidf-analysis` - TF-IDF analizi
- `POST /similarity` - Benzerlik analizi
- `POST /bert-embeddings` - BERT embeddings
- `POST /mask-pii` - PII maskeleme
- `POST /health-data-analysis` - Kapsamlı sağlık verisi analizi
- `GET /health` - Servis durumu

### Örnek API Kullanımı

```python
import requests

# Metin temizleme
response = requests.post(
    "http://localhost:8000/clean-text",
    json={"text": "Dr. Ahmet YILMAZ hastayı muayene etti!!!"}
)

# TF-IDF analizi
response = requests.post(
    "http://localhost:8000/tfidf-analysis",
    json={
        "texts": [
            "Hasta hipertansiyon tedavisi görüyor",
            "Diyabet hastası insulin kullanıyor"
        ]
    }
)

# PII maskeleme
response = requests.post(
    "http://localhost:8000/mask-pii",
    json={
        "text": "TC: 12345678901, Tel: 0532-123-4567",
        "method": "regex"
    }
)
```

## 📈 Performans Optimizasyonu

### Model Boyutu Azaltma
- **BERT**: ~440MB → **DistilBERT**: ~250MB (%43 azaltma)
- **Quantization**: Ek %20-30 boyut azaltma
- **Embedding sıkıştırma**: %50-70 boyut azaltma

### Hız Optimizasyonu
- **DistilBERT**: 2-3x daha hızlı
- **Batch processing**: Büyük veri setleri için
- **GPU kullanımı**: CUDA desteği ile

## 🛠️ Geliştirme

### Yeni Özellik Ekleme

1. `examples/` dizinine yeni script ekleyin
2. `api/main.py` dosyasına yeni endpoint ekleyin
3. `requirements.txt` dosyasını güncelleyin
4. Test script'ini `api/test_api.py` dosyasına ekleyin

### Test Etme

```bash
# Tüm scriptleri test et
python examples/01_text_cleaning_tokenization.py
python examples/02_tfidf_analysis.py
python examples/03_bert_analysis.py
python examples/04_model_optimization.py
python src/pii_masking.py

# API testleri
python api/test_api.py
```

## 📋 Önemli Notlar

### Bağımlılık Çakışmaları
- Tüm bağımlılıklar sabit sürümlerle belirtilmiştir
- Sanal ortam kullanımı zorunludur
- Conflict olması durumunda `requirements.txt` dosyasını güncelleyin

### GPU Kullanımı
- PyTorch CPU versiyonu yüklüdür
- GPU kullanımı için PyTorch CUDA versiyonunu yükleyin
- Model yükleme sırasında GPU kontrolü otomatik yapılır

### Bellek Kullanımı
- BERT modelleri yüksek bellek kullanır (~2-4GB)
- Büyük veri setleri için batch processing kullanın
- Model optimizasyonu ile bellek kullanımı azaltılabilir

## 🔍 Troubleshooting

### Yaygın Hatalar

1. **Model yüklenmiyor**: İnternet bağlantınızı kontrol edin
2. **NLTK verisi bulunamıyor**: `nltk.download()` komutlarını çalıştırın
3. **API çalışmıyor**: Port 8000'in açık olduğunu kontrol edin
4. **Bellek yetersiz**: Daha küçük batch size kullanın

### Çözümler

```bash
# Bağımlılıkları yeniden yükle
pip install --force-reinstall -r requirements.txt

# Cache temizle
pip cache purge

# Sanal ortamı yeniden oluştur
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📚 Kaynaklar

- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [NLTK Documentation](https://www.nltk.org/)
- [SpaCy Turkish Model](https://spacy.io/models/tr)

## 📄 Lisans

Bu proje eğitim amaçlıdır ve MIT lisansı altında dağıtılmaktadır.