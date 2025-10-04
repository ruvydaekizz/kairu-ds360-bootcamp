# DS360 Bootcamp: Iris Veri Seti Makine Öğrenimi Projesi


## 1. Proje Hakkında
Bu proje, DVC (Data Version Control) kullanılarak oluşturulmuş bir MLOps (Makine Öğrenimi Operasyonları) projesi şablonudur. Amacı, Iris veri setini kullanarak farklı sınıflandırma modellerini (KNN, Lojistik Regresyon, SVC, Random Forest) eğitmek ve sonuçları izlenebilir kılmaktır.


## 2. Kurulum
Projenin yerel makinenizde çalışması için aşağıdaki adımları izleyin:

git clone https://github.com/kullaniciAdi/projeAdi.git
cd projeAdi


### 2.1 Sanal Ortam Oluşturma (bash)
python -m venv iris_venv
source iris_venv/bin/activate  # Linux/Mac
source iris_venv/Scripts/activate   # Windows


### 2.2 Bağımlılıkları Yükleme
pip install -r requirements.txt


### 2.3 DVC Kurulumu ve Başlatma
Veri Versiyonlama Kontrolü (DVC) araçlarını kurun ve projeyi DVC için başlatın.

DVC'yi ve gerekli depolama bağlantılarını (ör: dvc-s3) yükleyin:
pip install dvc dvc-s3

DVC'yi projede başlatın:
dvc init


### 3. Çalıştırma
Tüm veri indirme, ön işleme ve model eğitim aşamalarını (pipeline) tek bir komutla çalıştırmak için DVC'yi kullanın.

Tüm pipeline'ı baştan sona çalıştırır ve çıktıları günceller:
dvc repro


### 4. Proje Yapısı
Projenin ana dizin yapısı, temizlik ve organizasyon için standartlaştırılmıştır.

DS360BOOTCAMP_IRIS_DATASET/
├── data/                  # Ham ve İşlenmiş Verilerin Tutulduğu Klasör
├── models/                # Eğitilmiş Model Çıktıları (.pkl, .json metrikler)
├── src/                   # Tüm Python Kod Betikleri (clean_data.py, download_data.py, train_model.py)
├── iris_venv/             # Python Sanal Ortamı (Git tarafından ignore edilir)
├── .gitignore             # Git tarafından takip edilmeyecek dosyaların listesi
├── requirements.txt       # Proje Bağımlılıkları Listesi
└── dvc.yaml               # DVC Pipeline Tanımı (Veri Akışı)

