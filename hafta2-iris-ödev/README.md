# KAIRU Data Science 360 Bootcamp 🚀
## DS360 Bootcamp: Iris Veri Seti Makine Öğrenimi Projesi


### 1. Proje Hakkında

Bu proje, DVC (Data Version Control) kullanılarak oluşturulmuş bir MLOps (Makine Öğrenimi Operasyonları) projesi şablonudur. Amacı, Iris veri setini kullanarak farklı sınıflandırma modellerini (KNN, Lojistik Regresyon, SVC, Random Forest) eğitmek ve sonuçları izlenebilir kılmaktır.


### 2. Kurulum

Projenin yerel makinenizde çalışması için aşağıdaki adımları izleyin:

git clone https://github.com/kullaniciAdi/projeAdi.git

cd projeAdi


### 2.1 Sanal Ortam Oluşturma (bash)

python -m venv sanal_ortam_adi

source sanal_ortam_adi/bin/activate  # Linux/Mac

source sanal_ortam_adi/Scripts/activate   # Windows


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

├── data/             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (Ham ve İşlenmiş Verilerin Tutulduğu Klasör)

├── models/           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (Eğitilmiş Model Çıktıları (.pkl, .json metrikler))

├── src/              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (Tüm Python Kod Betikleri (clean_data.py, download_data.py, train_model.py))

├── iris_venv/        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (Python Sanal Ortamı (Git tarafından ignore edilir))

├── .gitignore        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (Git tarafından takip edilmeyecek dosyaların listesi)

├── requirements.txt  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (Proje Bağımlılıkları Listesi)

└── dvc.yaml          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (DVC Pipeline Tanımı (Veri Akışı))
