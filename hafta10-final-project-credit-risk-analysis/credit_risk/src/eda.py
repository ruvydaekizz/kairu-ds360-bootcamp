import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Bu scriptin (eda.py) bulunduğu klasör yolunu alır
CURRENT_DIR = Path(__file__).resolve().parent

# Analiz edilecek işlenmiş verinin yolu (data/processed klasörü)
DATA_PATH = CURRENT_DIR.parent / "data" / "processed" / "credit_risk_dataset.csv"

def perform_eda():
    """Bu kısımda veri setini analiz eder ve temel istatistiklerini görmekteyiz."""
    
    # 1. Veri dosyasının varlığını kontrol ediyoruz.
    if not DATA_PATH.exists():
        # Veri dosyasını bulamazsa hata mesajı döndüreceğiz.
        raise FileNotFoundError(
            f"❌ Veri dosyası bulunamadı!\n"
            f"Aranan yol: {DATA_PATH}\n"
            f"Lütfen önce 'data_loader.py' çalıştırın."
        )
        
    # 2. CSV dosyasını DataFrame olarak yüklüyoruz.
    df = pd.read_csv(DATA_PATH)
    
    # 3. Genel Yapıyı İncele
    # (Satır/sütun sayısı, veri tipleri ve bellek kullanımı)
    print("\n" + "="*30 + " GENEL BİLGİLER " + "="*30)
    print(df.info())
    
    # 4. Eksik Veri Analizi
    # (Hangi sütunda kaç tane boş/NaN değer var?)
    print("\n" + "="*30 + " EKSİK DEĞERLER " + "="*30)
    print(df.isnull().sum())
    
    # 5. İstatistiksel Özet
    # (Sayısal sütunların ortalaması, standart sapması, min/max değerleri)
    # .T ile transpoze alarak tabloyu yan çevirip daha okunaklı yapıyoruz.
    print("\n" + "="*30 + " İSTATİSTİKSEL ÖZET " + "="*30)
    print(df.describe().T)
    
    # 6. Hedef Değişken Analizi (Dengesizlik Kontrolü)
    # (Kredi ödeyenler vs ödemeyenlerin oranı nedir?)
    print("\n" + "="*30 + " HEDEF DEĞİŞKEN (LOAN_STATUS) " + "="*30)
    print(df['loan_status'].value_counts(normalize=True))
    
    # 7. Mantıksız/Aykırı Değer Kontrolü
    print("\n" + "="*30 + " AYKIRI DEĞER KONTROLÜ " + "="*30)
    
    # Yaşı 100'den büyük olan kayıtlar (Hatalı giriş olabilir)
    outlier_age = df[df['person_age'] > 100]
    print(f"⚠️ 100 yaş üstü kayıt sayısı: {len(outlier_age)}")
    
    # İş deneyimi 60 yıldan fazla olan kayıtlar (Mantıksız olabilir)
    # Not: emp_length bazı verilerde float gelebilir, NaN kontrolü
    if 'person_emp_length' in df.columns:
        outlier_emp = df[df['person_emp_length'] > 60]
        print(f"⚠️ 60 yıl üzeri iş deneyimi sayısı: {len(outlier_emp)}")
        

if __name__ == "__main__":
    # Script doğrudan çalıştırıldığında analizi başlatacak.
    perform_eda()