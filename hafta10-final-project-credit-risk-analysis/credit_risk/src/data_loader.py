import os
from pathlib import Path
import pandas as pd
import shutil


# Bu script ile (data_loader.py) bulunduğu klasör yolunu alır
CURRENT_DIR = Path(__file__).resolve().parent

# Veri klasörlerinin ana dizini (src klasörünün bir üstündeki 'data' klasörü)
BASE_DATA_DIR = CURRENT_DIR.parent / "data" 

# Ham verilerin (raw) ve işlenmiş verilerin (processed) duracağı alt klasörler
RAW_DIR = BASE_DATA_DIR / "raw"
PROCESSED_DIR = BASE_DATA_DIR / "processed"

# Kullanılacak veri setinin dosya adı
DATA_FILENAME = "credit_risk_dataset.csv"

# Bilgisayarımdaki mutlak yedek yol - Eğer proje içinde dosya yoksa buradan otomatik kopyalanacak
LOCAL_SOURCE_PATH = Path(r"D:\Yeni Masaüstü\Kairu\DS-Final-Project-Credit\credit_risk\data\raw\credit_risk_dataset.csv")

def ensure_directories():
    """
    Gerekli veri klasörlerini (raw ve processed) oluşturur.
    parents=True: Aradaki eksik klasörleri de oluşturur.
    exist_ok=True: Klasör zaten varsa hata vermez.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def get_dataset_path():
    """
    Dataset yolunu bulmaya çalışır. 
    Önce proje içine bakar, yoksa yedek yoldan kopyalar.
    """
    target_path = RAW_DIR / DATA_FILENAME
    
    # 1. Kontrol: Proje içindeki raw klasöründe dosya var mı?
    if target_path.exists():
        return target_path
    
    # 2. Kontrol: Eğer proje içinde yoksa, yedek yolda (LOCAL_SOURCE_PATH) var mı?
    if LOCAL_SOURCE_PATH.exists():
        print(f"📦 Veri yedek konumdan proje içine kopyalanıyor...")
        ensure_directories() 
        shutil.copy(LOCAL_SOURCE_PATH, target_path) # Dosyayı kopyala
        return target_path
        
    # 3. Durum: Dosya hiçbir yerde bulunamadı ise None döner
    return None

def load_and_validate_raw_data():
    """
    Veriyi bulur, yükler ve boyutunu kontrol ederek DataFrame döndürür.
    """
    path = get_dataset_path()
    
    # Eğer dosya yolu bulunamadıysa raise ile hata fırlat diyoruz
    if not path:
        raise FileNotFoundError(
            f"❌ Dataset bulunamadı!\n"
            f"Lütfen '{DATA_FILENAME}' dosyasını şu klasöre koyun:\n"
            f"-> {RAW_DIR}"
        )
    
    # CSV dosyasını Pandas ile oku
    df = pd.read_csv(path)
    print(f"✅ Raw veri başarıyla yüklendi. Boyut: {df.shape}")
    return df

if __name__ == "__main__":
    # 1. Klasör yapısının hazır olduğundan emin ol
    ensure_directories()
    
    # 2. Veriyi yükle (Load)
    df = load_and_validate_raw_data()
    
    # 3. Veriyi işlenmiş (processed) klasörüne kaydet
    # Bu adım, ham verinin yedeğini processed klasörüne alarak sonraki adımlara hazırlar.
    output_path = PROCESSED_DIR / DATA_FILENAME
    df.to_csv(output_path, index=False)
    
    print(f"📂 Veri işlenmek üzere doğru konuma kaydedildi:")
    print(f"   -> {output_path}")