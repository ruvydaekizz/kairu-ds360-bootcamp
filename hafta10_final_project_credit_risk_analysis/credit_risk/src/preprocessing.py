import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Bu scriptin (preprocessing.py) bulunduğu klasör
CURRENT_DIR = Path(__file__).resolve().parent

# İşlenmiş verinin durduğu yol (data/processed/credit_risk_dataset.csv)
DATA_PATH = CURRENT_DIR.parent / "data" / "processed" / "credit_risk_dataset.csv"

def load_data():
    """
    İşlenmiş veri setini diskten okur.
    Dosya yoksa raise ile hata fırlatarak uyarı verir.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"❌ İşlenmiş veri bulunamadı!\n"
            f"Aranan yol: {DATA_PATH}\n"
            f"Lütfen önce 'data_loader.py' çalıştırın."
        )
    return pd.read_csv(DATA_PATH)

def clean_outliers(df):
    """
    Veri setindeki mantıksız veya aşırı uç (aykırı) değerleri temizliyoruz.
    """
    df = df.copy()
    
    # 1. Yaş Filtresi: 100 yaşından büyük kayıtları veri setinden tamamen çıkarıyoruz.
    df = df[df['person_age'] <= 100]
    
    # 2. İş Deneyimi Filtresi: 60 yıldan fazla deneyim mantıksız olduğu için
    # bu değerleri silmek yerine NaN (boş) yapar. Pipeline (Imputer) ile bunları sonradan dolduracağız.
    df.loc[df['person_emp_length'] > 60, 'person_emp_length'] = np.nan
    
    return df

def feature_engineering(df):
    """
    Modelin daha iyi öğrenmesi için mevcut verilerden yeni özellikler türetir
    veya veri tiplerini dönüştürür.
    """
    df = df.copy()
    
    # 1. Gelir Dönüşümü (Log Transformation): 
    # Gelir verisi genelde çarpıktır (çok yüksek gelirli az kişi vardır). 
    # Logaritma alarak dağılımı normale yaklaştırıyoruz..
    df['person_income_log'] = np.log1p(df['person_income'])
    
    # 2. Kategorik -> Sayısal Dönüşüm:
    # 'Y' (Yes) ve 'N' (No) değerlerini 1 ve 0'a çevirir.
    if 'cb_person_default_on_file' in df.columns:
        df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
        
    return df

def get_preprocessor():
    """
    Scikit-learn Pipeline nesnesi oluşturuyoruz.
    Bu pipeline, veriyi modele girmeden önce otomatik olarak işlenmesini sağlayacak.
    """
    
    # Sayısal Değişkenler Listesi
    numeric_features = ['person_age', 'person_income_log', 'person_emp_length', 
                        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                        'cb_person_cred_hist_length', 'cb_person_default_on_file']
    
    # Kategorik Değişkenler Listesi
    categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade']
    
    # Sayısal Değişkenler İçin İşlemler
    # 1. SimpleImputer: Eksik verileri medyan (ortanca) ile doldur.
    # 2. StandardScaler: Verileri standartlaştır (Ortalama=0, Std. Sapma=1 yap).
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Kategorik Değişkenler İçin İşlemler
    # 1. SimpleImputer: Eksik veri varsa 'MISSING' yazarak doldur.
    # 2. OneHotEncoder: Kategorileri sayısal matrise çevir (Kira -> [1, 0, 0, 0] gibi).
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Tüm işlemleri birleştir (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Listede olmayan diğer sütunları atıyor yani Hedef Değişeni.
    )
    return preprocessor

def get_train_test_data():
    """
    Ana fonksiyon:
    1. Veriyi yükler.
    2. Temizler (Outliers).
    3. Yeni özellikler ekler (Feature Engineering).
    4. Hedef değişkeni (y) ve özellikleri (X) ayırır.
    5. Train/Test ayrımı yapar.
    """
    df = load_data()
    df = clean_outliers(df)
    df = feature_engineering(df)
    # veri sızıntısını engellemek amacıylar get_preprocessor()'a yer vermedik burada. Bu kısmı train.py'de kullanacağız çünkü.
    
    # Hedef Değişken: Kredi Durumu (0: Ödedi, 1: Ödemedi/Batık)
    y = df['loan_status']
    # Özellikler Matrisi (Hedef değişken hariç her şey)
    X = df.drop('loan_status', axis=1)
    
    # Veriyi %80 Eğitim, %20 Test olarak ayır (stratify=y ile oranları koru)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Bu dosya doğrudan çalıştırıldığında (import edilmediğinde) test amaçlı çalışır.
    try:
        X_train, X_test, y_train, y_test = get_train_test_data()
        print(f"✅ Preprocessing ve veri yolu doğrulandı.")
        print(f"   Train Boyutu: {X_train.shape}")
        print(f"   Test Boyutu:  {X_test.shape}")
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")