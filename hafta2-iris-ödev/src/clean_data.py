import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os


def clean_iris_data(input_path = r"D:\Yeni Masaüstü\Kairu\Kairu_Code_DS360\ds360bootcamp_iris_dataset\data\raw\iris.csv", output_path='data/processed/iris_processed.csv'):
    """Iris veri setini temizle ve özellik mühendisliği yap"""
    
    # Data içerisinden raw dosyasından Iris datasetini yükle
    df = pd.read_csv(input_path)
    
    # Kopyasını alalım
    df_clean = df.copy()
    
    # Eksik değer kontrolü (iris datasetinde normalde eksik yok ama genelleştirelim)
    if df_clean.isnull().any().any():
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    
    # Kategorik değişken encode edelim (species)
    le_species = LabelEncoder()
    df_clean['species_encoded'] = le_species.fit_transform(df_clean['species'])
    
    # Yeni özellikler oluşturalım
    df_clean['sepal_area'] = df_clean['sepal_length'] * df_clean['sepal_width']
    df_clean['petal_area'] = df_clean['petal_length'] * df_clean['petal_width']
    df_clean['sepal_petal_ratio'] = df_clean['sepal_length'] / (df_clean['petal_length'] + 1e-6)
    
    # Çıktı dizinini oluşturalım
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Temizlenmiş veriyi kaydedelim
    df_clean.to_csv(output_path, index=False)
    
    print("✅ Iris verisi temizlendi ve kaydedildi:", output_path)
    print(f"Orijinal boyut: {df.shape}")
    print(f"Temizlenmiş boyut: {df_clean.shape}")
    print(f"Eksik değerler:\n{df_clean.isnull().sum().sum()} toplam eksik değer")
    
    # Özelliklerin listesi
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
                'sepal_area', 'petal_area', 'sepal_petal_ratio', 'species_encoded']
    
    print(f"Model özellikleri: {features}")
    
    return df_clean, features


if __name__ == "__main__":
    clean_iris_data()
