"""
Feature Scaling ve Encoding utilities
Fraud detection için veri ön işleme araçları
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Imputer'ların sadece fit/transform yapabilmesi için SimpleImputer kullanıldı
# KNNImputer'lar SimpleImputer ile değiştirildi.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Fraud Detection için kapsamlı feature preprocessing sınıfı"""
    
    def __init__(self, scaling_method='standard', encoding_method='onehot'):
        """
        Args:
            scaling_method (str): 'standard', 'robust', 'minmax'
            encoding_method (str): 'onehot', 'label', 'ordinal'
        """
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        
        # Scaler seçimi
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_method: 'standard', 'robust', veya 'minmax'")
        
        # Encoder seçimi
        if encoding_method == 'onehot':
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        elif encoding_method == 'label':
            # LabelEncoder tek bir sütunla çalışır, OneHotEncoder'a daha çok benzer yapıda OrdinalEncoder kullanmak daha iyidir
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', 
                                          unknown_value=-1, dtype=np.int64)
            logger.warning("Label Encoding, multi-column categorical features icin Ordinal Encoding'e cevrildi.")
        elif encoding_method == 'ordinal':
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', 
                                          unknown_value=-1, dtype=np.int64)
        else:
            raise ValueError("encoding_method: 'onehot', 'label', veya 'ordinal'")
        
        self.numerical_features = []
        self.categorical_features = []
        self.encoded_feature_names = []
        self.is_fitted = False
        
        # Missing Value Imputers (fit edilmiş değerleri tutmak için)
        self.num_imputer = SimpleImputer(strategy='mean')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        
    def identify_features(self, df):
        """
        Numerical ve categorical featureları otomatik tespit et
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        # Feature listelerini SADECE fit_transform/transform başlangıcında df'ye göre belirlemiyoruz.
        # Bunlar fit_transform sırasında belirlenmeli ve transform sırasında kullanılmalıdır.
        # Bu metod sadece dahili olarak kullanılmalıdır.
        
        # Target kolonu çıkarılmış dataframe beklenir
        numerical_features = df.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        categorical_features = df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        return numerical_features, categorical_features
        
    # handle_missing_values metodu, fit durumunu içerecek şekilde güncellenmeli.
    # Aksi halde transform sırasında fit edilmemiş imputer'lar kullanılır.
    def handle_missing_values(self, df, fit=False, strategy='mean', categorical_strategy='most_frequent'):
        """
        Eksik değerleri doldur
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Imputer'ları fit et
            strategy (str): Numerical için strateji (sadece fit=True ise kullanılır)
            categorical_strategy (str): Categorical için strateji (sadece fit=True ise kullanılır)
            
        Returns:
            pd.DataFrame: İşlenmiş dataframe
        """
        df_processed = df.copy()
        
        numerical_cols = [col for col in self.numerical_features if col in df_processed.columns]
        categorical_cols = [col for col in self.categorical_features if col in df_processed.columns]
        
        if fit:
            self.num_imputer = SimpleImputer(strategy=strategy)
            self.cat_imputer = SimpleImputer(strategy=categorical_strategy)

            if numerical_cols:
                df_processed[numerical_cols] = self.num_imputer.fit_transform(
                    df_processed[numerical_cols]
                )
            
            if categorical_cols:
                # Kategorik veriyi string olarak ele almasını sağla
                df_processed[categorical_cols] = df_processed[categorical_cols].astype(str)
                df_processed[categorical_cols] = self.cat_imputer.fit_transform(
                    df_processed[categorical_cols]
                )
        else:
            if numerical_cols:
                df_processed[numerical_cols] = self.num_imputer.transform(
                    df_processed[numerical_cols]
                )
            
            if categorical_cols:
                df_processed[categorical_cols] = df_processed[categorical_cols].astype(str)
                df_processed[categorical_cols] = self.cat_imputer.transform(
                    df_processed[categorical_cols]
                )
        
        logger.info("Eksik değerler dolduruldu")
        return df_processed
    
    # detect_outliers metodu, sadece bir utility'dir, fit_transform/transform içinde çağrılmasına gerek yoktur.
    # Çıkarıldı
    
    def create_features(self, df):
        """
        Feature engineering. Bu fonksiyon hem fit hem de transform'da çağrılır.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Feature engineered dataframe
        """
        df_new = df.copy()
        
        # Transaction amount için log transformation
        if 'Amount' in df_new.columns:
            # np.log1p kullanmak güvenlidir
            df_new['Amount_log'] = np.log1p(df_new['Amount']) 
        
        # Time features (eğer Time kolonu varsa)
        if 'Time' in df_new.columns:
            df_new['Hour'] = (df_new['Time'] // 3600) % 24
            # Gün sayısını 7'ye bölerek günün hangi gün olduğunu bulur (basit haftalık döngü)
            df_new['DayOfWeek'] = (df_new['Time'] // (3600 * 24)) % 7 
        
        # Interaction features için gerekli kolonlar
        # Hata veren kolonlar Amount_Customer_Age_ratio vb. idi. 
        # Sentetik data'da Customer_Age ve Transaction_Count_Day kolonları vardı.
        
        # Müşteri yaşı ve miktar etkileşimleri
        if 'Customer_Age' in df_new.columns and 'Amount' in df_new.columns:
            age_safe = df_new['Customer_Age'].replace(0, 1e-8) # Sifir bolunmelerini onle
            df_new['Amount_Customer_Age_ratio'] = df_new['Amount'] / age_safe
            df_new['Amount_Customer_Age_product'] = df_new['Amount'] * df_new['Customer_Age']
        
        # Önceki kodunuzdaki genel interaction feature mantığı silindi:
        # if len(self.numerical_features) >= 2: ... (Bu, feature listesi fit edilmeden önce çağrılırsa kararsızdır)
        
        logger.info("Feature engineering tamamlandı")
        return df_new
    
    def fit_transform(self, df, target_col=None):
        """
        Fit ve transform işlemlerini birlikte yap
        """
        df_processed = df.copy()
        
        # Target kolonu varsa ayır ve sonraki işlemlere targetsız devam et
        target = None
        if target_col and target_col in df_processed.columns:
            target = df_processed[target_col]
            df_processed = df_processed.drop(columns=[target_col])
        
        # 1. Feature Engineering (ilk ham veride yapılır)
        df_processed = self.create_features(df_processed)
        
        # 2. Feature types belirle ve KAYDET (fit aşamasının en kritik adımı)
        # Bu listeler transform'da kullanılacaktır!
        self.numerical_features, self.categorical_features = self.identify_features(df_processed)
        
        # 3. Missing values handle et (fit et)
        df_processed = self.handle_missing_values(df_processed, fit=True)
        
        # 4. Numerical scaling (fit et)
        if self.numerical_features:
            df_processed[self.numerical_features] = self.scaler.fit_transform(
                df_processed[self.numerical_features]
            )
            logger.info(f"Numerical features scaled with {self.scaling_method}")
        
        # 5. Categorical encoding (fit et)
        if self.categorical_features:
            if self.encoding_method == 'onehot':
                encoded_data = self.encoder.fit_transform(
                    df_processed[self.categorical_features]
                )
                
                # Feature names al ve KAYDET
                self.encoded_feature_names = self.encoder.get_feature_names_out(
                    self.categorical_features
                ).tolist()
                
                # Encoded dataframe oluştur
                encoded_df = pd.DataFrame(
                    encoded_data, 
                    columns=self.encoded_feature_names,
                    index=df_processed.index
                )
                
                # Categorical kolonları çıkar ve encoded olanları ekle
                df_processed = df_processed.drop(columns=self.categorical_features)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)
                
            else:  # label veya ordinal
                # OrdinalEncoder'i fit_transform sırasında kullanmak daha güvenlidir.
                encoded_data = self.encoder.fit_transform(df_processed[self.categorical_features])
                
                df_processed[self.categorical_features] = encoded_data
                
            logger.info(f"Categorical features encoded with {self.encoding_method}")
        
        self.is_fitted = True
        
        # Target kolonu varsa geri ekle
        if target_col and target is not None:
            df_processed[target_col] = target
        
        return df_processed
    
    def transform(self, df, target_col=None):
        """
        Sadece transform (fit edilmiş preprocessor için)
        """
        if not self.is_fitted:
            raise ValueError("Önce fit_transform metodunu çağırın")
        
        df_processed = df.copy()
        
        # Target kolonu varsa ayır
        target = None
        if target_col and target_col in df_processed.columns:
            target = df_processed[target_col]
            df_processed = df_processed.drop(columns=[target_col])
        
        # 1. Feature Engineering (Test verisine de yeni sütunları ekle)
        df_processed = self.create_features(df_processed)
        
        # Not: self.numerical_features ve self.categorical_features fit'ten gelir.
        
        # 2. Missing values handle et (transform et)
        df_processed = self.handle_missing_values(df_processed, fit=False)
        
        # 3. Numerical scaling (transform et)
        if self.numerical_features:
            # Sadece fit sırasında belirlenen ve transform edilecek feature'ları al.
            numerical_cols_to_transform = [f for f in self.numerical_features 
                                           if f in df_processed.columns]
            
            if numerical_cols_to_transform:
                df_processed[numerical_cols_to_transform] = self.scaler.transform(
                    df_processed[numerical_cols_to_transform]
                )
            
        # 4. Categorical encoding (transform et)
        if self.categorical_features:
            categorical_cols_to_transform = [f for f in self.categorical_features 
                                             if f in df_processed.columns]
            
            if categorical_cols_to_transform and self.encoding_method == 'onehot':
                encoded_data = self.encoder.transform(
                    df_processed[categorical_cols_to_transform]
                )
                
                encoded_df = pd.DataFrame(
                    encoded_data, 
                    columns=self.encoded_feature_names, # Fit'ten gelen isimleri kullan
                    index=df_processed.index
                )
                
                # Categorical kolonları çıkar ve encoded olanları ekle
                df_processed = df_processed.drop(columns=categorical_cols_to_transform)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)
                
            elif categorical_cols_to_transform:
                # label veya ordinal encoding (OrdinalEncoder fit'ten gelir)
                encoded_data = self.encoder.transform(df_processed[categorical_cols_to_transform])
                df_processed[categorical_cols_to_transform] = encoded_data
        
        # Target kolonu varsa geri ekle
        if target_col and target is not None:
            df_processed[target_col] = target
        
        return df_processed
    
    
    # ... (diğer metodlar olduğu gibi devam ediyor) ...
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """Outlier detection"""
        # ... (mevcut kodunuz) ...
        pass # Bu kısmı sadece yer tutması için bıraktım
        # Gerçek kodunuzda burayı silip mevcut detect_outliers içeriğinizi koyun.

    def visualize_distributions(self, df_original, df_processed, target_col=None):
        """Original vs processed data distributions"""
        # ... (mevcut kodunuz) ...
        pass # Gerçek kodunuzda burayı silip mevcut visualize_distributions içeriğinizi koyun.
        
    def get_feature_info(self):
        """Feature bilgilerini döndür"""
        # ... (mevcut kodunuz) ...
        pass # Gerçek kodunuzda burayı silip mevcut get_feature_info içeriğinizi koyun.


# ImbalanceHandler sınıfı, demo_preprocessing ve main fonksiyonları olduğu gibi korundu.

# ... (ImbalanceHandler sınıfı) ...
class ImbalanceHandler:
    # ... (mevcut ImbalanceHandler icerigi) ...
    @staticmethod
    def analyze_imbalance(y, target_names=None):
        # ... (mevcut kodunuz) ...
        pass # Gerçek kodunuzda burayı silip mevcut analyze_imbalance içeriğinizi koyun.
        
    @staticmethod
    def apply_smote(X, y, sampling_strategy='auto', random_state=42):
        try:
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"SMOTE uygulandı: {len(X)} -> {len(X_resampled)} samples")
            return X_resampled, y_resampled
        except Exception as e:
            logger.warning(f"SMOTE hatası: {e}. Orijinal veriler kullanılıyor.")
            # Hata durumunda (None yerine) orijinal veriyi döndürün
            return X, y 

    @staticmethod
    def apply_adasyn(X, y, sampling_strategy='auto', random_state=42):
        # ... (mevcut kodunuz) ...
        pass # Gerçek kodunuzda burayı silip mevcut apply_adasyn içeriğinizi koyun.

    @staticmethod
    def apply_smotetomek(X, y, sampling_strategy='auto', random_state=42):
        # ... (mevcut kodunuz) ...
        pass # Gerçek kodunuzda burayı silip mevcut apply_smotetomek içeriğinizi koyun.


def demo_preprocessing(data="/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta4/fraud-detection/data/processed/dataset_with_anomaly_scores_raw.csv"):
    # ... (mevcut demo_preprocessing icerigi) ...
    pass # Gerçek kodunuzda burayı silip mevcut demo_preprocessing içeriğinizi koyun.


if __name__ == "__main__":
    # demo_preprocessing'inize ihtiyacınız yoksa bu kısmı atlayın
    # demo_results = demo_preprocessing() 
    pass