import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os

def train_model(model_type='random_forest'):
    """Model eğit ve kaydet"""
    
    # İşlenmiş veriyi yükleyelim
    df = pd.read_csv(r'D:\Yeni Masaüstü\Kairu\Kairu_Code_DS360\ds360bootcamp_iris_dataset\data\processed\iris_processed.csv')
    
    # Özellikler listesi
    features_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'sepal_area', 'petal_area', 'sepal_petal_ratio']
    
    # Bağımlı ve bağımsız (X ve y'yi) ayır    
    X = df[features_cols]
    y = df['species_encoded']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model seçme adımı
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    elif model_type == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=5
        )
    elif model_type == 'svc':
        model = SVC(
            kernel='linear',  
            random_state=42
        )
    else:
        raise ValueError(f"Bilinmeyen model tipi: {model_type}")
    
    # Model eğitme adımı
    model.fit(X_train, y_train)
    
    # Tahmin değerini alalım
    y_pred = model.predict(X_test)
    
    # Metrik değerimiz (accuracy)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Model kaydetme adımı
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_type}_model.pkl'
    joblib.dump(model, model_path)
    
    # Metrikler
    metrics = {
        'accuracy': float(accuracy),
        'n_features': len(features_cols),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    print(f"✅ Model eğitildi: {model_type}")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"💾 Model kaydedildi: {model_path}")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics

if __name__ == "__main__":
    # Models metrics dictionary
    all_metrics = {}
    
    # Random Forest modeli eğit
    all_metrics['random_forest'] = train_model('random_forest')
    
    # Logistic Regression modeli eğit
    all_metrics['logistic_regression'] = train_model('logistic_regression')
    
    # KNN modeli eğit
    all_metrics['knn'] = train_model('knn')
    
    # SVC modeli eğit
    all_metrics['svc'] = train_model('svc')
    
    # Tüm metrics'i tek JSON dosyasında kaydet
    os.makedirs('models', exist_ok=True)
    with open('models/metrics_train_model.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\n🎯 Tüm modeller eğitildi ve metrics dosyası kaydedildi!")
