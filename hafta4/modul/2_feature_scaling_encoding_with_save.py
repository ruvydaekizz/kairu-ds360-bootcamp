"""
Feature Scaling ve Encoding
Eğitim amaçlı detaylı script - Görseller kaydedilir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    OneHotEncoder, LabelEncoder, OrdinalEncoder,
    PowerTransformer, QuantileTransformer
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
import os
warnings.filterwarnings('ignore')

# Create visualization directory
VIZ_DIR = "visualizations/script2_feature_scaling"
os.makedirs(VIZ_DIR, exist_ok=True)

def save_and_show(fig_name, dpi=300):
    """Save figure and show"""
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/{fig_name}.png", dpi=dpi, bbox_inches='tight')
    plt.show()

print("=" * 60)
print("FEATURE SCALING VE ENCODING")
print("=" * 60)

# 1. ÖRNEK VERİ SETİ OLUŞTURMA
print("\n1. ÖRNEK VERİ SETİ OLUŞTURMA")
print("-" * 30)

# Karma veri tiplerine sahip fraud dataset benzeri
np.random.seed(42)
n_samples = 1000

# Numerical features (farklı scale'lerde)
transaction_amount = np.random.lognormal(3, 1.5, n_samples)  # 0-1000 arası
account_balance = np.random.normal(5000, 2000, n_samples)    # 1000-9000 arası
transaction_count = np.random.poisson(10, n_samples)        # 0-30 arası
account_age_days = np.random.uniform(30, 3650, n_samples)   # 30-3650 gün

# Categorical features
payment_methods = np.random.choice(['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet'], n_samples)
merchant_categories = np.random.choice(['grocery', 'restaurant', 'gas_station', 'online', 'retail'], n_samples)
risk_levels = np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.7, 0.2, 0.1])

# Target variable (fraud)
fraud_prob = (
    (transaction_amount > 500) * 0.3 +
    (risk_levels == 'high') * 0.4 +
    (payment_methods == 'digital_wallet') * 0.2 +
    np.random.random(n_samples) * 0.1
)
is_fraud = (fraud_prob > 0.5).astype(int)

# DataFrame oluştur
df = pd.DataFrame({
    'transaction_amount': transaction_amount,
    'account_balance': account_balance,
    'transaction_count': transaction_count,
    'account_age_days': account_age_days,
    'payment_method': payment_methods,
    'merchant_category': merchant_categories,
    'risk_level': risk_levels,
    'is_fraud': is_fraud
})

print("Veri seti oluşturuldu:")
print(df.head())
print(f"\nVeri şekli: {df.shape}")
print(f"Fraud oranı: %{(df['is_fraud'].sum() / len(df)) * 100:.1f}")

# Veri tiplerine bak
print(f"\nVeri tipleri:")
print(df.dtypes)

# 2. SCALING ÖNCESİ VERİ ANALİZİ
print("\n2. SCALING ÖNCESİ VERİ ANALİZİ")
print("-" * 30)

# Numerical features'ların istatistikleri
numerical_cols = ['transaction_amount', 'account_balance', 'transaction_count', 'account_age_days']
print("Numerical Features İstatistikleri:")
print(df[numerical_cols].describe())

# Scale farklılıklarını görselleştir
plt.figure(figsize=(15, 10))

# Orijinal distribution'lar
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 4, i+1)
    plt.hist(df[col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'{col} - Original')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

# Box plots
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 4, i+5)
    plt.boxplot(df[col])
    plt.title(f'{col} - Box Plot')
    plt.ylabel(col)
    plt.grid(True, alpha=0.3)

save_and_show("01_original_data_distributions")

print("Scale Farklılıkları:")
for col in numerical_cols:
    print(f"{col}: Min={df[col].min():.2f}, Max={df[col].max():.2f}, Range={df[col].max()-df[col].min():.2f}")

# 3. FEATURE SCALING METODLARı
print("\n3. FEATURE SCALING METODLARI")
print("-" * 30)

X = df[numerical_cols].copy()

# 3.1 STANDARD SCALER (Z-score normalization)
print("\n3.1 STANDARD SCALER")
print("Formül: (x - μ) / σ")
print("Sonuç: Ortalama=0, Standart sapma=1")

standard_scaler = StandardScaler()
X_standard = pd.DataFrame(
    standard_scaler.fit_transform(X),
    columns=[f"{col}_standard" for col in X.columns]
)

print("Standard Scaler Sonuçları:")
print(X_standard.describe().round(3))

# 3.2 MIN-MAX SCALER
print("\n3.2 MIN-MAX SCALER")
print("Formül: (x - min) / (max - min)")
print("Sonuç: Değerler 0-1 arasında")

minmax_scaler = MinMaxScaler()
X_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(X),
    columns=[f"{col}_minmax" for col in X.columns]
)

print("Min-Max Scaler Sonuçları:")
print(X_minmax.describe().round(3))

# 3.3 ROBUST SCALER
print("\n3.3 ROBUST SCALER")
print("Formül: (x - median) / IQR")
print("Sonuç: Outlier'lara dayanıklı")

robust_scaler = RobustScaler()
X_robust = pd.DataFrame(
    robust_scaler.fit_transform(X),
    columns=[f"{col}_robust" for col in X.columns]
)

print("Robust Scaler Sonuçları:")
print(X_robust.describe().round(3))

# 3.4 POWER TRANSFORMER
print("\n3.4 POWER TRANSFORMER (Yeo-Johnson)")
print("Amaç: Verileri normal dağılıma yaklaştır")

power_transformer = PowerTransformer(method='yeo-johnson')
X_power = pd.DataFrame(
    power_transformer.fit_transform(X),
    columns=[f"{col}_power" for col in X.columns]
)

print("Power Transformer Sonuçları:")
print(X_power.describe().round(3))

# Scaling yöntemlerini görselleştir
plt.figure(figsize=(20, 12))

scalers = [
    ('Original', X, 'blue'),
    ('Standard', X_standard, 'red'),
    ('MinMax', X_minmax, 'green'),
    ('Robust', X_robust, 'orange'),
    ('Power', X_power, 'purple')
]

for i, col in enumerate(numerical_cols):
    for j, (scaler_name, data, color) in enumerate(scalers):
        plt.subplot(4, 5, i*5 + j + 1)
        
        if scaler_name == 'Original':
            values = data[col]
        else:
            values = data[f"{col}_{scaler_name.lower()}"]
        
        plt.hist(values, bins=30, alpha=0.7, color=color, edgecolor='black')
        plt.title(f'{col}\n{scaler_name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

save_and_show("02_scaling_methods_comparison")

# 4. CATEGORICAL ENCODING
print("\n4. CATEGORICAL ENCODING")
print("-" * 30)

categorical_cols = ['payment_method', 'merchant_category', 'risk_level']
print(f"Categorical columns: {categorical_cols}")

# Her categorical column'un unique değerlerini göster
for col in categorical_cols:
    print(f"\n{col} unique values: {df[col].unique()}")
    print(f"Value counts:\n{df[col].value_counts()}")

# 4.1 LABEL ENCODING
print("\n4.1 LABEL ENCODING")
print("Her kategoriyi sayıya çevirir (0, 1, 2, ...)")

label_encoders = {}
df_label = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_label[f"{col}_label"] = le.fit_transform(df[col])
    label_encoders[col] = le
    
    print(f"\n{col} Label Encoding:")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    for category, label in mapping.items():
        print(f"  {category} -> {label}")

# 4.2 ONE-HOT ENCODING
print("\n4.2 ONE-HOT ENCODING")
print("Her kategori için ayrı binary column oluşturur")

# One-hot encoder
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid multicollinearity
categorical_data = df[categorical_cols]
onehot_encoded = onehot_encoder.fit_transform(categorical_data)

# Column names oluştur
onehot_columns = []
for i, col in enumerate(categorical_cols):
    categories = onehot_encoder.categories_[i][1:]  # drop='first' nedeniyle ilk kategoriyi atla
    for cat in categories:
        onehot_columns.append(f"{col}_{cat}")

df_onehot = pd.DataFrame(onehot_encoded, columns=onehot_columns)

print("One-Hot Encoded Columns:")
print(df_onehot.columns.tolist())
print("\nOne-Hot Encoded Data Sample:")
print(df_onehot.head())

# 4.3 ORDINAL ENCODING
print("\n4.3 ORDINAL ENCODING")
print("Kategorilere anlamlı sıra verir")

# Risk level için ordinal encoding (low < medium < high)
risk_order = [['low', 'medium', 'high']]
ordinal_encoder = OrdinalEncoder(categories=risk_order)

df_ordinal = df.copy()
df_ordinal['risk_level_ordinal'] = ordinal_encoder.fit_transform(
    df[['risk_level']]
)

print("Risk Level Ordinal Encoding:")
for i, level in enumerate(['low', 'medium', 'high']):
    print(f"  {level} -> {i}")

# Encoding methods visualization
plt.figure(figsize=(15, 10))

# Categorical data distributions
plt.subplot(2, 3, 1)
df['payment_method'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Payment Method Distribution')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
df['merchant_category'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Merchant Category Distribution')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
df['risk_level'].value_counts().plot(kind='bar', color='lightcoral')
plt.title('Risk Level Distribution')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Label encoded visualizations
plt.subplot(2, 3, 4)
plt.scatter(df_label['payment_method_label'], df_label['is_fraud'], alpha=0.6)
plt.xlabel('Payment Method (Label Encoded)')
plt.ylabel('Fraud')
plt.title('Label Encoding Example')
plt.grid(True, alpha=0.3)

# One-hot encoded visualization
plt.subplot(2, 3, 5)
onehot_sample = df_onehot.head(10)
sns.heatmap(onehot_sample.T, annot=True, cmap='Blues', cbar=False)
plt.title('One-Hot Encoding Sample')
plt.ylabel('Features')
plt.xlabel('Samples')

# Ordinal encoding
plt.subplot(2, 3, 6)
plt.scatter(df_ordinal['risk_level_ordinal'], df_ordinal['is_fraud'], alpha=0.6)
plt.xlabel('Risk Level (Ordinal)')
plt.ylabel('Fraud')
plt.title('Ordinal Encoding Example')
plt.grid(True, alpha=0.3)

save_and_show("03_encoding_methods")

# 5. ENCODING YÖNTEMLERİNİ KARŞILAŞTIRMA
print("\n5. ENCODING YÖNTEMLERİNİ KARŞILAŞTIRMA")
print("-" * 30)

# Model performansını karşılaştır
y = df['is_fraud']

# Veri setlerini hazırla
datasets = {
    'Label Encoding': pd.concat([
        X_standard,
        df_label[['payment_method_label', 'merchant_category_label', 'risk_level_label']]
    ], axis=1),
    'One-Hot Encoding': pd.concat([X_standard, df_onehot], axis=1),
    'Ordinal Encoding': pd.concat([
        X_standard,
        df_label[['payment_method_label', 'merchant_category_label']],
        df_ordinal[['risk_level_ordinal']]
    ], axis=1)
}

# Model eğitimi ve performans karşılaştırması
results = {}

for encoding_name, X_encoded in datasets.items():
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_score = accuracy_score(y_test, rf.predict(X_test))
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_score = accuracy_score(y_test, lr.predict(X_test))
    
    results[encoding_name] = {
        'Random Forest': rf_score,
        'Logistic Regression': lr_score,
        'Feature Count': X_encoded.shape[1]
    }

# Sonuçları göster
results_df = pd.DataFrame(results).T
print("Model Performans Karşılaştırması:")
print(results_df.round(3))

# Görselleştir
plt.figure(figsize=(12, 6))

# Accuracy karşılaştırması
plt.subplot(1, 2, 1)
x_pos = np.arange(len(results))
rf_scores = [results[enc]['Random Forest'] for enc in results.keys()]
lr_scores = [results[enc]['Logistic Regression'] for enc in results.keys()]

bar_width = 0.35
plt.bar(x_pos - bar_width/2, rf_scores, bar_width, label='Random Forest', alpha=0.8)
plt.bar(x_pos + bar_width/2, lr_scores, bar_width, label='Logistic Regression', alpha=0.8)

plt.xlabel('Encoding Method')
plt.ylabel('Accuracy')
plt.title('Model Performance by Encoding Method')
plt.xticks(x_pos, results.keys(), rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Feature count karşılaştırması
plt.subplot(1, 2, 2)
feature_counts = [results[enc]['Feature Count'] for enc in results.keys()]
plt.bar(results.keys(), feature_counts, color='lightcoral', alpha=0.8)
plt.xlabel('Encoding Method')
plt.ylabel('Number of Features')
plt.title('Feature Count by Encoding Method')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

save_and_show("04_encoding_performance_comparison")

# 6. SCALING YÖNTEMLERİNİ KARŞILAŞTIRMA
print("\n6. SCALING YÖNTEMLERİNİ KARŞILAŞTIRMA")
print("-" * 30)

# One-hot encoded categorical data ile combine et
categorical_encoded = df_onehot

scaling_datasets = {
    'No Scaling': pd.concat([X, categorical_encoded], axis=1),
    'Standard Scaling': pd.concat([X_standard, categorical_encoded], axis=1),
    'MinMax Scaling': pd.concat([X_minmax, categorical_encoded], axis=1),
    'Robust Scaling': pd.concat([X_robust, categorical_encoded], axis=1),
    'Power Transform': pd.concat([X_power, categorical_encoded], axis=1)
}

scaling_results = {}

for scaling_name, X_scaled in scaling_datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Logistic Regression (scaling'e hassas)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_score = accuracy_score(y_test, lr.predict(X_test))
    
    scaling_results[scaling_name] = lr_score

print("Scaling Method Karşılaştırması (Logistic Regression):")
for method, score in scaling_results.items():
    print(f"{method}: {score:.3f}")

# Görselleştir
plt.figure(figsize=(10, 6))
plt.bar(scaling_results.keys(), scaling_results.values(), alpha=0.8, color='lightblue')
plt.xlabel('Scaling Method')
plt.ylabel('Accuracy')
plt.title('Logistic Regression Performance by Scaling Method')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

save_and_show("05_scaling_performance_comparison")

# 7. BEST PRACTICES VE ÖNERİLER
print("\n7. BEST PRACTICES VE ÖNERİLER")
print("-" * 30)

print("SCALING SEÇİMİ:")
print("✓ StandardScaler:")
print("  - Data normal dağılımlı")
print("  - Logistic Regression, SVM, Neural Networks")
print("  - Outlier'lar az")

print("\n✓ MinMaxScaler:")
print("  - Data belirli aralıkta olmalı (0-1)")
print("  - Neural Networks")
print("  - Tüm değerler pozitif")

print("\n✓ RobustScaler:")
print("  - Çok outlier var")
print("  - Median ve IQR kullanır")
print("  - Outlier'lara dayanıklı")

print("\n✓ PowerTransformer:")
print("  - Skewed data")
print("  - Normal dağılıma yaklaştırmak")

print("\nENCODING SEÇİMİ:")
print("✓ Label Encoding:")
print("  - Ordinal ilişki var (low, medium, high)")
print("  - Tree-based modeller")
print("  - Az memory kullanımı")

print("\n✓ One-Hot Encoding:")
print("  - Nominal kategoriler")
print("  - Linear modeller")
print("  - Az kategori sayısı (<10)")

print("\n✓ Ordinal Encoding:")
print("  - Açık sıralama var")
print("  - Kategoriler arası mesafe önemli")

print("\nDİKKAT EDİLECEKLER:")
print("⚠️  Train set'te fit, test set'te transform")
print("⚠️  Data leakage'dan kaçın")
print("⚠️  Categorical encoding öncesi missing values")
print("⚠️  High cardinality kategoriler için özel yöntemler")
print("⚠️  Feature correlation'ı kontrol et")

# 8. PIPELINE ÖRNEĞİ
print("\n8. COMPLETE PREPROCESSING PIPELINE")
print("-" * 30)

# Pipeline tanımla
numerical_features = ['transaction_amount', 'account_balance', 'transaction_count', 'account_age_days']
categorical_features = ['payment_method', 'merchant_category', 'risk_level']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Complete pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train pipeline
X_features = df[numerical_features + categorical_features]
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.3, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
pipeline_score = pipeline.score(X_test, y_test)

print(f"Pipeline Accuracy: {pipeline_score:.3f}")
print("Pipeline tamamlandı!")

# Pipeline visualization
plt.figure(figsize=(12, 8))
plt.text(0.5, 0.9, 'PREPROCESSING PIPELINE', fontsize=20, ha='center', weight='bold')

pipeline_steps = [
    'Raw Data',
    'Numerical Features → StandardScaler',
    'Categorical Features → OneHotEncoder', 
    'Combined Features',
    'RandomForest Classifier',
    'Predictions'
]

y_positions = np.linspace(0.8, 0.1, len(pipeline_steps))
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']

for i, (step, y_pos, color) in enumerate(zip(pipeline_steps, y_positions, colors)):
    plt.text(0.5, y_pos, step, fontsize=14, ha='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
    
    if i < len(pipeline_steps) - 1:
        plt.arrow(0.5, y_pos-0.05, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.title('Complete ML Pipeline Flow', fontsize=16, weight='bold', pad=20)

save_and_show("06_pipeline_visualization")

print(f"\n📊 Toplam {len(os.listdir(VIZ_DIR))} görsel kaydedildi: {VIZ_DIR}/")
print("\n" + "=" * 60)
print("FEATURE SCALING VE ENCODING EĞİTİMİ TAMAMLANDI!")
print("=" * 60)