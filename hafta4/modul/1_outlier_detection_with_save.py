"""
Outlier Detection - Isolation Forest ve LOF
Eğitim amaçlı detaylı script - Görseller kaydedilir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# Create visualization directory
VIZ_DIR = "visualizations/script1_outlier_detection"
os.makedirs(VIZ_DIR, exist_ok=True)

def save_and_show(fig_name, dpi=300):
    """Save figure and show"""
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/{fig_name}.png", dpi=dpi, bbox_inches='tight')
    plt.show()

print("=" * 60)
print("OUTLIER DETECTION - ISOLATION FOREST VE LOF")
print("=" * 60)

# 1. VERİ SETİ HAZIRLIĞI
print("\n1. VERİ SETİ HAZIRLIĞI")
print("-" * 30)

# Fraud benzeri veri seti oluştur
X, y = make_classification(
    n_samples=1000, 
    n_features=2, 
    n_informative=2,
    n_redundant=0, 
    n_clusters_per_class=1,
    weights=[0.9, 0.1],  # %10 fraud
    random_state=42
)

# DataFrame'e çevir
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df['is_fraud'] = y

print(f"Toplam işlem sayısı: {len(df)}")
print(f"Normal işlem: {len(df[df['is_fraud'] == 0])}")
print(f"Fraud işlem: {len(df[df['is_fraud'] == 1])}")
print(f"Fraud oranı: %{(df['is_fraud'].sum() / len(df)) * 100:.1f}")

# 2. VERİ GÖRSELLEŞTIRME
print("\n2. VERİ GÖRSELLEŞTIRME")
print("-" * 30)

plt.figure(figsize=(12, 5))

# Original data
plt.subplot(1, 2, 1)
colors = ['blue', 'red']
labels = ['Normal', 'Fraud']
for i in range(2):
    mask = df['is_fraud'] == i
    plt.scatter(df[mask]['Feature_1'], df[mask]['Feature_2'], 
               c=colors[i], label=labels[i], alpha=0.6)
plt.title('Original Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Data statistics
plt.subplot(1, 2, 2)
df.boxplot(column=['Feature_1', 'Feature_2'], ax=plt.gca())
plt.title('Feature Distributions')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

save_and_show("01_data_distribution")

# 3. ISOLATION FOREST
print("\n3. ISOLATION FOREST")
print("-" * 30)

print("Isolation Forest Nedir?")
print("- Ağaç tabanlı anomali tespit algoritması")
print("- Outlier'ları izole etmek için random splits kullanır")
print("- Outlier'lar daha az split ile izole edilir")
print("- Normal noktalar daha çok split gerektirir")

# Isolation Forest uygula
iso_forest = IsolationForest(
    n_estimators=100,      # Ağaç sayısı
    contamination=0.1,     # Beklenen outlier oranı
    random_state=42,
    n_jobs=-1
)

# Fit ve predict
X_features = df[['Feature_1', 'Feature_2']].values
iso_predictions = iso_forest.fit_predict(X_features)
iso_scores = iso_forest.decision_function(X_features)

# Sonuçları dataframe'e ekle
df['iso_outlier'] = iso_predictions  # -1: outlier, 1: normal
df['iso_score'] = iso_scores

print(f"\nIsolation Forest Sonuçları:")
print(f"Tespit edilen outlier sayısı: {sum(iso_predictions == -1)}")
print(f"Outlier oranı: %{(sum(iso_predictions == -1) / len(df)) * 100:.1f}")

# Performans analizi
iso_outliers = iso_predictions == -1
actual_frauds = df['is_fraud'] == 1

tp = sum(iso_outliers & actual_frauds)  # True Positive
fp = sum(iso_outliers & ~actual_frauds) # False Positive
fn = sum(~iso_outliers & actual_frauds) # False Negative

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPerformans Metrikleri:")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1_score:.3f}")

# 4. LOCAL OUTLIER FACTOR (LOF)
print("\n4. LOCAL OUTLIER FACTOR (LOF)")
print("-" * 30)

print("LOF Nedir?")
print("- Local density tabanlı anomali tespit")
print("- Bir noktanın komşularının density'si ile karşılaştırır")
print("- LOF > 1: Outlier (komşularından daha az dense)")
print("- LOF ≈ 1: Normal (komşularıyla benzer density)")

# LOF uygula
lof = LocalOutlierFactor(
    n_neighbors=20,        # Komşu sayısı
    contamination=0.1,     # Beklenen outlier oranı
    n_jobs=-1
)

# Fit ve predict
lof_predictions = lof.fit_predict(X_features)
lof_scores = lof.negative_outlier_factor_  # Negatif LOF skorları

# Sonuçları dataframe'e ekle
df['lof_outlier'] = lof_predictions  # -1: outlier, 1: normal
df['lof_score'] = lof_scores

print(f"\nLOF Sonuçları:")
print(f"Tespit edilen outlier sayısı: {sum(lof_predictions == -1)}")
print(f"Outlier oranı: %{(sum(lof_predictions == -1) / len(df)) * 100:.1f}")

# LOF performans analizi
lof_outliers = lof_predictions == -1

tp_lof = sum(lof_outliers & actual_frauds)
fp_lof = sum(lof_outliers & ~actual_frauds)
fn_lof = sum(~lof_outliers & actual_frauds)

precision_lof = tp_lof / (tp_lof + fp_lof) if (tp_lof + fp_lof) > 0 else 0
recall_lof = tp_lof / (tp_lof + fn_lof) if (tp_lof + fn_lof) > 0 else 0
f1_score_lof = 2 * (precision_lof * recall_lof) / (precision_lof + recall_lof) if (precision_lof + recall_lof) > 0 else 0

print(f"\nLOF Performans Metrikleri:")
print(f"Precision: {precision_lof:.3f}")
print(f"Recall: {recall_lof:.3f}")
print(f"F1-Score: {f1_score_lof:.3f}")

# 5. SONUÇLARI GÖRSELLEŞTIRME
print("\n5. SONUÇLARI GÖRSELLEŞTIRME")
print("-" * 30)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Isolation Forest sonuçları
ax1 = axes[0, 0]
colors_iso = ['blue' if x == 1 else 'red' for x in iso_predictions]
scatter = ax1.scatter(df['Feature_1'], df['Feature_2'], c=colors_iso, alpha=0.6)
ax1.set_title('Isolation Forest Results')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.grid(True, alpha=0.3)

# LOF sonuçları
ax2 = axes[0, 1]
colors_lof = ['blue' if x == 1 else 'red' for x in lof_predictions]
ax2.scatter(df['Feature_1'], df['Feature_2'], c=colors_lof, alpha=0.6)
ax2.set_title('LOF Results')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.grid(True, alpha=0.3)

# Score dağılımları
ax3 = axes[1, 0]
ax3.hist(iso_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax3.axvline(x=iso_forest.offset_, color='red', linestyle='--', label='Threshold')
ax3.set_title('Isolation Forest Scores')
ax3.set_xlabel('Anomaly Score')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.hist(lof_scores, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
ax4.axvline(x=-1, color='red', linestyle='--', label='Normal Threshold')
ax4.set_title('LOF Scores (Negative)')
ax4.set_xlabel('LOF Score')
ax4.set_ylabel('Frequency')
ax4.legend()
ax4.grid(True, alpha=0.3)

save_and_show("02_outlier_detection_results")

# 6. MODEL KARŞILAŞTIRMA
print("\n6. MODEL KARŞILAŞTIRMA")
print("-" * 30)

comparison_df = pd.DataFrame({
    'Model': ['Isolation Forest', 'LOF'],
    'Precision': [precision, precision_lof],
    'Recall': [recall, recall_lof],
    'F1-Score': [f1_score, f1_score_lof],
    'Detected_Outliers': [sum(iso_predictions == -1), sum(lof_predictions == -1)]
})

print(comparison_df.round(3))

# Confusion matrix karşılaştırması
plt.figure(figsize=(12, 5))

# Isolation Forest confusion matrix
plt.subplot(1, 2, 1)
confusion_iso = pd.crosstab(
    df['is_fraud'], 
    iso_predictions, 
    rownames=['Actual'], 
    colnames=['Predicted']
)
sns.heatmap(confusion_iso, annot=True, fmt='d', cmap='Blues')
plt.title('Isolation Forest Confusion Matrix')

# LOF confusion matrix
plt.subplot(1, 2, 2)
confusion_lof = pd.crosstab(
    df['is_fraud'], 
    lof_predictions, 
    rownames=['Actual'], 
    colnames=['Predicted']
)
sns.heatmap(confusion_lof, annot=True, fmt='d', cmap='Reds')
plt.title('LOF Confusion Matrix')

save_and_show("03_confusion_matrices")

# 7. HİPERPARAMETRE ETKİSİ
print("\n7. HİPERPARAMETRE ETKİSİ")
print("-" * 30)

print("Isolation Forest Parametreleri:")
print("- n_estimators: Ağaç sayısı (↑ daha stabil)")
print("- contamination: Beklenen outlier oranı")
print("- max_samples: Her ağaç için kullanılan sample sayısı")

print("\nLOF Parametreleri:")
print("- n_neighbors: Komşu sayısı (↑ daha global, ↓ daha local)")
print("- contamination: Beklenen outlier oranı")

# Contamination etkisi
contaminations = [0.05, 0.1, 0.15, 0.2]
results = []

for cont in contaminations:
    # Isolation Forest
    iso_temp = IsolationForest(contamination=cont, random_state=42)
    iso_pred_temp = iso_temp.fit_predict(X_features)
    iso_f1 = 2 * sum((iso_pred_temp == -1) & actual_frauds) / (sum(iso_pred_temp == -1) + sum(actual_frauds))
    
    # LOF
    lof_temp = LocalOutlierFactor(contamination=cont, n_neighbors=20)
    lof_pred_temp = lof_temp.fit_predict(X_features)
    lof_f1 = 2 * sum((lof_pred_temp == -1) & actual_frauds) / (sum(lof_pred_temp == -1) + sum(actual_frauds))
    
    results.append({
        'contamination': cont,
        'iso_f1': iso_f1,
        'lof_f1': lof_f1
    })

results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
plt.plot(results_df['contamination'], results_df['iso_f1'], 'o-', label='Isolation Forest', linewidth=2)
plt.plot(results_df['contamination'], results_df['lof_f1'], 's-', label='LOF', linewidth=2)
plt.xlabel('Contamination Rate')
plt.ylabel('F1-Score')
plt.title('Contamination Rate vs Performance')
plt.legend()
plt.grid(True, alpha=0.3)

save_and_show("04_contamination_analysis")

print("\nContamination Rate Analizi:")
print(results_df.round(3))

# 8. GERÇEK DÜNYA UYGULAMALARI
print("\n8. GERÇEK DÜNYA UYGULAMALARI")
print("-" * 30)

print("Isolation Forest Kullanım Alanları:")
print("✓ Fraud detection")
print("✓ Network intrusion detection")
print("✓ Medical anomaly detection")
print("✓ Industrial equipment monitoring")
print("✓ High-dimensional data")

print("\nLOF Kullanım Alanları:")
print("✓ Local cluster anomalies")
print("✓ Spatial data analysis")
print("✓ Image processing")
print("✓ Social network analysis")
print("✓ Time series anomalies")

print("\nHangisini Seçmeli?")
print("Isolation Forest:")
print("- Büyük veri setleri")
print("- Yüksek boyutlu veriler")
print("- Global anomaliler")
print("- Hızlı sonuç gerekli")

print("\nLOF:")
print("- Küçük-orta veri setleri")
print("- Local anomaliler önemli")
print("- Density-based anomaliler")
print("- Detaylı analiz gerekli")

print(f"\n📊 Toplam {len(os.listdir(VIZ_DIR))} görsel kaydedildi: {VIZ_DIR}/")
print("\n" + "=" * 60)
print("OUTLIER DETECTION EĞİTİMİ TAMAMLANDI!")
print("=" * 60)