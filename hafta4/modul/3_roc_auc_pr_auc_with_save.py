"""
ROC-AUC ve PR-AUC Metrikleri
Eğitim amaçlı detaylı script - Görseller kaydedilir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import warnings
import os
warnings.filterwarnings('ignore')

# Create visualization directory
VIZ_DIR = "visualizations/script3_roc_pr_auc"
os.makedirs(VIZ_DIR, exist_ok=True)

def save_and_show(fig_name, dpi=300):
    """Save figure and show"""
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/{fig_name}.png", dpi=dpi, bbox_inches='tight')
    plt.show()

print("=" * 60)
print("ROC-AUC VE PR-AUC METRİKLERİ")
print("=" * 60)

# 1. IMBALANCED FRAUD DATASET OLUŞTURMA
print("\n1. IMBALANCED FRAUD DATASET OLUŞTURMA")
print("-" * 30)

# Farklı imbalance oranlarında veri setleri oluştur
datasets = {}
imbalance_ratios = [0.5, 0.1, 0.05, 0.01]  # Fraud oranları

for ratio in imbalance_ratios:
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=8,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[1-ratio, ratio],
        random_state=42
    )
    datasets[ratio] = (X, y)
    print(f"Fraud ratio {ratio:.0%}: Normal={sum(y==0)}, Fraud={sum(y==1)}")

# Ana veri seti olarak %5 fraud kullanalım
X, y = datasets[0.05]
print(f"\nAna veri seti: {len(y)} transaction, {sum(y)} fraud (%{(sum(y)/len(y)*100):.1f})")

# 2. BASIC CLASSIFICATION METRICS
print("\n2. BASIC CLASSIFICATION METRICS")
print("-" * 30)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Model eğit
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Fraud probability

print("CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Confusion matrix'i görselleştir
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraud'], 
            yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.subplot(1, 2, 2)
sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Reds',
            xticklabels=['Normal', 'Fraud'], 
            yticklabels=['Normal', 'Fraud'])
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

save_and_show("01_confusion_matrices")

# Confusion matrix'ten metrikler
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Breakdown:")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nBASIC METRICS:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall (Sensitivity): {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# Additional metrics
specificity = tn / (tn + fp)
npv = tn / (tn + fn)  # Negative Predictive Value

print(f"Specificity: {specificity:.3f}")
print(f"Negative Predictive Value: {npv:.3f}")

print("\nMETRİK AÇIKLAMALARI:")
print("Precision = TP / (TP + FP) - Fraud dediğimizin kaçı gerçekten fraud")
print("Recall = TP / (TP + FN) - Gerçek fraudların kaçını yakaladık")
print("Specificity = TN / (TN + FP) - Normal işlemleri doğru tanıma")
print("F1-Score = 2 * (Precision * Recall) / (Precision + Recall)")

# 3. ROC CURVE VE ROC-AUC
print("\n3. ROC CURVE VE ROC-AUC")
print("-" * 30)

print("ROC (Receiver Operating Characteristic) Curve:")
print("- X-axis: False Positive Rate (FPR) = FP / (FP + TN)")
print("- Y-axis: True Positive Rate (TPR) = TP / (TP + FN) = Recall")
print("- Farklı threshold'larda TPR vs FPR")

# ROC curve hesapla
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"\nROC-AUC Score: {roc_auc:.3f}")

# ROC curve görselleştir
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Threshold analizi
plt.subplot(1, 3, 2)
# Ensure arrays have same length for ROC
min_len_roc = min(len(thresholds_roc), len(fpr), len(tpr))
thresholds_roc_sync = thresholds_roc[:min_len_roc-1]
fpr_sync = fpr[:min_len_roc-1]
tpr_sync = tpr[:min_len_roc-1]

plt.plot(thresholds_roc_sync, fpr_sync, 'b-', label='False Positive Rate')
plt.plot(thresholds_roc_sync, tpr_sync, 'r-', label='True Positive Rate')
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.title('TPR and FPR vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

# Probability distribution
plt.subplot(1, 3, 3)
plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Normal', color='blue')
plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Fraud', color='red')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Probability Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

save_and_show("02_roc_analysis")

# ROC-AUC Interpretation
print("\nROC-AUC YORUMLAMA:")
print("1.0: Mükemmel classifier")
print("0.9-1.0: Mükemmel")
print("0.8-0.9: İyi")
print("0.7-0.8: Adil")
print("0.6-0.7: Zayıf")
print("0.5: Random (rastgele tahmin)")
print("< 0.5: Random'dan kötü")

# 4. PRECISION-RECALL CURVE VE PR-AUC
print("\n4. PRECISION-RECALL CURVE VE PR-AUC")
print("-" * 30)

print("PR (Precision-Recall) Curve:")
print("- X-axis: Recall = TP / (TP + FN)")
print("- Y-axis: Precision = TP / (TP + FP)")
print("- Imbalanced dataset'lerde daha informatif")

# Precision-Recall curve hesapla
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

print(f"\nPR-AUC Score: {pr_auc:.3f}")

# No-skill baseline (fraud oranı)
no_skill = len(y_test[y_test==1]) / len(y_test)
print(f"No-skill baseline: {no_skill:.3f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
plt.axhline(y=no_skill, color='red', linestyle='--', label=f'No Skill = {no_skill:.3f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# Threshold analysis for PR
plt.subplot(1, 3, 2)
# Ensure arrays have same length for PR
min_len_pr = min(len(thresholds_pr), len(precision_vals), len(recall_vals))
thresholds_pr_sync = thresholds_pr[:min_len_pr-1]
precision_pr_sync = precision_vals[:min_len_pr-1]
recall_pr_sync = recall_vals[:min_len_pr-1]

plt.plot(thresholds_pr_sync, precision_pr_sync, 'b-', label='Precision')
plt.plot(thresholds_pr_sync, recall_pr_sync, 'r-', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

# F1-Score vs Threshold
# Use the same synchronized arrays from PR analysis
f1_scores = 2 * (precision_pr_sync * recall_pr_sync) / (precision_pr_sync + recall_pr_sync + 1e-8)  # Add small epsilon to avoid division by zero
plt.subplot(1, 3, 3)
plt.plot(thresholds_pr_sync, f1_scores, 'g-', label='F1-Score')
best_f1_idx = np.argmax(f1_scores)
best_threshold = thresholds_pr_sync[best_f1_idx]
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best threshold = {best_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('F1-Score')
plt.title('F1-Score vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

save_and_show("03_precision_recall_analysis")

print(f"Best F1-Score threshold: {best_threshold:.3f}")
print(f"Best F1-Score: {f1_scores[best_f1_idx]:.3f}")

# 5. ROC-AUC VS PR-AUC KARŞILAŞTIRMA
print("\n5. ROC-AUC VS PR-AUC KARŞILAŞTIRMA")
print("-" * 30)

# Farklı imbalance oranlarında performans
results = []

for ratio in imbalance_ratios:
    X_temp, y_temp = datasets[ratio]
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
        X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp
    )
    
    # Model eğit
    model_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    model_temp.fit(X_train_temp, y_train_temp)
    y_pred_proba_temp = model_temp.predict_proba(X_test_temp)[:, 1]
    
    # Metrics
    roc_auc_temp = auc(*roc_curve(y_test_temp, y_pred_proba_temp)[:2])
    pr_auc_temp = average_precision_score(y_test_temp, y_pred_proba_temp)
    baseline_temp = sum(y_test_temp) / len(y_test_temp)
    
    results.append({
        'fraud_ratio': ratio,
        'roc_auc': roc_auc_temp,
        'pr_auc': pr_auc_temp,
        'baseline': baseline_temp
    })

results_df = pd.DataFrame(results)
print("Imbalance Ratio Analizi:")
print(results_df.round(3))

# Görselleştir
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(results_df['fraud_ratio'], results_df['roc_auc'], 'o-', label='ROC-AUC', linewidth=2)
plt.plot(results_df['fraud_ratio'], results_df['pr_auc'], 's-', label='PR-AUC', linewidth=2)
plt.plot(results_df['fraud_ratio'], results_df['baseline'], '^--', label='No-skill baseline', linewidth=2)
plt.xlabel('Fraud Ratio')
plt.ylabel('AUC Score')
plt.title('ROC-AUC vs PR-AUC by Imbalance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')

plt.subplot(1, 2, 2)
bars1 = plt.bar([f"{r:.0%}" for r in results_df['fraud_ratio']], 
                results_df['roc_auc'], alpha=0.7, label='ROC-AUC')
bars2 = plt.bar([f"{r:.0%}" for r in results_df['fraud_ratio']], 
                results_df['pr_auc'], alpha=0.7, label='PR-AUC')
plt.xlabel('Fraud Ratio')
plt.ylabel('AUC Score')
plt.title('AUC Scores Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

save_and_show("04_imbalance_comparison")

# 6. THRESHOLD OPTİMİZASYONU
print("\n6. THRESHOLD OPTİMİZASYONU")
print("-" * 30)

print("Threshold seçimi fraud detection'da kritik:")
print("- Düşük threshold: Daha fazla fraud yakalama, ama daha fazla false alarm")
print("- Yüksek threshold: Daha az false alarm, ama fraud kaçırma riski")

# Farklı thresholds test et
thresholds_test = np.linspace(0.1, 0.9, 9)
threshold_results = []

for threshold in thresholds_test:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    
    # Metrics
    cm_thresh = confusion_matrix(y_test, y_pred_threshold)
    tn, fp, fn, tp = cm_thresh.ravel()
    
    precision_thresh = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_thresh = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_thresh = 2 * (precision_thresh * recall_thresh) / (precision_thresh + recall_thresh) if (precision_thresh + recall_thresh) > 0 else 0
    
    # Business metrics
    fraud_detected = tp
    false_alarms = fp
    fraud_missed = fn
    
    threshold_results.append({
        'threshold': threshold,
        'precision': precision_thresh,
        'recall': recall_thresh,
        'f1_score': f1_thresh,
        'fraud_detected': fraud_detected,
        'false_alarms': false_alarms,
        'fraud_missed': fraud_missed,
        'total_alerts': tp + fp
    })

threshold_df = pd.DataFrame(threshold_results)
print("Threshold Analysis:")
print(threshold_df.round(3))

# Threshold görselleştirme
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(threshold_df['threshold'], threshold_df['precision'], 'o-', label='Precision')
plt.plot(threshold_df['threshold'], threshold_df['recall'], 's-', label='Recall')
plt.plot(threshold_df['threshold'], threshold_df['f1_score'], '^-', label='F1-Score')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.bar(threshold_df['threshold'], threshold_df['total_alerts'], alpha=0.7, color='orange')
plt.xlabel('Threshold')
plt.ylabel('Total Alerts')
plt.title('Alert Volume vs Threshold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.bar(threshold_df['threshold'], threshold_df['fraud_detected'], alpha=0.7, color='green', label='Fraud Detected')
plt.bar(threshold_df['threshold'], threshold_df['fraud_missed'], alpha=0.7, color='red', label='Fraud Missed')
plt.xlabel('Threshold')
plt.ylabel('Count')
plt.title('Fraud Detection vs Missed')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.bar(threshold_df['threshold'], threshold_df['false_alarms'], alpha=0.7, color='lightcoral')
plt.xlabel('Threshold')
plt.ylabel('False Alarms')
plt.title('False Alarms vs Threshold')
plt.grid(True, alpha=0.3)

save_and_show("05_threshold_optimization")

# Optimal threshold seçimi
print("\nOPTIMAL THRESHOLD SEÇİMİ:")

# F1-Score'a göre
best_f1_idx = threshold_df['f1_score'].idxmax()
best_f1_threshold = threshold_df.loc[best_f1_idx, 'threshold']
print(f"Best F1-Score threshold: {best_f1_threshold}")

# Youden's Index (TPR + TNR - 1)
youdens_scores = []
for threshold in thresholds_test:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    cm_thresh = confusion_matrix(y_test, y_pred_threshold)
    tn, fp, fn, tp = cm_thresh.ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    youdens_index = tpr + tnr - 1
    youdens_scores.append(youdens_index)

best_youdens_idx = np.argmax(youdens_scores)
best_youdens_threshold = thresholds_test[best_youdens_idx]
print(f"Best Youden's Index threshold: {best_youdens_threshold}")

# 7. MULTIPLE MODELS COMPARISON
print("\n7. MULTIPLE MODELS COMPARISON")
print("-" * 30)

# Farklı modeller
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

model_results = {}

plt.figure(figsize=(15, 10))

# ROC curves
plt.subplot(2, 2, 1)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_proba_model = model.predict_proba(X_test)[:, 1]
    
    fpr_model, tpr_model, _ = roc_curve(y_test, y_pred_proba_model)
    roc_auc_model = auc(fpr_model, tpr_model)
    
    plt.plot(fpr_model, tpr_model, lw=2, label=f'{name} (AUC = {roc_auc_model:.3f})')
    
    model_results[name] = {
        'roc_auc': roc_auc_model,
        'pr_auc': average_precision_score(y_test, y_pred_proba_model)
    }

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# PR curves
plt.subplot(2, 2, 2)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_proba_model = model.predict_proba(X_test)[:, 1]
    
    precision_model, recall_model, _ = precision_recall_curve(y_test, y_pred_proba_model)
    pr_auc_model = average_precision_score(y_test, y_pred_proba_model)
    
    plt.plot(recall_model, precision_model, lw=2, label=f'{name} (AUC = {pr_auc_model:.3f})')

plt.axhline(y=no_skill, color='red', linestyle='--', label=f'No Skill = {no_skill:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Model comparison bar chart
plt.subplot(2, 2, 3)
model_names = list(model_results.keys())
roc_scores = [model_results[name]['roc_auc'] for name in model_names]
pr_scores = [model_results[name]['pr_auc'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, roc_scores, width, label='ROC-AUC')
plt.bar(x + width/2, pr_scores, width, label='PR-AUC')

plt.xlabel('Models')
plt.ylabel('AUC Score')
plt.title('Model Performance Comparison')
plt.xticks(x, model_names)
plt.legend()
plt.grid(True, alpha=0.3)

# Results table
plt.subplot(2, 2, 4)
plt.axis('tight')
plt.axis('off')
table_data = []
for name in model_names:
    table_data.append([name, f"{model_results[name]['roc_auc']:.3f}", f"{model_results[name]['pr_auc']:.3f}"])

table = plt.table(cellText=table_data,
                 colLabels=['Model', 'ROC-AUC', 'PR-AUC'],
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.title('Performance Summary')

save_and_show("06_multiple_models_comparison")

# 8. BUSINESS METRICS DEMO
print("\n8. BUSINESS METRICS DEMO")
print("-" * 30)

print("Business Impact Analysis:")
print("-" * 30)

# Business cost analysis
y_pred_proba_business = rf_model.predict_proba(X_test)[:, 1]

# Different thresholds
thresholds_business = [0.1, 0.3, 0.5, 0.7, 0.9]

# Cost parameters
fraud_loss_per_case = 1000  # Average fraud loss
investigation_cost = 50     # Cost to investigate each alert

print(f"Assumptions:")
print(f"- Average fraud loss: ${fraud_loss_per_case}")
print(f"- Investigation cost per alert: ${investigation_cost}")
print(f"- Total frauds in test set: {np.sum(y_test)}")
print(f"- Total test transactions: {len(y_test)}")

print(f"\nThreshold Analysis:")
print("Threshold | Alerts | Caught | Missed | Total Cost | Cost per Transaction")
print("-" * 75)

business_results = []

for threshold in thresholds_business:
    y_pred_business = (y_pred_proba_business >= threshold).astype(int)
    
    # Confusion matrix
    tp = np.sum((y_test == 1) & (y_pred_business == 1))  # Frauds caught
    fp = np.sum((y_test == 0) & (y_pred_business == 1))  # False alarms
    fn = np.sum((y_test == 1) & (y_pred_business == 0))  # Frauds missed
    
    # Business metrics
    total_alerts = tp + fp
    frauds_caught = tp
    frauds_missed = fn
    
    # Costs
    investigation_costs = total_alerts * investigation_cost
    fraud_losses = frauds_missed * fraud_loss_per_case
    total_cost = investigation_costs + fraud_losses
    cost_per_transaction = total_cost / len(y_test)
    
    print(f"{threshold:8.1f} | {total_alerts:6d} | {frauds_caught:6d} | {frauds_missed:6d} | "
          f"${total_cost:8.0f} | ${cost_per_transaction:8.2f}")
    
    business_results.append({
        'threshold': threshold,
        'total_cost': total_cost,
        'alerts': total_alerts,
        'caught': frauds_caught,
        'missed': frauds_missed
    })

# Business visualization
business_df = pd.DataFrame(business_results)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(business_df['threshold'], business_df['total_cost'], 'o-', linewidth=2, color='red')
plt.xlabel('Threshold')
plt.ylabel('Total Cost ($)')
plt.title('Total Cost vs Threshold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.bar(business_df['threshold'], business_df['alerts'], alpha=0.7, color='orange')
plt.xlabel('Threshold')
plt.ylabel('Number of Alerts')
plt.title('Alert Volume vs Threshold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.bar(business_df['threshold'] - 0.02, business_df['caught'], width=0.04, 
        alpha=0.7, color='green', label='Frauds Caught')
plt.bar(business_df['threshold'] + 0.02, business_df['missed'], width=0.04, 
        alpha=0.7, color='red', label='Frauds Missed')
plt.xlabel('Threshold')
plt.ylabel('Count')
plt.title('Fraud Detection Performance')
plt.legend()
plt.grid(True, alpha=0.3)

# ROI Analysis
plt.subplot(2, 2, 4)
baseline_cost = len(y_test[y_test == 1]) * fraud_loss_per_case  # All frauds missed
cost_savings = [baseline_cost - cost for cost in business_df['total_cost']]
plt.bar(business_df['threshold'], cost_savings, alpha=0.7, color='lightblue')
plt.xlabel('Threshold')
plt.ylabel('Cost Savings ($)')
plt.title('Cost Savings vs No Model')
plt.grid(True, alpha=0.3)

save_and_show("07_business_impact_analysis")

# Optimal threshold (minimum cost)
optimal_idx = business_df['total_cost'].idxmin()
optimal_threshold_business = business_df.loc[optimal_idx, 'threshold']
min_cost = business_df.loc[optimal_idx, 'total_cost']

print(f"\n🎯 Optimal threshold (minimum cost): {optimal_threshold_business}")
print(f"💰 Minimum total cost: ${min_cost:.0f}")
print(f"💰 Cost per transaction: ${min_cost/len(y_test):.2f}")

# 9. PRACTICAL RECOMMENDATIONS
print("\n9. PRACTICAL RECOMMENDATIONS")
print("-" * 30)

print("FRAUD DETECTION İÇİN METRİK SEÇİMİ:")

print("\n✓ ROC-AUC kullan:")
print("  - Balanced dataset")
print("  - False positive ve false negative eşit maliyette")
print("  - Genel model performansı")

print("\n✓ PR-AUC kullan:")
print("  - Imbalanced dataset (fraud detection)")
print("  - Positive class (fraud) daha önemli")
print("  - Precision önemli (false alarm maliyeti yüksek)")

print("\n✓ Threshold seçimi:")
print("  - Business requirements")
print("  - Investigation capacity")
print("  - False positive vs false negative maliyeti")

print("\n✓ Best practices:")
print("  - Her iki metriği de rapor et")
print("  - Threshold analizi yap")
print("  - Business context'i göz önünde bulundur")
print("  - Cross-validation kullan")
print("  - Multiple models karşılaştır")

print("\nBUSINESS CONSIDERATIONS:")
print("- Investigation team kapasitesi")
print("- Alert fatigue")
print("- Customer experience impact")
print("- Regulatory requirements")
print("- Cost of fraud vs cost of investigation")

print(f"\n📊 Toplam {len(os.listdir(VIZ_DIR))} görsel kaydedildi: {VIZ_DIR}/")
print("\n" + "=" * 60)
print("ROC-AUC VE PR-AUC EĞİTİMİ TAMAMLANDI!")
print("=" * 60)