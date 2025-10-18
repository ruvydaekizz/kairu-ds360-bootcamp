"""
SHAP & LIME — Baştan Düzenlenmiş Stabil Script (v3)
• Amaç: Fraud benzeri veri → RF modeli → Global/Local açıklamalar → Görselleri kaydet
• SHAP yeni/eski API uyumlu + sınıf ekseni (n_classes) güvenli seçimi
• Tüm importance vektörleri 1D garanti (Per-column arrays hatası yok)
• LIME feature adı parsing fix ("feat <= val" → "feat")
• Görseller: visualizations/script4_shap_lime/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# =============================
# 0) Klasör & Yardımcılar
# =============================
VIZ_DIR = "visualizations/script4_shap_lime"
os.makedirs(VIZ_DIR, exist_ok=True)


def save_fig(name: str, dpi: int = 300):
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/{name}.png", dpi=dpi, bbox_inches="tight")
    plt.show()


def to1d(arr, name="array"):
    a = np.asarray(arr)
    if a.ndim == 1:
        return a
    if a.ndim == 2 and 1 in a.shape:
        return a.reshape(-1,)
    raise ValueError(f"{name} 1D olmalı, shape={a.shape}. Kaynağı kontrol edin.")


# =============================
# 1) Opsiyonel Kütüphaneler
# =============================
SHAP_AVAILABLE = False
LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
    print("✓ SHAP hazır")
except Exception as e:
    print("⚠️ SHAP yok:", e)

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
    print("✓ LIME hazır")
except Exception as e:
    print("⚠️ LIME yok:", e)

print("=" * 60)
print("BAŞTAN DÜZENLENEN SHAP & LIME SCRIPT — v3")
print("=" * 60)

# =============================
# 2) Veri & Model (Fraud benzeri)
# =============================
np.random.seed(42)
X, y = make_classification(
    n_samples=1200,
    n_features=10,
    n_informative=8,
    n_redundant=1,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    random_state=42,
)

feature_names = [
    'transaction_amount',
    'account_age',
    'transaction_hour',
    'merchant_risk_score',
    'payment_method_risk',
    'location_risk',
    'frequency_score',
    'amount_deviation',
    'weekend_flag',
    'velocity_score'
]

X_df = pd.DataFrame(X, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.30, random_state=42, stratify=y
)

# y'yi Series'e çevir ve indexleri hizala (numpy olsa bile)
y_train_s = pd.Series(y_train, index=X_train.index)
y_test_s  = pd.Series(y_test,  index=X_test.index)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
print(f"Train acc: {rf.score(X_train, y_train):.3f} | Test acc: {rf.score(X_test, y_test):.3f}")

# =============================
# 3) Built-in, Permutation, Korelasyon
# =============================
feat_imp = to1d(rf.feature_importances_, "feature_importances")
imp_df = (
    pd.DataFrame({"feature": feature_names, "importance": feat_imp})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_mean = to1d(perm.importances_mean, "perm_mean")
perm_std = to1d(perm.importances_std, "perm_std")
perm_df = (
    pd.DataFrame({"feature": feature_names, "importance": perm_mean, "std": perm_std})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.barh(imp_df['feature'], imp_df['importance'])
plt.title('Built-in Feature Importance')
plt.gca().invert_yaxis()

plt.subplot(1,3,2)
plt.barh(perm_df['feature'], perm_df['importance'], xerr=perm_df['std'])
plt.title('Permutation Importance')
plt.gca().invert_yaxis()

plt.subplot(1,3,3)
sns.heatmap(X_df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlations')

save_fig("01_baseline_feature_importance")

# =============================
# 4) SHAP — Sınıf Ekseni Güvenli (n_classes)
# =============================
mean_abs_shap = None
base_values_vec = None  # varsa per-sample base values
CLASS_IDX = 1           # ikili sınıflamada pozitif sınıf

if SHAP_AVAILABLE:
    print("SHAP çalışıyor… (n_classes için güvenli)")
    X_sample = X_test.iloc[:200]

    try:
        # Yeni API
        explainer = shap.Explainer(rf)
        exp = explainer(X_sample)
        raw_vals = getattr(exp, 'values', exp)
        base_vals = getattr(exp, 'base_values', None)
    except Exception:
        # Eski API
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X_sample)
        raw_vals = sv  # list veya np.array olabilir
        base_vals = getattr(explainer, 'expected_value', None)

    # ---- SHAP matrisini 2D: (n_samples, n_features) hale getir ----
    def _to_2d_shap(values):
        v = values
        # Eski API list → sınıf seç
        if isinstance(v, list):
            v = v[CLASS_IDX]
        v = np.asarray(v)
        if v.ndim == 3:  # (n_samples, n_features, n_classes)
            v = v[..., CLASS_IDX]
        elif v.ndim == 2:  # (n_samples, n_features)
            pass
        else:
            raise ValueError(f"Beklenmeyen SHAP shape: {v.shape}")
        return v

    shap_matrix = _to_2d_shap(raw_vals)

    # ---- Base value(lar) ----
    base_value = 0.0
    if base_vals is not None:
        b = np.asarray(base_vals)
        if isinstance(base_vals, list):
            # Eski API: list halinde sınıf başına değer
            b = np.atleast_1d(base_vals[CLASS_IDX])
        if b.ndim == 2 and b.shape[1] >= 1:  # (n_samples, n_classes)
            base_values_vec = b[:, CLASS_IDX]
            base_value = float(np.mean(base_values_vec))
        elif b.ndim == 1:
            base_value = float(np.mean(b))
        else:
            base_value = float(np.mean(b))

    # ---- Global önem (1D) ----
    mean_abs_shap = to1d(np.mean(np.abs(shap_matrix), axis=0), "mean_abs_shap")
    shap_imp_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=True)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(15,10))
    # (1) Bar
    plt.subplot(2,2,1)
    plt.barh(shap_imp_df['feature'], shap_imp_df['mean_abs_shap'])
    plt.title('SHAP Feature Importance (Mean |SHAP|)')

    # (2) Beeswarm (mümkünse)
    plt.subplot(2,2,2)
    try:
        shap.summary_plot(shap_matrix, X_sample, feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot')
    except Exception as e:
        for i in range(min(6, len(feature_names))):
            plt.scatter(shap_matrix[:, i], [i]*len(X_sample), alpha=0.6)
        plt.yticks(range(min(6, len(feature_names))), feature_names[:6])
        plt.title(f'SHAP Dist. (Fallback) — {type(e).__name__}')

    # (3) SHAP vs Built-in
    plt.subplot(2,2,3)
    mrg = pd.merge(imp_df, shap_imp_df, on='feature')
    plt.scatter(mrg['importance'], mrg['mean_abs_shap'])
    for _, r in mrg.iterrows():
        plt.annotate(r['feature'][:9], (r['importance'], r['mean_abs_shap']), fontsize=8, alpha=0.7)
    plt.xlabel('Built-in'); plt.ylabel('Mean |SHAP|'); plt.title('SHAP vs Built-in')

    # (4) Dependence (Top-2)
    plt.subplot(2,2,4)
    top2 = shap_imp_df.tail(2)['feature'].tolist()
    for f in top2:
        idx = feature_names.index(f)
        try:
            shap.dependence_plot(idx, shap_matrix, X_sample, feature_names=feature_names, show=False)
        except Exception:
            plt.scatter(X_sample.iloc[:, idx], shap_matrix[:, idx], alpha=0.6, label=f)
    plt.title('SHAP Dependence (Top-2)'); plt.legend()

    save_fig("02_shap_analysis")

    # ---- Tekil örnekler (fraud & normal) ----
# X_sample içinde yer alan indeksler arasından first fraud & first normal seç
fraud_idx_global = next((idx for idx in X_sample.index if y_test_s.loc[idx] == 1), None)
normal_idx_global = next((idx for idx in X_sample.index if y_test_s.loc[idx] == 0), None)

# =============================
# 4) SHAP — Sınıf ekseni güvenli
# =============================
mean_abs_shap = None
base_values_vec = None
CLASS_IDX = 1  # pozitif sınıf

if SHAP_AVAILABLE:
    print("\nSHAP çalışıyor… (n_classes için güvenli)")
    X_sample = X_test.iloc[:200]

    try:
        explainer = shap.Explainer(rf)
        exp = explainer(X_sample)
        raw_vals  = getattr(exp, "values", exp)
        base_vals = getattr(exp, "base_values", None)
    except Exception:
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X_sample)
        raw_vals  = sv
        base_vals = getattr(explainer, "expected_value", None)

    # ---- SHAP matrisini 2D'ye indir
    def _to_2d_shap(values):
        v = values
        if isinstance(v, list):
            v = v[CLASS_IDX]
        v = np.asarray(v)
        if v.ndim == 3:              # (n_samples, n_features, n_classes)
            v = v[..., CLASS_IDX]
        elif v.ndim != 2:            # (n_samples, n_features) beklenir
            raise ValueError(f"Beklenmeyen SHAP shape: {v.shape}")
        return v

    shap_matrix = _to_2d_shap(raw_vals)

    # ---- Base value(lar)
    base_value = 0.0
    if base_vals is not None:
        b = base_vals
        if isinstance(b, list):      # eski API
            b = np.atleast_1d(b[CLASS_IDX])
        b = np.asarray(b)
        if b.ndim == 2:              # (n_samples, n_classes)
            base_values_vec = b[:, CLASS_IDX]
            base_value = float(np.mean(base_values_vec))
        else:
            base_value = float(np.mean(b))

    # ---- Global önem
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    shap_imp_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=True)
        .reset_index(drop=True)
    )

    # (grafikler… bar/beeswarm/compare/dependence)  # mevcut kodunla devam et

    # ---- Tekil örnekler (fraud & normal)
    # y_test'i Series yap (üstte yaptıysan gerek yok):
    # y_test_s = pd.Series(y_test, index=X_test.index)
    fraud_idx_global  = next((idx for idx in X_sample.index if y_test_s.loc[idx] == 1), None)
    normal_idx_global = next((idx for idx in X_sample.index if y_test_s.loc[idx] == 0), None)

    if fraud_idx_global is not None and normal_idx_global is not None:
        fi = X_sample.index.get_loc(fraud_idx_global)
        ni = X_sample.index.get_loc(normal_idx_global)

        fraud_base  = float(base_values_vec[fi]) if isinstance(base_values_vec, np.ndarray) and base_values_vec.ndim==1 else base_value
        normal_base = float(base_values_vec[ni]) if isinstance(base_values_vec, np.ndarray) and base_values_vec.ndim==1 else base_value

        plt.figure(figsize=(14,6))
        plt.subplot(1,2,1)
        try:
            shap.waterfall_plot(shap.Explanation(values=shap_matrix[fi], base_values=fraud_base,
                                                 data=X_sample.iloc[fi].values, feature_names=feature_names), show=False)
            plt.title('Fraud — SHAP Waterfall')
        except Exception:
            vals = shap_matrix[fi]; idxs = np.argsort(np.abs(vals))[-10:]
            plt.barh([feature_names[i] for i in idxs], vals[idxs]); plt.title('Fraud — SHAP (Fallback)')

        plt.subplot(1,2,2)
        try:
            shap.waterfall_plot(shap.Explanation(values=shap_matrix[ni], base_values=normal_base,
                                                 data=X_sample.iloc[ni].values, feature_names=feature_names), show=False)
            plt.title('Normal — SHAP Waterfall')
        except Exception:
            vals = shap_matrix[ni]; idxs = np.argsort(np.abs(vals))[-10:]
            plt.barh([feature_names[i] for i in idxs], vals[idxs]); plt.title('Normal — SHAP (Fallback)')

        save_fig("03_shap_individual")
    else:
        print("Uyarı: X_sample içinde hem fraud hem normal örnek yok — tekil grafik atlandı.")

else:
    # SHAP yoksa burası çalışır (tek bir else!)
    print("SHAP kurulumu öneri: pip install shap==0.41.0")
    plt.figure(figsize=(8,5))
    plt.text(0.5,0.5,'SHAP yok — pip install shap==0.41.0', ha='center', va='center')
    plt.axis('off')
    save_fig("02_shap_not_available")


# =============================
# 5) LIME — Stabil Toplu Önem
# =============================
mean_lime = None

if LIME_AVAILABLE:
    print("LIME çalışıyor…")
    expl = LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=['Normal','Fraud'],
        mode='classification',
        discretize_continuous=True,
    )

    def _parse_feat(raw: str) -> str:
        # 'feature <= 0.12' → 'feature'
        return raw.split("<=")[0].split(">=")[0].split("<")[0].split(">")[0].strip()

    n = min(30, len(X_test))
    buf = np.zeros((n, len(feature_names)))

    ok = 0
    for i in range(n):
        try:
            e = expl.explain_instance(X_test.iloc[i].values, rf.predict_proba, num_features=len(feature_names))
            for raw_name, w in e.as_list(label=1):  # fraud perspektifi
                fn = _parse_feat(raw_name)
                if fn in feature_names:
                    j = feature_names.index(fn)
                    buf[ok, j] = w
            ok += 1
        except Exception:
            continue

    if ok > 0:
        mean_lime = to1d(np.mean(np.abs(buf[:ok]), axis=0), "mean_lime")
        lime_df = (
            pd.DataFrame({"feature": feature_names, "lime_importance": mean_lime})
            .sort_values("lime_importance", ascending=True)
            .reset_index(drop=True)
        )
        plt.figure(figsize=(10,6))
        plt.barh(lime_df['feature'], lime_df['lime_importance'])
        plt.title(f"Mean LIME Importance ({ok} explanations)")
        save_fig("06_lime_feature_importance")
    else:
        print("LIME açıklaması üretilemedi")
        plt.figure(figsize=(8,5)); plt.text(0.5,0.5,'LIME failed — parametreleri değiştirin', ha='center', va='center'); plt.axis('off')
        save_fig("06_lime_failed")
else:
    print("LIME kurulumu öneri: pip install lime==0.2.0.1")
    plt.figure(figsize=(8,5)); plt.text(0.5,0.5,'LIME yok — pip install lime==0.2.0.1', ha='center', va='center'); plt.axis('off')
    save_fig("05_lime_not_available")

# =============================
# 6) Yöntem Karşılaştırma (hepsi 1D)
# =============================
comp = {
    'Feature': feature_names,
    'Built-in': to1d(imp_df.set_index('feature').loc[feature_names, 'importance'].values, 'built_in'),
    'Permutation': to1d(perm_df.set_index('feature').loc[feature_names, 'importance'].values, 'perm'),
}
if mean_abs_shap is not None:
    comp['SHAP'] = to1d(mean_abs_shap, 'shap')
if mean_lime is not None:
    comp['LIME'] = to1d(mean_lime, 'lime')

cmp_df = pd.DataFrame(comp)

num_cols = [c for c in cmp_df.columns if c != 'Feature']
for c in num_cols:
    m = cmp_df[c].max() or 1.0
    cmp_df[f'{c}_norm'] = cmp_df[c] / m

print("Karşılaştırma (ilk 5):", cmp_df[['Feature'] + num_cols].head())

if len(num_cols) >= 2:
    plt.figure(figsize=(12,10))
    plt.subplot(2,2,1)
    sns.heatmap(cmp_df[num_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Method Correlations')

    k = min(3, len(num_cols)-1)
    for i in range(k):
        plt.subplot(2,2,i+2)
        x, y = num_cols[0], num_cols[i+1]
        plt.scatter(cmp_df[x], cmp_df[y])
        plt.xlabel(f'{x}'); plt.ylabel(f'{y}'); plt.title(f'{x} vs {y}')

    save_fig("07_explainability_comparison")

# =============================
# 7) Business Case — Örnek Ekran
# =============================
hi_idx = X_test['transaction_amount'].idxmax()
hi = X_test.loc[hi_idx]
proba = rf.predict_proba([hi])[0,1]
pred = int(proba >= 0.5)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
cols = ['transaction_amount','account_age','merchant_risk_score','location_risk']
plt.bar(cols, [hi[c] for c in cols]); plt.xticks(rotation=25); plt.title('Transaction Features'); plt.grid(alpha=0.3)

plt.subplot(2,2,2)
probs = rf.predict_proba([hi])[0]
plt.pie(probs, labels=['Normal','Fraud'], autopct='%1.1f%%'); plt.title('Prediction Probabilities')

plt.subplot(2,2,3)
plt.barh(perm_df.head(5)['feature'], perm_df.head(5)['importance'])
plt.title('Top Features (Global)')

plt.subplot(2,2,4)
msg = f"""
BUSINESS EXPLANATION:

Prediction: {'FRAUD' if pred==1 else 'NORMAL'}
Confidence: {proba:.1%}

Top Risk Factors:
• Transaction Amount: {hi['transaction_amount']:.2f}
• Merchant Risk: {hi['merchant_risk_score']:.2f}
• Location Risk: {hi['location_risk']:.2f}

Recommendation: {'BLOCK' if pred==1 else 'APPROVE'}
"""
plt.text(0.05,0.95,msg,va='top',ha='left',bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
plt.axis('off'); plt.title('Business Explanation')

save_fig("08_business_case_example")

print(f"Görseller: {VIZ_DIR}/ dizinine kaydedildi. Tamam")