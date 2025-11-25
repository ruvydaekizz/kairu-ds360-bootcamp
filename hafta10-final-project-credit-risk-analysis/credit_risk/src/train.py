# train.py
import joblib
import os
import json
from pathlib import Path
# imblearn.pipeline: Normal sklearn pipeline'dan farkı, SMOTE gibi örnekleme yöntemlerini
# sadece eğitim (train) setine uygulayıp test setine dokunmamasıdır. Veri sızıntısını önlemiş olur.
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from preprocessing import get_train_test_data, get_preprocessor
import warnings

# Gereksiz uyarı mesajlarını (özellikle versiyon veya özellik ismi uyarıları) gizliyoruz.
warnings.filterwarnings("ignore", category=UserWarning)

CURRENT_DIR = Path(__file__).resolve().parent

# Modellerin kaydedileceği 'artifacts' klasörünü belirler ve yoksa oluşturuyoruz.
ARTIFACTS_DIR = CURRENT_DIR.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def save_artifacts(model, columns, tag):
    """Eğitilen modeli ve kullanılan sütun isimlerini diske kaydeder."""
    # Modeli .pkl uzantısıyla sıkıştırıp kaydet
    joblib.dump(model, ARTIFACTS_DIR / f"model_{tag}.pkl")
    # Modelin hangi sütun sırasıyla eğitildiğini kaydediyoruz (İleride tahmin yaparken gerekli)
    with open(ARTIFACTS_DIR / f"features_{tag}.json", "w") as f:
        json.dump(list(columns), f)
    print(f"💾 Kaydedildi: model_{tag}.pkl")
    print(f"   -> Konum: {ARTIFACTS_DIR}")

def train_models():
    print("🚀 Eğitim başlıyor...")
    
    # Preprocessing.py'dan gelen fonksiyonla veriyi yükle ve ayır
    X_train, X_test, y_train, y_test = get_train_test_data()
    
    # Veriyi işleyecek (standartlaştırma, one-hot encoding) pipeline nesnesini al
    preprocessor = get_preprocessor()
    
    # Dengesizlik Oranı Hesaplama:
    # Veri setimizde "Kredi Ödeyenler" (0) çok, "Ödemeyenler" (1) azdır.
    # Bu oran, XGBoost ve LightGBM'in azınlık sınıfına daha fazla odaklanmasını sağlar.
    ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

    # MODEL 1: XGBoost (Weighted)
    print("\n" + "="*10 + " XGBoost (Weighted) " + "="*10)
    
    # Pipeline Kurulumu: Önce veri işlenir, sonra modele girer.
    # scale_pos_weight=ratio: Dengesiz veri için ağırlıklandırma yapar.
    xgb_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss'))
    ])                          # scale_pos_weight ile dengeliyoruz(kredi ödemeyenler (1) azınlıkta), Model "Herkes ödüyor" diyip geçmesin diye, azınlık olan "1" sınıfının hatasını daha ağır cezalandırır.
                                # eval_metric='logloss' : Modelin hata yapma oranını en aza indirmek için kullanır.
    
    # Modeli Eğit (Fit)
    xgb_pipeline.fit(X_train, y_train)
    
    # Tahminler (Raporlama için)
    y_pred_xgb = xgb_pipeline.predict(X_test)         # Sınıf Tahmini (0 veya 1)
    y_prob_xgb = xgb_pipeline.predict_proba(X_test)[:, 1] # Olasılık Tahmini (Örn: 0.85 risk)
    
    # Performans Metriklerini Yazdır
    print(classification_report(y_test, y_pred_xgb))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob_xgb):.3f}")
    
    # Modeli artifacts kalsörüne kaydeder.
    save_artifacts(xgb_pipeline, X_train.columns, "xgboost_weighted")

    # MODEL 2: LightGBM (Weighted)
    print("\n" + "="*10 + " LightGBM (Weighted) " + "="*10)

    # LightGBM Pipeline: XGBoost'a benzer şekilde ağırlıklandırılmış (weighted) eğitim.
    
    lgbm_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(scale_pos_weight=ratio, random_state=42, verbose=-1))
    ])                                  # scale_pos_weight ile dengeliyoruz(kredi ödemeyenler (1) azınlıkta), Model "Herkes ödüyor" diyip geçmesin diye, azınlık olan "1" sınıfının hatasını daha ağır cezalandırır.
                                        # verbose=-1: Gereksiz logları kapatır.
    
    lgbm_pipeline.fit(X_train, y_train)
    
    y_pred_lgbm = lgbm_pipeline.predict(X_test)
    y_prob_lgbm = lgbm_pipeline.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred_lgbm))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob_lgbm):.3f}")
    
    save_artifacts(lgbm_pipeline, X_train.columns, "lightgbm_weighted")

    # MODEL 3: Logistic Regression (SMOTE)
    print("\n" + "="*10 + " Logistic Regression (SMOTE) " + "="*10)
    
    # Logistic Regression Pipeline:
    # Burada 'ratio' yerine SMOTE kullanıyoruz.
    # SMOTE: Azınlık sınıfından (ödenmemiş krediler) sentetik veriler üreterek veriyi dengeler.
    lr_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(max_iter=2000))
    ])
    
    lr_pipeline.fit(X_train, y_train)
    
    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred_lr))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob_lr):.3f}")
    
    save_artifacts(lr_pipeline, X_train.columns, "lr_smote")

if __name__ == "__main__":
    train_models()