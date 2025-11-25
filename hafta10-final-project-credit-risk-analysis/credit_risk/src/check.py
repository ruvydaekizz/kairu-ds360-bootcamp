import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
# Veriyi yüklemek için preprocessing modülünü kullanıyoruz (Veri seti kaynağımız)
from preprocessing import get_train_test_data

import warnings
# Terminal çıktısını kirletmemesi için gereksiz uyarıları gizliyoruz
warnings.filterwarnings("ignore", category=UserWarning)

# Bu dosyanın olduğu klasörü buluyoruz
CURRENT_DIR = Path(__file__).resolve().parent
# Modellerin kayıtlı olduğu 'artifacts' klasörünün yolunu belirliyoruz
ARTIFACTS_DIR = CURRENT_DIR.parent / "artifacts"

def evaluate_overfitting(model_tag, model, X_train, X_test, y_train, y_test):
    """
    Modelin Train (Eğitim) ve Test performansını karşılaştırarak 
    ezberleyip ezberlemediğini (Overfitting) kontrol ediyoruz.
    """
    print(f"\n🔍 ANALİZ EDİLİYOR: {model_tag.upper()} ...")
    
    # TRAIN (Eğitim) Seti Performansı
    # Modelin eğitimde gördüğü verilerle tahmin yapması
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    # Eğitim skorlarını hesapla
    train_auc = roc_auc_score(y_train, y_train_prob)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # TEST Seti Performansı
    # Modelin hiç görmediği test verileriyle tahmin yapması
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    # Test skorlarını hesapla
    test_auc = roc_auc_score(y_test, y_test_prob)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # FARK ANALİZİ
    # Eğitim başarısı ile Test başarısı arasındaki farkı bul
    auc_diff = train_auc - test_auc
    
    # Sonuçları tablo olarak ekrana yazdır
    print("-" * 40)
    print(f"📊 METRİK       |  TRAIN (Eğitim)  |   TEST   |  FARK")
    print("-" * 40)
    print(f"🎯 ROC AUC      |      {train_auc:.3f}       |  {test_auc:.3f}   | {auc_diff:+.3f}")
    print(f"✅ Accuracy     |      {train_acc:.3f}       |  {test_acc:.3f}   | {(train_acc - test_acc):+.3f}")
    print("-" * 40)
    
    # YORUM VE KARAR
    # Farkın büyüklüğüne göre modelin durumunu yorumla
    if auc_diff > 0.10:
        # Fark %10'dan büyükse model ezberlemiştir
        print("⚠️  SONUÇ: CİDDİ AŞIRI ÖĞRENME (OVERFITTING) VAR!")
        print("    -> Model eğitimi ezberlemiş, yeni veride başarısız.")
    elif auc_diff > 0.05:
        # Fark %5 - %10 arasındaysa risk vardır
        print("⚠️ SONUÇ: HAFİF AŞIRI ÖĞRENME OLABİLİR.")
        print("    -> Train ve Test arasında fark açılmaya başlamış.")
    else:
        # Fark %5'ten küçükse model sağlıklıdır
        print("✅ SONUÇ: MODEL DENGELİ (BAŞARILI).")
        print("    -> Train ve Test skorları birbirine yakın, ezber yok.")

def check_models():
    # 1. Adım: Veriyi Yükle
    print("📦 Veriler yükleniyor...")
    X_train, X_test, y_train, y_test = get_train_test_data()
    
    # 2. Adım: Kontrol edilecek modellerin isim listesi
    models_to_check = ["xgboost_weighted", "lightgbm_weighted", "lr_smote"]
    
    # 3. Adım: Her bir model için döngü başlat
    for tag in models_to_check:
        # Model dosyasının yolunu oluştur
        model_path = ARTIFACTS_DIR / f"model_{tag}.pkl"
        
        # Eğer model dosyası yoksa uyarı ver ve sonraki modele geç
        if not model_path.exists():
            print(f"\n❌ Model bulunamadı: {tag} ({model_path})")
            continue
            
        # 4. Adım: Modeli Yükle ve Analiz Et
        try:
            model = joblib.load(model_path) # Kayıtlı modeli yükle
            # Yukarıdaki analiz fonksiyonunu çağır
            evaluate_overfitting(tag, model, X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"❌ Model yüklenirken hata oluştu ({tag}): {e}")

if __name__ == "__main__":
    check_models()