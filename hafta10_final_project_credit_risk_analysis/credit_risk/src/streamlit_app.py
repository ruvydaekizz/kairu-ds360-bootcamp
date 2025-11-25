import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =============
# AYARLAR
# =============
# Sayfa başlığını ve düzenini (geniş ekran) ayarlıyoruz.
st.set_page_config(layout="wide", page_title="Kredi Risk Merkezi")

# Yollar: Model dosyasının nerede olduğunu dinamik olarak buluyoruz.
CURRENT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = CURRENT_DIR.parent / "artifacts"
# Eğittiğimiz ve seçtiğimiz en iyi model olan XGBoost'u kullanacağız. pkl uzantılı olanı alıyoruz.
MODEL_PATH = ARTIFACTS_DIR / "model_xgboost_weighted.pkl"

# =============
# TASARIM
# =============
# Arayüzü iyileştirmek için CSS kodları ekliyoruz.
st.markdown("""
<style>
    /* Genel Arka Planı Beyaz Yap */
    .stApp {
        background-color: #ffffff;
    }

    /* --- HEADER (BAŞLIK) TASARIMI --- */
    .header-container {
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
    }
    .header-box {
        background-color: #2c3e50; /* Lacivert */
        color: white;
        padding: 15px 40px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 24px;
        font-family: 'Arial', sans-serif;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        text-align: center;
    }

    /* --- SÜTUN TASARIMI (KUTULAR) --- */
    /* İçinde kart başlığı (.card-title) olan sütunları Turkuaz yap */
    div[data-testid="stColumn"]:has(.card-title) {
        background-color: #3caea3;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0px 5px;
    }

    /* --- KART BAŞLIKLARI --- */
    .card-title {
        color: white;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.3);
        padding-bottom: 10px;
    }

    /* --- INPUT ETİKETLERİ (LABELS) --- */
    /* Yazıları beyaz yap */
    .stNumberInput label, .stSelectbox label, .stTextInput label, .stRadio label {
        color: white !important;
        font-weight: 500;
        font-size: 14px;
    }
    /* Input kutusunun içindeki yazıyı koyu yap (Okunabilirlik için) */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        color: #333;
    }

    /* --- BUTON DÜZENİ (KESİN HİZALAMA) --- */
    /* Butonu sütun içinde ortala */
    div.stButton {
        display: flex;
        justify-content: center; 
        width: 100%;
    }
    
    /* Butonun Görsel Tasarımı */
    div.stButton > button {
        background-color: #2c3e50; /* Lacivert */
        color: white;
        border: none;
        padding: 15px 50px;
        font-size: 16px;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: 0.3s;
        width: auto; /* Genişlik içeriğe göre olsun */
        margin-left: 230px;
    }
    
    div.stButton > button:hover {
        background-color: #1a252f;
        border: 1px solid white;
    }

    /* --- SONUÇ KUTULARI (BAŞARILI / BAŞARISIZ) --- */
    .result-bar-success {
        background: linear-gradient(90deg, #76b852 0%, #8DC26F 100%); /* Yeşil */
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .result-bar-fail {
        background: linear-gradient(90deg, #cb2d3e 0%, #ef473a 100%); /* Kırmızı */
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .result-title { font-size: 16px; font-weight: bold; opacity: 0.9; }
    .result-score { font-size: 48px; font-weight: bold; line-height: 1.2; }
    .result-desc { font-size: 12px; font-weight: normal; opacity: 0.8; }

    /* Radio Button Tasarımı */
    div[role="radiogroup"] {
        background-color: rgba(255,255,255,0.1);
        padding: 5px;
        border-radius: 5px;
    }

</style>
""", unsafe_allow_html=True)

# ==============
# MODELİ YÜKLE
# ==============
@st.cache_resource
def load_model():
    """Modeli sadece bir kez yükler ve önbelleğe alır (Hız için)."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# ==============
# SAYFA DÜZENİ (INTERFACE)
# ==============

# 1. Başlık Alanı
st.markdown("""
    <div class="header-container">
        <div class="header-box">KREDİ RİSK MERKEZİ</div>
    </div>
""", unsafe_allow_html=True)

# 2. Giriş Kartları (3 Sütunlu Yapımız)
# gap="medium" ile sütunlar arası boşluğu ayarlıyoruz.
col1, col2, col3 = st.columns(3, gap="medium")

# SOL SÜTUN: Kişisel Profil
with col1:
    st.markdown('<div class="card-title">👤 Kişisel Profil</div>', unsafe_allow_html=True)
    age = st.number_input("Yaş", 18, 100, 30)
    emp_length = st.number_input("İş Deneyimi (Yıl)", 0, 60, 5)
    # Kullanıcıya 'KİRA' gösterip arka planda 'RENT' değerini alıyoruz. OWN= kendi evi, MORTGAGE= İpotek, Other: Ailesiyle / Akrabasıyla Yaşayanlar veya  Bilinmeyen de olabilir.
    home_ownership = st.selectbox("Ev Durumu", ["RENT", "OWN", "MORTGAGE", "OTHER"], 
                                format_func=lambda x: "KİRA (RENT)" if x == "RENT" else ("EV SAHİBİ (OWN)" if x == "OWN" else x))

# ORTA SÜTUN: Kredi Başvurusu
with col2:
    st.markdown('<div class="card-title">📄 Kredi Başvurusu</div>', unsafe_allow_html=True)
    loan_intent = st.selectbox("Kredi Amacı", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_amount = st.number_input("İstenen Tutar ($)", 100, 1000000, 15000, step=500)
    loan_int_rate = st.number_input("Faiz Oranı (%)", 0.0, 100.0, 12.50, step=0.1)
    loan_grade = st.selectbox("Kredi Notu (Grade)", ["A", "B", "C", "D", "E", "F", "G"])

# SAĞ SÜTUN: Finansal Geçmiş
with col3:
    st.markdown('<div class="card-title">💰 Finansal Geçmiş</div>', unsafe_allow_html=True)
    annual_income = st.number_input("Yıllık Gelir ($)", 0, 10000000, 60000, step=1000)
    cred_hist_len = st.number_input("Kredi Geçmişi (Yıl)", 0, 50, 8)
    st.write("Geçmişte Temerrüt Var Mı?")
    # Radio butonu yatay olarak gösteriyoruz
    default_on_file = st.radio("Temerrüt Durumu", ["EVET", "HAYIR"], index=1, horizontal=True, label_visibility="collapsed")

# 3. Buton Alanı
st.markdown("<br>", unsafe_allow_html=True)
# Butonu ortalamak için yine 3 sütun açıp ortadakini (b2'yi) kullanıyoruz.
b1, b2, b3 = st.columns(3, gap="large")

with b2:
    analyze_clicked = st.button("RİSKİ ANALİZ ET")

# 4. Sonuç Alanı (Tahmin İşlemi)
if analyze_clicked:
    if model is None:
        st.error("Model dosyası bulunamadı! Lütfen önce eğitimi tamamlayın.")
    else:
        # Girdileri Hazırla: Ekranda 'EVET' yazanı modele 'Y' olarak veriyoruz.
        default_val = 'Y' if default_on_file == "EVET" else 'N'
        # Ev durumunu temizliyoruz (Kullanıcı dostu metinden orijinal koda dönüş)
        home_clean = "RENT" if "RENT" in str(home_ownership) else ("OWN" if "OWN" in str(home_ownership) else home_ownership)

        # Modelin beklediği formatta bir DataFrame oluşturuyoruz
        input_data = pd.DataFrame({
            'person_age': [age],
            'person_income': [annual_income],
            'person_home_ownership': [home_clean],
            'person_emp_length': [emp_length],
            'loan_intent': [loan_intent],
            'loan_grade': [loan_grade],
            'loan_amnt': [loan_amount],
            'loan_int_rate': [loan_int_rate],
            'loan_status': [0], # Dummy kolon (Pipeline hatası olmasın diye, sonra atacağız)
            'loan_percent_income': [loan_amount / annual_income if annual_income > 0 else 0],
            'cb_person_default_on_file': [default_val], 
            'cb_person_cred_hist_length': [cred_hist_len]
        })

        # ÖNEMLİ: Feature Engineering (Manuel)
        # Model eğitilirken yapılan işlemlerin aynısını burada tekil veri için yapıyoruz.
        # 1. Gelirin logaritmasını al
        input_data['person_income_log'] = np.log1p(input_data['person_income'])
        # 2. Y/N değerini 1/0'a çevir
        input_data['cb_person_default_on_file'] = input_data['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
        
        # Hedef değişkeni veriden çıkar (Tahmin edeceğimiz şey bu zaten)
        if 'loan_status' in input_data.columns:
            input_data = input_data.drop('loan_status', axis=1)

        try:
            # Tahmin Yap: Olasılık (Probability) alıyoruz.
            # [0][1] -> 1. sınıfın (Temerrüt/Riskli/Ödeyip Ödememe) olasılığını verir.
            prob_default = model.predict_proba(input_data)[0][1]
            prob_percentage = prob_default * 100
            
            # Sonucu Göster (Eşik Değer %50)
            if prob_percentage < 50:
                # Yeşil Kutu (Onay)
                st.markdown(f"""
                <div class="result-bar-success">
                    <div class="result-title">✅ KREDİ ONAYLANABİLİR</div>
                    <div class="result-score">%{prob_percentage:.1f}</div>
                    <div class="result-desc">Temerrüt Riski (Düşük)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Kırmızı Kutu (Red)
                st.markdown(f"""
                <div class="result-bar-fail">
                    <div class="result-title">❌ KREDİ REDDEDİLDİ</div>
                    <div class="result-score">%{prob_percentage:.1f}</div>
                    <div class="result-desc">Temerrüt Riski (Yüksek)</div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Hata: {e}")