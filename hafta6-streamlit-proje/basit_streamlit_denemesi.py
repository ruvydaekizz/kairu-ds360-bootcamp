"""
STREAMLIT UYGULAMASI
Market Basket Analysis için kullanıcı dostu arayüz
"""

import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import plotly.express as px

# Sayfa ayarları
st.set_page_config(page_title="Market Sepeti Analizi",
                page_icon="🛒",
                layout="wide")

# Küçük stiller
st.markdown(
    "<style>"
    "body {background-color: #fbf7ff;}"
    "h1, h2, h3 {color: #4c1d95;}"
    ".st-badge {background-color:#7c3aed;}"
    "</style>",
    unsafe_allow_html=True
)

# -------------------------
# Yardımcı Fonksiyonlar
# -------------------------
@st.cache_data
def veri_yukle(dosya='data/basket_analysis.csv'):
    try:
        veri = pd.read_csv(dosya, index_col=0)
    except FileNotFoundError:
        return None, None

    sepetler = []
    for _, satir in veri.iterrows():
        sepet = [urun for urun in veri.columns if satir[urun] in (True, 'True', 1, '1')]
        if sepet:
            sepetler.append(sepet)

    return veri, sepetler


def urun_sayilarini_hesapla(sepetler):
    urun_sayilari = {}
    for sepet in sepetler:
        for urun in sepet:
            urun_sayilari[urun] = urun_sayilari.get(urun, 0) + 1
    return urun_sayilari


def birliktelik_hesapla(sepetler, min_support=0.05):
    toplam_sepet = len(sepetler)
    min_sepet_sayisi = max(1, int(min_support * toplam_sepet))

    birliktelik_sayilari = {}
    for sepet in sepetler:
        if len(sepet) >= 2:
            for urun1, urun2 in combinations(sorted(sepet), 2):
                cift = (urun1, urun2)
                birliktelik_sayilari[cift] = birliktelik_sayilari.get(cift, 0) + 1

    onemli = {}
    for cift, sayi in birliktelik_sayilari.items():
        if sayi >= min_sepet_sayisi:
            onemli[cift] = {'sepet_sayisi': sayi, 'support': sayi / toplam_sepet}
    return onemli


def kural_olustur(birliktelikler, urun_sayilari, toplam_sepet, min_confidence=0.3):
    kurallar = []
    for (u1, u2), bilgi in birliktelikler.items():
        birlikte_sayi = bilgi['sepet_sayisi']

        # u1 -> u2
        if urun_sayilari.get(u1, 0) > 0:
            conf1 = birlikte_sayi / urun_sayilari[u1]
            if conf1 >= min_confidence:
                lift1 = conf1 / (urun_sayilari[u2] / toplam_sepet)
                kurallar.append({'antecedent': u1, 'consequent': u2, 'support': bilgi['support'], 'confidence': conf1, 'lift': lift1})

        # u2 -> u1
        if urun_sayilari.get(u2, 0) > 0:
            conf2 = birlikte_sayi / urun_sayilari[u2]
            if conf2 >= min_confidence:
                lift2 = conf2 / (urun_sayilari[u1] / toplam_sepet)
                kurallar.append({'antecedent': u2, 'consequent': u1, 'support': bilgi['support'], 'confidence': conf2, 'lift': lift2})

    kurallar.sort(key=lambda x: x['confidence'], reverse=True)
    return kurallar


# -------------------------
# Başlık
# -------------------------
st.markdown(
    """
    <div style='display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap'>
    <div style='display:flex;align-items:center;gap:12px'>
        <div style='background:linear-gradient(90deg,#c4b5fd,#7c3aed);padding:12px;border-radius:10px;color:white;font-weight:700'>🛒</div>
        <div>
        <h1 style='margin:0'>Market Sepeti Analizi</h1>
        </div>
    </div>
    <div style='color:#6b21a8;font-weight:600'>data/basket_analysis.csv</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("---")

# -------------------------
# NAV - Üst Menü (selectbox)
# -------------------------
pages = ["Ana Sayfa", "Veri", "Popüler Ürünler", "Birliktelik", "Kurallar", "Öneriler"]
sayfa = st.selectbox("🔀 Sayfa seçin", pages, index=0, help="Sayfa seçimi (üst menü)")

# -------------------------
# Veri yükle
# -------------------------
veri, sepetler = veri_yukle()
if veri is None or sepetler is None:
    st.error("❌ data/basket_analysis.csv bulunamadı. Lütfen dosyayı data klasörüne ekleyin.")
    st.stop()

urun_sayilari = urun_sayilarini_hesapla(sepetler)

# -------------------------
# Sayfalar
# -------------------------
if sayfa == "Ana Sayfa":
    st.subheader("Hoş geldiniz 🎓")
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Sepet", len(sepetler))
    c2.metric("Ürün Çeşidi", len(urun_sayilari))
    c3.metric("Ortalama Ürün/Sepet", f"{np.mean([len(s) for s in sepetler]):.1f}")

    st.markdown("---")
    st.info("Market Basket Analysis: Support, Confidence, Lift gibi metriklerle ürün birlikteliklerini analiz eder.")

    with st.expander("📌 Nasıl kullanılır?", expanded=False):
        st.write("1. Veri sekmesinden ham veriyi inceleyin.\n2. Popüler Ürünler sekmesinde grafik ayarını yapın.\n3. Birliktelik sekmesinde Support belirleyip analizi çalıştırın.\n4. Kurallar sekmesinde Confidence seçip kuralları üretin.\n5. Öneriler sekmesinde ürün bazlı öneriler alın.")

elif sayfa == "Veri":
    st.subheader("📋 Ham Veri & Özet")
    left, right = st.columns([2, 1])

    with left:
        st.table(veri.head(10))

    with right:
        st.write("**Veri Özeti**")
        st.write(f"Sepet sayısı: {veri.shape[0]}")
        st.write(f"Ürün sayısı: {veri.shape[1]}")
        sepet_boyutlari = [len(s) for s in sepetler]
        st.write(f"Ortalama ürün/sepet: {np.mean(sepet_boyutlari):.1f}")
        st.write(f"En fazla ürün: {max(sepet_boyutlari)}")
        st.write(f"En az ürün: {min(sepet_boyutlari)}")

    st.markdown("---")
    st.subheader("Örnek Sepetler")
    for i, s in enumerate(sepetler[:8], 1):
        st.markdown(f"**Sepet {i}:** {', '.join(s)}")

elif sayfa == "Popüler Ürünler":
    st.subheader("🔍 Popüler Ürünler")
    gosterilecek = st.slider("Grafikte gösterilecek üst ürün sayısı:", 5, 30, 10)
    sorted_urunler = sorted(urun_sayilari.items(), key=lambda x: x[1], reverse=True)
    top = sorted_urunler[:gosterilecek]
    df_top = pd.DataFrame(top, columns=["Ürün", "Sepet Sayısı"])
    fig = px.bar(df_top, x="Sepet Sayısı", y="Ürün", orientation="h", title="Popüler Ürünler")
    # mor tonunda
    fig.update_traces(marker_color="#7b2cbf")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

elif sayfa == "Birliktelik":
    st.subheader("🔗 Birliktelik Analizi")
    min_support = st.slider("Min Support (%):", 1, 20, 5)
    if st.button("Analizi Çalıştır"):
        birliktelikler = birliktelik_hesapla(sepetler, min_support/100)
        if birliktelikler:
            rows = [{'Ürün 1': u1, 'Ürün 2': u2, 'Sepet Sayısı': b['sepet_sayisi'], 'Support': f"%{b['support']*100:.1f}"} 
                    for (u1, u2), b in birliktelikler.items()]
            df = pd.DataFrame(rows).sort_values('Sepet Sayısı', ascending=False)
            st.dataframe(df.head(30), use_container_width=True)
            top10 = df.head(10).copy()
            top10['Çift'] = top10['Ürün 1'] + ' + ' + top10['Ürün 2']
            fig = px.bar(top10, x='Sepet Sayısı', y='Çift', orientation='h', title='En Güçlü 10 Birliktelik')
            fig.update_traces(marker_color='#9d4edd')
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)
            st.session_state['birliktelikler'] = birliktelikler
        else:
            st.warning("Hiç birliktelik bulunamadı. Support'u düşürmeyi deneyin.")

elif sayfa == "Kurallar":
    st.subheader("📋 Kural Analizi")
    if 'birliktelikler' not in st.session_state:
        st.warning("Önce Birliktelik analizini çalıştırın.")
    else:
        min_confidence = st.slider("Min Confidence (%):", 10, 90, 30)
        if st.button("Kuralları Üret"):
            birliktelikler = st.session_state['birliktelikler']
            kurallar = kural_olustur(birliktelikler, urun_sayilari, len(sepetler), min_confidence/100)
            if kurallar:
                dfk = pd.DataFrame(kurallar)
                st.dataframe(dfk.head(40), use_container_width=True)
                st.session_state['kurallar'] = kurallar
            else:
                st.warning("Hiç kural bulunamadı. Confidence seviyesini düşürmeyi deneyin.")

elif sayfa == "Öneriler":
    st.subheader("🎯 Ürün Önerileri")
    if 'kurallar' not in st.session_state:
        st.warning("Önce Kuralları üretin.")
    else:
        tum_urunler = sorted(list(urun_sayilari.keys()))
        secilen = st.selectbox("Öneri alınacak ürünü seçin:", tum_urunler)
        adet = st.slider("Gösterilecek öneri sayısı:", 1, 12, 5)
        if st.button("Önerileri Göster"):
            kurallar = st.session_state['kurallar']
            uygun = [k for k in kurallar if k['antecedent'] == secilen]
            if not uygun:
                st.warning(f"{secilen} için öneri bulunamadı.")
            else:
                uygun = sorted(uygun, key=lambda x: x['confidence'], reverse=True)[:adet]
                df_on = pd.DataFrame({'Ürün':[k['consequent'] for k in uygun],
                                    'Güven':[k['confidence']*100 for k in uygun]})
                fig = px.bar(df_on, x='Güven', y='Ürün', orientation='h', title=f"{secilen} için Öneriler")
                fig.update_traces(marker_color='#5a189a')
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.write("---")
st.markdown("💜 Bu uygulama eğitim amaçlı hazırlanmıştır.")
