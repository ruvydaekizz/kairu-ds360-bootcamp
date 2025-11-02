"""
BASIT STREAMLIT UYGULAMASI
Market Basket Analysis için öğrenci dostu web arayüzü
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go


# Sayfa ayarları
st.set_page_config(
    page_title="Market Sepeti Analizi",
    page_icon="🛒",
    layout="wide"
)

# Ana başlık
st.title("🛒 Market Sepeti Analizi")
st.markdown("---")
st.markdown("Bu uygulama market sepetlerindeki ürün birlikteliklerini analiz eder.")

# Yan menü
st.sidebar.title("📋 Menü")
sayfa = st.sidebar.selectbox(
    "Analiz türünü seçin:",
    ["🏠 Ana Sayfa", "📊 Veri Görüntüleme", "🔍 Popüler Ürünler", "🔗 Birliktelik Analizi", "📋 Kural Analizi", "🎯 Ürün Önerileri"]
)

@st.cache_data
def veri_yukle():
    """Veriyi yükler ve işler"""
    try:
        veri = pd.read_csv('data/basket_analysis.csv', index_col=0)
        
        # Sepetleri oluştur
        sepetler = []
        for i, satir in veri.iterrows():
            sepet = []
            for urun in veri.columns:
                if satir[urun] == True or satir[urun] == 'True':
                    sepet.append(urun)
            if sepet:
                sepetler.append(sepet)
        
        return veri, sepetler
    except FileNotFoundError:
        st.error("❌ data/basket_analysis.csv dosyası bulunamadı!")
        return None, None

def urun_sayilarini_hesapla(sepetler):
    """Her ürünün kaç sepette olduğunu hesaplar"""
    urun_sayilari = {}
    for sepet in sepetler:
        for urun in sepet:
            urun_sayilari[urun] = urun_sayilari.get(urun, 0) + 1
    return urun_sayilari

def birliktelik_hesapla(sepetler, min_support=0.05):
    """Ürün birlikteliklerini hesaplar"""
    toplam_sepet = len(sepetler)
    min_sepet_sayisi = int(min_support * toplam_sepet)
    
    birliktelik_sayilari = {}
    
    for sepet in sepetler:
        if len(sepet) >= 2:
            for urun1, urun2 in combinations(sepet, 2):
                if urun1 > urun2:
                    urun1, urun2 = urun2, urun1
                
                cift = (urun1, urun2)
                birliktelik_sayilari[cift] = birliktelik_sayilari.get(cift, 0) + 1
    
    # Minimum desteği geçenleri filtrele
    onemli_birliktelikler = {}
    for cift, sayi in birliktelik_sayilari.items():
        if sayi >= min_sepet_sayisi:
            support = sayi / toplam_sepet
            onemli_birliktelikler[cift] = {
                'sepet_sayisi': sayi,
                'support': support
            }
    
    return onemli_birliktelikler

def kural_olustur(birliktelikler, urun_sayilari, toplam_sepet, min_confidence=0.3):
    """Association rules oluşturur"""
    kurallar = []
    
    for (urun1, urun2), bilgi in birliktelikler.items():
        birlikte_sayi = bilgi['sepet_sayisi']
        
        # Kural 1: urun1 → urun2
        confidence1 = birlikte_sayi / urun_sayilari[urun1]
        if confidence1 >= min_confidence:
            lift1 = confidence1 / (urun_sayilari[urun2] / toplam_sepet)
            kurallar.append({
                'antecedent': urun1,
                'consequent': urun2,
                'support': bilgi['support'],
                'confidence': confidence1,
                'lift': lift1
            })
        
        # Kural 2: urun2 → urun1
        confidence2 = birlikte_sayi / urun_sayilari[urun2]
        if confidence2 >= min_confidence:
            lift2 = confidence2 / (urun_sayilari[urun1] / toplam_sepet)
            kurallar.append({
                'antecedent': urun2,
                'consequent': urun1,
                'support': bilgi['support'],
                'confidence': confidence2,
                'lift': lift2
            })
    
    return sorted(kurallar, key=lambda x: x['confidence'], reverse=True)

# Veriyi yükle
veri, sepetler = veri_yukle()

if veri is not None and sepetler is not None:
    urun_sayilari = urun_sayilarini_hesapla(sepetler)
    
    # Sayfa içeriği
    if sayfa == "🏠 Ana Sayfa":
        st.header("Hoş Geldiniz!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Toplam Sepet Sayısı", len(sepetler))
        
        with col2:
            st.metric("Toplam Ürün Çeşidi", len(urun_sayilari))
        
        with col3:
            ortalama_urun = np.mean([len(sepet) for sepet in sepetler])
            st.metric("Ortalama Ürün/Sepet", f"{ortalama_urun:.1f}")
        
        st.markdown("---")
        
        st.subheader("📖 Market Basket Analysis Nedir?")
        st.write("""
        Market Basket Analysis (Market Sepeti Analizi), müşterilerin hangi ürünleri birlikte aldığını 
        anlamamızı sağlayan bir veri analizi tekniğidir.
        
        **Ana Kavramlar:**
        - **Support (Destek)**: Bir ürün veya ürün çiftinin ne sıklıkla alındığı
        - **Confidence (Güven)**: A ürününü alan müşterilerin kaçta kaçının B ürününü de aldığı
        - **Lift**: İki ürünün birlikte alınma olasılığının tesadüfi olma durumuna göre ne kadar yüksek olduğu
        """)
        
        st.subheader("🎯 Bu Analiz Ne İşe Yarar?")
        st.write("""
        - 🛍️ **Cross-selling**: Hangi ürünleri birlikte önereceğimizi anlarız
        - 📦 **Ürün yerleştirme**: Mağazada ürünleri nereye koyacağımızı planlarız
        - 💡 **Kampanya planlama**: Hangi ürünlerde birlikte indirim yapacağımızı belirleriz
        - 📊 **Müşteri davranışı**: Müşteri alışkanlıklarını anlarız
        """)
    
    elif sayfa == "📊 Veri Görüntüleme":
        st.header("📊 Veri Görüntüleme")
        
        st.subheader("Ham Veri")
        st.write("İlk 10 sepet:")
        st.dataframe(veri.head(10))
        
        st.subheader("Veri Özeti")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Veri Boyutu:**")
            st.write(f"- Sepet sayısı: {veri.shape[0]}")
            st.write(f"- Ürün sayısı: {veri.shape[1]}")
            
        with col2:
            st.write("**Sepet İstatistikleri:**")
            sepet_boyutlari = [len(sepet) for sepet in sepetler]
            st.write(f"- Ortalama ürün/sepet: {np.mean(sepet_boyutlari):.1f}")
            st.write(f"- En fazla ürün: {max(sepet_boyutlari)}")
            st.write(f"- En az ürün: {min(sepet_boyutlari)}")
        
        st.subheader("Örnek Sepetler")
        st.write("İlk 5 sepet ve içerdikleri ürünler:")
        for i, sepet in enumerate(sepetler[:5], 1):
            st.write(f"**Sepet {i}:** {', '.join(sepet)}")
    
    elif sayfa == "🔍 Popüler Ürünler":
        st.header("🔍 Popüler Ürünler")
        
        # Popüler ürünler listesi
        sorted_urunler = sorted(urun_sayilari.items(), key=lambda x: x[1], reverse=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("En Popüler 10 Ürün")
            for i, (urun, sayi) in enumerate(sorted_urunler[:10], 1):
                yuzde = (sayi / len(sepetler)) * 100
                st.write(f"{i}. **{urun}**: {sayi} sepet (%{yuzde:.1f})")
        
        with col2:
            st.subheader("Popülarite Grafiği")
            
            # Kaç ürün gösterileceği seçimi
            gosterilecek_sayi = st.slider("Gösterilecek ürün sayısı:", 5, 16, 10)
            
            top_urunler = sorted_urunler[:gosterilecek_sayi]
            urunler = [item[0] for item in top_urunler]
            sayilar = [item[1] for item in top_urunler]
            
            # Plotly bar chart
            fig = px.bar(
                x=sayilar, 
                y=urunler, 
                orientation='h',
                title=f'En Popüler {gosterilecek_sayi} Ürün',
                labels={'x': 'Sepet Sayısı', 'y': 'Ürünler'},
                color=sayilar,
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    elif sayfa == "🔗 Birliktelik Analizi":
        st.header("🔗 Birliktelik Analizi")
        
        st.subheader("Parametreler")
        min_support = st.slider(
            "Minimum Support (Destek) Oranı:", 
            0.01, 0.20, 0.05, 0.01,
            help="Bir ürün çiftinin analiz edilmesi için minimum sepet yüzdesi"
        )
        
        if st.button("🔍 Birliktelik Analizi Yap"):
            birliktelikler = birliktelik_hesapla(sepetler, min_support)
            
            if birliktelikler:
                st.success(f"✅ {len(birliktelikler)} birliktelik bulundu!")
                
                # Birliktelikleri DataFrame'e çevir
                birliktelik_listesi = []
                for (urun1, urun2), bilgi in birliktelikler.items():
                    birliktelik_listesi.append({
                        'Ürün 1': urun1,
                        'Ürün 2': urun2,
                        'Sepet Sayısı': bilgi['sepet_sayisi'],
                        'Support': f"%{bilgi['support']*100:.1f}"
                    })
                
                df_birliktelik = pd.DataFrame(birliktelik_listesi)
                df_birliktelik = df_birliktelik.sort_values('Sepet Sayısı', ascending=False)
                
                st.subheader("En Güçlü Birliktelikler")
                st.dataframe(df_birliktelik.head(15), use_container_width=True)
                
                # Grafik
                st.subheader("Birliktelik Grafiği")
                top_10 = df_birliktelik.head(10)
                
                # Ürün çiftlerini tek string haline getir
                top_10['Ürün Çifti'] = top_10['Ürün 1'] + ' + ' + top_10['Ürün 2']
                
                fig = px.bar(
                    top_10,
                    x='Sepet Sayısı',
                    y='Ürün Çifti',
                    orientation='h',
                    title='En Güçlü 10 Birliktelik',
                    labels={'Sepet Sayısı': 'Sepet Sayısı', 'Ürün Çifti': 'Ürün Çiftleri'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Session state'e kaydet
                st.session_state['birliktelikler'] = birliktelikler
                
            else:
                st.warning("❌ Hiç birliktelik bulunamadı. Support oranını düşürmeyi deneyin.")
    
    elif sayfa == "📋 Kural Analizi":
        st.header("📋 Association Rules (Birliktelik Kuralları)")
        
        # Önce birliktelik analizi yapılmış mı kontrol et
        if 'birliktelikler' not in st.session_state:
            st.warning("⚠️ Önce 'Birliktelik Analizi' sayfasında analiz yapmalısınız!")
        else:
            st.subheader("Parametreler")
            min_confidence = st.slider(
                "Minimum Confidence (Güven) Oranı:", 
                0.1, 0.9, 0.3, 0.05,
                help="Bir kuralın geçerli sayılması için minimum güven yüzdesi"
            )
            
            if st.button("📋 Kural Analizi Yap"):
                birliktelikler = st.session_state['birliktelikler']
                kurallar = kural_olustur(birliktelikler, urun_sayilari, len(sepetler), min_confidence)
                
                if kurallar:
                    st.success(f"✅ {len(kurallar)} kural bulundu!")
                    
                    # Kuralları DataFrame'e çevir
                    kural_listesi = []
                    for kural in kurallar:
                        kural_listesi.append({
                            'Öncül': kural['antecedent'],
                            'Sonuç': kural['consequent'],
                            'Support': f"%{kural['support']*100:.1f}",
                            'Confidence': f"%{kural['confidence']*100:.1f}",
                            'Lift': f"{kural['lift']:.2f}"
                        })
                    
                    df_kurallar = pd.DataFrame(kural_listesi)
                    
                    st.subheader("En Güçlü Kurallar")
                    st.dataframe(df_kurallar.head(20), use_container_width=True)
                    
                    # Kural açıklaması
                    st.subheader("📖 Kural Açıklaması")
                    st.write("**En güçlü 3 kural:**")
                    
                    for i, kural in enumerate(kurallar[:3], 1):
                        st.write(f"""
                        **{i}. {kural['antecedent']} → {kural['consequent']}**
                        - '{kural['antecedent']}' alan müşterilerin %{kural['confidence']*100:.0f}'i '{kural['consequent']}' da alıyor
                        - Bu birliktelik tesadüften {kural['lift']:.1f} kat daha güçlü
                        """)
                    
                    # Session state'e kaydet
                    st.session_state['kurallar'] = kurallar
                    
                else:
                    st.warning("❌ Hiç kural bulunamadı. Confidence oranını düşürmeyi deneyin.")
    
    elif sayfa == "🎯 Ürün Önerileri":
        st.header("🎯 Ürün Önerileri")
        
        # Kurallar var mı kontrol et
        if 'kurallar' not in st.session_state:
            st.warning("⚠️ Önce 'Kural Analizi' sayfasında analiz yapmalısınız!")
        else:
            kurallar = st.session_state['kurallar']
            
            st.subheader("Ürün Seçimi")
            
            # Tüm ürünleri listele
            tum_urunler = sorted(urun_sayilari.keys())
            secilen_urun = st.selectbox("Öneri almak istediğiniz ürünü seçin:", tum_urunler)
            
            oneri_sayisi = st.slider("Gösterilecek öneri sayısı:", 1, 10, 5)
            
            if st.button("🎯 Önerileri Göster"):
                # Bu ürün için uygun kuralları bul
                uygun_kurallar = [kural for kural in kurallar 
                                 if kural['antecedent'] == secilen_urun]
                
                if uygun_kurallar:
                    st.success(f"✅ '{secilen_urun}' için {len(uygun_kurallar)} öneri bulundu!")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader(f"'{secilen_urun}' için Öneriler")
                        
                        for i, kural in enumerate(uygun_kurallar[:oneri_sayisi], 1):
                            st.write(f"""
                            **{i}. {kural['consequent']}**
                            - Güven: %{kural['confidence']*100:.1f}
                            - Lift: {kural['lift']:.2f}
                            """)
                    
                    with col2:
                        st.subheader("Öneri Gücü Grafiği")
                        
                        top_oneriler = uygun_kurallar[:oneri_sayisi]
                        urun_isimleri = [kural['consequent'] for kural in top_oneriler]
                        guven_skorlari = [kural['confidence']*100 for kural in top_oneriler]
                        
                        fig = px.bar(
                            x=guven_skorlari,
                            y=urun_isimleri,
                            orientation='h',
                            title=f"'{secilen_urun}' için Öneri Gücü",
                            labels={'x': 'Güven Skoru (%)', 'y': 'Önerilen Ürünler'},
                            color=guven_skorlari,
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("📊 Detaylı Açıklama")
                    en_iyi_kural = uygun_kurallar[0]
                    
                    st.info(f"""
                    **En güçlü öneri: {en_iyi_kural['consequent']}**
                    
                    📈 **Analiz Sonucu:**
                    - '{secilen_urun}' satın alan müşterilerin %{en_iyi_kural['confidence']*100:.0f}'i aynı zamanda '{en_iyi_kural['consequent']}' da satın alıyor
                    - Bu birliktelik tesadüften {en_iyi_kural['lift']:.1f} kat daha güçlü
                    - Bu iki ürün sepetlerin %{en_iyi_kural['support']*100:.1f}'inde birlikte görülüyor
                    
                    💡 **İş Önerisi:**
                    - '{secilen_urun}' alan müşterilere '{en_iyi_kural['consequent']}' önerebilirsiniz
                    - Bu ürünleri mağazada yakın yerlere koyabilirsiniz
                    - Bu ürünlerde birlikte kampanya yapabilirsiniz
                    """)
                    
                else:
                    st.warning(f"❌ '{secilen_urun}' için öneri bulunamadı.")
                    st.write("Bu durum şu sebeplerden olabilir:")
                    st.write("- Bu ürün başka ürünlerle güçlü birliktelik oluşturmuyor")
                    st.write("- Confidence threshold'u çok yüksek olabilir")
                    st.write("- Bu ürün genellikle tek başına alınıyor")

else:
    st.error("Veri yüklenemedi. Lütfen data/basket_analysis.csv dosyasının mevcut olduğundan emin olun.")

# Footer
st.markdown("---")
st.markdown("💡 **İpucu:** Sol menüden farklı analiz türlerini deneyebilirsiniz!")
st.markdown("📖 **Not:** Bu uygulama eğitim amaçlı hazırlanmıştır.")