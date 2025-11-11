"""
BERT Tabanlı Metin Analizi Örneği
Bu script BERT modelini kullanarak Türkçe sağlık verilerinde metin analizini gösterir
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModel, 
    pipeline, AutoModelForSequenceClassification
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BERTAnalyzer:
    def __init__(self, model_name='bert-base-multilingual-cased'):
        """
        BERT tabanlı analizci sınıfı
        
        Args:
            model_name: Kullanılacak BERT modeli
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.embeddings = None
        
    def load_model(self):
        """BERT modelini yükle"""
        print(f"BERT modeli yükleniyor: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            print("Model başarıyla yüklendi!")
            return True
        except Exception as e:
            print(f"Model yüklenirken hata: {e}")
            # Fallback to a smaller model
            print("Alternatif model deneniyor...")
            try:
                self.model_name = 'distilbert-base-multilingual-cased'
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                print("Alternatif model başarıyla yüklendi!")
                return True
            except Exception as e2:
                print(f"Alternatif model de yüklenemedi: {e2}")
                return False
    
    def get_embeddings(self, texts, max_length=128):
        """Metinler için BERT embeddings'lerini al"""
        if self.model is None:
            return None
        
        embeddings = []
        
        print(f"BERT embeddings hesaplanıyor... ({len(texts)} metin)")
        
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"İşlenen: {i}/{len(texts)}")
            
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
            # Model çıktısını al
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # CLS token embedding'ini al (ilk token)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(embedding)
        
        self.embeddings = np.array(embeddings)
        print(f"Embeddings hesaplandı. Boyut: {self.embeddings.shape}")
        
        return self.embeddings
    
    def find_similar_texts(self, query_index, n_similar=5):
        """Benzer metinleri bul"""
        if self.embeddings is None:
            return None
        
        # Cosine similarity hesapla
        query_embedding = self.embeddings[query_index].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
        # En benzer olanları bul (kendisi hariç)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        results = []
        for idx in similar_indices:
            results.append({
                'index': idx,
                'similarity': similarities[idx]
            })
        
        return results
    
    def cluster_texts(self, n_clusters=5):
        """Metinleri kümeleyerek grupla"""
        if self.embeddings is None:
            return None
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.embeddings)
        
        return clusters
    
    def reduce_dimensions_pca(self, n_components=2):
        """PCA ile boyut azaltma"""
        if self.embeddings is None:
            return None
        
        pca = PCA(n_components=n_components, random_state=42)
        reduced_embeddings = pca.fit_transform(self.embeddings)
        
        return reduced_embeddings, pca

def setup_sentiment_pipeline():
    """Duygu analizi için pipeline oluştur"""
    try:
        # Türkçe duygu analizi modeli
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )
        return sentiment_pipeline
    except Exception as e:
        print(f"Duygu analizi modeli yüklenemedi: {e}")
        return None

def analyze_health_texts_with_bert():
    """Sağlık metinlerinde BERT analizi"""
    print("=" * 80)
    print("BERT TABANLI METİN ANALİZİ ÖRNEĞİ")
    print("=" * 80)
    
    # Sentetik sağlık metinleri
    health_texts = [
        "Hasta hipertansiyon nedeniyle acil servise başvurdu. Kan basıncı 180/110 mmHg ölçüldü.",
        "Diyabet hastası kan şeker kontrolünde. HbA1c değeri normal aralıkta.",
        "Astım krizi geçiren çocuk hasta bronkodilatör tedavi ile iyileşti.",
        "Depresyon tanısı alan hasta psikiyatrik tedavi başladı.",
        "Migren ağrıları nedeniyle başvuran hasta ağrı kesici aldı.",
        "Kalp hastalığı bulunan yaşlı hasta kardiyoloji kontrolünde.",
        "Obez hasta diyet programına alındı. Kilo verme hedeflendi.",
        "Artrit ağrıları olan hasta fizik tedavi başlayacak.",
        "COVID-19 pozitif hasta izolasyona alındı. Semptomlar hafif.",
        "Alerji krizi geçiren hasta antihistaminik tedavi aldı.",
        "Kanser hastası kemoterapi sürecinde. Yan etkiler takip ediliyor.",
        "Böbrek hastalığı olan hasta diyaliz tedavisine başladı."
    ]
    
    # BERT analizörünü oluştur
    analyzer = BERTAnalyzer()
    
    # Modeli yükle
    if not analyzer.load_model():
        print("BERT modeli yüklenemedi. Analiz durduruluyor.")
        return
    
    # Embeddings'leri hesapla
    embeddings = analyzer.get_embeddings(health_texts)
    
    if embeddings is None:
        print("Embeddings hesaplanamadı.")
        return
    
    # Benzerlik analizi
    print("\n--- BENZERLİK ANALİZİ ---")
    query_index = 0
    print(f"Sorgu metni: {health_texts[query_index]}")
    
    similar_results = analyzer.find_similar_texts(query_index, n_similar=3)
    print("\nEn benzer metinler:")
    for i, result in enumerate(similar_results, 1):
        idx = result['index']
        similarity = result['similarity']
        print(f"{i}. (Benzerlik: {similarity:.3f}) {health_texts[idx]}")
    
    # Kümeleme analizi
    print("\n--- KÜMELEME ANALİZİ ---")
    clusters = analyzer.cluster_texts(n_clusters=4)
    
    # Küme sonuçlarını göster
    cluster_df = pd.DataFrame({
        'text': health_texts,
        'cluster': clusters
    })
    
    print("Küme dağılımı:")
    for cluster_id in range(4):
        print(f"\nKüme {cluster_id}:")
        cluster_texts = cluster_df[cluster_df['cluster'] == cluster_id]['text']
        for text in cluster_texts:
            print(f"  - {text}")
    
    # Boyut azaltma ve görselleştirme
    print("\n--- BOYUT AZALTMA VE GÖRSELLEŞTİRME ---")
    reduced_embeddings, pca = analyzer.reduce_dimensions_pca(n_components=2)
    
    if reduced_embeddings is not None:
        plt.figure(figsize=(12, 8))
        
        # Kümeleme sonuçlarıyla renklendir
        colors = ['red', 'blue', 'green', 'orange']
        for cluster_id in range(4):
            cluster_points = reduced_embeddings[clusters == cluster_id]
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1], 
                c=colors[cluster_id], 
                label=f'Küme {cluster_id}',
                alpha=0.7
            )
        
        # Metinleri etiketle
        for i, (x, y) in enumerate(reduced_embeddings):
            plt.annotate(
                f'T{i}', 
                (x, y), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8
            )
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varyans)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varyans)')
        plt.title('BERT Embeddings - PCA Görselleştirme')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"Açıklanan toplam varyans: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Duygu analizi (eğer mevcut ise)
    print("\n--- DUYGU ANALİZİ ---")
    sentiment_pipeline = setup_sentiment_pipeline()
    
    if sentiment_pipeline:
        print("Metinlerin duygu analizi:")
        for i, text in enumerate(health_texts[:5]):  # İlk 5 metin
            try:
                result = sentiment_pipeline(text[:200])  # Uzun metinleri kırp
                label = result[0]['label']
                score = result[0]['score']
                print(f"T{i}: {label} (skor: {score:.3f})")
            except Exception as e:
                print(f"T{i}: Hata - {e}")

def compare_embedding_methods():
    """Farklı embedding yöntemlerini karşılaştır"""
    print("\n" + "=" * 80)
    print("EMBEDDING YÖNTEMLERİ KARŞILAŞTIRMA")
    print("=" * 80)
    
    sample_texts = [
        "Hasta hipertansiyon tedavisi görüyor.",
        "Kan basıncı yüksek hasta tedavi alıyor.",
        "Diyabet hastası insulin kullanıyor.",
        "Şeker hastalığı için ilaç alıyor."
    ]
    
    analyzer = BERTAnalyzer()
    if analyzer.load_model():
        embeddings = analyzer.get_embeddings(sample_texts)
        
        if embeddings is not None:
            print("\nBERT benzerlik matrisi:")
            similarity_matrix = cosine_similarity(embeddings)
            
            # DataFrame olarak göster
            similarity_df = pd.DataFrame(
                similarity_matrix,
                index=[f"T{i}" for i in range(len(sample_texts))],
                columns=[f"T{i}" for i in range(len(sample_texts))]
            )
            
            print(similarity_df.round(3))
            
            # Isı haritası
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                similarity_df, 
                annot=True, 
                cmap='coolwarm', 
                center=0.5,
                fmt='.3f'
            )
            plt.title('BERT Embeddings Benzerlik Matrisi')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    analyze_health_texts_with_bert()
    compare_embedding_methods()