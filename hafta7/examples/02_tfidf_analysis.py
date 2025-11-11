"""
TF-IDF Analizi Örneği
Bu script sağlık verilerinde TF-IDF kullanarak metin analizini gösterir
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

class TFIDFAnalyzer:
    def __init__(self, max_features=1000, ngram_range=(1, 2)):
        """
        TF-IDF analizci sınıfı
        
        Args:
            max_features: Maksimum özellik sayısı
            ngram_range: N-gram aralığı (1,1) unigram, (1,2) unigram+bigram
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None
    
    def preprocess_text(self, text):
        """Basit metin ön işleme"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Sadece harfleri ve sayıları tut
        text = re.sub(r'[^a-zA-Z0-9çğıöşüÇĞIİÖŞÜ\s]', ' ', text)
        # Çoklu boşlukları tek boşlukla değiştir
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def fit_transform(self, texts):
        """TF-IDF vektörleştirme"""
        # Metinleri ön işle
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # TF-IDF vectorizer'ı oluştur
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words=None,  # Türkçe için None
            lowercase=True,
            min_df=2,  # En az 2 dokümanda geçen terimler
            max_df=0.95  # %95'ten fazla dokümanda geçen terimleri çıkar
        )
        
        # TF-IDF matrix'ini hesapla
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        return self.tfidf_matrix
    
    def get_top_terms(self, n_terms=20):
        """En önemli terimleri getir"""
        if self.tfidf_matrix is None:
            return None
        
        # Her terimin ortalama TF-IDF skorunu hesapla
        mean_scores = np.array(self.tfidf_matrix.mean(axis=0)).flatten()
        
        # En yüksek skorları al
        top_indices = np.argsort(mean_scores)[::-1][:n_terms]
        
        top_terms = []
        for idx in top_indices:
            top_terms.append({
                'term': self.feature_names[idx],
                'score': mean_scores[idx]
            })
        
        return top_terms
    
    def find_similar_documents(self, doc_index, n_similar=5):
        """Belirli bir dokümana benzer dokümanlar bul"""
        if self.tfidf_matrix is None:
            return None
        
        # Cosine similarity hesapla
        similarities = cosine_similarity(
            self.tfidf_matrix[doc_index], 
            self.tfidf_matrix
        ).flatten()
        
        # Kendisi hariç en benzer dokümanları bul
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        results = []
        for idx in similar_indices:
            results.append({
                'index': idx,
                'similarity': similarities[idx]
            })
        
        return results
    
    def cluster_documents(self, n_clusters=5):
        """Dokümanları kümeleyerek grupla"""
        if self.tfidf_matrix is None:
            return None
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.tfidf_matrix)
        
        return clusters
    
    def reduce_dimensions(self, n_components=50):
        """Boyut azaltma için SVD uygula"""
        if self.tfidf_matrix is None:
            return None
        
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_data = svd.fit_transform(self.tfidf_matrix)
        
        return reduced_data, svd
    
    def visualize_top_terms(self, n_terms=15):
        """En önemli terimleri görselleştir"""
        top_terms = self.get_top_terms(n_terms)
        if not top_terms:
            return
        
        terms = [t['term'] for t in top_terms]
        scores = [t['score'] for t in top_terms]
        
        plt.figure(figsize=(10, 6))
        plt.barh(terms[::-1], scores[::-1])
        plt.xlabel('Ortalama TF-IDF Skoru')
        plt.title(f'En Önemli {n_terms} Terim')
        plt.tight_layout()
        plt.show()

def analyze_health_data():
    """Sağlık verilerinde TF-IDF analizi"""
    print("=" * 80)
    print("TF-IDF ANALİZİ ÖRNEĞİ - SAĞLIK VERİLERİ")
    print("=" * 80)
    
    # Sağlık verilerini yükle
    try:
        df = pd.read_csv('data/synthetic_health_data.csv')
        print(f"Veri seti yüklendi: {len(df)} kayıt")
    except FileNotFoundError:
        print("Sağlık verisi bulunamadı. Sentetik veri oluşturuluyor...")
        
        # Basit sentetik veri oluştur
        diagnoses = [
            "hastada şiddetli hipertansiyon görülmektedir kan basıncı yüksek",
            "diyabet tip 2 tanısı konulmuştur kan şekeri kontrolü gerekli",
            "astım krizi geçiren hasta bronkodilatör tedavi başlandı",
            "depresyon belirtileri gösteren hasta psikiyatrik destek alacak",
            "migren ağrıları şiddetlenen hasta ağrı kesici verildi",
            "kalp hastalığı riski yüksek hasta kardiyoloji kontrolü",
            "obezite problemi olan hasta diyet programına alındı",
            "artrit ağrıları olan hasta fizik tedavi başlayacak"
        ] * 10  # 80 kayıt oluştur
        
        df = pd.DataFrame({
            'tani': diagnoses,
            'notlar': [f"hasta için özel tedavi notları {i}" for i in range(len(diagnoses))]
        })
    
    # Tanı verilerini analiz et
    analyzer = TFIDFAnalyzer(max_features=500, ngram_range=(1, 2))
    
    # TF-IDF matrix'ini hesapla
    print("\nTF-IDF matrix'i hesaplanıyor...")
    tfidf_matrix = analyzer.fit_transform(df['tani'])
    print(f"Matrix boyutu: {tfidf_matrix.shape}")
    
    # En önemli terimleri göster
    print("\n--- EN ÖNEMLİ TERİMLER ---")
    top_terms = analyzer.get_top_terms(15)
    for i, term_info in enumerate(top_terms, 1):
        print(f"{i:2d}. {term_info['term']:20} (skor: {term_info['score']:.4f})")
    
    # Benzer dokümanları bul
    print("\n--- BENZERLİK ANALİZİ ---")
    sample_doc_idx = 0
    similar_docs = analyzer.find_similar_documents(sample_doc_idx, n_similar=3)
    
    print(f"Referans doküman ({sample_doc_idx}): {df.iloc[sample_doc_idx]['tani']}")
    print("En benzer dokümanlar:")
    for i, sim_info in enumerate(similar_docs, 1):
        idx = sim_info['index']
        similarity = sim_info['similarity']
        print(f"{i}. (Benzerlik: {similarity:.3f}) {df.iloc[idx]['tani']}")
    
    # Kümeleme analizi
    print("\n--- KÜMELEME ANALİZİ ---")
    n_clusters = 5
    clusters = analyzer.cluster_documents(n_clusters)
    
    # Küme dağılımını göster
    cluster_counts = Counter(clusters)
    print("Küme dağılımı:")
    for cluster_id, count in sorted(cluster_counts.items()):
        print(f"Küme {cluster_id}: {count} doküman")
    
    # Her kümeden örnek dokümanlar göster
    print("\nKüme örnekleri:")
    for cluster_id in range(n_clusters):
        cluster_docs = df[clusters == cluster_id]['tani'].head(2)
        print(f"\nKüme {cluster_id} örnekleri:")
        for doc in cluster_docs:
            print(f"  - {doc}")
    
    # Boyut azaltma
    print("\n--- BOYUT AZALTMA ---")
    reduced_data, svd = analyzer.reduce_dimensions(n_components=10)
    print(f"Orijinal boyut: {tfidf_matrix.shape}")
    print(f"Azaltılmış boyut: {reduced_data.shape}")
    print(f"Açıklanan varyans oranı: {svd.explained_variance_ratio_.sum():.3f}")

if __name__ == "__main__":
    analyze_health_data()