"""
Model Boyutunu Küçültme ve Optimizasyon Örneği
Bu script BERT modellerini optimizasyon tekniklerini gösterir
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, 
    DistilBertTokenizer, DistilBertModel
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import time
import pickle
import os
from typing import List, Tuple
import matplotlib.pyplot as plt

class ModelOptimizer:
    def __init__(self):
        """Model optimizasyon sınıfı"""
        self.original_model = None
        self.optimized_model = None
        self.tokenizer = None
        self.pca_reducer = None
        
    def load_model(self, model_name='bert-base-multilingual-cased'):
        """Orijinal modeli yükle"""
        print(f"Model yükleniyor: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.original_model = AutoModel.from_pretrained(model_name)
            print(f"Model başarıyla yüklendi!")
            return True
        except Exception as e:
            print(f"Model yüklenirken hata: {e}")
            # Fallback modeli dene
            try:
                fallback_model = 'distilbert-base-multilingual-cased'
                print(f"Fallback model deneniyor: {fallback_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self.original_model = AutoModel.from_pretrained(fallback_model)
                print(f"Fallback model başarıyla yüklendi!")
                return True
            except Exception as e2:
                print(f"Fallback model de yüklenemedi: {e2}")
                return False
    
    def load_distilled_model(self):
        """Distilled (küçültülmüş) model yükle"""
        print("DistilBERT modeli yükleniyor...")
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
            self.optimized_model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
            print("DistilBERT başarıyla yüklendi!")
            return True
        except Exception as e:
            print(f"DistilBERT yüklenirken hata: {e}")
            return False
    
    def get_model_size(self, model):
        """Model boyutunu hesapla"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def get_embeddings_with_timing(self, texts: List[str], model, tokenizer, max_length=128):
        """Zamanlama ile embeddings al"""
        start_time = time.time()
        embeddings = []
        
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(embedding)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return np.array(embeddings), processing_time
    
    def apply_quantization(self, model):
        """Model quantization uygula"""
        print("Model quantization uygulanıyor...")
        try:
            # PyTorch quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            return quantized_model
        except Exception as e:
            print(f"Quantization hatası: {e}")
            return model
    
    def apply_pruning(self, embeddings, variance_threshold=0.95):
        """PCA ile boyut azaltma (pruning benzeri)"""
        print(f"PCA ile boyut azaltma uygulanıyor (varyans: {variance_threshold})")
        
        pca = PCA()
        pca.fit(embeddings)
        
        # Kümülatif varyansı hesapla
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        
        # Threshold'u geçen component sayısını bul
        n_components = np.argmax(cumsum >= variance_threshold) + 1
        
        print(f"Orijinal boyut: {embeddings.shape[1]}")
        print(f"Azaltılmış boyut: {n_components}")
        print(f"Boyut azaltma oranı: {(1 - n_components/embeddings.shape[1])*100:.1f}%")
        
        # Yeni PCA transformer
        self.pca_reducer = PCA(n_components=n_components)
        reduced_embeddings = self.pca_reducer.fit_transform(embeddings)
        
        return reduced_embeddings
    
    def create_vocabulary_subset(self, texts: List[str], vocab_size=5000):
        """Sınırlı vocabulary oluştur"""
        print(f"Vocabulary subset oluşturuluyor (boyut: {vocab_size})")
        
        from collections import Counter
        import re
        
        # Tüm metinleri birleştir ve tokenize et
        all_text = " ".join(texts).lower()
        words = re.findall(r'\b\w+\b', all_text)
        
        # En sık kullanılan kelimeleri bul
        word_counts = Counter(words)
        top_words = [word for word, count in word_counts.most_common(vocab_size)]
        
        print(f"Vocabulary oluşturuldu: {len(top_words)} kelime")
        return top_words
    
    def compress_embeddings(self, embeddings, method='svd', target_dim=128):
        """Embeddings'leri sıkıştır"""
        print(f"Embeddings sıkıştırılıyor ({method}, hedef boyut: {target_dim})")
        
        if method == 'svd':
            reducer = TruncatedSVD(n_components=target_dim, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=target_dim)
        else:
            raise ValueError("Desteklenen methodlar: 'svd', 'pca'")
        
        compressed = reducer.fit_transform(embeddings)
        
        print(f"Sıkıştırma tamamlandı: {embeddings.shape} -> {compressed.shape}")
        print(f"Açıklanan varyans: {reducer.explained_variance_ratio_.sum():.3f}")
        
        return compressed, reducer
    
    def benchmark_models(self, test_texts: List[str]):
        """Modelleri karşılaştır"""
        print("=" * 80)
        print("MODEL KARŞILAŞTIRMA")
        print("=" * 80)
        
        results = {}
        
        # Orijinal BERT
        if self.original_model:
            print("\n--- Orijinal BERT ---")
            size_mb = self.get_model_size(self.original_model)
            embeddings, processing_time = self.get_embeddings_with_timing(
                test_texts, self.original_model, self.tokenizer
            )
            
            results['original_bert'] = {
                'size_mb': size_mb,
                'processing_time': processing_time,
                'embedding_shape': embeddings.shape,
                'embeddings': embeddings
            }
            
            print(f"Model boyutu: {size_mb:.1f} MB")
            print(f"İşleme zamanı: {processing_time:.3f} saniye")
            print(f"Embedding boyutu: {embeddings.shape}")
        
        # DistilBERT
        if self.load_distilled_model():
            print("\n--- DistilBERT ---")
            size_mb = self.get_model_size(self.optimized_model)
            embeddings, processing_time = self.get_embeddings_with_timing(
                test_texts, self.optimized_model, self.tokenizer
            )
            
            results['distilbert'] = {
                'size_mb': size_mb,
                'processing_time': processing_time,
                'embedding_shape': embeddings.shape,
                'embeddings': embeddings
            }
            
            print(f"Model boyutu: {size_mb:.1f} MB")
            print(f"İşleme zamanı: {processing_time:.3f} saniye")
            print(f"Embedding boyutu: {embeddings.shape}")
        
        # Quantized model (eğer orijinal model varsa)
        if self.original_model:
            print("\n--- Quantized BERT ---")
            quantized_model = self.apply_quantization(self.original_model)
            size_mb = self.get_model_size(quantized_model)
            embeddings, processing_time = self.get_embeddings_with_timing(
                test_texts, quantized_model, self.tokenizer
            )
            
            results['quantized_bert'] = {
                'size_mb': size_mb,
                'processing_time': processing_time,
                'embedding_shape': embeddings.shape,
                'embeddings': embeddings
            }
            
            print(f"Model boyutu: {size_mb:.1f} MB")
            print(f"İşleme zamanı: {processing_time:.3f} saniye")
            print(f"Embedding boyutu: {embeddings.shape}")
        
        return results
    
    def visualize_compression_effects(self, original_embeddings, compressed_embeddings):
        """Sıkıştırma etkilerini görselleştir"""
        print("\nSıkıştırma etkilerini görselleştiriyor...")
        
        # 2D PCA ile görselleştirme
        pca_2d = PCA(n_components=2)
        
        original_2d = pca_2d.fit_transform(original_embeddings)
        compressed_2d = pca_2d.fit_transform(compressed_embeddings)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Orijinal embeddings
        ax1.scatter(original_2d[:, 0], original_2d[:, 1], alpha=0.7)
        ax1.set_title(f'Orijinal Embeddings\n({original_embeddings.shape[1]} boyut)')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        
        # Sıkıştırılmış embeddings
        ax2.scatter(compressed_2d[:, 0], compressed_2d[:, 1], alpha=0.7, color='red')
        ax2.set_title(f'Sıkıştırılmış Embeddings\n({compressed_embeddings.shape[1]} boyut)')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        
        plt.tight_layout()
        plt.show()
    
    def save_optimized_components(self, output_dir='optimized_models'):
        """Optimize edilmiş bileşenleri kaydet"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.pca_reducer:
            with open(f'{output_dir}/pca_reducer.pkl', 'wb') as f:
                pickle.dump(self.pca_reducer, f)
            print(f"PCA reducer kaydedildi: {output_dir}/pca_reducer.pkl")
        
        print(f"Optimizasyon bileşenleri kaydedildi: {output_dir}/")

def demonstrate_optimization():
    """Model optimizasyon demonstrasyonu"""
    print("=" * 80)
    print("MODEL OPTİMİZASYON DEMONSTRASİYONU")
    print("=" * 80)
    
    # Test metinleri
    test_texts = [
        "Hasta hipertansiyon nedeniyle tedavi görüyor",
        "Diyabet tip 2 hastası kan şeker kontrolünde",
        "Astım krizi geçiren hasta bronkodilatör aldı",
        "Depresyon tanısı alan hasta psikiyatrik destek alıyor",
        "Migren ağrıları olan hasta ağrı kesici verildi",
        "Kalp hastalığı bulunan hasta kardiyoloji kontrolünde",
        "Obez hasta diyet programına alındı kilo verme",
        "Artrit ağrıları olan hasta fizik tedavi başlayacak"
    ]
    
    # Optimizör oluştur
    optimizer = ModelOptimizer()
    
    # Model yükle
    if not optimizer.load_model():
        print("Model yüklenemedi. Demo durduruluyor.")
        return
    
    # Model karşılaştırması
    print("\nModel performans karşılaştırması yapılıyor...")
    benchmark_results = optimizer.benchmark_models(test_texts)
    
    # Sonuçları özetle
    print("\n" + "=" * 60)
    print("KARŞILAŞTIRMA ÖZETİ")
    print("=" * 60)
    
    for model_name, results in benchmark_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Boyut: {results['size_mb']:.1f} MB")
        print(f"  Süre: {results['processing_time']:.3f} sn")
        print(f"  Embedding: {results['embedding_shape']}")
        
        if 'original_bert' in benchmark_results:
            orig_size = benchmark_results['original_bert']['size_mb']
            orig_time = benchmark_results['original_bert']['processing_time']
            
            size_reduction = (1 - results['size_mb'] / orig_size) * 100
            time_improvement = (1 - results['processing_time'] / orig_time) * 100
            
            print(f"  Boyut azaltma: {size_reduction:+.1f}%")
            print(f"  Hız artışı: {time_improvement:+.1f}%")
    
    # Embeddings sıkıştırma
    if 'original_bert' in benchmark_results:
        print("\n" + "=" * 60)
        print("EMBEDDİNGS SIKIŞTIRRMA")
        print("=" * 60)
        
        original_embeddings = benchmark_results['original_bert']['embeddings']
        
        # SVD ile sıkıştırma
        compressed_svd, svd_reducer = optimizer.compress_embeddings(
            original_embeddings, method='svd', target_dim=128
        )
        
        # PCA ile boyut azaltma
        compressed_pca = optimizer.apply_pruning(original_embeddings, variance_threshold=0.90)
        
        # Görselleştirme
        optimizer.visualize_compression_effects(original_embeddings, compressed_svd)
        
        # Vocabulary subset
        subset_vocab = optimizer.create_vocabulary_subset(test_texts, vocab_size=1000)
        print(f"Vocabulary subset örneği: {subset_vocab[:10]}")
    
    # Bileşenleri kaydet
    optimizer.save_optimized_components()
    
    print("\nOptimizasyon demonstrasyonu tamamlandı!")

def compare_inference_speeds():
    """Farklı model boyutlarında çıkarım hızlarını karşılaştır"""
    print("\n" + "=" * 80)
    print("ÇIKARIM HIZI KARŞILAŞTIRMA")
    print("=" * 80)
    
    # Farklı boyutlarda test metinleri
    text_sizes = [1, 5, 10, 20, 50]
    base_text = "Hasta hipertansiyon nedeniyle tedavi görüyor. Kan basıncı kontrol altında tutulmalı."
    
    optimizer = ModelOptimizer()
    
    if not optimizer.load_model():
        return
    
    results = []
    
    for size in text_sizes:
        test_texts = [base_text] * size
        
        # Orijinal model
        _, time_original = optimizer.get_embeddings_with_timing(
            test_texts, optimizer.original_model, optimizer.tokenizer
        )
        
        # DistilBERT
        if optimizer.load_distilled_model():
            _, time_distil = optimizer.get_embeddings_with_timing(
                test_texts, optimizer.optimized_model, optimizer.tokenizer
            )
        else:
            time_distil = None
        
        results.append({
            'size': size,
            'original': time_original,
            'distilbert': time_distil
        })
        
        print(f"Metin sayısı {size:2d}: BERT {time_original:.3f}s", end="")
        if time_distil:
            speedup = time_original / time_distil
            print(f", DistilBERT {time_distil:.3f}s (Hızlanma: {speedup:.1f}x)")
        else:
            print()
    
    # Grafik çiz
    if len(results) > 1:
        sizes = [r['size'] for r in results]
        original_times = [r['original'] for r in results]
        distil_times = [r['distilbert'] for r in results if r['distilbert']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, original_times, 'b-o', label='BERT', linewidth=2)
        if distil_times and len(distil_times) == len(sizes):
            plt.plot(sizes, distil_times, 'r-o', label='DistilBERT', linewidth=2)
        
        plt.xlabel('Metin Sayısı')
        plt.ylabel('İşleme Zamanı (saniye)')
        plt.title('Model Boyutu ve Çıkarım Hızı Karşılaştırma')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    demonstrate_optimization()
    compare_inference_speeds()