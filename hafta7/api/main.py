"""
FastAPI Metin Analiz Servisi
Bu servis metin temizleme, TF-IDF analizi, BERT embeddings ve PII maskeleme API'larını sunar
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import re
import sys
import os

# Parent directory'yi path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Metin Analiz API",
    description="Sağlık verilerinde metin analizi için kapsamlı API servisi",
    version="1.0.0"
)

# Global değişkenler
tfidf_vectorizer = None
bert_tokenizer = None
bert_model = None

# Pydantic modelleri
class TextRequest(BaseModel):
    text: str
    language: str = "tr"

class TextListRequest(BaseModel):
    texts: List[str]
    language: str = "tr"

class TFIDFRequest(BaseModel):
    texts: List[str]
    max_features: int = 1000
    ngram_range: tuple = (1, 2)

class SimilarityRequest(BaseModel):
    query_text: str
    corpus_texts: List[str]
    method: str = "tfidf"  # "tfidf" or "bert"

class PIIMaskRequest(BaseModel):
    text: str
    method: str = "regex"  # "regex" or "presidio"

# Utility fonksiyonları
def clean_text(text: str) -> str:
    """Temel metin temizleme"""
    if not text:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def mask_pii_regex(text: str) -> str:
    """Regex ile PII maskeleme"""
    text = re.sub(r'\b\d{11}\b', '[TC_KIMLIK]', text)
    text = re.sub(r'\b0?\d{3}[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}\b', '[TELEFON]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    return text

def load_bert_model():
    """BERT modelini yükle"""
    global bert_tokenizer, bert_model
    try:
        model_name = 'bert-base-multilingual-cased'
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)
        return True
    except Exception as e:
        print(f"BERT model yüklenemedi: {e}")
        # Fallback model dene
        try:
            fallback_model = 'distilbert-base-multilingual-cased'
            print(f"Fallback model deneniyor: {fallback_model}")
            bert_tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            bert_model = AutoModel.from_pretrained(fallback_model)
            print("Fallback model başarıyla yüklendi!")
            return True
        except Exception as e2:
            print(f"Fallback model de yüklenemedi: {e2}")
            return False

def get_bert_embeddings(texts: List[str]) -> np.ndarray:
    """BERT embeddings hesapla"""
    if not bert_model or not bert_tokenizer:
        return None
    
    embeddings = []
    for text in texts:
        inputs = bert_tokenizer(
            text,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(embedding)
    
    return np.array(embeddings)

# API Endpoints
@app.get("/")
async def root():
    """Ana sayfa"""
    return {
        "message": "Metin Analiz API'sine hoş geldiniz!",
        "version": "1.0.0",
        "endpoints": [
            "/docs - API dokumentasyonu",
            "/clean-text - Metin temizleme",
            "/tfidf-analysis - TF-IDF analizi",
            "/similarity - Benzerlik analizi",
            "/bert-embeddings - BERT embeddings",
            "/mask-pii - PII maskeleme"
        ]
    }

@app.post("/clean-text")
async def clean_text_endpoint(request: TextRequest):
    """Metin temizleme servisi"""
    try:
        cleaned = clean_text(request.text)
        return {
            "original_text": request.text,
            "cleaned_text": cleaned,
            "language": request.language,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metin temizleme hatası: {str(e)}")

@app.post("/tfidf-analysis")
async def tfidf_analysis_endpoint(request: TFIDFRequest):
    """TF-IDF analizi servisi"""
    try:
        # Metinleri temizle
        cleaned_texts = [clean_text(text) for text in request.texts]
        
        # TF-IDF vektörleştir
        vectorizer = TfidfVectorizer(
            max_features=request.max_features,
            ngram_range=request.ngram_range,
            lowercase=True
        )
        
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # En önemli terimleri bul
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = np.argsort(mean_scores)[::-1][:20]
        
        top_terms = []
        for idx in top_indices:
            top_terms.append({
                "term": feature_names[idx],
                "score": float(mean_scores[idx])
            })
        
        return {
            "top_terms": top_terms,
            "matrix_shape": tfidf_matrix.shape,
            "total_features": len(feature_names),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TF-IDF analizi hatası: {str(e)}")

@app.post("/similarity")
async def similarity_analysis_endpoint(request: SimilarityRequest):
    """Benzerlik analizi servisi"""
    try:
        all_texts = [request.query_text] + request.corpus_texts
        
        if request.method == "tfidf":
            # TF-IDF benzerliği
            cleaned_texts = [clean_text(text) for text in all_texts]
            
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
        elif request.method == "bert":
            # BERT benzerliği
            if not bert_model:
                load_bert_model()
            
            if not bert_model:
                raise HTTPException(status_code=500, detail="BERT modeli yüklenemedi")
            
            embeddings = get_bert_embeddings(all_texts)
            query_embedding = embeddings[0:1]
            corpus_embeddings = embeddings[1:]
            
            similarities = cosine_similarity(query_embedding, corpus_embeddings).flatten()
            
        else:
            raise HTTPException(status_code=400, detail="Geçersiz method. 'tfidf' veya 'bert' kullanın.")
        
        # Sonuçları sırala
        results = []
        for i, similarity in enumerate(similarities):
            results.append({
                "text": request.corpus_texts[i],
                "similarity": float(similarity),
                "rank": i + 1
            })
        
        # Benzerlik skoruna göre sırala
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        return {
            "query_text": request.query_text,
            "method": request.method,
            "results": results,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benzerlik analizi hatası: {str(e)}")

@app.post("/bert-embeddings")
async def bert_embeddings_endpoint(request: TextListRequest):
    """BERT embeddings servisi"""
    try:
        if not bert_model:
            load_bert_model()
        
        if not bert_model:
            raise HTTPException(status_code=500, detail="BERT modeli yüklenemedi")
        
        embeddings = get_bert_embeddings(request.texts)
        
        if embeddings is None:
            raise HTTPException(status_code=500, detail="Embeddings hesaplanamadı")
        
        return {
            "embeddings": embeddings.tolist(),
            "shape": embeddings.shape,
            "texts_count": len(request.texts),
            "embedding_size": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BERT embeddings hatası: {str(e)}")

@app.post("/mask-pii")
async def mask_pii_endpoint(request: PIIMaskRequest):
    """PII maskeleme servisi"""
    try:
        if request.method == "regex":
            masked_text = mask_pii_regex(request.text)
        else:
            # Presidio implementasyonu burada olabilir
            masked_text = mask_pii_regex(request.text)
        
        return {
            "original_text": request.text,
            "masked_text": masked_text,
            "method": request.method,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PII maskeleme hatası: {str(e)}")

@app.post("/health-data-analysis")
async def health_data_analysis_endpoint(request: TextListRequest):
    """Sağlık verilerinde kapsamlı analiz"""
    try:
        # Metinleri temizle
        cleaned_texts = [clean_text(text) for text in request.texts]
        
        # TF-IDF analizi
        vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            lowercase=True
        )
        
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # En önemli terimleri bul
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = np.argsort(mean_scores)[::-1][:10]
        
        top_terms = []
        for idx in top_indices:
            top_terms.append({
                "term": feature_names[idx],
                "score": float(mean_scores[idx])
            })
        
        # Kümeleme (basit)
        from sklearn.cluster import KMeans
        
        if len(request.texts) >= 3:
            n_clusters = min(3, len(request.texts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            cluster_info = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_info:
                    cluster_info[cluster_id] = []
                cluster_info[cluster_id].append({
                    "index": i,
                    "text": request.texts[i][:100] + "..." if len(request.texts[i]) > 100 else request.texts[i]
                })
        else:
            cluster_info = {"0": [{"index": i, "text": text[:100] + "..."} for i, text in enumerate(request.texts)]}
        
        return {
            "analysis_summary": {
                "total_texts": len(request.texts),
                "top_terms": top_terms,
                "clusters": cluster_info
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sağlık verisi analizi hatası: {str(e)}")

@app.get("/health")
async def health_check():
    """Servis sağlık kontrolü"""
    return {
        "status": "healthy",
        "bert_model_loaded": bert_model is not None,
        "services": [
            "text_cleaning",
            "tfidf_analysis", 
            "similarity_analysis",
            "bert_embeddings",
            "pii_masking"
        ]
    }

# Başlangıçta BERT modelini yükle
@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışacak fonksiyon"""
    print("Metin Analiz API başlatılıyor...")
    print("BERT modeli yükleniyor...")
    success = load_bert_model()
    if success:
        print("BERT modeli başarıyla yüklendi!")
    else:
        print("BERT modeli yüklenemedi. Sadece TF-IDF servisleri çalışacak.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)