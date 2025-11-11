"""
Metin Temizleme ve Tokenization Örneği
Bu script Türkçe metinlerde temel metin işleme operasyonlarını gösterir
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import spacy

# NLTK verilerini indir
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK verilerini indiriyor...")
    nltk.download('punkt')
    nltk.download('stopwords')

# SpaCy Türkçe modelini yükle (opsiyonel)
try:
    nlp = spacy.load("tr_core_news_sm")
    SPACY_AVAILABLE = True
except OSError:
    print("SpaCy Türkçe modeli bulunamadı. 'python -m spacy download tr_core_news_sm' komutu ile yükleyebilirsiniz.")
    SPACY_AVAILABLE = False

class TextProcessor:
    def __init__(self):
        self.turkish_stopwords = set(stopwords.words('turkish'))
        self.english_stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Türkçe karakterler için regex pattern
        self.turkish_chars = 'çğıöşüÇĞIİÖŞÜ'
        
    def clean_text(self, text):
        """Metni temizle"""
        if not text or pd.isna(text):
            return ""
        
        # String'e çevir
        text = str(text)
        
        # Küçük harfe çevir
        text = text.lower()
        
        # E-mail ve URL'leri kaldır
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+', '', text)
        
        # Telefon numaralarını kaldır
        text = re.sub(r'\b\d{10,}\b', '', text)
        text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
        
        # Noktalama işaretlerini kaldır (Türkçe karakterleri koru)
        text = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ]', ' ', text)
        
        # Çoklu boşlukları tek boşlukla değiştir
        text = re.sub(r'\s+', ' ', text)
        
        # Baştan ve sondan boşlukları kaldır
        text = text.strip()
        
        return text
    
    def remove_stopwords(self, tokens):
        """Türkçe ve İngilizce stop word'leri kaldır"""
        all_stopwords = self.turkish_stopwords.union(self.english_stopwords)
        return [token for token in tokens if token not in all_stopwords and len(token) > 2]
    
    def tokenize_with_nltk(self, text):
        """NLTK ile tokenization"""
        # Cümle tokenization
        sentences = sent_tokenize(text, language='turkish')
        
        # Kelime tokenization
        tokens = word_tokenize(text, language='turkish')
        
        return {
            'sentences': sentences,
            'tokens': tokens
        }
    
    def tokenize_with_spacy(self, text):
        """SpaCy ile tokenization (eğer mevcut ise)"""
        if not SPACY_AVAILABLE:
            return None
        
        doc = nlp(text)
        
        tokens = [token.text for token in doc if not token.is_space]
        lemmas = [token.lemma_ for token in doc if not token.is_space]
        pos_tags = [(token.text, token.pos_) for token in doc if not token.is_space]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return {
            'tokens': tokens,
            'lemmas': lemmas,
            'pos_tags': pos_tags,
            'entities': entities
        }
    
    def process_text(self, text):
        """Tam metin işleme pipeline'ı"""
        # Temizle
        cleaned_text = self.clean_text(text)
        
        # NLTK ile tokenize et
        nltk_results = self.tokenize_with_nltk(cleaned_text)
        
        # Stop word'leri kaldır
        filtered_tokens = self.remove_stopwords(nltk_results['tokens'])
        
        # SpaCy sonuçları (eğer mevcut ise)
        spacy_results = self.tokenize_with_spacy(cleaned_text) if SPACY_AVAILABLE else None
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'nltk_sentences': nltk_results['sentences'],
            'nltk_tokens': nltk_results['tokens'],
            'filtered_tokens': filtered_tokens,
            'spacy_results': spacy_results
        }

def demonstrate_text_processing():
    """Metin işleme örnekleri"""
    processor = TextProcessor()
    
    # Örnek metinler
    sample_texts = [
        "Merhaba! Ben Dr. Ahmet Yılmaz. E-mail adresim: ahmet.yilmaz@hastane.gov.tr ve telefonum: 0532-123-4567",
        "Hasta Ayşe Kaya'nın tanısı hipertansiyon. Adres: Ankara Çankaya Mahallesi No:15",
        "COVID-19 pandemisi sırasında sağlık çalışanlarımız büyük fedakarlık gösterdi.",
        "Bu ilaçları günde 3 kez, yemekten sonra alınız. Dr. Mehmet Özkan - 0312-456-7890"
    ]
    
    print("=" * 80)
    print("METİN TEMİZLEME VE TOKENİZATION ÖRNEĞİ")
    print("=" * 80)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n--- ÖRNEK {i} ---")
        result = processor.process_text(text)
        
        print(f"Orijinal metin: {result['original_text']}")
        print(f"Temizlenmiş metin: {result['cleaned_text']}")
        print(f"Cümle sayısı: {len(result['nltk_sentences'])}")
        print(f"Token sayısı: {len(result['nltk_tokens'])}")
        print(f"Filtrelenmiş token sayısı: {len(result['filtered_tokens'])}")
        print(f"Filtrelenmiş tokenler: {result['filtered_tokens'][:10]}")  # İlk 10 token
        
        if result['spacy_results']:
            print(f"SpaCy entities: {result['spacy_results']['entities']}")
            print(f"POS tags (ilk 5): {result['spacy_results']['pos_tags'][:5]}")

if __name__ == "__main__":
    import pandas as pd
    demonstrate_text_processing()