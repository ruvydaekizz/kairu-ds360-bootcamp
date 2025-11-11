"""
PII (Personally Identifiable Information) Maskeleme Scripti
Bu script sağlık verilerindeki kişisel bilgileri maskeleyerek gizliliği korur
"""

import pandas as pd
import re
import hashlib
from faker import Faker
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import warnings
warnings.filterwarnings('ignore')

class PIIMasker:
    def __init__(self):
        """PII Maskeleme sınıfı"""
        self.fake = Faker('tr_TR')
        self.analyzer = None
        self.anonymizer = None
        self.init_presidio()
        
    def init_presidio(self):
        """Presidio kütüphanelerini başlat"""
        try:
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            
            # NLP engine konfigürasyonu
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "en", "model_name": "en_core_web_sm"},
                    {"lang_code": "tr", "model_name": "en_core_web_sm"}  # Türkçe için İngilizce model kullan
                ]
            }
            
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()
            
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            self.anonymizer = AnonymizerEngine()
            print("Presidio başarıyla başlatıldı!")
            return True
        except Exception as e:
            # Basit konfigürasyonla dene
            try:
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                print("Presidio basit konfigürasyonla başlatıldı!")
                return True
            except Exception as e2:
                print(f"Presidio başlatılamadı: {e2}")
                return False
    
    def mask_with_regex(self, text):
        """Regex ile temel PII maskeleme"""
        if pd.isna(text):
            return text
        
        text = str(text)
        
        # TC Kimlik No (11 haneli sayılar)
        text = re.sub(r'\b\d{11}\b', '[TC_KIMLIK]', text)
        
        # Telefon numaraları
        text = re.sub(r'\b0?\d{3}[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}\b', '[TELEFON]', text)
        text = re.sub(r'\b0?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[TELEFON]', text)
        
        # E-mail adresleri
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # IBAN (TR ile başlayan)
        text = re.sub(r'TR\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{2}', '[IBAN]', text)
        
        # Kredi kartı numaraları (16 haneli)
        text = re.sub(r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b', '[KART_NO]', text)
        
        # İsimler (Dr. + isim formatı)
        text = re.sub(r'\bDr\.?\s+[A-ZÜĞIÖŞÇI][a-züğıöşçi]+\s+[A-ZÜĞIÖŞÇI][a-züğıöşçi]+\b', '[DOKTOR_ADI]', text)
        
        # Hasta + isim formatı
        text = re.sub(r'\bHasta\s+[A-ZÜĞIÖŞÇI][a-züğıöşçi]+\s+[A-ZÜĞIÖŞÇI][a-züğıöşçi]+\b', 'Hasta [HASTA_ADI]', text)
        
        # Genel isim pattern (büyük harfle başlayan kelimeler)
        text = re.sub(r'\b[A-ZÜĞIÖŞÇI][a-züğıöşçi]{2,}\s+[A-ZÜĞIÖŞÇI][a-züğıöşçi]{2,}\b', '[AD_SOYAD]', text)
        
        # Adres bilgileri (yaygın sokak/cadde isimleri)
        address_patterns = [
            r'\b\d+\.\s*sokak\b',
            r'\b\d+\.\s*cadde\b',
            r'\b\w+\s+mahalle\w*\b',
            r'\bno:\s*\d+\b',
            r'\bdaire\s*\d+\b'
        ]
        for pattern in address_patterns:
            text = re.sub(pattern, '[ADRES]', text, flags=re.IGNORECASE)
        
        return text
    
    def mask_with_presidio(self, text):
        """Presidio ile gelişmiş PII maskeleme"""
        if not self.analyzer or pd.isna(text):
            return text
        
        text = str(text)
        
        try:
            # Önce English ile dene
            results_en = self.analyzer.analyze(
                text=text,
                language='en',
                entities=['PERSON', 'PHONE_NUMBER', 'EMAIL_ADDRESS', 'CREDIT_CARD']
            )
            
            # Türkçe karakterli metinler için özel işlem
            results_custom = []
            
            # Email adresleri için özel pattern
            email_matches = re.finditer(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            for match in email_matches:
                from presidio_analyzer import RecognizerResult
                results_custom.append(RecognizerResult(
                    entity_type="EMAIL_ADDRESS",
                    start=match.start(),
                    end=match.end(),
                    score=0.9
                ))
            
            # Telefon numaraları için özel pattern
            phone_matches = re.finditer(r'\b0?\d{3}[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}\b', text)
            for match in phone_matches:
                from presidio_analyzer import RecognizerResult
                results_custom.append(RecognizerResult(
                    entity_type="PHONE_NUMBER",
                    start=match.start(),
                    end=match.end(),
                    score=0.9
                ))
            
            # Tüm sonuçları birleştir
            all_results = results_en + results_custom
            
            if all_results:
                anonymized_result = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=all_results
                )
                return anonymized_result.text
            else:
                return self.mask_with_regex(text)
            
        except Exception as e:
            return self.mask_with_regex(text)
    
    def generate_fake_data(self, original_data, column_mappings):
        """Orijinal veriler yerine sahte veriler üret"""
        fake_data = original_data.copy()
        
        for column, data_type in column_mappings.items():
            if column not in fake_data.columns:
                continue
                
            if data_type == 'name':
                fake_data[column] = [self.fake.first_name() for _ in range(len(fake_data))]
            elif data_type == 'surname':
                fake_data[column] = [self.fake.last_name() for _ in range(len(fake_data))]
            elif data_type == 'email':
                fake_data[column] = [self.fake.email() for _ in range(len(fake_data))]
            elif data_type == 'phone':
                fake_data[column] = [self.fake.phone_number() for _ in range(len(fake_data))]
            elif data_type == 'address':
                fake_data[column] = [self.fake.address().replace('\n', ', ') for _ in range(len(fake_data))]
            elif data_type == 'ssn':
                fake_data[column] = [self.fake.ssn() for _ in range(len(fake_data))]
            elif data_type == 'hash':
                # Orijinal veriyi hash'le (tutarlılık için)
                fake_data[column] = fake_data[column].apply(self.hash_data)
        
        return fake_data
    
    def hash_data(self, data):
        """Veriyi hash'leyerek maskele"""
        if pd.isna(data):
            return data
        
        # SHA-256 hash
        hash_object = hashlib.sha256(str(data).encode())
        return hash_object.hexdigest()[:16]  # İlk 16 karakter
    
    def mask_dataframe(self, df, text_columns, pii_columns, method='regex'):
        """DataFrame'deki PII'ları maskele"""
        masked_df = df.copy()
        
        # Metin sütunlarındaki PII'ları maskele
        for column in text_columns:
            if column in masked_df.columns:
                print(f"Maskeleniyor: {column}")
                if method == 'presidio' and self.analyzer:
                    masked_df[column] = masked_df[column].apply(self.mask_with_presidio)
                else:
                    masked_df[column] = masked_df[column].apply(self.mask_with_regex)
        
        # PII sütunlarını fake data ile değiştir
        if pii_columns:
            fake_df = self.generate_fake_data(masked_df, pii_columns)
            for column in pii_columns.keys():
                if column in masked_df.columns:
                    masked_df[column] = fake_df[column]
        
        return masked_df
    
    def anonymize_dataset(self, df, config):
        """Veri setini tamamen anonimleştir"""
        anonymized_df = df.copy()
        
        # ID sütunlarını değiştir
        if 'id_columns' in config:
            for id_col in config['id_columns']:
                if id_col in anonymized_df.columns:
                    anonymized_df[id_col] = [f"ID_{i:06d}" for i in range(len(anonymized_df))]
        
        # Sensitive sütunları kaldır veya değiştir
        if 'remove_columns' in config:
            anonymized_df = anonymized_df.drop(columns=config['remove_columns'], errors='ignore')
        
        # Kategorik verileri genelleştir
        if 'generalize_columns' in config:
            for col, mapping in config['generalize_columns'].items():
                if col in anonymized_df.columns:
                    anonymized_df[col] = anonymized_df[col].map(mapping).fillna('Diğer')
        
        return anonymized_df

def demonstrate_pii_masking():
    """PII maskeleme demonstrasyonu"""
    print("=" * 80)
    print("PII MASKELEME DEMONSTRASİYONU")
    print("=" * 80)
    
    # Sentetik sağlık verilerini yükle
    try:
        df = pd.read_csv('../data/synthetic_health_data.csv')
        print(f"Veri seti yüklendi: {len(df)} kayıt")
    except FileNotFoundError:
        try:
            df = pd.read_csv('data/synthetic_health_data.csv')
            print(f"Veri seti yüklendi: {len(df)} kayıt")
        except FileNotFoundError:
            print("Sağlık verisi bulunamadı.")
            return
    
    # PII maskeleme sınıfını oluştur
    masker = PIIMasker()
    
    print("\n--- ORİJİNAL VERİ (İlk 3 kayıt) ---")
    print(df[['ad', 'soyad', 'tc_kimlik', 'telefon', 'email', 'notlar']].head(3))
    
    # Maskeleme konfigürasyonu
    text_columns = ['notlar', 'adres']  # Metin içindeki PII'ları maskele
    pii_columns = {
        'ad': 'name',
        'soyad': 'surname', 
        'tc_kimlik': 'hash',
        'telefon': 'phone',
        'email': 'email',
        'adres': 'address'
    }
    
    # Regex ile maskeleme
    print("\n--- REGEX İLE MASKELENMİŞ VERİ ---")
    masked_regex = masker.mask_dataframe(df, text_columns, pii_columns, method='regex')
    print(masked_regex[['ad', 'soyad', 'tc_kimlik', 'telefon', 'email', 'notlar']].head(3))
    
    # Presidio ile maskeleme (eğer mevcut ise)
    if masker.analyzer:
        print("\n--- PRESTİDIO İLE MASKELENMİŞ VERİ ---")
        masked_presidio = masker.mask_dataframe(df, text_columns, pii_columns, method='presidio')
        print(masked_presidio[['ad', 'soyad', 'tc_kimlik', 'telefon', 'email', 'notlar']].head(3))
    
    # Tam anonimleştirme
    print("\n--- TAM ANONİMLEŞTİRME ---")
    anonymize_config = {
        'id_columns': ['hasta_id'],
        'remove_columns': ['sigorta_no'],
        'generalize_columns': {
            'kan_grubu': {'A+': 'A', 'A-': 'A', 'B+': 'B', 'B-': 'B', 'AB+': 'AB', 'AB-': 'AB', '0+': '0', '0-': '0'},
            'cinsiyet': {'E': 'E', 'K': 'K'}
        }
    }
    
    anonymized_df = masker.anonymize_dataset(masked_regex, anonymize_config)
    print(anonymized_df[['hasta_id', 'ad', 'kan_grubu', 'tani', 'hastane']].head(3))
    
    # Maskelenmiş veriyi kaydet
    try:
        output_file = '../data/masked_health_data.csv'
        anonymized_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nMaskelenmiş veri kaydedildi: {output_file}")
    except:
        output_file = 'masked_health_data.csv'
        anonymized_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nMaskelenmiş veri kaydedildi: {output_file}")
    
    # İstatistikler
    print("\n--- MASKELEME İSTATİSTİKLERİ ---")
    print(f"Orijinal kayıt sayısı: {len(df)}")
    print(f"Maskelenmiş kayıt sayısı: {len(anonymized_df)}")
    print(f"Sütun sayısı: {len(df.columns)} -> {len(anonymized_df.columns)}")
    
    # PII korunma oranı
    original_unique_names = len(df['ad'].unique())
    masked_unique_names = len(anonymized_df['ad'].unique())
    print(f"Ad çeşitliliği korunma: {masked_unique_names/original_unique_names:.2%}")

def test_masking_methods():
    """Farklı maskeleme yöntemlerini test et"""
    print("\n" + "=" * 80)
    print("MASKELEME YÖNTEMLERİ TEST")
    print("=" * 80)
    
    test_texts = [
        "Dr. Ahmet Yılmaz (TC: 12345678901) hastayı muayene etti. Tel: 0532-123-4567",
        "Hasta e-mail: hasta@example.com, Adres: Ankara Çankaya 1. Sokak No: 15",
        "Kredi kartı: 1234 5678 9012 3456, IBAN: TR33 0006 1005 1978 6457 8413 26",
        "Dr. Mehmet Özkan'ın raporu: Hasta Ayşe Kaya için tedavi planı hazırlandı."
    ]
    
    masker = PIIMasker()
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Orijinal: {text}")
        
        # Regex maskeleme
        masked_regex = masker.mask_with_regex(text)
        print(f"Regex:    {masked_regex}")
        
        # Presidio maskeleme (eğer mevcut ise)
        if masker.analyzer:
            masked_presidio = masker.mask_with_presidio(text)
            print(f"Presidio: {masked_presidio}")

if __name__ == "__main__":
    demonstrate_pii_masking()
    test_masking_methods()