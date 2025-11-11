"""
FastAPI Test Scripti
Bu script API servislerini test eder
"""

import requests
import json
import time

class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def test_health_check(self):
        """Servis sağlık kontrolü"""
        print("=" * 60)
        print("SAĞLIK KONTROLÜ")
        print("=" * 60)
        
        try:
            response = requests.get(f"{self.base_url}/health")
            result = response.json()
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return response.status_code == 200
        except Exception as e:
            print(f"Sağlık kontrolü hatası: {e}")
            return False
    
    def test_clean_text(self):
        """Metin temizleme testi"""
        print("\n" + "=" * 60)
        print("METİN TEMİZLEME TEST")
        print("=" * 60)
        
        test_data = {
            "text": "Dr. Ahmet YILMAZ hastayı muayene etti!!! E-mail: ahmet@hastane.gov.tr",
            "language": "tr"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/clean-text",
                json=test_data
            )
            result = response.json()
            print(f"Status: {response.status_code}")
            print(f"Orijinal: {test_data['text']}")
            print(f"Temizlenmiş: {result.get('cleaned_text', 'N/A')}")
            return response.status_code == 200
        except Exception as e:
            print(f"Metin temizleme test hatası: {e}")
            return False
    
    def test_tfidf_analysis(self):
        """TF-IDF analizi testi"""
        print("\n" + "=" * 60)
        print("TF-IDF ANALİZİ TEST")
        print("=" * 60)
        
        test_data = {
            "texts": [
                "Hasta hipertansiyon nedeniyle tedavi görüyor",
                "Diyabet hastası kan şeker kontrolünde",
                "Astım krizi geçiren hasta bronkodilatör aldı",
                "Depresyon tanısı olan hasta psikiyatrik destek alıyor"
            ],
            "max_features": 100,
            "ngram_range": [1, 2]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/tfidf-analysis",
                json=test_data
            )
            result = response.json()
            print(f"Status: {response.status_code}")
            print(f"Matrix shape: {result.get('matrix_shape', 'N/A')}")
            print("En önemli terimler:")
            for i, term in enumerate(result.get('top_terms', [])[:5], 1):
                print(f"  {i}. {term['term']}: {term['score']:.4f}")
            return response.status_code == 200
        except Exception as e:
            print(f"TF-IDF analizi test hatası: {e}")
            return False
    
    def test_similarity_analysis(self):
        """Benzerlik analizi testi"""
        print("\n" + "=" * 60)
        print("BENZERLİK ANALİZİ TEST")
        print("=" * 60)
        
        test_data = {
            "query_text": "Hasta hipertansiyon tedavisi alıyor",
            "corpus_texts": [
                "Diyabet hastası insulin kullanıyor",
                "Kan basıncı yüksek hasta ilaç alıyor",
                "Astım krizi geçiren çocuk hasta",
                "Hipertansiyon için tedavi başlandı"
            ],
            "method": "tfidf"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/similarity",
                json=test_data
            )
            result = response.json()
            print(f"Status: {response.status_code}")
            print(f"Query: {test_data['query_text']}")
            print("En benzer metinler:")
            for i, res in enumerate(result.get('results', [])[:3], 1):
                print(f"  {i}. {res['text']} (Benzerlik: {res['similarity']:.3f})")
            return response.status_code == 200
        except Exception as e:
            print(f"Benzerlik analizi test hatası: {e}")
            return False
    
    def test_pii_masking(self):
        """PII maskeleme testi"""
        print("\n" + "=" * 60)
        print("PII MASKELEME TEST")
        print("=" * 60)
        
        test_data = {
            "text": "Dr. Mehmet Özkan (TC: 12345678901) hastayı muayene etti. Tel: 0532-123-4567, E-mail: doktor@hastane.com",
            "method": "regex"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/mask-pii",
                json=test_data
            )
            result = response.json()
            print(f"Status: {response.status_code}")
            print(f"Orijinal: {test_data['text']}")
            print(f"Maskelenmiş: {result.get('masked_text', 'N/A')}")
            return response.status_code == 200
        except Exception as e:
            print(f"PII maskeleme test hatası: {e}")
            return False
    
    def test_health_data_analysis(self):
        """Sağlık verisi analizi testi"""
        print("\n" + "=" * 60)
        print("SAĞLIK VERİSİ ANALİZİ TEST")
        print("=" * 60)
        
        test_data = {
            "texts": [
                "Hasta hipertansiyon nedeniyle acil servise başvurdu. Kan basıncı 180/110 mmHg.",
                "Diyabet tip 2 hastası kan şeker kontrolünde. HbA1c değeri yüksek.",
                "Astım krizi geçiren 8 yaşında çocuk hasta bronkodilatör tedavi aldı.",
                "Depresyon tanısı alan hasta antidepresan başlandı psikiyatrik takip.",
                "Migren ağrıları olan hasta ağrı kesici verildi. Tetikleyici faktörler araştırılıyor.",
                "Kalp hastalığı bulunan yaşlı hasta kardiyoloji poliklinik kontrolünde."
            ],
            "language": "tr"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/health-data-analysis",
                json=test_data
            )
            result = response.json()
            print(f"Status: {response.status_code}")
            
            summary = result.get('analysis_summary', {})
            print(f"Toplam metin sayısı: {summary.get('total_texts', 'N/A')}")
            
            print("En önemli terimler:")
            for i, term in enumerate(summary.get('top_terms', [])[:5], 1):
                print(f"  {i}. {term['term']}: {term['score']:.4f}")
            
            print("Küme bilgileri:")
            clusters = summary.get('clusters', {})
            for cluster_id, docs in clusters.items():
                print(f"  Küme {cluster_id}: {len(docs)} doküman")
            
            return response.status_code == 200
        except Exception as e:
            print(f"Sağlık verisi analizi test hatası: {e}")
            return False
    
    def run_all_tests(self):
        """Tüm testleri çalıştır"""
        print("API TEST BAŞLATILIJYOR...")
        print(f"Base URL: {self.base_url}")
        
        tests = [
            ("Sağlık Kontrolü", self.test_health_check),
            ("Metin Temizleme", self.test_clean_text),
            ("TF-IDF Analizi", self.test_tfidf_analysis),
            ("Benzerlik Analizi", self.test_similarity_analysis),
            ("PII Maskeleme", self.test_pii_masking),
            ("Sağlık Verisi Analizi", self.test_health_data_analysis)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n{test_name} testi çalıştırılıyor...")
            try:
                success = test_func()
                results[test_name] = "✓ BAŞARILI" if success else "✗ BAŞARISIZ"
            except Exception as e:
                results[test_name] = f"✗ HATA: {e}"
            
            time.sleep(1)  # API'ye yük binmemesi için
        
        # Sonuç özeti
        print("\n" + "=" * 80)
        print("TEST SONUÇLARI")
        print("=" * 80)
        
        for test_name, result in results.items():
            print(f"{test_name:25}: {result}")
        
        successful_tests = sum(1 for r in results.values() if "✓" in r)
        total_tests = len(results)
        
        print(f"\nBaşarı oranı: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="API Test Scripti")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000", 
        help="API base URL (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    tester = APITester(base_url=args.url)
    tester.run_all_tests()