#!/usr/bin/env python3
"""
Hafta 7 Örneklerini Çalıştırma Scripti
Bu script tüm örnekleri sırasıyla çalıştırarak demo yapar
"""

import subprocess
import sys
import time
import os

def run_script(script_path, timeout=60):
    """Script'i çalıştır ve sonucu göster"""
    print(f"\n{'='*80}")
    print(f"ÇALIŞTIRILACAK: {script_path}")
    print(f"{'='*80}")
    
    try:
        # Virtual environment'ı aktif et ve script'i çalıştır
        cmd = f"source venv/bin/activate && python {script_path}"
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Timeout ile bekle
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode == 0:
            print("✅ BAŞARILI")
            print("\nÇıktı:")
            print(stdout[:1000] + "..." if len(stdout) > 1000 else stdout)
        else:
            print("❌ HATA")
            print("\nHata mesajı:")
            print(stderr[:1000] + "..." if len(stderr) > 1000 else stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (İşlem çok uzun sürdü)")
        process.kill()
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
    
    time.sleep(2)  # Scriptler arası bekleme

def main():
    """Ana fonksiyon"""
    print("🚀 HAFTA 7 - METİN İŞLEME VE NLP OPTİMİZASYONU")
    print("📁 Proje Örnekleri Demo Başlıyor...")
    
    # Mevcut dizini kontrol et
    if not os.path.exists("venv"):
        print("❌ Virtual environment bulunamadı!")
        print("Lütfen önce 'python -m venv venv && source venv/bin/activate && pip install -r requirements.txt' komutunu çalıştırın.")
        return
    
    # Çalıştırılacak scriptler
    examples = [
        ("📊 Veri Üretimi", "data/synthetic_health_data.py"),
        ("🧹 Metin Temizleme", "examples/01_text_cleaning_tokenization.py"),
        ("📈 TF-IDF Analizi", "examples/02_tfidf_analysis.py"),
        ("🔐 PII Maskeleme", "src/pii_masking.py"),
        # BERT örnekleri çok yavaş olduğu için atlıyoruz
        # ("🤖 BERT Analizi", "examples/03_bert_analysis.py"),
        # ("⚡ Model Optimizasyonu", "examples/04_model_optimization.py"),
    ]
    
    success_count = 0
    total_count = len(examples)
    
    for name, script_path in examples:
        print(f"\n🔄 {name} çalıştırılıyor...")
        
        if os.path.exists(script_path):
            run_script(script_path, timeout=30)
            success_count += 1
        else:
            print(f"⚠️  Dosya bulunamadı: {script_path}")
    
    # Özet
    print(f"\n{'='*80}")
    print("📋 ÖZET")
    print(f"{'='*80}")
    print(f"Toplam script: {total_count}")
    print(f"Çalıştırılan: {success_count}")
    print(f"Başarı oranı: {success_count/total_count*100:.1f}%")
    
    # API testi önerisi
    print(f"\n🌐 API SERVİSİ TESTİ:")
    print("1. Terminal'de: python api/main.py")
    print("2. Başka terminal'de: python api/test_api.py")
    
    # Manuel testler
    print(f"\n📝 MANUEL TESTLER:")
    print("• BERT Analizi: python examples/03_bert_analysis.py")
    print("• Model Optimizasyonu: python examples/04_model_optimization.py")
    print("• Sentetik veri görüntüleme: head -5 data/synthetic_health_data.csv")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Kullanıcı tarafından durduruldu!")
    except Exception as e:
        print(f"\n\n💥 Beklenmeyen hata: {e}")
    finally:
        print("\n👋 Demo tamamlandı!")