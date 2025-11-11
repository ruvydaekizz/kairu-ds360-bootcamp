import pandas as pd
import random
from faker import Faker
import numpy as np
from datetime import datetime, timedelta

fake = Faker('tr_TR')  # Turkish locale

def generate_synthetic_health_data(n_records=1000):
    """Generate synthetic health data with PII information"""
    
    data = []
    
    for i in range(n_records):
        # Generate patient data
        first_name = fake.first_name()
        last_name = fake.last_name()
        
        record = {
            'hasta_id': f"H{i+1:06d}",
            'tc_kimlik': fake.ssn(),
            'ad': first_name,
            'soyad': last_name,
            'dogum_tarihi': fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d'),
            'telefon': fake.phone_number(),
            'email': fake.email(),
            'adres': fake.address().replace('\n', ', '),
            'cinsiyet': random.choice(['E', 'K']),
            'kan_grubu': random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', '0+', '0-']),
            'boy': random.randint(150, 200),
            'kilo': random.randint(45, 120),
            'tani': random.choice([
                'Hipertansiyon', 'Diyabet Tip 2', 'Astım', 'Depresyon', 
                'Migren', 'Artrit', 'Kalp hastalığı', 'Obezite'
            ]),
            'doktor_adi': fake.name(),
            'hastane': random.choice([
                'Ankara Şehir Hastanesi', 'İstanbul Medeniyet Üniversitesi Hastanesi',
                'Hacettepe Üniversitesi Hastanesi', 'Gazi Üniversitesi Hastanesi'
            ]),
            'muayene_tarihi': fake.date_between(start_date='-2y', end_date='today').strftime('%Y-%m-%d'),
            'sigorta_no': f"SIG{random.randint(100000, 999999)}",
            'notlar': f"Hasta {first_name} {last_name} için özel notlar ve tedavi planı"
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate data
    df = generate_synthetic_health_data(500)
    
    # Save to CSV
    output_file = 'synthetic_health_data.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Synthetic health data generated and saved to {output_file}")
    print(f"Generated {len(df)} records")
    print("\nSample data:")
    print(df.head(3).to_string())