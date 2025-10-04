#!/bin/bash

# Loan Risk Analysis - Start Script
echo "🚀 Loan Risk Analysis Başlatılıyor..."

# Virtual environment kontrolü
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment oluşturuluyor..."
    python3 -m venv venv
fi

# Virtual environment aktivasyonu
echo "🔧 Virtual environment aktif ediliyor..."
source venv/bin/activate

# Dependencies kurulumu
echo "📚 Dependencies kuruluyor..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Models klasörü kontrolü
if [ ! -d "models" ]; then
    echo "📁 Models klasörü oluşturuluyor..."
    mkdir -p models
fi

# Data klasörü kontrolü  
if [ ! -d "data" ]; then
    echo "📁 Data klasörü oluşturuluyor..."
    mkdir -p data
fi

echo "✅ Kurulum tamamlandı!"
echo ""
echo "🎯 Kullanım adımları:"
echo "1. EDA için:"
echo "   - Script ile: cd src && python eda.py"
echo "   - Notebook ile: jupyter notebook notebooks/01_eda.ipynb"
echo "2. Preprocessing için: cd src && python preprocessing.py"
echo "3. Modelleri eğitmek için: cd src && python models.py"
echo "4. Streamlit uygulamasını çalıştırmak için: streamlit run streamlit_app/app.py"
echo ""
echo "🌐 Render/Railway deployment için:"
echo "   - requirements.txt dosyası hazır"
echo "   - Start command: streamlit run streamlit_app/app.py --server.port=\$PORT --server.address=0.0.0.0"