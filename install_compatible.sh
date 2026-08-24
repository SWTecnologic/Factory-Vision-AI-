#!/bin/bash

echo "🔧 Instalando versões compatíveis do PyTorch e Ultralytics..."
echo "=================================================="

# Desinstala versões atuais
echo "📦 Removendo versões antigas..."
pip uninstall -y ultralytics torch torchvision torchaudio

# Instala PyTorch 2.0.1 (compatível)
echo "📦 Instalando PyTorch 2.0.1..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Instala Ultralytics compatível
echo "📦 Instalando Ultralytics 8.0.196..."
pip install ultralytics==8.0.196

# Instala outras dependências
echo "📦 Instalando outras dependências..."
pip install opencv-python-headless numpy requests python-dotenv

# Verifica instalação
echo ""
echo "✅ Verificando instalação:"
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'✅ Ultralytics: {ultralytics.__version__}')"
python -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')"

echo ""
echo "=================================================="
echo "✅ Instalação concluída!"
echo ""
echo "Execute: python main_fixed.py"
