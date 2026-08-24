#!/bin/bash
# setup.sh - Instalação completa do Factory Vision AI

echo "🚀 Iniciando instalação do Factory Vision AI"
echo "================================================"

# Detecta o sistema operacional
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_ID
else
    echo "⚠️  Não foi possível detectar o sistema operacional"
    OS="unknown"
fi

echo "📋 Sistema detectado: $OS $VERSION"

# Instala dependências do sistema
echo ""
echo "📦 Instalando dependências do sistema..."

if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    # Para Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install -y \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libx11-6 \
        libxcb1 \
        libxcb-xinerama0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
        libxkbcommon0 \
        libgl1-mesa-dev \
        wget \
        curl

elif [ "$OS" = "rhel" ] || [ "$OS" = "fedora" ] || [ "$OS" = "centos" ]; then
    # Para RHEL/Fedora/CentOS
    sudo yum install -y \
        mesa-libGL \
        glibc \
        libX11 \
        libXext \
        libXrender \
        libgomp \
        wget \
        curl
else
    echo "⚠️  Sistema não reconhecido. Tentando instalar pacotes básicos..."
    # Tenta com apt (fallback)
    sudo apt-get update || true
    sudo apt-get install -y libgl1 libglib2.0-0 || true
fi

# Instala dependências Python
echo ""
echo "📦 Instalando dependências Python..."

# Cria ambiente virtual (opcional)
# python -m venv venv
# source venv/bin/activate

# Atualiza pip
pip install --upgrade pip

# Instala dependências
pip install -r requirements.txt

# Cria diretório para modelos
echo ""
echo "📁 Criando diretórios..."
mkdir -p models
mkdir -p logs
mkdir -p output

echo ""
echo "================================================"
echo "✅ INSTALAÇÃO CONCLUÍDA!"
echo ""
echo "📝 Próximos passos:"
echo "  1. Coloque seus modelos .pt na pasta models/"
echo "  2. Configure o arquivo .env (já criado)"
echo "  3. Coloque um vídeo de teste na raiz (video_teste.mp4)"
echo "  4. Execute: python main.py"
echo ""
echo "🔧 Comandos úteis:"
echo "  - Usar webcam: VIDEO_SOURCE=0 python main.py"
echo "  - Usar vídeo: VIDEO_SOURCE=video.mp4 python main.py"
echo "  - Ver eventos: cat events.jsonl | jq '.'"
echo "================================================"