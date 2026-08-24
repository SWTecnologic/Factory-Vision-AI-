import cv2
import numpy as np
import os

# Carrega o vídeo
cap = cv2.VideoCapture('video_teste.mp4')
ret, frame = cap.read()
if not ret:
    print("❌ Erro ao carregar vídeo")
    exit()

height, width = frame.shape[:2]

print("=" * 60)
print("🎯 AJUSTE DE ZONAS - MODO SEM GUI")
print("=" * 60)
print("\n📋 ZONAS ATUAIS:")
print("  collection: (0.00, 0.25, 0.30, 0.95)")
print("  sewing:     (0.30, 0.20, 0.70, 0.95)")
print("  packaging:  (0.70, 0.20, 1.00, 0.95)")
print("  needle_hazard: (0.45, 0.45, 0.55, 0.70)")
print("\n🔧 Para ajustar, edite as coordenadas no main.py")
print("   Os valores vão de 0 a 1 (0% a 100% da imagem)")
print("\n📐 EXEMPLO DE AJUSTE:")
print("   Se a agulha está mais à esquerda, mude:")
print("   'needle_hazard': (0.35, 0.45, 0.45, 0.70)")
print("\n   Se a zona de coleta é maior, mude:")
print("   'collection': (0.00, 0.20, 0.35, 1.00)")

# Desenha as zonas atuais na imagem
ZONES = {
    "collection": (0.00, 0.25, 0.30, 0.95),
    "sewing": (0.30, 0.20, 0.70, 0.95),
    "packaging": (0.70, 0.20, 1.00, 0.95),
    "needle_hazard": (0.45, 0.45, 0.55, 0.70),
}

colors = {
    "collection": (255, 170, 0),   # Laranja
    "sewing": (0, 220, 255),       # Ciano
    "packaging": (0, 210, 0),      # Verde
    "needle_hazard": (0, 0, 255),  # Vermelho
}

for name, zone in ZONES.items():
    x1 = int(zone[0] * width)
    y1 = int(zone[1] * height)
    x2 = int(zone[2] * width)
    y2 = int(zone[3] * height)
    
    color = colors.get(name, (255, 255, 255))
    
    # Overlay transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    
    # Borda
    thickness = 3 if name == "needle_hazard" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Label
    label = "⚠️ PERIGO" if name == "needle_hazard" else name.upper()
    cv2.putText(frame, label, (x1 + 5, y1 + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Salva a imagem
output_file = "zonas_atuais.jpg"
cv2.imwrite(output_file, frame)
print(f"\n📸 Imagem com zonas salva em: {output_file}")
print(f"\n📂 No Codespace, clique com botão direito no arquivo e baixe")
print("   Ou abra no navegador para ver as zonas")

print("\n" + "=" * 60)
print("📝 COMO AJUSTAR AS COORDENADAS:")
print("=" * 60)
print("""
1. Baixe a imagem 'zonas_atuais.jpg'
2. Veja onde cada zona está desenhada
3. Abra o main.py e ajuste os valores em ZONES
4. Rode o sistema novamente

📐 FORMATO: (x1, y1, x2, y2)
   x1 = canto superior esquerdo (horizontal)
   y1 = canto superior esquerdo (vertical)
   x2 = canto inferior direito (horizontal)
   y2 = canto inferior direito (vertical)

   Valores de 0 a 1 (0% a 100%)
""")

# Salva também um arquivo de texto com as coordenadas
with open("coordenadas_zonas.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("COORDENADAS DAS ZONAS\n")
    f.write("=" * 60 + "\n\n")
    for name, zone in ZONES.items():
        f.write(f"{name}: {zone}\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write("Para ajustar, edite estes valores no main.py\n")
    f.write("=" * 60 + "\n")

print("✅ Arquivo 'coordenadas_zonas.txt' criado!")
print("\n🔄 Para testar novos valores, edite o main.py e rode:")
print("   python main.py")

