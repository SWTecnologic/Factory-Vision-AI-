# test_zones.py
import cv2

# Carrega o vídeo para mostrar as coordenadas
cap = cv2.VideoCapture('video_teste.mp4')
ret, frame = cap.read()
if not ret:
    print("Erro ao carregar vídeo")
    exit()

height, width = frame.shape[:2]

print("=" * 60)
print("📐 ZONAS SUGERIDAS PARA SUA ESTAÇÃO")
print("=" * 60)
print("\n📋 COPIE E COLE ISSO NO main.py:\n")

ZONES = {
    "collection": (0.00, 0.25, 0.30, 0.95),
    "sewing": (0.30, 0.20, 0.70, 0.95),
    "packaging": (0.70, 0.20, 1.00, 0.95),
    "needle_hazard": (0.45, 0.45, 0.55, 0.70),
}

print("ZONES = {")
for name, coords in ZONES.items():
    print(f'    "{name}": {coords},')
print("}")

print("\n" + "=" * 60)
print("✅ Zonas atualizadas!")
print("=" * 60)

# Mostra as zonas na imagem (salva em arquivo)
for name, coords in ZONES.items():
    x1 = int(coords[0] * width)
    y1 = int(coords[1] * height)
    x2 = int(coords[2] * width)
    y2 = int(coords[3] * height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(frame, name.upper(), (x1+5, y1+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Salva a imagem com as zonas
cv2.imwrite('zonas_visuais.jpg', frame)
print("\n📸 Imagem com as zonas salva como 'zonas_visuais.jpg'")