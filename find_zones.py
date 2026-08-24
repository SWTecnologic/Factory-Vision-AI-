# find_zones.py
import cv2
import numpy as np

# Carrega o vídeo
cap = cv2.VideoCapture('video_teste.mp4')
ret, frame = cap.read()
if not ret:
    print("Erro ao carregar vídeo")
    exit()

height, width = frame.shape[:2]
frame_copy = frame.copy()

# Pontos para marcar as zonas
points = []
current_zone = 0
zone_names = ["COLETA", "COSTURA", "EMBALAGEM", "PERIGO"]

def mouse_callback(event, x, y, flags, param):
    global points, frame_copy, current_zone
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Marca o ponto
        points.append((x, y))
        cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame_copy, f"P{len(points)}", (x+5, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if len(points) == 2:
            # Desenha o retângulo
            x1, y1 = points[0]
            x2, y2 = points[1]
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Mostra coordenadas normalizadas
            norm_x1 = x1/width
            norm_y1 = y1/height
            norm_x2 = x2/width
            norm_y2 = y2/height
            print(f"\n📍 ZONA {zone_names[current_zone]}:")
            print(f"   ({norm_x1:.2f}, {norm_y1:.2f}, {norm_x2:.2f}, {norm_y2:.2f})")
            
            # Reseta para próxima zona
            points = []
            current_zone += 1
            cv2.putText(frame_copy, f"{zone_names[current_zone-1]} CONFIGURADA", 
                        (10, 30 + current_zone*30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)

# Mostra instruções
print("🎯 INSTRUÇÕES:")
print("1. Clique no CANTO SUPERIOR ESQUERDO da zona")
print("2. Clique no CANTO INFERIOR DIREITO da zona")
print("3. Ordens: COLETA → COSTURA → EMBALAGEM → PERIGO")
print("\nPressione ESC para sair\n")

cv2.imshow("Ajuste as Zonas - Clique 2 pontos por zona", frame_copy)
cv2.setMouseCallback("Ajuste as Zonas - Clique 2 pontos por zona", mouse_callback)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()