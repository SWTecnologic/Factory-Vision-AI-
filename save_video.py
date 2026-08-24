# save_video.py
import cv2
import time

# Configurações
video_source = 'video_teste.mp4'
output_file = 'video_processado.mp4'

cap = cv2.VideoCapture(video_source)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Configura o writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

print("🎬 Processando vídeo...")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Desenha informações no frame
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out.write(frame)
    frame_count += 1
    
    if frame_count % 50 == 0:
        print(f"📊 Processado: {frame_count} frames")

cap.release()
out.release()

print(f"✅ Vídeo salvo em: {output_file}")
print(f"📊 Total de frames: {frame_count}")