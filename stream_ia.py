import cv2
import time
import json
import os
import requests
import numpy as np
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# ============================================================
# CONFIGURAÇÕES DA API ROBOFLOW
# ============================================================

ROBOFLOW_API_KEY = "1cMfVGpVHccfndIOgfqX"
ROBOFLOW_MODEL_ID = "safety-glove-workstation-monitor/1"
OBJECT_CONFIDENCE = 0.45

# ============================================================
# FUNÇÃO DE DETECÇÃO
# ============================================================

def detect_objects(frame):
    """Detecta objetos usando a API do Roboflow"""
    temp_path = "temp_frame.jpg"
    cv2.imwrite(temp_path, frame)
    
    try:
        url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}?api_key={ROBOFLOW_API_KEY}&confidence={int(OBJECT_CONFIDENCE*100)}"
        
        with open(temp_path, "rb") as f:
            response = requests.post(url, files={"file": f})
        
        result = response.json()
        os.remove(temp_path)
        
        detections = []
        if result and 'predictions' in result:
            for pred in result['predictions']:
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                w = pred.get('width', 0)
                h = pred.get('height', 0)
                class_name = pred.get('class', 'unknown').lower()
                confidence = pred.get('confidence', 0)
                
                if confidence >= OBJECT_CONFIDENCE:
                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    x2 = int(x + w/2)
                    y2 = int(y + h/2)
                    
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': (x1, y1, x2, y2)
                    })
        
        return detections
        
    except Exception as e:
        print(f"⚠️ Erro API: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return []

# ============================================================
# CORES DAS CLASSES
# ============================================================

COLORS = {
    'hand': (255, 80, 80),      # Vermelho
    'glove': (0, 220, 255),     # Amarelo
    'plastic_bag': (255, 0, 255), # Magenta
}

DEFAULT_COLOR = (180, 180, 180)

# ============================================================
# ZONAS
# ============================================================

ZONES = {
    "collection": (0.00, 0.25, 0.30, 0.95),
    "sewing": (0.30, 0.20, 0.70, 0.95),
    "packaging": (0.70, 0.20, 1.00, 0.95),
    "needle_hazard": (0.45, 0.45, 0.55, 0.70),
}

STAGE_COLORS = {1: (255, 170, 0), 2: (0, 220, 255), 3: (0, 210, 0)}

def zone_to_pixels(zone, width, height):
    x1, y1, x2, y2 = zone
    return (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))

def draw_zones(frame):
    height, width = frame.shape[:2]
    
    for name, zone in ZONES.items():
        x1, y1, x2, y2 = zone_to_pixels(zone, width, height)
        
        if name == "needle_hazard":
            color = (0, 0, 255)
            label = "⚠️ PERIGO"
            thickness = 3
        else:
            stage_num = 1 if name == "collection" else 2 if name == "sewing" else 3
            color = STAGE_COLORS.get(stage_num, (255, 255, 255))
            label = name.upper()
            thickness = 2
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1 + 5, max(22, y1 + 22)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        
        color = COLORS.get(class_name, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 6)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

# ============================================================
# STREAMING
# ============================================================

video_source = 'video_teste.mp4'
cap = cv2.VideoCapture(video_source)

def generate_frames():
    frame_count = 0
    skip_frames = 2
    
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        if frame_count % skip_frames == 0:
            detections = detect_objects(frame)
        else:
            detections = []
        
        draw_zones(frame)
        draw_detections(frame, detections)
        
        cv2.putText(frame, f"Factory Vision AI - Detecções: {len(detections)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Factory Vision AI - Monitor</title>
        <style>
            body { background: #0d1117; color: white; font-family: Arial; text-align: center; padding: 20px; }
            img { border: 2px solid #30363d; border-radius: 8px; max-width: 100%; }
            .info { margin-top: 20px; padding: 15px; background: #161b22; border-radius: 8px; }
            .online { color: #2ea043; }
        </style>
    </head>
    <body>
        <h1>🏭 Factory Vision AI - Monitor Industrial</h1>
        <div class="info">
            <span class="online">🟢 AO VIVO</span> | IA em tempo real
        </div>
        <img src="/video_feed" width="900">
        <p style="color: #8b949e; font-size: 12px;">Clique no botão "Ports" no Codespace para ver a URL</p>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 SERVIDOR COM IA INICIADO!")
    print("📹 Vídeo: video_teste.mp4")
    print("🤖 Detecções: Mãos, Luvas e Sacos Plásticos")
    print("=" * 60)
    print("\n🌐 Para ver o vídeo com IA:")
    print("  1. Clique na aba 'PORTS'")
    print("  2. Porta 5000 -> Clique no 🌐")
    print("\n🛑 Pressione Ctrl+C para parar")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
