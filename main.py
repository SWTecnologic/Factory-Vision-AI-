import json
import math
import os
import queue
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Suprime avisos
warnings.filterwarnings("ignore")

# Carrega variáveis de ambiente
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Arquivo .env carregado com sucesso!")
except ImportError:
    print("⚠️ python-dotenv não instalado")

# Tenta importar cv2
try:
    import cv2
except ImportError:
    print("⚠️ cv2 não encontrado, instalando headless...")
    import subprocess
    subprocess.check_call(["pip", "install", "opencv-python-headless"])
    import cv2

import numpy as np
import requests

# ============================================================
# CONFIGURAÇÕES DA API ROBOFLOW
# ============================================================

ROBOFLOW_API_KEY = "1cMfVGpVHccfndIOgfqX"
ROBOFLOW_MODEL_ID = "safety-glove-workstation-monitor/1"

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

CAMERA_ID = os.getenv("CAMERA_ID", "posto-costura-01")
OPERATOR_ID = os.getenv("OPERATOR_ID", "operador-001")
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "video_teste.mp4")
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "models/pose_model.pt")
DEVICE = os.getenv("DEVICE", "cpu")
OBJECT_CONFIDENCE = float(os.getenv("OBJECT_CONFIDENCE", "0.45"))
POSE_CONFIDENCE = float(os.getenv("POSE_CONFIDENCE", "0.40"))
KEYPOINT_CONFIDENCE = float(os.getenv("KEYPOINT_CONFIDENCE", "0.35"))
INFERENCE_EVERY_N_FRAMES = int(os.getenv("INFERENCE_EVERY_N_FRAMES", "2"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
STAGE_DEBOUNCE_SECONDS = float(os.getenv("STAGE_DEBOUNCE_SECONDS", "0.50"))
HAZARD_ALERT_COOLDOWN_SECONDS = float(os.getenv("HAZARD_ALERT_COOLDOWN_SECONDS", "2.0"))
MAX_TRUNK_ANGLE_DEGREES = float(os.getenv("MAX_TRUNK_ANGLE_DEGREES", "20"))
POSTURE_HOLD_SECONDS = float(os.getenv("POSTURE_HOLD_SECONDS", "3"))
POSTURE_ALERT_COOLDOWN_SECONDS = float(os.getenv("POSTURE_ALERT_COOLDOWN_SECONDS", "15"))
LOCAL_EVENT_FILE = os.getenv("LOCAL_EVENT_FILE", "events.jsonl")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "vision_events")

# Controle de streaming
ENABLE_STREAMING = True
STREAM_PORT = 5000

print("=" * 60)
print("📋 CONFIGURAÇÕES CARREGADAS:")
print(f"  📹 Video Source: {VIDEO_SOURCE}")
print(f"  🆔 Camera ID: {CAMERA_ID}")
print(f"  👤 Operator ID: {OPERATOR_ID}")
print(f"  💻 Device: {DEVICE}")
print(f"  🤖 Roboflow Model: {ROBOFLOW_MODEL_ID}")
print(f"  ☁️  Supabase: {'✅ Configurado' if SUPABASE_URL and SUPABASE_KEY else '❌ Offline'}")
print(f"  🌐 Streaming: {'✅ Ativo' if ENABLE_STREAMING else '❌ Desativado'}")
print("=" * 60)

# ============================================================
# ZONAS
# ============================================================

ZONES = {
    "collection": (0.00, 0.20, 0.30, 0.95),
    "sewing": (0.30, 0.20, 0.65, 0.95),
    "packaging": (0.65, 0.20, 1.00, 0.95),
    "needle_hazard": (0.42, 0.45, 0.52, 0.70),
}

STAGE_NAMES = {0: "aguardando", 1: "coleta", 2: "costura", 3: "embalagem"}
STAGE_COLORS = {1: (255, 170, 0), 2: (0, 220, 255), 3: (0, 210, 0)}
HAND_CLASSES = {"hand", "mao", "mão"}
GLOVE_CLASSES = {"glove", "luva", "safety_glove"}
BAG_CLASSES = {"plastic_bag", "plastic bag", "bag", "saco", "saco_plastico"}

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]

# ============================================================
# CLASSES E FUNÇÕES
# ============================================================

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

Keypoints = Dict[str, Tuple[int, int, float]]

def zone_to_pixels(zone, width, height):
    x1, y1, x2, y2 = zone
    return (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))

def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def boxes_intersect(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0

def point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

def angle_from_vertical(top, bottom):
    dx = top[0] - bottom[0]
    dy = bottom[1] - top[1]
    return abs(math.degrees(math.atan2(dx, max(abs(dy), 1e-6))))

def get_keypoint(keypoints, name):
    point = keypoints.get(name)
    if point is None:
        return None
    return (point[0], point[1])

def get_detections_by_classes(detections, accepted_classes):
    return [d for d in detections if d.class_name in accepted_classes]

# ============================================================
# EVENT WRITER
# ============================================================

class EventWriter:
    def __init__(self, local_file, supabase_url, supabase_key, supabase_table):
        self.local_file = Path(local_file)
        self.local_file.parent.mkdir(parents=True, exist_ok=True)
        self.supabase_enabled = bool(supabase_url and supabase_key)
        self.supabase_endpoint = f"{supabase_url}/rest/v1/{supabase_table}" if self.supabase_enabled else ""
        self.supabase_headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        self.event_queue = queue.Queue()
        self.running = True
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def write(self, event_type, payload, severity="info"):
        local_event = {
            "event_id": str(uuid.uuid4()),
            "camera_id": CAMERA_ID,
            "operator_id": OPERATOR_ID,
            "event_type": event_type,
            "severity": severity,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        
        with self.local_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(local_event, ensure_ascii=False) + "\n")
        print(f"[EVENTO] {event_type}")
        
        if self.supabase_enabled:
            supabase_event = self._convert_to_supabase_format(event_type, payload)
            if supabase_event:
                self.event_queue.put(supabase_event)

    def _convert_to_supabase_format(self, event_type, payload):
        supabase_event = {
            "event_type": event_type,
            "event_timestamp": datetime.now(timezone.utc).isoformat(),
            "additional_data": payload,
        }
        
        if event_type == "needle_hazard":
            supabase_event["object_detected"] = "hand"
            supabase_event["confidence"] = 0.95
        elif event_type == "stage_started":
            stage_num = payload.get("stage_number", 0)
            stage_name = payload.get("stage_name", "unknown")
            supabase_event["object_detected"] = f"stage_{stage_num}_{stage_name}"
            supabase_event["confidence"] = 1.0
        elif event_type == "stage_completed":
            stage_num = payload.get("stage_number", 0)
            stage_name = payload.get("stage_name", "unknown")
            supabase_event["object_detected"] = f"stage_{stage_num}_{stage_name}_completed"
            supabase_event["confidence"] = 1.0
            supabase_event["additional_data"]["duration_seconds"] = payload.get("duration_seconds", 0)
        elif event_type == "cycle_completed":
            supabase_event["object_detected"] = "cycle_completed"
            supabase_event["confidence"] = 1.0
            supabase_event["additional_data"]["cycle_number"] = payload.get("cycle_number", 0)
            supabase_event["additional_data"]["cycle_duration"] = payload.get("cycle_duration_seconds", 0)
        elif event_type == "ergonomic_risk":
            supabase_event["object_detected"] = "bad_posture"
            supabase_event["confidence"] = 1.0
            supabase_event["additional_data"]["trunk_angle"] = payload.get("trunk_angle_degrees", 0)
            supabase_event["additional_data"]["duration_seconds"] = payload.get("duration_seconds", 0)
        elif event_type == "camera_started":
            supabase_event["object_detected"] = "system_start"
            supabase_event["confidence"] = 1.0
            supabase_event["additional_data"]["video_source"] = payload.get("video_source", "")
            supabase_event["additional_data"]["device"] = payload.get("device", "")
        elif event_type == "camera_stopped":
            supabase_event["object_detected"] = "system_stop"
            supabase_event["confidence"] = 1.0
            supabase_event["additional_data"]["processed_frames"] = payload.get("processed_frames", 0)
            supabase_event["additional_data"]["completed_cycles"] = payload.get("completed_cycles", 0)
        else:
            supabase_event["object_detected"] = event_type
            supabase_event["confidence"] = 1.0
        
        return supabase_event

    def _worker_loop(self):
        while self.running or not self.event_queue.empty():
            try:
                event = self.event_queue.get(timeout=0.5)
                try:
                    response = requests.post(
                        self.supabase_endpoint,
                        headers=self.supabase_headers,
                        json=event,
                        timeout=5,
                    )
                    response.raise_for_status()
                    print(f"✅ Evento enviado para Supabase: {event['event_type']}")
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 400:
                        print(f"⚠️ Erro 400 - Dados inválidos:")
                        print(f"   Resposta: {e.response.text}")
                    else:
                        print(f"⚠️ Supabase erro {e.response.status_code}: {e}")
                    backup_file = Path("supabase_failed_events.jsonl")
                    with backup_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "event": event,
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"⚠️ Supabase erro: {e}")
                finally:
                    self.event_queue.task_done()
            except queue.Empty:
                continue

    def close(self):
        self.running = False
        self.worker.join(timeout=3)

# ============================================================
# DETECTION - CORRIGIDO (SEM REMOVER ARQUIVO)
# ============================================================

def run_object_detection_api(frame):
    """Detecta objetos usando a API do Roboflow via requests"""
    
    temp_path = "temp_frame.jpg"
    cv2.imwrite(temp_path, frame)
    
    try:
        url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}?api_key={ROBOFLOW_API_KEY}&confidence={int(OBJECT_CONFIDENCE*100)}"
        
        with open(temp_path, "rb") as f:
            response = requests.post(url, files={"file": f})
        
        result = response.json()
        
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
                    
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox=(x1, y1, x2, y2)
                    ))
        
        # NÃO REMOVE O ARQUIVO - CORRIGIDO!
        # os.remove(temp_path)  # COMENTADO
        
        return detections
        
    except Exception as e:
        print(f"⚠️ Erro API: {e}")
        return []

def run_pose_estimation(model, frame):
    if model is None:
        return {}
    
    try:
        results = model(
            frame,
            conf=POSE_CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )
    except Exception as e:
        print(f"⚠️ Erro na inferência de pose: {e}")
        return {}

    if len(results) == 0 or results[0].keypoints is None:
        return {}

    keypoint_data = results[0].keypoints
    keypoint_xy = keypoint_data.xy.cpu().numpy()
    
    if len(keypoint_xy) == 0:
        return {}

    if keypoint_data.conf is not None and len(keypoint_data.conf) > 0:
        keypoint_confidence = keypoint_data.conf.cpu().numpy()
    else:
        keypoint_confidence = np.ones(keypoint_xy.shape[:2], dtype=np.float32)

    person_scores = keypoint_confidence.mean(axis=1)
    person_index = int(np.argmax(person_scores))

    person_xy = keypoint_xy[person_index]
    person_confidence = keypoint_confidence[person_index]

    keypoints = {}
    max_points = min(len(person_xy), len(COCO_KEYPOINT_NAMES))

    for idx in range(max_points):
        confidence = float(person_confidence[idx])
        if confidence < KEYPOINT_CONFIDENCE:
            continue
        x, y = person_xy[idx]
        keypoints[COCO_KEYPOINT_NAMES[idx]] = (int(round(x)), int(round(y)), confidence)

    return keypoints

def identify_candidate_stage(detections, frame_width, frame_height):
    collection_zone = zone_to_pixels(ZONES["collection"], frame_width, frame_height)
    sewing_zone = zone_to_pixels(ZONES["sewing"], frame_width, frame_height)
    packaging_zone = zone_to_pixels(ZONES["packaging"], frame_width, frame_height)

    hands = get_detections_by_classes(detections, HAND_CLASSES)
    gloves = get_detections_by_classes(detections, GLOVE_CLASSES)
    bags = get_detections_by_classes(detections, BAG_CLASSES)

    for glove in gloves:
        for bag in bags:
            if box_iou(glove.bbox, bag.bbox) >= 0.03 or \
               (point_in_box(glove.center, packaging_zone) and point_in_box(bag.center, packaging_zone)):
                return 3

    sewing_objects = hands + gloves
    if any(point_in_box(d.center, sewing_zone) for d in sewing_objects):
        return 2

    interaction_distance = int(0.12 * min(frame_width, frame_height))
    for glove in gloves:
        if not point_in_box(glove.center, collection_zone):
            continue
        for hand in hands:
            if point_distance(hand.center, glove.center) <= interaction_distance:
                return 1

    return None

# ============================================================
# DRAW FUNCTIONS
# ============================================================

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
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        if detection.class_name in HAND_CLASSES:
            color = (255, 80, 80)
        elif detection.class_name in GLOVE_CLASSES:
            color = (0, 220, 255)
        elif detection.class_name in BAG_CLASSES:
            color = (255, 0, 255)
        else:
            color = (180, 180, 180)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

def draw_pose(frame, keypoints):
    try:
        for first_name, second_name in POSE_CONNECTIONS:
            first = get_keypoint(keypoints, first_name)
            second = get_keypoint(keypoints, second_name)
            if first is not None and second is not None:
                cv2.line(frame, first, second, (0, 200, 0), 2, cv2.LINE_AA)

        for x, y, confidence in keypoints.values():
            if confidence >= KEYPOINT_CONFIDENCE:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    except:
        pass

# ============================================================
# MONITORS
# ============================================================

class ProductionStateMachine:
    def __init__(self, event_writer):
        self.event_writer = event_writer
        self.current_stage = 0
        self.stage_started_at = None
        self.cycle_started_at = None
        self.candidate_stage = None
        self.candidate_started_at = None
        self.completed_cycles = 0

    def update(self, candidate_stage, current_time):
        if candidate_stage is None:
            self.candidate_stage = None
            self.candidate_started_at = None
            return

        if candidate_stage != self.candidate_stage:
            self.candidate_stage = candidate_stage
            self.candidate_started_at = current_time
            return

        if self.candidate_started_at is None:
            self.candidate_started_at = current_time
            return

        stable_duration = current_time - self.candidate_started_at
        if stable_duration < STAGE_DEBOUNCE_SECONDS:
            return

        self._accept_stage(candidate_stage, current_time)

    def _accept_stage(self, new_stage, current_time):
        if new_stage == 1 and self.current_stage in {0, 3}:
            self.cycle_started_at = current_time
            self._transition(new_stage, current_time)
            return

        if new_stage == self.current_stage + 1:
            self._transition(new_stage, current_time)

    def _transition(self, new_stage, current_time):
        if self.current_stage > 0 and self.stage_started_at is not None:
            previous_duration = current_time - self.stage_started_at
            self.event_writer.write(
                "stage_completed",
                {
                    "stage_number": self.current_stage,
                    "stage_name": STAGE_NAMES[self.current_stage],
                    "duration_seconds": round(previous_duration, 3),
                },
            )

        self.current_stage = new_stage
        self.stage_started_at = current_time

        self.event_writer.write(
            "stage_started",
            {
                "stage_number": new_stage,
                "stage_name": STAGE_NAMES[new_stage],
            },
        )

        if new_stage == 3 and self.cycle_started_at is not None:
            cycle_duration = current_time - self.cycle_started_at
            self.completed_cycles += 1
            self.event_writer.write(
                "cycle_completed",
                {
                    "cycle_number": self.completed_cycles,
                    "cycle_duration_seconds": round(cycle_duration, 3),
                },
            )

class NeedleSafetyMonitor:
    def __init__(self, event_writer):
        self.event_writer = event_writer
        self.previous_hazard_state = False
        self.last_alert_at = 0.0

    def update(self, detections, keypoints, frame_width, frame_height, current_time):
        hazard_zone = zone_to_pixels(ZONES["needle_hazard"], frame_width, frame_height)

        hands = get_detections_by_classes(detections, HAND_CLASSES)
        hand_box_in_hazard = any(boxes_intersect(h.bbox, hazard_zone) for h in hands)

        left_wrist = get_keypoint(keypoints, "left_wrist")
        right_wrist = get_keypoint(keypoints, "right_wrist")
        wrist_in_hazard = any(
            wrist is not None and point_in_box(wrist, hazard_zone)
            for wrist in [left_wrist, right_wrist]
        )

        hazard_active = hand_box_in_hazard or wrist_in_hazard
        entered_hazard = hazard_active and not self.previous_hazard_state
        cooldown_finished = current_time - self.last_alert_at >= HAZARD_ALERT_COOLDOWN_SECONDS

        if entered_hazard and cooldown_finished:
            self.last_alert_at = current_time
            self.event_writer.write(
                "needle_hazard",
                {
                    "message": "Mão detectada na zona de risco da agulha",
                    "detected_by_hand_model": hand_box_in_hazard,
                    "detected_by_wrist_keypoint": wrist_in_hazard,
                },
                severity="critical",
            )

        self.previous_hazard_state = hazard_active
        return hazard_active

class ErgonomicsMonitor:
    def __init__(self, event_writer):
        self.event_writer = event_writer
        self.bad_posture_started_at = None
        self.last_alert_at = 0.0

    def update(self, keypoints, current_time):
        left_shoulder = get_keypoint(keypoints, "left_shoulder")
        right_shoulder = get_keypoint(keypoints, "right_shoulder")
        left_hip = get_keypoint(keypoints, "left_hip")
        right_hip = get_keypoint(keypoints, "right_hip")

        if any(p is None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
            self.bad_posture_started_at = None
            return None

        shoulder_center = midpoint(left_shoulder, right_shoulder)
        hip_center = midpoint(left_hip, right_hip)
        trunk_angle = angle_from_vertical(shoulder_center, hip_center)

        if trunk_angle > MAX_TRUNK_ANGLE_DEGREES:
            if self.bad_posture_started_at is None:
                self.bad_posture_started_at = current_time

            posture_duration = current_time - self.bad_posture_started_at
            cooldown_finished = current_time - self.last_alert_at >= POSTURE_ALERT_COOLDOWN_SECONDS

            if posture_duration >= POSTURE_HOLD_SECONDS and cooldown_finished:
                self.last_alert_at = current_time
                self.event_writer.write(
                    "ergonomic_risk",
                    {
                        "risk": "sustained_trunk_inclination",
                        "trunk_angle_degrees": round(trunk_angle, 2),
                        "duration_seconds": round(posture_duration, 2),
                        "angle_threshold": MAX_TRUNK_ANGLE_DEGREES,
                    },
                    severity="warning",
                )
        else:
            self.bad_posture_started_at = None

        return trunk_angle

def parse_video_source(source):
    if source.isdigit():
        return int(source)
    return source

# ============================================================
# STREAMING
# ============================================================

def start_streaming():
    """Inicia o servidor de streaming em uma thread separada"""
    try:
        from flask import Flask, Response
    except ImportError:
        print("⚠️ Flask não instalado. Streaming desativado.")
        print("💡 Instale: pip install flask")
        return
    
    app = Flask(__name__)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
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
                detections = run_object_detection_api(frame)
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
    
    print(f"\n🌐 Streaming disponível em: http://localhost:{STREAM_PORT}")
    print(f"   Abra no Codespace: Portas → {STREAM_PORT} → 🌐\n")
    
    app.run(host='0.0.0.0', port=STREAM_PORT, debug=False, use_reloader=False)

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Iniciando Monitor Industrial de Luvas")
    print("=" * 50)
    
    # Inicia streaming em thread separada
    if ENABLE_STREAMING:
        try:
            import flask
            stream_thread = threading.Thread(target=start_streaming, daemon=True)
            stream_thread.start()
            time.sleep(2)
        except ImportError:
            print("⚠️ Flask não instalado. Instale com: pip install flask")
            print("💡 Streaming desativado")
    
    # Carrega modelo de pose
    pose_model = None
    try:
        from ultralytics import YOLO
        pose_model = YOLO(POSE_MODEL_PATH) if os.path.exists(POSE_MODEL_PATH) else None
        if pose_model:
            print(f"✅ Modelo de pose carregado: {POSE_MODEL_PATH}")
        else:
            print("⚠️ Modelo de pose não encontrado. Baixando...")
            pose_model = YOLO("yolov8n-pose.pt")
            print("✅ Modelo de pose baixado!")
    except Exception as e:
        print(f"⚠️ Erro ao carregar modelo de pose: {e}")

    event_writer = EventWriter(LOCAL_EVENT_FILE, SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE)

    production_monitor = ProductionStateMachine(event_writer)
    safety_monitor = NeedleSafetyMonitor(event_writer)
    ergonomics_monitor = ErgonomicsMonitor(event_writer)

    video_source = parse_video_source(VIDEO_SOURCE)
    print(f"📹 Abrindo: {video_source}")
    capture = cv2.VideoCapture(video_source)

    if not capture.isOpened():
        event_writer.close()
        raise RuntimeError(f"Não foi possível abrir: {VIDEO_SOURCE}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        print(f"🎬 Vídeo: {total_frames} frames, {fps:.2f} FPS")
    else:
        print("🎥 Webcam ao vivo")

    frame_number = 0
    last_detections = []
    last_keypoints = {}

    event_writer.write("camera_started", {
        "video_source": VIDEO_SOURCE,
        "device": DEVICE
    })

    print("✅ Sistema iniciado! Pressione Ctrl+C para parar.")
    print("=" * 50)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("📹 Fim do vídeo.")
                break

            frame_number += 1
            current_time = time.monotonic()
            frame_height, frame_width = frame.shape[:2]

            if frame_number % max(INFERENCE_EVERY_N_FRAMES, 1) == 0:
                try:
                    last_detections = run_object_detection_api(frame)
                    if pose_model:
                        last_keypoints = run_pose_estimation(pose_model, frame)
                except Exception as error:
                    print(f"[ERRO] {error}")

            candidate_stage = identify_candidate_stage(last_detections, frame_width, frame_height)
            production_monitor.update(candidate_stage, current_time)

            safety_monitor.update(last_detections, last_keypoints, frame_width, frame_height, current_time)
            ergonomics_monitor.update(last_keypoints, current_time)

            if frame_number % 50 == 0:
                print(f"📊 Frame {frame_number} | Etapa: {STAGE_NAMES[production_monitor.current_stage]} | Ciclos: {production_monitor.completed_cycles} | Detecções: {len(last_detections)}")

            if total_frames > 0 and frame_number % 100 == 0:
                progress = int((frame_number / total_frames) * 100)
                print(f"📊 Progresso: {progress}%")

            # Exibe a janela (opcional)
            try:
                frame_display = frame.copy()
                draw_zones(frame_display)
                draw_detections(frame_display, last_detections)
                draw_pose(frame_display, last_keypoints)
                cv2.imshow("Factory Vision AI", frame_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass

    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário.")
    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

        event_writer.write("camera_stopped", {
            "processed_frames": frame_number,
            "completed_cycles": production_monitor.completed_cycles
        })
        event_writer.close()

        print("=" * 50)
        print("📊 RESUMO:")
        print(f"  📹 Frames: {frame_number}")
        print(f"  🔄 Ciclos: {production_monitor.completed_cycles}")
        print(f"  📁 Eventos: {LOCAL_EVENT_FILE}")
        if SUPABASE_URL and SUPABASE_KEY:
            print(f"  ☁️  Supabase: Enviando eventos para a tabela {SUPABASE_TABLE}")
        print(f"  🤖 Roboflow Model: {ROBOFLOW_MODEL_ID}")
        if ENABLE_STREAMING:
            print(f"  🌐 Streaming: http://localhost:{STREAM_PORT}")
        print("=" * 50)

if __name__ == "__main__":
    main()