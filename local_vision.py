"""
local_vision.py
================
Substitui completamente a dependencia do Roboflow Workflow API.

Gera objetos `Detection` (mesma classe usada no main.py: class_name,
confidence, bbox) a partir de visao computacional 100% local:

    - Deteccao de MAO  -> MediaPipe Hand Landmarker (modelo local .task)
    - Deteccao de LUVA -> segmentacao de cor (HSV) + tracking por centroide
                          + heuristica geometrica (aspect ratio / area) para
                          diferenciar "luva_aberta" de "luva_dobrada"
    - Deteccao de CAIXA -> nao e' detectada por classificador; e' uma ROI
                          (regiao) configuravel do frame. A classe "caixa"
                          so e' emitida quando uma luva JA CONFIRMADA COMO
                          DOBRADA tem seu centroide dentro dessa ROI. Isso
                          reproduz fielmente o evento #7 do pedido:
                          "detectar quando a luva dobrada e' colocada
                          dentro da caixa" (e nao apenas "caixa esta
                          visivel", que e' sempre verdade numa camera fixa).

Este modulo foi desenhado para ser um DROP-IN replacement de
`run_workflow_api()`: a funcao `detect_local(frame)` devolve
`List[Detection]` com os MESMOS nomes de classe que o main.py ja
espera (HAND_CLASSES / GLOVE_CLASSES / BAG_CLASSES), entao o
CycleManager, EventWriter, ProductionRepository e Supabase NAO
precisam mudar.

Quando um modelo YOLO local (.pt) treinado especificamente para essa
operacao estiver disponivel, basta substituir o corpo de
`GloveDetector.detect()` (e opcionalmente `HandDetector.detect()`)
pela inferencia do YOLO, mantendo o mesmo contrato de retorno.
"""

import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ============================================================
# Detection (mesma interface do main.py)
# ============================================================
# Importado de dentro do main.py em tempo de execucao para evitar
# import circular; aqui recriamos a mesma dataclass de forma
# independente e o main.py usa esta versao unica (ver integracao).


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))


# ============================================================
# CONFIGURACOES (todas via .env, com defaults calibrados no
# video de teste enviado — ajuste para a camera real)
# ============================================================

# --- MediaPipe Hand Landmarker ---
HAND_MODEL_PATH = os.getenv("HAND_MODEL_PATH", "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MIN_DETECTION_CONFIDENCE = float(
    os.getenv("HAND_MIN_DETECTION_CONFIDENCE", "0.5")
)
HAND_MAX_NUM_HANDS = int(os.getenv("HAND_MAX_NUM_HANDS", "2"))
HAND_BBOX_MARGIN_PX = int(os.getenv("HAND_BBOX_MARGIN_PX", "25"))

# --- Segmentacao de cor da luva (HSV) ---
# Default calibrado para luva VERDE. Se a luva real for de outra cor,
# ajuste GLOVE_HSV_LOWER / GLOVE_HSV_UPPER no .env (formato "H,S,V").
GLOVE_HSV_LOWER = np.array(
    [int(v) for v in os.getenv("GLOVE_HSV_LOWER", "40,40,40").split(",")]
)
GLOVE_HSV_UPPER = np.array(
    [int(v) for v in os.getenv("GLOVE_HSV_UPPER", "90,255,255").split(",")]
)
GLOVE_MIN_AREA_PX = int(os.getenv("GLOVE_MIN_AREA_PX", "1200"))

# --- Heuristica aberta -> dobrada ---
# Luva "aberta" tem aspect ratio alto (comprida/esticada).
# Luva "dobrada" fica mais compacta: aspect ratio cai E a area cai em
# relacao a maior area ja observada para aquele objeto rastreado.
GLOVE_OPEN_ASPECT_MIN = float(os.getenv("GLOVE_OPEN_ASPECT_MIN", "1.5"))
GLOVE_FOLDED_ASPECT_MAX = float(os.getenv("GLOVE_FOLDED_ASPECT_MAX", "1.3"))
GLOVE_FOLDED_AREA_RATIO_MAX = float(
    os.getenv("GLOVE_FOLDED_AREA_RATIO_MAX", "0.65")
)
GLOVE_FOLD_CONFIRM_FRAMES = int(os.getenv("GLOVE_FOLD_CONFIRM_FRAMES", "3"))

# --- ROI da caixa (fracoes 0..1 do frame: x1,y1,x2,y2) ---
# Default calibrado no video enviado (caixa no lado direito do frame).
# Ajuste para a posicao real da caixa na sua camera.
_box_roi_raw = os.getenv("BOX_ROI", "0.62,0.10,1.0,1.0")
BOX_ROI_FRAC = tuple(float(v) for v in _box_roi_raw.split(","))

# --- Tracking simples por centroide (para nao contar a mesma luva 2x) ---
TRACK_MAX_DISTANCE_PX = int(os.getenv("TRACK_MAX_DISTANCE_PX", "120"))
TRACK_MAX_MISSED_FRAMES = int(os.getenv("TRACK_MAX_MISSED_FRAMES", "10"))

# --- Margem extra ao redor da ROI da caixa para considerar que a MAO
# ainda esta "na caixa" (segurando/soltando a luva). Sem essa margem,
# a mao sairia da deteccao um pouco antes do previsto porque a ROI da
# caixa e' desenhada nas bordas internas do papelao.
HAND_BOX_MARGIN_PX = int(os.getenv("HAND_BOX_MARGIN_PX", "40"))


# ============================================================
# HAND DETECTOR (MediaPipe Tasks - HandLandmarker, 100% local)
# ============================================================

class HandDetector:
    """Detecta maos usando o MediaPipe Hand Landmarker (modelo local)."""

    def __init__(self):
        self._detector = None
        self._enabled = False
        self._init_error = None
        self._ensure_model()
        self._load_detector()

    def _ensure_model(self):
        if os.path.exists(HAND_MODEL_PATH):
            return
        try:
            print(
                f"Baixando modelo MediaPipe Hand Landmarker "
                f"(~8MB, uma vez) para {HAND_MODEL_PATH} ..."
            )
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
            print("Modelo de mao baixado com sucesso.")
        except Exception as e:
            self._init_error = e
            print(
                f"AVISO: nao foi possivel baixar o modelo de mao "
                f"automaticamente ({e}).\n"
                f"Baixe manualmente em:\n  {HAND_MODEL_URL}\n"
                f"e salve como '{HAND_MODEL_PATH}' na pasta do projeto."
            )

    def _load_detector(self):
        if not os.path.exists(HAND_MODEL_PATH):
            print(
                "Deteccao de MAO desativada (sem modelo local). "
                "O motor vai funcionar apenas com deteccao de luva/caixa."
            )
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_options = mp_python.BaseOptions(
                model_asset_path=HAND_MODEL_PATH
            )
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=HAND_MAX_NUM_HANDS,
                min_hand_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
            )
            self._detector = mp_vision.HandLandmarker.create_from_options(
                options
            )
            self._mp = mp
            self._enabled = True
            print("HandDetector (MediaPipe) inicializado com sucesso.")

        except Exception as e:
            self._init_error = e
            print(f"AVISO: falha ao inicializar MediaPipe Hands: {e}")

    def detect(self, frame_bgr) -> List[Detection]:
        if not self._enabled:
            return []

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )

        try:
            result = self._detector.detect(mp_image)
        except Exception as e:
            print(f"Erro na inferencia MediaPipe Hands: {e}")
            return []

        detections = []

        if not result or not result.hand_landmarks:
            return detections

        for hand_idx, landmarks in enumerate(result.hand_landmarks):
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]

            x1 = max(0, int(min(xs)) - HAND_BBOX_MARGIN_PX)
            y1 = max(0, int(min(ys)) - HAND_BBOX_MARGIN_PX)
            x2 = min(w, int(max(xs)) + HAND_BBOX_MARGIN_PX)
            y2 = min(h, int(max(ys)) + HAND_BBOX_MARGIN_PX)

            confidence = 1.0
            if result.handedness and len(result.handedness) > hand_idx:
                cats = result.handedness[hand_idx]
                if cats:
                    confidence = float(cats[0].score)

            detections.append(
                Detection(
                    class_name="mao",
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections


# ============================================================
# TRACKING SIMPLES POR CENTROIDE (evita contar a mesma luva 2x)
# ============================================================

@dataclass
class GloveTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    max_area: float
    consecutive_folded_frames: int = 0
    folded_confirmed: bool = False
    missed_frames: int = 0
    last_seen_monotonic: float = field(default_factory=time.monotonic)


class GloveTracker:
    """Tracker por centroide (leve, sem dependencias externas).

    Suficiente aqui porque tipicamente ha' NO MAXIMO UMA luva relevante
    em cena por vez. Mantem o estado (area maxima ja vista, quantos
    frames consecutivos em formato "dobrado") por ID persistente, o
    que e' a base da logica temporal pedida (nao contar a mesma luva
    varias vezes, e so' considerar "dobrada" apos evolucao real a
    partir do estado "aberta").
    """

    def __init__(self):
        self._tracks: dict[int, GloveTrack] = {}
        self._next_id = 1

    def update(self, raw_boxes: List[Tuple[int, int, int, int, float]]):
        """raw_boxes: lista de (x1,y1,x2,y2,area) detectados neste frame.

        Retorna lista de (track, is_new) para os tracks atualizados
        neste frame.
        """
        updated = []
        used_track_ids = set()

        for (x1, y1, x2, y2, area) in raw_boxes:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            best_id = None
            best_dist = TRACK_MAX_DISTANCE_PX

            for tid, track in self._tracks.items():
                if tid in used_track_ids:
                    continue
                tcx, tcy = track.center
                dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                track = self._tracks[best_id]
                track.bbox = (x1, y1, x2, y2)
                track.center = (cx, cy)
                track.max_area = max(track.max_area, area)
                track.missed_frames = 0
                track.last_seen_monotonic = time.monotonic()
                used_track_ids.add(best_id)
                updated.append((track, False))
            else:
                new_track = GloveTrack(
                    track_id=self._next_id,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    max_area=area,
                )
                self._tracks[self._next_id] = new_track
                used_track_ids.add(self._next_id)
                updated.append((new_track, True))
                self._next_id += 1

        # Envelhece tracks nao vistos neste frame; remove os antigos.
        stale_ids = []
        for tid, track in self._tracks.items():
            if tid not in used_track_ids:
                track.missed_frames += 1
                if track.missed_frames > TRACK_MAX_MISSED_FRAMES:
                    stale_ids.append(tid)

        for tid in stale_ids:
            del self._tracks[tid]

        return updated


# ============================================================
# GLOVE DETECTOR (segmentacao HSV + tracking + heuristica de dobra)
# ============================================================

class GloveDetector:
    """Detecta a luva por cor e classifica o estado (aberta/dobrada).

    Substituivel no futuro por um YOLO treinado: basta trocar o corpo
    de `_segment()` por inferencia do modelo e manter o resto (o
    tracking + heuristica de dobra continuam validos mesmo com boxes
    vindas de um YOLO real).
    """

    def __init__(self):
        self.tracker = GloveTracker()

    def _segment(self, frame_bgr) -> List[Tuple[int, int, int, int, float]]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GLOVE_HSV_LOWER, GLOVE_HSV_UPPER)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
        )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < GLOVE_MIN_AREA_PX:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h, area))

        return boxes

    def detect(self, frame_bgr) -> List[Detection]:
        h_frame, w_frame = frame_bgr.shape[:2]

        box_x1 = int(BOX_ROI_FRAC[0] * w_frame)
        box_y1 = int(BOX_ROI_FRAC[1] * h_frame)
        box_x2 = int(BOX_ROI_FRAC[2] * w_frame)
        box_y2 = int(BOX_ROI_FRAC[3] * h_frame)

        raw_boxes = self._segment(frame_bgr)
        tracked = self.tracker.update(raw_boxes)

        detections = []

        for track, _is_new in tracked:
            x1, y1, x2, y2 = track.bbox
            w = x2 - x1
            h = y2 - y1
            if h == 0:
                continue

            area = float(w * h)
            aspect = w / h

            area_ratio = (
                area / track.max_area if track.max_area > 0 else 1.0
            )

            is_folded_now = (
                aspect <= GLOVE_FOLDED_ASPECT_MAX
                and area_ratio <= GLOVE_FOLDED_AREA_RATIO_MAX
            )

            if is_folded_now:
                track.consecutive_folded_frames += 1
            else:
                track.consecutive_folded_frames = 0

            if track.consecutive_folded_frames >= GLOVE_FOLD_CONFIRM_FRAMES:
                track.folded_confirmed = True

            if track.folded_confirmed:
                class_name = "luva_dobrada"
            elif aspect >= GLOVE_OPEN_ASPECT_MIN:
                class_name = "luva_aberta"
            else:
                # estado intermediario (sendo dobrada agora)
                class_name = "luva_aberta"

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=0.9,
                    bbox=track.bbox,
                )
            )

            # Evento-chave #7: luva JA CONFIRMADA COMO DOBRADA cujo
            # centroide esta dentro da ROI da caixa -> emite "caixa".
            if track.folded_confirmed:
                cx, cy = track.center
                inside_box = (
                    box_x1 <= cx <= box_x2 and box_y1 <= cy <= box_y2
                )
                if inside_box:
                    detections.append(
                        Detection(
                            class_name="caixa",
                            confidence=0.9,
                            bbox=track.bbox,
                        )
                    )

        return detections

    def box_roi_pixels(self, frame_shape) -> Tuple[int, int, int, int]:
        h_frame, w_frame = frame_shape[:2]
        return (
            int(BOX_ROI_FRAC[0] * w_frame),
            int(BOX_ROI_FRAC[1] * h_frame),
            int(BOX_ROI_FRAC[2] * w_frame),
            int(BOX_ROI_FRAC[3] * h_frame),
        )


# ============================================================
# API PUBLICA — drop-in replacement de run_workflow_api()
# ============================================================

_hand_detector: Optional[HandDetector] = None
_glove_detector: Optional[GloveDetector] = None


def _get_detectors():
    global _hand_detector, _glove_detector
    if _hand_detector is None:
        _hand_detector = HandDetector()
    if _glove_detector is None:
        _glove_detector = GloveDetector()
    return _hand_detector, _glove_detector


def detect_local(frame_bgr) -> List[Detection]:
    """Roda a deteccao 100% local (mao + luva + caixa) em um frame.

    Retorna List[Detection] com class_name em {"mao", "mao_na_caixa",
    "luva_aberta", "luva_dobrada", "caixa"} — compativel com
    HAND_CLASSES, GLOVE_CLASSES e BAG_CLASSES ja existentes no
    main.py, mais a nova classe "mao_na_caixa" usada para confirmar
    com precisao o momento em que o operador solta a luva e afasta
    a mao da caixa (evento de producao real).
    """
    hand_detector, glove_detector = _get_detectors()

    hand_detections = hand_detector.detect(frame_bgr)
    glove_detections = glove_detector.detect(frame_bgr)

    detections = list(hand_detections) + list(glove_detections)

    # ------------------------------------------------------------
    # "mao_na_caixa": emitido quando o centro de QUALQUER mao
    # detectada esta dentro da ROI da caixa (com uma margem extra).
    # Isso e' o sinal usado pelo CycleManager para saber que o
    # operador AINDA esta segurando/posicionando a luva dentro da
    # caixa — so' quando esse sinal desaparecer por alguns frames
    # consecutivos e' que a producao e' confirmada (mao se afastou).
    # ------------------------------------------------------------
    if hand_detections:
        h_frame, w_frame = frame_bgr.shape[:2]
        bx1, by1, bx2, by2 = glove_detector.box_roi_pixels(
            frame_bgr.shape
        )
        bx1 -= HAND_BOX_MARGIN_PX
        by1 -= HAND_BOX_MARGIN_PX
        bx2 += HAND_BOX_MARGIN_PX
        by2 += HAND_BOX_MARGIN_PX

        for hand_det in hand_detections:
            hx, hy = hand_det.center
            if bx1 <= hx <= bx2 and by1 <= hy <= by2:
                detections.append(
                    Detection(
                        class_name="mao_na_caixa",
                        confidence=hand_det.confidence,
                        bbox=hand_det.bbox,
                    )
                )
                break  # uma ocorrencia ja basta como sinal

    return detections


def get_box_roi_pixels(frame_shape) -> Tuple[int, int, int, int]:
    """Exposto para desenhar a ROI da caixa no streaming (debug visual)."""
    _, glove_detector = _get_detectors()
    return glove_detector.box_roi_pixels(frame_shape)