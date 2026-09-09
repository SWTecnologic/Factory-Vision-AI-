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
    - ZONAS (caixas)   -> NAO sao detectadas por classificador; sao
                          regioes (ROIs) configuraveis, desenhadas
                          visualmente no frontend (StationZonesPage) e
                          salvas na tabela "zones" do Supabase. Cada
                          zona tem um zone_type (ex.: PACKAGING_BOX,
                          DISCARD_BOX) e so' emite deteccao quando uma
                          luva JA CONFIRMADA COMO DOBRADA tem seu
                          centroide dentro dela. Isso reproduz
                          fielmente o evento "detectar quando a luva
                          dobrada e' colocada dentro da caixa
                          [de embalagem/descarte]" (e nao apenas
                          "caixa esta visivel", que e' sempre
                          verdade numa camera fixa).

Este modulo foi desenhado para ser um DROP-IN replacement de
`run_workflow_api()`: a funcao `detect_local(frame)` devolve
`List[Detection]` com os MESMOS nomes de classe que o main.py ja
espera (HAND_CLASSES / GLOVE_CLASSES / PACKAGING_CLASSES /
DISCARD_CLASSES), entao o CycleManager, EventWriter,
ProductionRepository e Supabase NAO precisam mudar quando as zonas
sao alteradas — so' o que muda e' QUAIS zonas existem, vindas do
banco.

Quando um modelo YOLO local (.pt) treinado especificamente para essa
operacao estiver disponivel, basta substituir o corpo de
`GloveDetector.detect()` (e opcionalmente `HandDetector.detect()`)
pela inferencia do YOLO, mantendo o mesmo contrato de retorno.
"""

import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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

# --- Tracking simples por centroide (para nao contar a mesma luva 2x) ---
TRACK_MAX_DISTANCE_PX = int(os.getenv("TRACK_MAX_DISTANCE_PX", "120"))
TRACK_MAX_MISSED_FRAMES = int(os.getenv("TRACK_MAX_MISSED_FRAMES", "10"))

# --- Margem extra ao redor de cada zona de caixa para considerar que
# a MAO ainda esta "na caixa" (segurando/soltando a luva). Sem essa
# margem, a mao sairia da deteccao um pouco antes do previsto porque
# a ROI da caixa e' desenhada nas bordas internas do papelao.
HAND_BOX_MARGIN_PX = int(os.getenv("HAND_BOX_MARGIN_PX", "40"))


# ============================================================
# ZONAS DINAMICAS (vindas do Supabase, configuradas no frontend)
# ============================================================
# Cada zona: {"id":..., "name":..., "zone_type":..., "frac": (x1,y1,x2,y2)}
# "frac" esta em 0..1, independente da resolucao real do frame — o
# frontend salva coordinates em PORCENTAGEM (0..100) e o main.py
# converte para fracao (0..1) antes de chamar configure_zones().

_ZONES: List[dict] = []

# Fallback: se nenhuma zona vier do Supabase (ex.: banco offline, ou
# estacao ainda sem nenhuma zona cadastrada), usa o BOX_ROI do .env
# como uma unica zona do tipo PACKAGING_BOX, pra nao quebrar
# setups antigos/testes locais sem banco.
_fallback_roi_raw = os.getenv("BOX_ROI", "0.62,0.10,1.0,1.0")
_FALLBACK_ZONE = {
    "id": None,
    "name": "fallback_env (BOX_ROI)",
    "zone_type": "PACKAGING_BOX",
    "frac": tuple(float(v) for v in _fallback_roi_raw.split(",")),
}

# Mapeia o zone_type (igual ao que o frontend salva na coluna
# "zone_type" da tabela "zones") para a classe de deteccao que o
# CycleManager (main.py) reconhece via PACKAGING_CLASSES /
# DISCARD_CLASSES. Zonas com zone_type nao mapeado aqui (ex.:
# INSPECTION, DANGER, STORAGE, ASSEMBLY) sao ignoradas pela logica
# de ciclo, mas continuam sendo desenhadas no streaming.
ZONE_TYPE_CLASS_MAP: Dict[str, str] = {
    "PACKAGING_BOX": "caixa_embalagem",
    "DISCARD_BOX": "caixa_descarte",
}


def configure_zones(zones: List[dict]):
    """Chamado pelo main.py (na inicializacao e, opcionalmente, num
    refresh periodico) depois de buscar as zonas no Supabase.
    Substitui completamente o conjunto de zonas ativas."""

    global _ZONES
    _ZONES = zones


def _active_zones() -> List[dict]:
    """Zonas configuradas via Supabase, ou o fallback do .env se
    nao houver nenhuma."""

    return _ZONES if _ZONES else [_FALLBACK_ZONE]


def _zone_pixels(zone: dict, frame_shape) -> Tuple[int, int, int, int]:
    h_frame, w_frame = frame_shape[:2]
    x1, y1, x2, y2 = zone["frac"]
    return (
        int(x1 * w_frame),
        int(y1 * h_frame),
        int(x2 * w_frame),
        int(y2 * h_frame),
    )


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
                "O motor vai funcionar apenas com deteccao de luva/zonas."
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
# GLOVE DETECTOR (segmentacao HSV + tracking + heuristica de dobra
# + verificacao contra as ZONAS dinamicas do Supabase)
# ============================================================

class GloveDetector:
    """Detecta a luva por cor e classifica o estado (aberta/dobrada).

    Substituivel no futuro por um YOLO treinado: basta trocar o corpo
    de `_segment()` por inferencia do modelo e manter o resto (o
    tracking + heuristica de dobra + verificacao de zonas continuam
    validos mesmo com boxes vindas de um YOLO real).
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
        raw_boxes = self._segment(frame_bgr)
        tracked = self.tracker.update(raw_boxes)

        # Zonas ativas neste frame, ja convertidas para pixels (uma
        # unica vez por frame, nao por track).
        zone_pixels = [
            (_zone_pixels(z, frame_bgr.shape), z["zone_type"])
            for z in _active_zones()
        ]

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

            # Evento-chave: luva JA CONFIRMADA COMO DOBRADA cujo
            # centroide cai dentro de ALGUMA zona configurada no
            # Supabase -> emite a classe correspondente ao
            # zone_type daquela zona especifica (embalagem,
            # descarte, ...). Uma luva so' pode estar em uma zona
            # por vez, entao paramos no primeiro match.
            if track.folded_confirmed:
                cx, cy = track.center

                for (bx1, by1, bx2, by2), zone_type in zone_pixels:

                    if bx1 <= cx <= bx2 and by1 <= cy <= by2:

                        mapped_class = ZONE_TYPE_CLASS_MAP.get(zone_type)

                        if mapped_class:
                            detections.append(
                                Detection(
                                    class_name=mapped_class,
                                    confidence=0.9,
                                    bbox=track.bbox,
                                )
                            )

                        break

        return detections

    def zones_pixels(self, frame_shape):
        """Todas as zonas ativas em pixels, com nome e tipo — usado
        para desenhar no streaming (debug visual) e para o
        detect_local() verificar a mao contra todas as caixas."""

        return [
            (_zone_pixels(z, frame_shape), z["zone_type"], z["name"])
            for z in _active_zones()
        ]


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
    """Roda a deteccao 100% local (mao + luva + zonas) em um frame.

    Retorna List[Detection] com class_name em {"mao", "mao_na_caixa",
    "luva_aberta", "luva_dobrada", "caixa_embalagem",
    "caixa_descarte"} — compativel com HAND_CLASSES, GLOVE_CLASSES,
    PACKAGING_CLASSES e DISCARD_CLASSES do main.py, cujas zonas
    (posicao e tipo) vem dinamicamente do Supabase via
    configure_zones().
    """
    hand_detector, glove_detector = _get_detectors()

    hand_detections = hand_detector.detect(frame_bgr)
    glove_detections = glove_detector.detect(frame_bgr)

    detections = list(hand_detections) + list(glove_detections)

    # ------------------------------------------------------------
    # "mao_na_caixa": emitido quando o centro de QUALQUER mao
    # detectada esta dentro de QUALQUER zona de caixa configurada
    # (com uma margem extra). Isso e' o sinal usado pelo
    # CycleManager para saber que o operador AINDA esta
    # segurando/posicionando a luva dentro da caixa — so' quando
    # esse sinal desaparecer por alguns frames consecutivos e' que
    # a producao/descarte e' confirmado (mao se afastou).
    # ------------------------------------------------------------
    if hand_detections:

        for (bx1, by1, bx2, by2), _zone_type, _name in (
            glove_detector.zones_pixels(frame_bgr.shape)
        ):

            bx1 -= HAND_BOX_MARGIN_PX
            by1 -= HAND_BOX_MARGIN_PX
            bx2 += HAND_BOX_MARGIN_PX
            by2 += HAND_BOX_MARGIN_PX

            hand_in_this_zone = False

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
                    hand_in_this_zone = True
                    break  # uma ocorrencia ja basta como sinal

            if hand_in_this_zone:
                break  # nao precisa checar as outras zonas

    return detections


def get_zones_pixels(frame_shape):
    """Exposto para o main.py desenhar TODAS as zonas configuradas
    (embalagem, descarte, etc.) no streaming (debug visual)."""

    _, glove_detector = _get_detectors()
    return glove_detector.zones_pixels(frame_shape)