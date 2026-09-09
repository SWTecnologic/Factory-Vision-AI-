import json
import os
import queue
import threading
import time
import uuid
import warnings

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

warnings.filterwarnings("ignore")

# ============================================================
# DOTENV
# ============================================================

try:
    from dotenv import load_dotenv

    load_dotenv()
    print("Arquivo .env carregado com sucesso!")

except ImportError:
    print("python-dotenv nao instalado")

# ============================================================
# OPENCV
# ============================================================

try:
    import cv2

except ImportError:
    print("cv2 nao encontrado, instalando...")

    import subprocess

    subprocess.check_call(
        ["pip", "install", "opencv-python-headless"]
    )

    import cv2

# ============================================================
# FLASK
# ============================================================

try:
    from flask import Flask, Response

    FLASK_AVAILABLE = True

except ImportError:
    FLASK_AVAILABLE = False
    print("Flask nao instalado. Streaming desativado.")

# ============================================================
# OUTROS
# ============================================================

import numpy as np
import requests

# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    ""
)

SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    ""
)

if not SUPABASE_URL or not SUPABASE_KEY:
    print(
        "SUPABASE_URL ou SUPABASE_KEY nao configurados!"
    )

try:
    from supabase import create_client

    supabase = create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )

    print("Supabase conectado!")

except Exception as e:

    print(
        f"Erro ao conectar Supabase: {e}"
    )

    supabase = None

# ============================================================
# VISAO COMPUTACIONAL LOCAL (substitui o Roboflow por completo)
# ============================================================
# Toda a inferencia (mao / luva / zonas) agora roda localmente,
# via local_vision.py (MediaPipe + OpenCV). Nenhuma chamada de
# rede e' feita para detectar objetos. As ZONAS (caixa de
# embalagem, caixa de descarte, etc.) agora vem do Supabase,
# configuradas visualmente no frontend (StationZonesPage).

from local_vision import detect_local, get_zones_pixels, configure_zones

POSTURE_ENABLED = os.getenv(
    "POSTURE_ENABLED",
    "false"
).lower() == "true"

# ============================================================
# CONFIGURACOES GERAIS
# ============================================================

STATION_ID = os.getenv(
    "STATION_ID",
    "408c959c-fd2f-4096-8b36-6bcc65986394"
)

OPERATOR_ID = os.getenv(
    "OPERATOR_ID",
    "operador-001"
)

# ============================================================
# VIDEO USADO PELO MOTOR
# ============================================================

VIDEO_SOURCE = os.getenv(
    "VIDEO_SOURCE",
    "video_teste.mp4"
)

POSTURE_VIDEO_SOURCE = os.getenv(
    "POSTURE_VIDEO_SOURCE",
    "/workspaces/Factory-Vision-AI-/Camera de posturas.mp4"
)

DEVICE = os.getenv(
    "DEVICE",
    "cpu"
)

OBJECT_CONFIDENCE = float(
    os.getenv(
        "OBJECT_CONFIDENCE",
        "0.45"
    )
)

LOCAL_EVENT_FILE = os.getenv(
    "LOCAL_EVENT_FILE",
    "events.jsonl"
)

SUPABASE_TABLE = os.getenv(
    "SUPABASE_TABLE",
    "vision_events"
)

ENABLE_STREAMING = True
STREAM_PORT = 5000

# ============================================================
# CONFIGURACAO DO MOTOR
# ============================================================

MISSING_CONFIRMATIONS_REQUIRED = int(
    os.getenv(
        "MISSING_CONFIRMATIONS_REQUIRED",
        "3"
    )
)

START_CONFIRMATIONS_REQUIRED = int(
    os.getenv(
        "START_CONFIRMATIONS_REQUIRED",
        "2"
    )
)

# Intervalo (segundos) para o motor re-consultar as zonas no
# Supabase, sem precisar reiniciar o processo. 0 = desativado
# (zonas carregadas so' uma vez, na inicializacao).
ZONES_REFRESH_SECONDS = int(
    os.getenv(
        "ZONES_REFRESH_SECONDS",
        "60"
    )
)

# ============================================================
# LOG INICIAL
# ============================================================

print("=" * 70)
print("CONFIGURACOES CARREGADAS")
print("=" * 70)

print(
    f"Video Producao : {VIDEO_SOURCE}"
)

print(
    f"Video Postura  : {POSTURE_VIDEO_SOURCE}"
)

print(
    f"Station ID     : {STATION_ID}"
)

print(
    f"Operator ID    : {OPERATOR_ID}"
)

print(
    "Deteccao       : Local (MediaPipe + OpenCV, sem Roboflow)"
)

print(
    f"Postura ativa  : {POSTURE_ENABLED}"
)

print(
    f"Confianca      : {OBJECT_CONFIDENCE}"
)

print(
    f"Confirmacoes ausencia: "
    f"{MISSING_CONFIRMATIONS_REQUIRED}"
)

supabase_status = (
    "Configurado"
    if SUPABASE_URL and SUPABASE_KEY
    else "Offline"
)

print(
    f"Supabase       : {supabase_status}"
)

stream_status = (
    "Ativo"
    if ENABLE_STREAMING and FLASK_AVAILABLE
    else "Desativado"
)

print(
    f"Streaming      : {stream_status}"
)

print("=" * 70)

# ============================================================
# BUSCAR ESTACAO
# ============================================================

def get_station_config_by_id(station_uuid):

    if not supabase:
        return None

    try:

        stat_res = (
            supabase
            .table("stations")
            .select(
                "*, products(*), cameras!stations_camera_id_fkey(*)"
            )
            .eq(
                "id",
                station_uuid
            )
            .execute()
        )

        if not stat_res.data:

            print(
                f"Estacao ID {station_uuid} "
                f"nao encontrada no Supabase!"
            )

            return None

        station = stat_res.data[0]

        product = station.get(
            "products"
        )

        camera = station.get(
            "cameras"
        )

        if not product:

            prod_res = (
                supabase
                .table("products")
                .select("*")
                .limit(1)
                .execute()
            )

            if prod_res.data:

                product = prod_res.data[0]

                (
                    supabase
                    .table("stations")
                    .update(
                        {
                            "product_id":
                                product["id"]
                        }
                    )
                    .eq(
                        "id",
                        station["id"]
                    )
                    .execute()
                )

                station["product_id"] = product["id"]

            else:

                new_prod = (
                    supabase
                    .table("products")
                    .insert(
                        {
                            "name":
                                "Luva Padrao",

                            "code":
                                "LUVA-PADRAO",

                            "standard_time_seconds":
                                30
                        }
                    )
                    .execute()
                )

                if not new_prod.data:
                    return None

                product = new_prod.data[0]

                (
                    supabase
                    .table("stations")
                    .update(
                        {
                            "product_id":
                                product["id"]
                        }
                    )
                    .eq(
                        "id",
                        station["id"]
                    )
                    .execute()
                )

                station["product_id"] = product["id"]

        print(
            f"Estacao sincronizada: "
            f"{station['name']} | "
            f"Produto: {product['name']}"
        )

        return {
            "station":
                station,

            "product":
                product,

            "camera":
                camera
        }

    except Exception as e:

        print(
            f"Erro ao carregar estacao: {e}"
        )

        return None


print(
    f"Sincronizando configuracao da estacao: "
    f"{STATION_ID}"
)

config = get_station_config_by_id(
    STATION_ID
)

if (
    config
    and config.get("product")
    and config.get("station")
):

    product = config["product"]
    station = config["station"]

    camera_data = config.get(
        "camera"
    )

    if camera_data:

        CAMERA_NAME = camera_data.get(
            "name",
            "camera-setor"
        )

    else:

        CAMERA_NAME = "camera-setor"

else:

    print(
        "Usando fallback local."
    )

    product = {
        "id":
            str(uuid.uuid4()),

        "name":
            "Produto Local",

        "code":
            "LOCAL"
    }

    station = {
        "id":
            STATION_ID,

        "name":
            "Estacao Local",

        "code":
            "LOCAL"
    }

    CAMERA_NAME = "camera-local"

# ============================================================
# ZONAS DA ESTACAO (caixa de embalagem, caixa de descarte, etc.)
# ============================================================
# As zonas sao desenhadas visualmente no frontend
# (StationZonesPage) e salvas na tabela "zones" do Supabase, com
# coordinates em PORCENTAGEM (0..100) do frame: {x, y, width,
# height}. Aqui convertemos para fracao (0..1) e entregamos ao
# local_vision.py, que faz todo o calculo de pixels sozinho -
# assim funciona em qualquer resolucao de camera.

class ZoneRepository:

    def __init__(self, station_id):
        self.station_id = station_id

    def get_zones(self):

        if not supabase:
            return []

        try:

            result = (
                supabase
                .table("zones")
                .select("*")
                .eq(
                    "station_id",
                    self.station_id
                )
                .eq(
                    "status",
                    "ACTIVE"
                )
                .execute()
            )

            return result.data or []

        except Exception as e:

            print(
                f"Erro ao buscar zonas: {e}"
            )

            return []


def load_zones_for_local_vision(station_id):
    """Busca as zonas no Supabase e converte pra formato
    fracionario (0..1) que o local_vision.py usa pra calcular
    pixels em qualquer resolucao de frame."""

    zone_rows = ZoneRepository(station_id).get_zones()

    zones = []

    for row in zone_rows:

        coords = row.get("coordinates") or {}

        try:

            x1 = float(coords["x"]) / 100.0
            y1 = float(coords["y"]) / 100.0
            x2 = x1 + float(coords["width"]) / 100.0
            y2 = y1 + float(coords["height"]) / 100.0

        except (KeyError, TypeError, ValueError):

            print(
                f"Zona '{row.get('name')}' com coordinates "
                f"invalido, ignorando."
            )

            continue

        zones.append(
            {
                "id":
                    row.get("id"),

                "name":
                    row.get("name"),

                "zone_type":
                    row.get("zone_type"),

                "frac":
                    (x1, y1, x2, y2),
            }
        )

    if zones:

        print(
            f"{len(zones)} zona(s) carregada(s) do Supabase:"
        )

        for z in zones:

            print(
                f"   - {z['name']} ({z['zone_type']})"
            )

    else:

        print(
            "Nenhuma zona configurada no Supabase para esta "
            "estacao. Usando fallback do BOX_ROI do .env "
            "(se existir)."
        )

    return zones


zones = load_zones_for_local_vision(
    station["id"]
)

configure_zones(
    zones
)

# ============================================================
# CLASSES DA IA
# ============================================================

STAGE_NAMES = {
    0:
        "aguardando",

    1:
        "preparacao_dobra",

    2:
        "colocacao_caixa"
}

HAND_CLASSES = {
    "hand",
    "mao",
    "mão"
}

GLOVE_CLASSES = {
    "glove",
    "luva",
    "safety_glove",
    "luva_aberta",
    "luva_dobrada"
}

# Luva "nova/fresca", ainda nao processada — usada para permitir o
# INICIO de um novo ciclo. Note que "luva_dobrada" (a luva que ja foi
# entregue na caixa) NAO entra aqui de proposito: isso impede que a
# mesma luva, parada dentro da caixa, dispare um novo ciclo falso.
OPEN_GLOVE_CLASSES = {
    "glove",
    "luva",
    "safety_glove",
    "luva_aberta"
}

# Zonas do tipo "caixa de embalagem" — producao normal, boa.
# "caixa" e' mantido por compatibilidade com o fallback do
# BOX_ROI (.env), que emite essa classe generica quando nao ha'
# nenhuma zona configurada no Supabase.
PACKAGING_CLASSES = {
    "caixa_embalagem",
    "caixa"
}

# Zonas do tipo "caixa de descarte" — peca rejeitada, nao entra
# na contagem de producao (vira evento "item_descartado").
DISCARD_CLASSES = {
    "caixa_descarte"
}

# Mao detectada dentro/perto da ROI de QUALQUER zona de caixa —
# sinal de que o operador ainda esta segurando ou posicionando a
# luva dentro da caixa. So' quando esse sinal SOME por alguns
# frames consecutivos e' que a producao e' confirmada (mao se
# afastou).
HAND_IN_BOX_CLASSES = {
    "mao_na_caixa"
}

# ============================================================
# DETECTION
# ============================================================

@dataclass
class Detection:

    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]

    @property
    def center(self):

        x1, y1, x2, y2 = self.bbox

        return (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )


def get_detections_by_classes(
    detections,
    accepted_classes
):

    return [
        detection
        for detection in detections
        if detection.class_name in accepted_classes
    ]

# ============================================================
# EVENT WRITER
# ============================================================

class EventWriter:

    def __init__(
        self,
        local_file,
        supabase_url,
        supabase_key,
        supabase_table,
        product_id,
        station_id
    ):

        self.local_file = Path(
            local_file
        )

        self.local_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.supabase_enabled = bool(
            supabase_url
            and supabase_key
        )

        if self.supabase_enabled:

            self.supabase_endpoint = (
                f"{supabase_url}"
                f"/rest/v1/"
                f"{supabase_table}"
            )

        else:

            self.supabase_endpoint = ""

        self.supabase_headers = {
            "apikey":
                supabase_key,

            "Authorization":
                f"Bearer {supabase_key}",

            "Content-Type":
                "application/json",

            "Prefer":
                "return=minimal"
        }

        self.product_id = product_id
        self.station_id = station_id

        self.event_queue = queue.Queue()

        self.running = True

        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )

        self.worker.start()

    def write(
        self,
        event_type,
        payload,
        custom_camera_name=None,
        severity="info",
        current_production_id=None
    ):

        cam_to_use = (
            custom_camera_name
            or CAMERA_NAME
        )

        occurred_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        local_event = {

            "event_id":
                str(uuid.uuid4()),

            "camera_id":
                cam_to_use,

            "operator_id":
                OPERATOR_ID,

            "event_type":
                event_type,

            "severity":
                severity,

            "occurred_at":
                occurred_at,

            "payload":
                payload
        }

        try:

            with self.local_file.open(
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    json.dumps(
                        local_event,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        except Exception as e:

            print(
                f"Erro ao salvar evento local: {e}"
            )

        print(
            f"[EVENTO - {cam_to_use}] "
            f"{event_type}"
        )

        if not self.supabase_enabled:
            return

        supabase_event = {

            "event_type":
                event_type,

            "event_timestamp":
                occurred_at,

            "station_id":
                self.station_id,

            "object_detected":
                event_type,

            "confidence":
                1.0,

            "additional_data":
                payload
        }

        if current_production_id:

            supabase_event[
                "production_id"
            ] = current_production_id

        self.event_queue.put(
            supabase_event
        )

    def _worker_loop(self):

        while (
            self.running
            or not self.event_queue.empty()
        ):

            try:

                event = self.event_queue.get(
                    timeout=0.5
                )

            except queue.Empty:

                continue

            try:

                response = requests.post(
                    self.supabase_endpoint,
                    headers=self.supabase_headers,
                    json=event,
                    timeout=5
                )

                response.raise_for_status()

            except requests.exceptions.HTTPError as e:

                if e.response is not None:

                    print(
                        f"Supabase HTTP "
                        f"{e.response.status_code}: "
                        f"{e.response.text[:500]}"
                    )

                else:

                    print(
                        f"Supabase HTTP error: {e}"
                    )

            except Exception as e:

                print(
                    f"Supabase erro: {e}"
                )

            finally:

                self.event_queue.task_done()

    def close(self):

        self.running = False

        self.worker.join(
            timeout=3
        )

# ============================================================
# DESENHAR DETECCOES / ZONAS
# ============================================================

def draw_zones(frame):
    """Desenha TODAS as zonas configuradas (embalagem, descarte,
    etc.), lidas do Supabase, para calibracao visual no
    streaming. Cores diferentes por zone_type."""

    for (x1, y1, x2, y2), zone_type, name in get_zones_pixels(
        frame.shape
    ):

        if zone_type == "DISCARD_BOX":

            color = (
                0,
                0,
                255
            )

        elif zone_type == "PACKAGING_BOX":

            color = (
                0,
                165,
                255
            )

        else:

            color = (
                255,
                255,
                0
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            name or zone_type,
            (x1 + 4, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )


def draw_detections(
    frame,
    detections
):

    for detection in detections:

        x1, y1, x2, y2 = (
            detection.bbox
        )

        if detection.class_name in HAND_CLASSES:

            color = (
                255,
                80,
                80
            )

        elif detection.class_name in GLOVE_CLASSES:

            color = (
                0,
                220,
                255
            )

        elif detection.class_name in PACKAGING_CLASSES:

            color = (
                255,
                0,
                255
            )

        elif detection.class_name in DISCARD_CLASSES:

            color = (
                0,
                0,
                255
            )

        else:

            color = (
                180,
                180,
                180
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(
                    20,
                    y1 - 6
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA
        )

# ============================================================
# REPOSITORIO DE PRODUCAO
# ============================================================

class ProductionRepository:

    def __init__(
        self,
        product_id,
        station_id
    ):

        self.product_id = product_id
        self.station_id = station_id

    def create_production(self):

        if not supabase:
            return None

        try:

            production_data = {

                "product_id":
                    self.product_id,

                "station_id":
                    self.station_id,

                "production_date":
                    datetime.now().date().isoformat(),

                "start_time":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "status":
                    "IN_PROGRESS",

                "error_count":
                    0,

                "executed_steps":
                    0,

                "skipped_steps":
                    0
            }

            result = (
                supabase
                .table("productions")
                .insert(production_data)
                .execute()
            )

            if result.data:

                production_id = (
                    result.data[0]["id"]
                )

                print(
                    f"Producao criada: "
                    f"{production_id}"
                )

                return production_id

        except Exception as e:

            print(
                f"Erro ao criar producao: {e}"
            )

        return None

    def get_product_steps(self):

        if not supabase:
            return []

        try:

            result = (
                supabase
                .table("production_steps")
                .select("*")
                .eq(
                    "product_id",
                    self.product_id
                )
                .order(
                    "step_order"
                )
                .execute()
            )

            return result.data or []

        except Exception as e:

            print(
                f"Erro ao buscar production_steps: {e}"
            )

            return []

    def create_step_execution(
        self,
        production_id,
        step,
        executed_order,
        start_time
    ):

        if not supabase:
            return None

        try:

            data = {

                "production_id":
                    production_id,

                "step_id":
                    step["id"],

                "expected_order":
                    step["step_order"],

                "executed_order":
                    executed_order,

                "start_time":
                    start_time,

                "standard_time_seconds":
                    step[
                        "standard_time_seconds"
                    ],

                "status":
                    "IN_PROGRESS"
            }

            result = (
                supabase
                .table(
                    "production_step_executions"
                )
                .insert(data)
                .execute()
            )

            if result.data:

                return result.data[0]["id"]

        except Exception as e:

            print(
                f"Erro ao criar execucao da etapa: {e}"
            )

        return None

    def complete_step_execution(
        self,
        execution_id,
        end_time,
        duration_seconds,
        status="COMPLETED"
    ):

        if (
            not supabase
            or not execution_id
        ):

            return

        try:

            (
                supabase
                .table(
                    "production_step_executions"
                )
                .update(
                    {
                        "end_time":
                            end_time,

                        "duration_seconds":
                            int(
                                duration_seconds
                            ),

                        "status":
                            status
                    }
                )
                .eq(
                    "id",
                    execution_id
                )
                .execute()
            )

        except Exception as e:

            print(
                f"Erro ao finalizar execucao da etapa: {e}"
            )

    def complete_production(
        self,
        production_id,
        start_monotonic
    ):

        if (
            not supabase
            or not production_id
        ):

            return

        try:

            total_duration = int(
                time.monotonic()
                - start_monotonic
            )

            (
                supabase
                .table("productions")
                .update(
                    {
                        "end_time":
                            datetime.now(
                                timezone.utc
                            ).isoformat(),

                        "status":
                            "COMPLETED",

                        "total_time_seconds":
                            total_duration
                    }
                )
                .eq(
                    "id",
                    production_id
                )
                .execute()
            )

            print(
                f"Producao {production_id} "
                f"COMPLETADA em "
                f"{total_duration}s"
            )

        except Exception as e:

            print(
                f"Erro ao finalizar producao: {e}"
            )

# ============================================================
# CONTROLE DO CICLO
# ============================================================

class CycleManager:

    def __init__(
        self,
        event_writer,
        product_id,
        station_id,
        product_name
    ):

        self.event_writer = event_writer

        self.product_id = product_id

        self.station_id = station_id

        self.product_name = product_name

        self.repository = ProductionRepository(
            product_id,
            station_id
        )

        self.current_stage = 0

        self.cycle_number = 0

        self.total_gloves = 0

        self.total_discarded = 0

        self.production_start_monotonic = None

        self.current_production_id = None

        self.current_step_execution_id = None

        self.current_step_start_monotonic = None

        self.steps = []

        self.missing_confirmations = 0

        self.start_confirmations = 0

        self.last_stage_change = 0

        # "OK" = vai para caixa de embalagem (producao valida).
        # "DESCARTE" = vai para caixa de descarte (peca rejeitada).
        # Decidido na transicao ETAPA 1 -> 2, conforme QUAL zona
        # recebeu a luva dobrada, e consumido na ETAPA 2 -> ciclo.
        self.pending_outcome = "OK"

        self._load_steps()

    def _load_steps(self):

        self.steps = (
            self.repository
            .get_product_steps()
        )

        if not self.steps:

            print(
                "Nenhuma production_step encontrada."
            )

            print(
                "O MOTOR funcionara, "
                "mas nao gravara "
                "production_step_executions."
            )

            return

        print(
            f"{len(self.steps)} etapas carregadas."
        )

        for step in self.steps:

            print(
                f"   {step['step_order']} - "
                f"{step['name']} | "
                f"padrao: "
                f"{step['standard_time_seconds']}s"
            )

    def _start_production(self):

        if self.current_production_id:
            return

        self.production_start_monotonic = (
            time.monotonic()
        )

        self.current_production_id = (
            self.repository
            .create_production()
        )

        if self.current_production_id:

            print(
                f"PRODUCAO REAL INICIADA: "
                f"{self.current_production_id}"
            )

            if self.event_writer:

                self.event_writer.write(
                    "production_started",
                    {
                        "production_id":
                            self.current_production_id,

                        "product_id":
                            self.product_id,

                        "product_name":
                            self.product_name
                    },
                    current_production_id=
                        self.current_production_id
                )

    def _start_step(
        self,
        stage_number,
        current_time
    ):

        self.current_step_start_monotonic = (
            time.monotonic()
        )

        self.current_step_execution_id = None

        if self.steps:

            step = next(
                (
                    s
                    for s in self.steps
                    if s["step_order"]
                    == stage_number
                ),
                None
            )

            if step:

                self.current_step_execution_id = (
                    self.repository
                    .create_step_execution(
                        self.current_production_id,
                        step,
                        stage_number,
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    )
                )

                print(
                    f"ETAPA {stage_number}: "
                    f"{step['name']}"
                )

        if self.event_writer:

            stage_name = STAGE_NAMES.get(
                stage_number,
                "desconhecida"
            )

            self.event_writer.write(
                "stage_started",
                {
                    "stage":
                        stage_number,

                    "stage_name":
                        stage_name
                },
                current_production_id=
                    self.current_production_id
            )

    def _complete_step(
        self,
        stage_number
    ):

        if (
            self.current_step_start_monotonic
            is None
        ):

            return

        duration = (
            time.monotonic()
            - self.current_step_start_monotonic
        )

        end_time = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        if self.current_step_execution_id:

            self.repository.complete_step_execution(
                self.current_step_execution_id,
                end_time,
                duration,
                "COMPLETED"
            )

        if self.event_writer:

            stage_name = STAGE_NAMES.get(
                stage_number,
                "desconhecida"
            )

            self.event_writer.write(
                "stage_completed",
                {
                    "stage":
                        stage_number,

                    "stage_name":
                        stage_name,

                    "duration_seconds":
                        round(
                            duration,
                            2
                        )
                },
                current_production_id=
                    self.current_production_id
            )

        print(
            f"ETAPA {stage_number} "
            f"finalizada em "
            f"{duration:.2f}s"
        )

        self.current_step_execution_id = None

        self.current_step_start_monotonic = None

    def _complete_production(self):

        if not self.current_production_id:
            return

        production_id = (
            self.current_production_id
        )

        self.repository.complete_production(
            production_id,
            self.production_start_monotonic
        )

        if self.event_writer:

            self.event_writer.write(
                "production_completed",
                {
                    "production_id":
                        production_id,

                    "cycle_number":
                        self.cycle_number,

                    "total_produced":
                        self.total_gloves,

                    "total_discarded":
                        self.total_discarded
                },
                current_production_id=
                    production_id
            )

        self.current_production_id = None

        self.production_start_monotonic = None

    def process(
        self,
        detections,
        current_time
    ):

        # --------------------------------------------------------
        # Sinais brutos extraidos das deteccoes deste frame
        # --------------------------------------------------------

        hands = get_detections_by_classes(
            detections,
            HAND_CLASSES
        )

        open_gloves = get_detections_by_classes(
            detections,
            OPEN_GLOVE_CLASSES
        )

        packaging_hits = get_detections_by_classes(
            detections,
            PACKAGING_CLASSES
        )

        discard_hits = get_detections_by_classes(
            detections,
            DISCARD_CLASSES
        )

        hand_in_box = get_detections_by_classes(
            detections,
            HAND_IN_BOX_CLASSES
        )

        has_hand = len(hands) > 0

        has_open_glove = len(open_gloves) > 0

        # "luva DOBRADA confirmada DENTRO de uma zona de caixa" —
        # emitido pelo local_vision.py somente apos a luva ja ter
        # passado por luva_aberta -> luva_dobrada e o centro dela
        # estar dentro de UMA das zonas configuradas no Supabase.
        # Cada tipo de zona (embalagem / descarte) gera uma classe
        # diferente, tratada separadamente aqui.
        has_folded_glove_in_packaging = len(packaging_hits) > 0

        has_folded_glove_in_discard = len(discard_hits) > 0

        # "a mao ainda esta dentro/perto de alguma caixa" — sinal
        # de que o operador esta segurando/posicionando a luva,
        # ainda NAO soltou. So' quando esse sinal sumir por N
        # frames seguidos e' que consideramos a luva efetivamente
        # solta e a mao afastada.
        has_hand_in_box = len(hand_in_box) > 0

        # Usado apenas para permitir o INICIO de um novo ciclo
        # (etapa 0). Note que propositalmente NAO inclui
        # "luva_dobrada": a luva que acabou de ser entregue na
        # caixa continua visivel ali, mas nao pode disparar um
        # novo ciclo sozinha — precisa aparecer uma luva NOVA
        # (aberta) e/ou uma mao.
        has_start_activity = (
            has_hand
            or has_open_glove
        )

        # ==========================================================
        # ETAPA 0 -> 1: uma mao e/ou luva aberta (nova, fora da
        # caixa) aparece em cena de forma sustentada.
        # ==========================================================

        if self.current_stage == 0:

            if has_start_activity:

                self.start_confirmations += 1

            else:

                self.start_confirmations = 0

            if (
                self.start_confirmations
                >= START_CONFIRMATIONS_REQUIRED
            ):

                self.start_confirmations = 0

                self._start_production()

                if not self.current_production_id:

                    return (
                        self.current_stage,
                        self.cycle_number,
                        self.total_gloves
                    )

                self.current_stage = 1

                self.last_stage_change = (
                    current_time
                )

                self._start_step(
                    1,
                    current_time
                )

                print(
                    "[ETAPA 1] "
                    "Preparacao/dobra iniciada."
                )

        # ==========================================================
        # ETAPA 1 -> 2: a luva DOBRADA entrou em alguma zona de
        # caixa (embalagem OU descarte). Guardamos qual foi, para
        # decidir o desfecho do ciclo mais adiante.
        # ==========================================================

        elif self.current_stage == 1:

            if (
                has_folded_glove_in_packaging
                or has_folded_glove_in_discard
            ):

                self.pending_outcome = (
                    "DESCARTE"
                    if has_folded_glove_in_discard
                    else "OK"
                )

                self._complete_step(
                    1
                )

                self.current_stage = 2

                self.last_stage_change = (
                    current_time
                )

                self.missing_confirmations = 0

                self._start_step(
                    2,
                    current_time
                )

                destino = (
                    "caixa de DESCARTE"
                    if self.pending_outcome == "DESCARTE"
                    else "caixa de EMBALAGEM"
                )

                print(
                    f"[ETAPA 2] Luva dobrada entrou na "
                    f"{destino}. Aguardando operador soltar "
                    f"e afastar a mao."
                )

        # ==========================================================
        # ETAPA 2 -> CICLO CONFIRMADO: a producao so' e' confirmada
        # quando a mao deixa de estar na regiao da caixa por
        # MISSING_CONFIRMATIONS_REQUIRED frames seguidos — ou seja,
        # o operador realmente soltou a luva e afastou a mao.
        # Isso evita contar a mesma luva varias vezes so' porque ela
        # continua visivel, parada, dentro da caixa.
        # ==========================================================

        elif self.current_stage == 2:

            if has_hand_in_box:

                # Operador ainda segurando/posicionando a luva.
                self.missing_confirmations = 0

            else:

                self.missing_confirmations += 1

            if (
                self.missing_confirmations
                >= MISSING_CONFIRMATIONS_REQUIRED
            ):

                self.missing_confirmations = 0

                self._complete_step(
                    2
                )

                self.cycle_number += 1

                if self.pending_outcome == "DESCARTE":

                    self.total_discarded += 1

                    print(
                        "1 PECA DESCARTADA "
                        "(dobrada -> caixa de descarte -> "
                        "mao afastada)"
                    )

                    print(
                        f"Total descartado: "
                        f"{self.total_discarded}"
                    )

                    if self.event_writer:

                        self.event_writer.write(
                            "item_descartado",
                            {
                                "cycle_number":
                                    self.cycle_number,

                                "total_discarded":
                                    self.total_discarded,

                                "product_name":
                                    self.product_name,

                                "station_id":
                                    self.station_id
                            },
                            current_production_id=
                                self.current_production_id
                        )

                else:

                    self.total_gloves += 1

                    print(
                        "1 LUVA PRODUZIDA "
                        "(dobrada -> caixa de embalagem -> "
                        "mao afastada)"
                    )

                    print(
                        f"Total produzido: "
                        f"{self.total_gloves}"
                    )

                    if self.event_writer:

                        cycle_duration = None

                        if (
                            self.production_start_monotonic
                            is not None
                        ):

                            cycle_duration = round(
                                time.monotonic()
                                - self.production_start_monotonic,
                                2
                            )

                        self.event_writer.write(
                            "cycle_completed",
                            {
                                "cycle_number":
                                    self.cycle_number,

                                "total_produced":
                                    self.total_gloves,

                                "product_name":
                                    self.product_name,

                                "station_id":
                                    self.station_id,

                                "cycle_duration_seconds":
                                    cycle_duration
                            },
                            current_production_id=
                                self.current_production_id
                        )

                self.pending_outcome = "OK"

                # Volta para AGUARDANDO. Um novo ciclo so' comeca
                # quando uma NOVA luva (aberta, fora da caixa) for
                # detectada — ver has_start_activity acima.
                self._complete_production()

                self.current_stage = 0

                self.last_stage_change = (
                    current_time
                )

        return (
            self.current_stage,
            self.cycle_number,
            self.total_gloves
        )

# ============================================================
# ESTADO GLOBAL
# ============================================================

global_status = {

    "prod_stage":
        0,

    "prod_cycles":
        0,

    "prod_total":
        0,

    "latest_prod_detections":
        [],

    "posture_detected":
        "Normal",

    "last_production_id":
        None
}

# ============================================================
# THREAD DE ATUALIZACAO DE ZONAS (recarrega do Supabase de
# tempos em tempos, sem precisar reiniciar o motor)
# ============================================================

def zones_refresh_worker():

    if ZONES_REFRESH_SECONDS <= 0:
        return

    while True:

        time.sleep(
            ZONES_REFRESH_SECONDS
        )

        updated_zones = load_zones_for_local_vision(
            station["id"]
        )

        configure_zones(
            updated_zones
        )

# ============================================================
# WORKER PRODUCAO
# ============================================================

def production_worker(
    event_writer,
    cycle_manager
):

    cap = cv2.VideoCapture(
        VIDEO_SOURCE
    )

    if not cap.isOpened():

        print(
            f"Nao foi possivel abrir "
            f"o video: {VIDEO_SOURCE}"
        )

        return

    frame_count = 0

    print(
        f"Thread de Producao iniciada: "
        f"{VIDEO_SOURCE}"
    )

    while cap.isOpened():

        success, frame = cap.read()

        if not success:

            print(
                "Fim do video. Reiniciando..."
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            continue

        frame_count += 1

        current_time = time.monotonic()

        if frame_count % 2 != 0:

            time.sleep(0.01)

            continue

        detections = detect_local(frame)

        detected_classes = [
            d.class_name
            for d in detections
        ]

        print(
            f"[frame {frame_count}] "
            f"deteccao local: "
            f"{detected_classes}"
        )

        global_status[
            "latest_prod_detections"
        ] = detections

        (
            stage,
            cycle,
            total_produced
        ) = cycle_manager.process(
            detections,
            current_time
        )

        global_status[
            "prod_stage"
        ] = stage

        global_status[
            "prod_cycles"
        ] = cycle

        global_status[
            "prod_total"
        ] = total_produced

        global_status[
            "last_production_id"
        ] = (
            cycle_manager.current_production_id
        )

        time.sleep(0.03)

# ============================================================
# WORKER POSTURA
# ============================================================

def posture_worker(
    event_writer
):

    # A analise de postura ainda nao foi migrada para visao local
    # (era feita via Roboflow Workflow, agora removido). Decisao do
    # usuario: resolver a producao/dobra primeiro e avaliar depois
    # se/como portar a postura (ex.: MediaPipe Pose). Para nao
    # quebrar nada, a thread so' roda de fato se POSTURE_ENABLED=true
    # no .env, e mesmo assim so' exibe o video sem inferencia de IA.

    if not POSTURE_ENABLED:

        print(
            "Thread de Postura desativada "
            "(POSTURE_ENABLED=false). "
            "A deteccao de postura via Roboflow foi removida e "
            "ainda nao tem substituto local — defina "
            "POSTURE_ENABLED=true no .env quando quiser reativar "
            "so' o streaming do video de postura."
        )

        global_status["posture_detected"] = "Desativado (sem Roboflow)"

        return

    cap = cv2.VideoCapture(
        POSTURE_VIDEO_SOURCE
    )

    if not cap.isOpened():

        print(
            f"Nao foi possivel abrir "
            f"video de postura: "
            f"{POSTURE_VIDEO_SOURCE}"
        )

        return

    print(
        f"Thread de Postura iniciada (somente streaming, "
        f"sem inferencia de IA): "
        f"{POSTURE_VIDEO_SOURCE}"
    )

    global_status["posture_detected"] = "Sem analise de IA (a implementar)"

    while cap.isOpened():

        success, frame = cap.read()

        if not success:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            continue

        time.sleep(0.05)

# ============================================================
# STREAMING
# ============================================================

def start_streaming():

    if not FLASK_AVAILABLE:
        return

    app = Flask(
        __name__
    )

    cap_prod = cv2.VideoCapture(
        VIDEO_SOURCE
    )

    cap_posture = cv2.VideoCapture(
        POSTURE_VIDEO_SOURCE
    )

    def generate_prod_frames():

        while True:

            success, frame = (
                cap_prod.read()
            )

            if not success:

                cap_prod.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )

                continue

            stage = global_status[
                "prod_stage"
            ]

            cycle = global_status[
                "prod_cycles"
            ]

            total_produced = global_status[
                "prod_total"
            ]

            detections = global_status[
                "latest_prod_detections"
            ]

            draw_detections(
                frame,
                detections
            )

            draw_zones(frame)

            cv2.putText(
                frame,
                f"Factory AI - Produzidas: {total_produced}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            stage_name = STAGE_NAMES.get(
                stage,
                "AGUARDANDO"
            )

            cv2.putText(
                frame,
                f"Etapa Atual: {stage_name}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 170, 0),
                2
            )

            cv2.putText(
                frame,
                f"Ciclos: {cycle}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            ret, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    def generate_posture_frames():

        while True:

            success, frame = (
                cap_posture.read()
            )

            if not success:

                cap_posture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )

                continue

            posture = global_status[
                "posture_detected"
            ]

            cv2.putText(
                frame,
                f"Postura (Ergonomia): {posture}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 100, 255),
                2
            )

            ret, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    @app.route("/")
    def index():

        return """
        <html>
        <head>
            <title>Factory Vision AI</title>
        </head>

        <body
            style="
                background:#0d1117;
                color:white;
                text-align:center;
                font-family:sans-serif;
            "
        >

            <h1>
                Factory Vision AI
            </h1>

            <div
                style="
                    display:flex;
                    justify-content:center;
                    gap:20px;
                    flex-wrap:wrap;
                    margin-top:20px;
                "
            >

                <div>

                    <h3>
                        Camera de Producao
                    </h3>

                    <img
                        src="/video_feed_prod"
                        width="600"
                        style="
                            border:2px solid #3b82f6;
                            border-radius:8px;
                        "
                    >

                </div>

                <div>

                    <h3>
                        Camera de Postura
                    </h3>

                    <img
                        src="/video_feed_posture"
                        width="600"
                        style="
                            border:2px solid #a855f7;
                            border-radius:8px;
                        "
                    >

                </div>

            </div>

        </body>
        </html>
        """

    @app.route(
        "/video_feed_prod"
    )
    def video_feed_prod():

        return Response(
            generate_prod_frames(),
            mimetype=
                "multipart/x-mixed-replace; boundary=frame"
        )

    @app.route(
        "/video_feed_posture"
    )
    def video_feed_posture():

        return Response(
            generate_posture_frames(),
            mimetype=
                "multipart/x-mixed-replace; boundary=frame"
        )

    print(
        f"Streaming disponivel em "
        f"http://localhost:{STREAM_PORT}"
    )

    app.run(
        host="0.0.0.0",
        port=STREAM_PORT,
        debug=False,
        use_reloader=False
    )

# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nINICIANDO FACTORY VISION AI"
    )

    print(
        f"Fonte da IA: {VIDEO_SOURCE}"
    )

    print(
        "Deteccao: Local (MediaPipe + OpenCV) - sem Roboflow"
    )

    print(
        "Backend: Supabase"
    )

    event_writer = EventWriter(
        LOCAL_EVENT_FILE,
        SUPABASE_URL,
        SUPABASE_KEY,
        SUPABASE_TABLE,
        product["id"],
        station["id"]
    )

    cycle_manager = CycleManager(
        event_writer,
        product["id"],
        station["id"],
        product["name"]
    )

    threading.Thread(
        target=production_worker,
        args=(
            event_writer,
            cycle_manager
        ),
        daemon=True
    ).start()

    threading.Thread(
        target=posture_worker,
        args=(
            event_writer,
        ),
        daemon=True
    ).start()

    threading.Thread(
        target=zones_refresh_worker,
        daemon=True
    ).start()

    if (
        ENABLE_STREAMING
        and FLASK_AVAILABLE
    ):

        threading.Thread(
            target=start_streaming,
            daemon=True
        ).start()

        time.sleep(2)

    event_writer.write(
        "camera_started",
        {
            "video_source":
                VIDEO_SOURCE,

            "posture_source":
                POSTURE_VIDEO_SOURCE,

            "device":
                DEVICE,

            "detection":
                "Local (MediaPipe + OpenCV)",

            "backend":
                "Supabase"
        }
    )

    try:

        while True:

            time.sleep(1)

    except KeyboardInterrupt:

        print(
            "\nMOTOR encerrado."
        )

    finally:

        event_writer.close()


if __name__ == "__main__":
    main()