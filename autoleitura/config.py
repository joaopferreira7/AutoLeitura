"""Configurações centrais do AutoLeitura.

Todos os caminhos são relativos à raiz do projeto, para funcionar igual
no notebook local e no Google Colab.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # dataset original (UFPR-AMR, Roboflow, sintético)
DET_DIR = DATA_DIR / "det"            # dataset no formato YOLO para o detector do contador
REC_DIR = DATA_DIR / "rec"            # recortes do contador + labels.csv para o reconhecedor
DIG_DIR = DATA_DIR / "dig"            # recortes do contador no formato YOLO, um box por dígito (classes 0–9)

WEIGHTS_DIR = ROOT / "weights"
DET_WEIGHTS = WEIGHTS_DIR / "detector.pt"
REC_WEIGHTS = WEIGHTS_DIR / "recognizer.pt"
DIG_WEIGHTS = WEIGHTS_DIR / "digits.pt"

# AUTOLEITURA_DB permite usar outro arquivo (ex.: um banco de demonstração para a apresentação)
DB_PATH = Path(os.environ.get("AUTOLEITURA_DB", ROOT / "autoleitura.db"))

# Detector (YOLOv8)
DET_IMG_SIZE = 640
DET_BASE_MODEL = "yolov8n.pt"         # 'n' é leve o suficiente p/ CPU; troque por yolov8s.pt se tiver GPU

# Leitor de dígitos (YOLOv8 com 10 classes, roda sobre o recorte do contador)
DIG_IMG_SIZE = 320

# Reconhecedor (CRNN)
REC_HEIGHT = 32
REC_WIDTH = 160
ALPHABET = "0123456789"               # índice 0 do CTC é o "blank"; dígitos são 1..10
NUM_DIGITS = 5                        # contadores do UFPR-AMR têm 5 dígitos

# Margem adicionada ao recorte do contador antes do reconhecimento
CROP_MARGIN = 0.08
