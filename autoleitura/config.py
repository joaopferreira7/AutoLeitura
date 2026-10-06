"""Configurações centrais do AutoLeitura.

Todos os caminhos são relativos à raiz do projeto, para funcionar igual
no notebook local e no Google Colab.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # dataset original (UFPR-AMR, Roboflow, sintético)
DET_DIR = DATA_DIR / "det"            # dataset no formato YOLO para o detector do contador
REC_DIR = DATA_DIR / "rec"            # recortes do contador + labels.csv para o reconhecedor

WEIGHTS_DIR = ROOT / "weights"
DET_WEIGHTS = WEIGHTS_DIR / "detector.pt"
REC_WEIGHTS = WEIGHTS_DIR / "recognizer.pt"

DB_PATH = ROOT / "autoleitura.db"

# Detector (YOLOv8)
DET_IMG_SIZE = 640
DET_BASE_MODEL = "yolov8n.pt"         # 'n' é leve o suficiente p/ CPU; troque por yolov8s.pt se tiver GPU

# Reconhecedor (CRNN)
REC_HEIGHT = 32
REC_WIDTH = 160
ALPHABET = "0123456789"               # índice 0 do CTC é o "blank"; dígitos são 1..10
NUM_DIGITS = 5                        # contadores do UFPR-AMR têm 5 dígitos

# Margem adicionada ao recorte do contador antes do reconhecimento
CROP_MARGIN = 0.08
