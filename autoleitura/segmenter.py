"""Segmentação da janela dos dígitos de hidrômetros com YOLOv8-seg.

Hidrômetros aparecem girados na foto (caixas no chão, foto de qualquer lado). Uma caixa reta
pegaria a janela torta e com sobra; aqui a janela é segmentada e endireitada, que é como foram
feitos os recortes de treino da CRNN de água.
"""
import cv2
from ultralytics import YOLO

from . import config
from .preprocess import endireitar


class WindowSegmenter:
    def __init__(self, weights=config.AGUA_SEG_WEIGHTS, conf=0.25):
        self.model = YOLO(str(weights))
        self.conf = conf

    def segment(self, img):
        """Retorna (recorte endireitado, confiança, caixa (x1, y1, x2, y2)) ou (None, 0.0, None)."""
        r = self.model.predict(img, imgsz=config.DET_IMG_SIZE, conf=self.conf, verbose=False)[0]
        if r.masks is None or len(r.boxes) == 0:
            return None, 0.0, None
        i = int(r.boxes.conf.argmax())
        poligono = r.masks.xy[i]                       # contorno em coordenadas da imagem
        if len(poligono) < 3:
            return None, 0.0, None
        x, y, w, h = cv2.boundingRect(poligono.astype("float32"))
        return endireitar(img, poligono), float(r.boxes.conf[i]), (x, y, x + w, y + h)
