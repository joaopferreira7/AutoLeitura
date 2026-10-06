"""Detecção da região do contador com YOLOv8 (etapa 3 da arquitetura)."""
from ultralytics import YOLO

from . import config


class CounterDetector:
    def __init__(self, weights=config.DET_WEIGHTS, conf=0.25):
        self.model = YOLO(str(weights))
        self.conf = conf

    def detect(self, img):
        """Retorna a caixa de maior confiança (x1, y1, x2, y2) e a confiança, ou (None, 0.0)."""
        result = self.model.predict(img, imgsz=config.DET_IMG_SIZE, conf=self.conf, verbose=False)[0]
        if len(result.boxes) == 0:
            return None, 0.0
        best = int(result.boxes.conf.argmax())
        box = result.boxes.xyxy[best].tolist()
        return tuple(box), float(result.boxes.conf[best])
