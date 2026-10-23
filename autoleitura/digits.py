"""Leitura dos dígitos com YOLOv8 de 10 classes sobre o recorte do contador.

Complementa a CRNN: aqui cada dígito é uma caixa com classe (0–9), e a leitura é a
sequência das classes ordenadas da esquerda para a direita.
"""
import numpy as np
from ultralytics import YOLO

from . import config


class DigitReader:
    def __init__(self, weights=config.DIG_WEIGHTS, conf=0.1):
        self.model = YOLO(str(weights))
        self.conf = conf

    def read(self, crop):
        """Recorte BGR do contador -> (leitura, confiança, confiança por dígito)."""
        # agnostic_nms: num rolete no meio da troca, o modelo pode propor duas classes para o mesmo
        # lugar (ex.: 6 e 7); fica só a caixa de maior confiança
        r = self.model.predict(crop, imgsz=config.DIG_IMG_SIZE, conf=self.conf, agnostic_nms=True, verbose=False)[0]
        dets = [(float(b[0] + b[2]) / 2, int(c), float(p))
                for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        if len(dets) > config.NUM_DIGITS:              # sobra caixa: fica com as mais confiáveis
            dets = sorted(dets, key=lambda d: -d[2])[:config.NUM_DIGITS]
        dets.sort(key=lambda d: d[0])
        text = "".join(str(c) for _, c, _ in dets)
        confs = [p for _, _, p in dets]
        return text, (float(np.prod(confs)) if confs else 0.0), confs
