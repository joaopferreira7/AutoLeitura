"""Pipeline completo: foto -> pré-processamento -> detecção -> recorte -> reconhecimento -> leitura."""
import time

import cv2

from . import config
from .detector import CounterDetector
from .preprocess import crop_box, limit_size
from .recognizer import Recognizer


class AutoLeitura:
    def __init__(self, det_weights=config.DET_WEIGHTS, rec_weights=config.REC_WEIGHTS):
        self.detector = CounterDetector(det_weights)
        self.recognizer = Recognizer(rec_weights)

    def read(self, img):
        """Recebe uma imagem BGR e devolve um dicionário com a leitura e as informações de apoio."""
        t0 = time.perf_counter()
        small, scale = limit_size(img)
        box, det_conf = self.detector.detect(small)

        if box is None:
            return {"reading": None, "confidence": 0.0, "det_confidence": 0.0, "bbox": None,
                    "digit_confidences": [], "crop": None,
                    "time_ms": (time.perf_counter() - t0) * 1000,
                    "warning": "Contador não encontrado na imagem."}

        box = tuple(v / scale for v in box)                 # volta para a escala original
        crop = crop_box(img, box)
        text, rec_conf, digit_confs = self.recognizer.read(crop)

        # Pós-processamento: confiança final combina detecção e reconhecimento;
        # leituras com quantidade de dígitos inesperada são sinalizadas.
        confidence = det_conf * rec_conf
        warning = None
        if len(text) != config.NUM_DIGITS:
            warning = f"Foram lidos {len(text)} dígitos (esperado {config.NUM_DIGITS}). Confira a leitura."
            confidence *= 0.5

        return {
            "reading": text,
            "confidence": confidence,
            "det_confidence": det_conf,
            "bbox": [round(v, 1) for v in box],
            "digit_confidences": digit_confs,
            "crop": crop,
            "time_ms": (time.perf_counter() - t0) * 1000,
            "warning": warning,
        }


def draw_result(img, result):
    """Desenha a caixa do contador e a leitura sobre a imagem (para exibir na interface)."""
    out = img.copy()
    if result["bbox"] is None:
        return out
    x1, y1, x2, y2 = map(int, result["bbox"])
    thick = max(2, img.shape[1] // 300)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), thick)
    label = f"{result['reading']}  ({result['confidence'] * 100:.0f}%)"
    scale = max(0.6, img.shape[1] / 900)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y_text = max(th + 6, y1 - 6)
    cv2.rectangle(out, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), (0, 200, 0), -1)
    cv2.putText(out, label, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick)
    return out
