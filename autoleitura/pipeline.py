"""Pipeline completo: foto -> pré-processamento -> detecção -> recorte -> reconhecimento -> leitura."""
import time

import cv2

from . import config
from .detector import CounterDetector
from .preprocess import crop_box, limit_size
from .recognizer import Recognizer


def combine(crnn, yolo):
    """Junta as duas leituras do recorte. Regra escolhida no conjunto de validação do UFPR-AMR:

    - a leitura é a do YOLO de dígitos (93,8% na validação, contra 91,0% da CRNN);
      se ele não achar os 5 dígitos e a CRNN achar, fica a da CRNN;
    - quando os dois leitores concordam (88,8% das fotos), a leitura está certa em 99,2% dos casos,
      então a confiança sobe; quando discordam, ela cai e o app pede conferência.

    Cada leitura é (texto, confiança, confiança por dígito). Retorna (texto, confiança, conf. por dígito).
    """
    (tc, cc, dcc), (ty, cy, dcy) = crnn, yolo
    if len(ty) == config.NUM_DIGITS or len(tc) != config.NUM_DIGITS:
        text, digit_confs = ty, dcy
    else:
        text, digit_confs = tc, dcc
    if tc == ty:
        conf = 1 - (1 - cc) * (1 - cy)
    elif tc and ty:
        conf = 0.5 * min(cc, cy)
    else:
        conf = 0.25 * max(cc, cy)
    return text, conf, digit_confs


class AutoLeitura:
    def __init__(self, det_weights=config.DET_WEIGHTS, rec_weights=config.REC_WEIGHTS, dig_weights=config.DIG_WEIGHTS):
        self.detector = CounterDetector(det_weights)
        self.recognizer = Recognizer(rec_weights)
        self.digit_reader = None
        if dig_weights and dig_weights.exists():          # sem o YOLO de dígitos, usa só a CRNN
            from .digits import DigitReader
            self.digit_reader = DigitReader(dig_weights)

    def read(self, img):
        """Recebe uma imagem BGR e devolve um dicionário com a leitura e as informações de apoio."""
        t0 = time.perf_counter()
        small, scale = limit_size(img)
        box, det_conf = self.detector.detect(small)

        if box is None:
            return {"reading": None, "confidence": 0.0, "det_confidence": 0.0, "bbox": None,
                    "digit_confidences": [], "crop": None, "leitores": {},
                    "time_ms": (time.perf_counter() - t0) * 1000,
                    "warning": "Contador não encontrado na imagem."}

        box = tuple(v / scale for v in box)                 # volta para a escala original
        crop = crop_box(img, box)
        crnn = self.recognizer.read(crop)
        leitores = {"CRNN": crnn[0]}
        if self.digit_reader is not None:
            yolo = self.digit_reader.read(crop)
            leitores["YOLO de dígitos"] = yolo[0]
            text, rec_conf, digit_confs = combine(crnn, yolo)
        else:
            text, rec_conf, digit_confs = crnn

        # Pós-processamento: confiança final combina detecção e reconhecimento;
        # leituras com quantidade de dígitos inesperada são sinalizadas.
        confidence = det_conf * rec_conf
        warning = None
        if len(text) != config.NUM_DIGITS:
            warning = f"Foram lidos {len(text)} dígitos (esperado {config.NUM_DIGITS}). Confira a leitura."
            confidence *= 0.5
        elif len(set(leitores.values())) > 1:
            warning = ("Os dois leitores discordaram (" + ", ".join(f"{k}: {v or '—'}" for k, v in leitores.items())
                       + "). Confira os números com o medidor.")

        return {
            "reading": text,
            "confidence": confidence,
            "det_confidence": det_conf,
            "bbox": [round(v, 1) for v in box],
            "digit_confidences": digit_confs,
            "crop": crop,
            "leitores": leitores,
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
