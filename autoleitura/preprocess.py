"""Pré-processamento com OpenCV (etapas 2 e 4 da arquitetura)."""
import cv2
import numpy as np

from . import config


def load_image(path_or_bytes):
    """Lê uma imagem de um caminho ou de bytes (upload da interface) em BGR."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não foi possível ler a imagem.")
    return img


def enhance(img):
    """Realce de contraste com CLAHE no canal L (LAB). Ajuda em fotos escuras/estouradas."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def limit_size(img, max_side=1600):
    """Fotos de celular podem ter 4000px; reduzimos para acelerar sem perder os dígitos."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img, 1.0
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale


def crop_box(img, box, margin=config.CROP_MARGIN):
    """Recorta a caixa (x1, y1, x2, y2) com uma margem proporcional."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return img[y1:y2, x1:x2]


def deskew(crop):
    """Retificação: estima a inclinação dos dígitos e gira o recorte para deixá-los na horizontal.

    Usa o retângulo de área mínima das regiões de borda; se a estimativa for
    pouco confiável (ângulo grande), devolve o recorte sem alteração.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    edges = cv2.Canny(gray, 50, 150)
    pts = cv2.findNonZero(edges)
    if pts is None or len(pts) < 50:
        return crop
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(pts)
    if rw < rh:
        angle -= 90
    if abs(angle) > 15 or abs(angle) < 0.5:
        return crop
    m = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), angle, 1.0)
    return cv2.warpAffine(crop, m, (crop.shape[1], crop.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def to_recognizer_input(crop):
    """Converte o recorte do contador no tensor esperado pela CRNN: 1xHxW, valores 0–1."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, (config.REC_WIDTH, config.REC_HEIGHT), interpolation=cv2.INTER_AREA)
    return (gray.astype(np.float32) / 255.0)[None, :, :]
