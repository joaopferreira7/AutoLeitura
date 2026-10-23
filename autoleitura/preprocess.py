"""Pré-processamento com OpenCV (etapas 2 e 4 da arquitetura)."""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import config


def load_image(path_or_bytes):
    """Lê uma imagem de um caminho ou de bytes (upload da interface) em BGR.

    Fotos de celular costumam vir "deitadas", com a rotação só nos metadados EXIF;
    o exif_transpose aplica essa rotação nos pixels antes de qualquer processamento.
    """
    try:
        src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes)
        pil = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    except Exception as e:
        raise ValueError("Não foi possível ler a imagem.") from e
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


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


def crop_coords(shape, box, margin=config.CROP_MARGIN):
    """Coordenadas inteiras (x1, y1, x2, y2) do recorte da caixa com margem, limitadas à imagem."""
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return max(0, int(x1 - mx)), max(0, int(y1 - my)), min(w, int(x2 + mx)), min(h, int(y2 + my))


def crop_box(img, box, margin=config.CROP_MARGIN):
    """Recorta a caixa (x1, y1, x2, y2) com uma margem proporcional."""
    x1, y1, x2, y2 = crop_coords(img.shape, box, margin)
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


def shrink_crop(crop, max_width=2 * config.REC_WIDTH):
    """Reduz o recorte (que pode ter ~1000 px nas fotos originais) para 2x a largura de entrada da CRNN.
    O CLAHE, o deskew e a augmentation ficam bem mais rápidos e se comportam igual em qualquer resolução."""
    h, w = crop.shape[:2]
    if w <= max_width:
        return crop
    return cv2.resize(crop, (max_width, max(1, round(h * max_width / w))), interpolation=cv2.INTER_AREA)


def prepare_crop(crop):
    """Tratamento do recorte do contador antes da CRNN. É usado igual no treino e na inferência,
    para o modelo ver na leitura o mesmo tipo de imagem que viu no treino."""
    return deskew(enhance(shrink_crop(crop)))


def to_recognizer_input(crop):
    """Converte o recorte do contador no tensor esperado pela CRNN: 1xHxW, valores 0–1."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, (config.REC_WIDTH, config.REC_HEIGHT), interpolation=cv2.INTER_AREA)
    return (gray.astype(np.float32) / 255.0)[None, :, :]
