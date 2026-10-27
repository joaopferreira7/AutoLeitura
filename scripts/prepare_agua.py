"""Converte o dataset de medidores de água (word-wheel, Dryad) para o formato do projeto.

Dataset: Zhao, S.; Gao, Y.; Liu, F.; Zhang, Y.; Li, J. "A comprehensive dataset for word-wheel water meter
reading under challenging conditions". Dryad, 2026. https://doi.org/10.5061/dryad.7d7wm3860 (licença CC0).

  data/agua_det/  -> segmentação no formato YOLO (1 classe: contador): o polígono da janela dos dígitos
  data/agua_rec/  -> recortes do contador (5 e 6 dígitos) + labels.csv, no mesmo formato de data/rec

O conjunto de teste oficial do dataset fica intocado como teste; 10% do treino vira validação.

  python scripts/prepare_agua.py --input data/raw/agua/Word-Wheel_Water_Meter_Dataset
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoleitura import config  # noqa: E402

FRACAO_VAL = 0.10


def caixa_da_mascara(caminho):
    """Caixa (x1, y1, x2, y2) que envolve a região marcada na máscara, ou None se ela estiver vazia."""
    m = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None, None
    pts = cv2.findNonZero((m > 0).astype("uint8"))
    if pts is None:
        return None, m.shape
    x, y, w, h = cv2.boundingRect(pts)
    return (x, y, x + w, y + h), m.shape


def poligono_da_mascara(caminho):
    """Contorno da janela dos dígitos (o maior da máscara), simplificado, em coordenadas 0–1.

    O segmentador aprende essa janela; na leitura, ela é endireitada (retângulo girado + perspectiva),
    que é como os recortes de treino da CRNN foram feitos.
    """
    m = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    contornos, _ = cv2.findContours((m > 0).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    c = max(contornos, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True)
    if len(c) < 3:
        return None
    h, w = m.shape
    return [(float(x) / w, float(y) / h) for x, y in c.reshape(-1, 2)]


def separar_validacao(nomes, seed=42):
    nomes = sorted(nomes)
    random.Random(seed).shuffle(nomes)
    corte = int(len(nomes) * FRACAO_VAL)
    return set(nomes[:corte])


def preparar_deteccao(raiz):
    """Dataset de segmentação no formato YOLO: um polígono (a janela dos dígitos) por foto."""
    for split in ("train", "val", "test"):
        (config.AGUA_DET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        shutil.rmtree(config.AGUA_DET_DIR / "labels" / split, ignore_errors=True)
        (config.AGUA_DET_DIR / "labels" / split).mkdir(parents=True)
    for cache in config.AGUA_DET_DIR.glob("labels/*.cache"):
        cache.unlink()

    contagem = {"train": 0, "val": 0, "test": 0, "sem_mascara": 0}
    for origem in ("train", "test"):
        imgs = sorted((raiz / "detection" / origem / f"{origem}_img").glob("*.png"))
        val = separar_validacao([p.name for p in imgs]) if origem == "train" else set()
        for i, img in enumerate(imgs):
            poligono = poligono_da_mascara(raiz / "detection" / origem / f"{origem}_seg_label" / img.name)
            if poligono is None:
                contagem["sem_mascara"] += 1
                continue
            split = "test" if origem == "test" else ("val" if img.name in val else "train")
            nome = f"agua_{origem}_{img.stem}"
            destino = config.AGUA_DET_DIR / "images" / split / f"{nome}.png"
            if not destino.exists():                     # imagens já copiadas numa preparação anterior
                shutil.copy(img, destino)
            (config.AGUA_DET_DIR / "labels" / split / f"{nome}.txt").write_text(
                "0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poligono) + "\n")
            contagem[split] += 1
            if (i + 1) % 5000 == 0:
                print(f"  segmentação {origem}: {i + 1}/{len(imgs)}")
    (config.AGUA_DET_DIR / "data.yaml").write_text(
        f"path: {config.AGUA_DET_DIR.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: contador\n", encoding="utf-8")
    return contagem


def preparar_reconhecimento(raiz):
    shutil.rmtree(config.AGUA_REC_DIR, ignore_errors=True)
    (config.AGUA_REC_DIR / "images").mkdir(parents=True)
    contagem = {"train": 0, "val": 0, "test": 0, "descartados": 0}
    with open(config.AGUA_REC_DIR / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "leitura", "split", "origem"])
        for n in (5, 6):
            for origem in ("train", "test"):
                pasta = raiz / "recognition" / f"recognition_{n}-digit" / origem
                rotulos = list(csv.reader(open(pasta / f"{n}-digit_{origem}_rec_label_CSV.csv", encoding="utf-8-sig")))
                val = separar_validacao([r[0] for r in rotulos], seed=42 + n) if origem == "train" else set()
                for r in rotulos:
                    arq, leitura = r[0], r[1].strip()
                    img = pasta / f"{n}-digit_{origem}_img" / arq
                    if not (leitura.isdigit() and len(leitura) == n and img.exists()):
                        contagem["descartados"] += 1
                        continue
                    split = "test" if origem == "test" else ("val" if arq in val else "train")
                    nome = f"agua{n}_{origem}_{Path(arq).stem}.jpg"
                    shutil.copy(img, config.AGUA_REC_DIR / "images" / nome)
                    w.writerow([nome, leitura, split, f"{n}-digit/{origem}/{arq}"])
                    contagem[split] += 1
    return contagem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="pasta Word-Wheel_Water_Meter_Dataset extraída")
    ap.add_argument("--so", choices=["deteccao", "reconhecimento"], help="prepara só uma das partes")
    args = ap.parse_args()
    raiz = Path(args.input)
    if args.so != "reconhecimento":
        print("Detecção:", preparar_deteccao(raiz))
    if args.so != "deteccao":
        print("Reconhecimento:", preparar_reconhecimento(raiz))


if __name__ == "__main__":
    main()
