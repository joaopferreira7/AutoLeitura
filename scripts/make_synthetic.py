"""Gera medidores sintéticos no mesmo formato de anotação do UFPR-AMR.

Serve para testar o pipeline inteiro (preparação -> treino -> avaliação -> interface)
enquanto o dataset real não chega. NÃO use os resultados sintéticos como resultado
final do trabalho: eles são apenas um teste de fumaça.

  python scripts/make_synthetic.py --n 600 --out data/raw/synthetic
  python scripts/prepare_data.py --source ufpr --input data/raw/synthetic
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN]


def make_one(rng, w=640, h=480):
    img = np.full((h, w, 3), rng.integers(60, 200, 3), np.uint8)
    img = cv2.add(img, rng.integers(0, 40, (h, w, 3), dtype=np.uint8))

    # corpo do medidor
    cx, cy = w // 2 + rng.integers(-60, 60), h // 2 + rng.integers(-40, 40)
    r = int(rng.integers(160, 210))
    cv2.circle(img, (cx, cy), r, tuple(int(v) for v in rng.integers(170, 245, 3)), -1)
    cv2.circle(img, (cx, cy), r, (40, 40, 40), 3)

    # contador de 5 dígitos
    reading = "".join(str(rng.integers(0, 10)) for _ in range(5))
    cell_w, cell_h = int(rng.integers(30, 42)), int(rng.integers(44, 58))
    gap = int(rng.integers(2, 7))
    cw = 5 * cell_w + 4 * gap + 12
    x1, y1 = cx - cw // 2, cy - r // 3 - cell_h // 2
    dark = rng.random() < 0.7
    bg, fg = ((20, 20, 20), (235, 235, 235)) if dark else ((235, 235, 235), (20, 20, 20))
    cv2.rectangle(img, (x1, y1), (x1 + cw, y1 + cell_h + 12), (90, 90, 90), -1)
    font = FONTS[rng.integers(len(FONTS))]
    digit_boxes = []
    for i, d in enumerate(reading):
        dx = x1 + 6 + i * (cell_w + gap)
        cv2.rectangle(img, (dx, y1 + 6), (dx + cell_w, y1 + 6 + cell_h), bg, -1)
        scale = cell_h / 30 if font != cv2.FONT_HERSHEY_PLAIN else cell_h / 15
        (tw, th), _ = cv2.getTextSize(d, font, scale, 2)
        tx, ty = dx + (cell_w - tw) // 2, y1 + 6 + (cell_h + th) // 2
        ty += int(rng.integers(-3, 4))          # roletes desalinhados, como em medidores reais
        cv2.putText(img, d, (tx, ty), font, scale, fg, 2, cv2.LINE_AA)
        digit_boxes.append((dx, y1 + 6, cell_w, cell_h))
    counter = np.array([[x1, y1], [x1 + cw, y1], [x1 + cw, y1 + cell_h + 12], [x1, y1 + cell_h + 12]], np.float32)

    # textos de distração (número de série etc.)
    cv2.putText(img, f"N {rng.integers(10**6, 10**7)}", (cx - 70, cy + r // 2), FONTS[0], 0.7, (60, 60, 60), 2)
    cv2.putText(img, "kWh", (x1 + cw + 8, y1 + cell_h), FONTS[0], 0.7, (30, 30, 30), 2)

    # perspectiva/rotação leve
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-35, 35, (4, 2)).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, src + jitter)
    img = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def warp(pts):
        return cv2.perspectiveTransform(np.asarray(pts, np.float32)[None], m)[0]

    c = warp(counter)
    cbox = (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())
    dboxes = []
    for (dx, dy, dw, dh) in digit_boxes:
        p = warp([[dx, dy], [dx + dw, dy + dh]])
        dboxes.append((p[:, 0].min(), p[:, 1].min(), p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min()))

    # condições de foto: brilho, desfoque, ruído
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.6, 1.3), beta=rng.uniform(-40, 40))
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    noise = rng.normal(0, rng.uniform(2, 12), img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img, cbox, reading, dboxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="data/raw/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    splits = [("training", 0.4), ("validation", 0.2), ("testing", 0.4)]   # mesma proporção 800/400/800 do UFPR-AMR
    i = 0
    for split, frac in splits:
        d = Path(args.out) / split
        d.mkdir(parents=True, exist_ok=True)
        for _ in range(int(args.n * frac)):
            img, (x1, y1, x2, y2), reading, dboxes = make_one(rng)
            name = f"meter{i:05d}"
            cv2.imwrite(str(d / f"{name}.jpg"), img)
            lines = ["camera: synthetic",
                     f"position: {int(x1)} {int(y1)} {int(x2 - x1)} {int(y2 - y1)}",
                     f"value: {reading}"]
            lines += [f"digit {k + 1}: {int(a)} {int(b)} {int(c)} {int(e)}" for k, (a, b, c, e) in enumerate(dboxes)]
            (d / f"{name}.txt").write_text("\n".join(lines) + "\n")
            i += 1
    print(f"{i} imagens sintéticas geradas em {args.out}")


if __name__ == "__main__":
    main()
