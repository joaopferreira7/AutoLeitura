"""Converte um dataset bruto nos dois datasets usados pelo projeto.

  data/det/  -> formato YOLO (1 classe: "contador") para treinar o detector
  data/rec/  -> recortes do contador + labels.csv (arquivo, leitura, split) para a CRNN
  data/dig/  -> recortes do contador no formato YOLO, um box por dígito (classes 0–9)

Fontes suportadas (--source):
  ufpr      UFPR-AMR (pastas training/validation/testing, cada imagem com um .txt de anotação)
  roboflow  Export do Roboflow no formato "YOLOv8" com classes 0–9 (um box por dígito)
            e, opcionalmente, uma classe de contador ("counter", "contador", "display"...)

Exemplos:
  python scripts/prepare_data.py --source ufpr --input data/raw/UFPR-AMR
  python scripts/prepare_data.py --source roboflow --input data/raw/water-meter --append
"""
import argparse
import csv
import random
import re
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoleitura import config  # noqa: E402
from autoleitura.preprocess import crop_box, crop_coords, limit_size  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_ALIASES = {"training": "train", "train": "train", "validation": "val", "valid": "val",
                 "val": "val", "testing": "test", "test": "test"}


# ---------------------------------------------------------------- UFPR-AMR
def parse_ufpr_txt(path):
    """Lê o .txt do UFPR-AMR. Ele traz câmera, posição do contador (x y w h),
    a leitura e a posição de cada dígito. O parser procura as chaves por padrão
    (e não por nome exato) para tolerar pequenas variações entre versões do dataset."""
    counter, reading, digits = None, None, []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, val = (s.strip().lower() for s in line.split(":", 1))
        nums = [int(float(n)) for n in re.findall(r"-?\d+(?:\.\d+)?", val)]
        if "digit" in key and len(nums) >= 4:
            digits.append(nums[:4])
        elif ("counter" in key or "position" in key) and "digit" not in key and len(nums) >= 4:
            counter = nums[:4]
        elif "reading" in key or "value" in key:
            m = re.search(r"\d+", val.replace(" ", ""))
            if m:
                reading = m.group(0)
    if counter is None and digits:                   # fallback: união das caixas dos dígitos
        x1 = min(d[0] for d in digits); y1 = min(d[1] for d in digits)
        x2 = max(d[0] + d[2] for d in digits); y2 = max(d[1] + d[3] for d in digits)
        counter = [x1, y1, x2 - x1, y2 - y1]
    if counter is None or reading is None:
        return None
    x, y, w, h = counter
    digit_boxes = [(dx, dy, dx + dw, dy + dh) for dx, dy, dw, dh in sorted(digits, key=lambda d: d[0])]
    return (x, y, x + w, y + h), reading, digit_boxes


def load_ufpr(root):
    samples = []
    for img_path in sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS):
        txt = img_path.with_suffix(".txt")
        if not txt.exists():
            continue
        parsed = parse_ufpr_txt(txt)
        if parsed is None:
            print(f"[aviso] anotação incompleta, ignorada: {txt}")
            continue
        split = next((SPLIT_ALIASES[p.lower()] for p in img_path.parts if p.lower() in SPLIT_ALIASES), None)
        box, reading, digit_boxes = parsed
        # cada caixa de dígito recebe o caractere da leitura na mesma posição (esquerda -> direita)
        digits = list(zip(digit_boxes, reading)) if len(digit_boxes) == len(reading) else []
        samples.append({"image": img_path, "box": box, "reading": reading, "digits": digits, "split": split})
    return samples


# ---------------------------------------------------------------- Roboflow (YOLO export)
def load_roboflow(root):
    import yaml
    root = Path(root)
    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    digit_cls = {i: str(n) for i, n in enumerate(names) if str(n) in config.ALPHABET}
    counter_cls = {i for i, n in enumerate(names)
                   if str(n).lower() in {"counter", "contador", "display", "reading", "meter", "digits"}}
    if len(digit_cls) < 10:
        print(f"[aviso] o data.yaml tem {len(digit_cls)} classes de dígito: {names}")

    samples = []
    for img_path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS):
        lbl = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"   # train/images/x.jpg -> train/labels/x.txt
        if not lbl.exists():
            continue
        h, w = cv2.imread(str(img_path)).shape[:2]
        dig, cnt = [], None
        for line in lbl.read_text().split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            if c in digit_cls:
                dig.append((box, digit_cls[c]))
            elif c in counter_cls:
                cnt = box
        if not dig:
            continue
        dig.sort(key=lambda d: d[0][0])                # leitura = dígitos ordenados da esquerda p/ direita
        reading = "".join(d[1] for d in dig)
        if cnt is None:
            cnt = (min(d[0][0] for d in dig), min(d[0][1] for d in dig),
                   max(d[0][2] for d in dig), max(d[0][3] for d in dig))
        split = next((SPLIT_ALIASES[p.lower()] for p in img_path.parts if p.lower() in SPLIT_ALIASES), None)
        samples.append({"image": img_path, "box": cnt, "reading": reading, "digits": dig, "split": split})
    return samples


# ---------------------------------------------------------------- escrita dos datasets
def assign_splits(samples, seed=42):
    """Mantém o split original quando existe; senão divide 70/15/15."""
    missing = [s for s in samples if s["split"] is None]
    random.Random(seed).shuffle(missing)
    n = len(missing)
    for i, s in enumerate(missing):
        s["split"] = "train" if i < 0.70 * n else ("val" if i < 0.85 * n else "test")


def write_datasets(samples, prefix, append):
    if not append:
        for d in (config.DET_DIR, config.REC_DIR, config.DIG_DIR):
            shutil.rmtree(d, ignore_errors=True)
    for split in ("train", "val", "test"):
        for d in (config.DET_DIR, config.DIG_DIR):
            (d / "images" / split).mkdir(parents=True, exist_ok=True)
            (d / "labels" / split).mkdir(parents=True, exist_ok=True)
    (config.REC_DIR / "images").mkdir(parents=True, exist_ok=True)

    labels_csv = config.REC_DIR / "labels.csv"
    new_file = not labels_csv.exists()
    with open(labels_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["arquivo", "leitura", "split", "origem"])
        for i, s in enumerate(samples):
            img = cv2.imread(str(s["image"]))
            if img is None:
                continue
            h, w = img.shape[:2]
            name = f"{prefix}_{i:05d}"
            x1, y1, x2, y2 = s["box"]
            cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h

            # Fotos de celular chegam a 4160 px; o YOLO treina em 640, e decodificar JPGs enormes a cada
            # época vira gargalo (o Colab tem só 2 núcleos). Os rótulos são normalizados e não mudam.
            # 1600 px é o mesmo limite que o pipeline usa na inferência (preprocess.limit_size).
            small, _ = limit_size(img, max_side=1600)
            cv2.imwrite(str(config.DET_DIR / "images" / s["split"] / f"{name}.jpg"), small,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (config.DET_DIR / "labels" / s["split"] / f"{name}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            crop = crop_box(img, s["box"])
            cv2.imwrite(str(config.REC_DIR / "images" / f"{name}.png"), crop)

            # dataset de dígitos: o mesmo recorte, com a caixa de cada dígito relativa ao recorte
            if s.get("digits"):
                ox, oy, _, _ = crop_coords(img.shape, s["box"])
                ch, cw = crop.shape[:2]
                lines = []
                for (dx1, dy1, dx2, dy2), d in s["digits"]:
                    dx1, dx2 = max(0, dx1 - ox), min(cw, dx2 - ox)
                    dy1, dy2 = max(0, dy1 - oy), min(ch, dy2 - oy)
                    if dx2 > dx1 and dy2 > dy1:
                        lines.append(f"{d} {(dx1 + dx2) / 2 / cw:.6f} {(dy1 + dy2) / 2 / ch:.6f} "
                                     f"{(dx2 - dx1) / cw:.6f} {(dy2 - dy1) / ch:.6f}")
                cv2.imwrite(str(config.DIG_DIR / "images" / s["split"] / f"{name}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                (config.DIG_DIR / "labels" / s["split"] / f"{name}.txt").write_text("\n".join(lines) + "\n")
            writer.writerow([f"{name}.png", s["reading"], s["split"], s["image"].name])
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(samples)} imagens processadas")

    (config.DET_DIR / "data.yaml").write_text(
        f"path: {config.DET_DIR.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: contador\n", encoding="utf-8")
    (config.DIG_DIR / "data.yaml").write_text(
        f"path: {config.DIG_DIR.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: '{i}'\n" for i in range(10)), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["ufpr", "roboflow"], required=True)
    ap.add_argument("--input", required=True, help="pasta do dataset bruto")
    ap.add_argument("--append", action="store_true", help="adiciona aos datasets existentes (ex.: energia + água)")
    args = ap.parse_args()

    samples = load_ufpr(args.input) if args.source == "ufpr" else load_roboflow(args.input)
    if not samples:
        sys.exit("Nenhuma amostra encontrada. Confira o caminho e o formato do dataset.")
    assign_splits(samples)
    write_datasets(samples, prefix=args.source, append=args.append)

    counts = {sp: sum(s["split"] == sp for s in samples) for sp in ("train", "val", "test")}
    print(f"{len(samples)} amostras convertidas {counts}")
    print(f"  detector:      {config.DET_DIR / 'data.yaml'}")
    print(f"  reconhecedor:  {config.REC_DIR / 'labels.csv'}")
    print(f"  dígitos:       {config.DIG_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
