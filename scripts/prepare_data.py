"""Converte um dataset bruto nos dois datasets usados pelo projeto.

  data/det/  -> formato YOLO (1 classe: "contador") para treinar o detector
  data/rec/  -> recortes do contador + labels.csv (arquivo, leitura, split) para a CRNN

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
from autoleitura.preprocess import crop_box  # noqa: E402

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
    return (x, y, x + w, y + h), reading


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
        samples.append({"image": img_path, "box": parsed[0], "reading": parsed[1], "split": split})
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
        samples.append({"image": img_path, "box": cnt, "reading": reading, "split": split})
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
        shutil.rmtree(config.DET_DIR, ignore_errors=True)
        shutil.rmtree(config.REC_DIR, ignore_errors=True)
    for split in ("train", "val", "test"):
        (config.DET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.DET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
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

            shutil.copy(s["image"], config.DET_DIR / "images" / s["split"] / f"{name}{s['image'].suffix.lower()}")
            (config.DET_DIR / "labels" / s["split"] / f"{name}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            cv2.imwrite(str(config.REC_DIR / "images" / f"{name}.png"), crop_box(img, s["box"]))
            writer.writerow([f"{name}.png", s["reading"], s["split"], s["image"].name])

    (config.DET_DIR / "data.yaml").write_text(
        f"path: {config.DET_DIR.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: contador\n", encoding="utf-8")


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


if __name__ == "__main__":
    main()
