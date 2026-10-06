"""Treina a CRNN que lê os dígitos do recorte do contador.

Usa os recortes gerados por scripts/prepare_data.py (data/rec). A augmentation
é aplicada só no treino, como definido na proposta.

  python train_recognizer.py --epochs 60          # Colab / GPU
  python train_recognizer.py --epochs 10          # teste rápido em CPU
"""
import argparse
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from autoleitura import config
from autoleitura.metrics import evaluate_readings
from autoleitura.preprocess import to_recognizer_input
from autoleitura.recognizer import CRNN, ctc_greedy_decode, encode


def augment(img, rng):
    h, w = img.shape[:2]
    # rotação leve + pequena distorção de perspectiva
    ang = rng.uniform(-4, 4)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, rng.uniform(0.95, 1.05))
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + rng.uniform(-0.04, 0.04, (4, 2)) * [w, h]).astype(np.float32)
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    # recorte com folga variável (simula caixas do detector menos precisas)
    x1, x2 = int(w * rng.uniform(0, 0.05)), w - int(w * rng.uniform(0, 0.05))
    y1, y2 = int(h * rng.uniform(0, 0.08)), h - int(h * rng.uniform(0, 0.08))
    img = img[y1:y2, x1:x2]
    # brilho / contraste / desfoque / ruído
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-30, 30))
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        img = 255 - img                       # contadores com fundo claro e com fundo escuro
    return img


class CropDataset(Dataset):
    def __init__(self, split, train=False):
        with open(config.REC_DIR / "labels.csv", encoding="utf-8") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        self.train = train
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(str(config.REC_DIR / "images" / r["arquivo"]))
        if self.train:
            img = augment(img, self.rng)
        return torch.from_numpy(to_recognizer_input(img)), r["leitura"]


def collate(batch):
    imgs, texts = zip(*batch)
    targets = torch.tensor([c for t in texts for c in encode(t)], dtype=torch.long)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    return torch.stack(imgs), targets, lengths, list(texts)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, gts = [], []
    for imgs, _, _, texts in loader:
        log_probs = model(imgs.to(device)).log_softmax(2).cpu()
        preds += [ctc_greedy_decode(lp)[0] for lp in log_probs]
        gts += texts
    return preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--samples-per-epoch", type=int, default=4000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds = CropDataset("train", train=True), CropDataset("val")
    print(f"treino={len(train_ds)}  val={len(val_ds)}  device={device}")
    # O CTC precisa de alguns milhares de passos para sair do "platô do blank". Com datasets
    # pequenos, cada época sorteia pelo menos --samples-per-epoch recortes (cada um com augmentation diferente).
    sampler = RandomSampler(train_ds, replacement=True, num_samples=max(len(train_ds), args.samples_per_epoch))
    train_dl = DataLoader(train_ds, args.batch, sampler=sampler, num_workers=args.workers, collate_fn=collate, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch, num_workers=args.workers, collate_fn=collate)

    model = CRNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    config.WEIGHTS_DIR.mkdir(exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for imgs, targets, lengths, _ in train_dl:
            log_probs = model(imgs.to(device)).log_softmax(2).permute(1, 0, 2)   # T x B x C
            input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long)
            loss = ctc(log_probs, targets, input_lengths, lengths)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item()

        m = evaluate_readings(*predict(model, val_dl, device))
        print(f"época {epoch:3d}  loss={total / len(train_dl):.4f}  "
              f"val leitura={m['reading_acc']:.4f}  dígito={m['digit_acc']:.4f}  lev={m['mean_lev']:.3f}")
        if m["reading_acc"] > best:
            best = m["reading_acc"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": m}, config.REC_WEIGHTS)

    print(f"\nMelhor acurácia de leitura na validação: {best:.4f}  ->  {config.REC_WEIGHTS}")


if __name__ == "__main__":
    main()
