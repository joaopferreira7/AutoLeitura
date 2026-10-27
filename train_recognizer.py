"""Treina a CRNN que lê os dígitos do recorte do contador.

Usa os recortes gerados por scripts/prepare_data.py (data/rec). A augmentation
é aplicada só no treino, como definido na proposta.

  python train_recognizer.py --epochs 60          # Colab / GPU
  python train_recognizer.py --epochs 10          # teste rápido em CPU
  python train_recognizer.py --tipo agua --epochs 40   # medidores de água
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
from autoleitura.preprocess import prepare_crop, shrink_crop, to_recognizer_input
from autoleitura.recognizer import CRNN, ctc_greedy_decode, encode


def augment(img, rng, girar180=False):
    # Hidrômetros em caixas no chão são fotografados de qualquer lado: no dataset de água, parte dos
    # recortes está de cabeça para baixo. Girar metade dos recortes ensina a CRNN a ler nos dois sentidos.
    if girar180 and rng.random() < 0.5:
        img = cv2.rotate(img, cv2.ROTATE_180)
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


DADOS = {"energia": (config.REC_DIR, config.REC_WEIGHTS), "agua": (config.AGUA_REC_DIR, config.AGUA_REC_WEIGHTS)}


class CropDataset(Dataset):
    def __init__(self, split, train=False, pasta=config.REC_DIR, girar180=False):
        with open(pasta / "labels.csv", encoding="utf-8") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        self.train = train
        self.pasta = pasta
        self.girar180 = girar180

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = shrink_crop(cv2.imread(str(self.pasta / "images" / r["arquivo"])))
        if self.train:
            # gerador novo a cada amostra, semeado pelo torch: cada worker do DataLoader tem
            # sua própria semente, então as augmentations não se repetem entre workers
            img = augment(img, np.random.default_rng(torch.randint(0, 2**31, (1,)).item()), self.girar180)
        # mesmo tratamento da inferência (Recognizer.read)
        return torch.from_numpy(to_recognizer_input(prepare_crop(img))), r["leitura"]


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
    ap.add_argument("--tipo", choices=list(DADOS), default="energia")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--samples-per-epoch", type=int, default=4000)
    args = ap.parse_args()

    pasta, saida = DADOS[args.tipo]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = CropDataset("train", train=True, pasta=pasta, girar180=args.tipo == "agua")
    val_ds = CropDataset("val", pasta=pasta)
    print(f"treino={len(train_ds)}  val={len(val_ds)}  device={device}")
    # O CTC precisa de alguns milhares de passos para sair do "platô do blank". Com datasets
    # pequenos, cada época sorteia pelo menos --samples-per-epoch recortes (cada um com augmentation diferente).
    sampler = RandomSampler(train_ds, replacement=True, num_samples=max(len(train_ds), args.samples_per_epoch))
    # persistent_workers: no Windows, recriar os processos do DataLoader a cada época custa vários segundos
    extra = {"persistent_workers": True} if args.workers > 0 else {}
    train_dl = DataLoader(train_ds, args.batch, sampler=sampler, num_workers=args.workers, collate_fn=collate,
                          drop_last=True, **extra)
    val_dl = DataLoader(val_ds, args.batch, num_workers=args.workers, collate_fn=collate, **extra)

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
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": m}, saida)

    print(f"\nMelhor acurácia de leitura na validação: {best:.4f}  ->  {saida}")


if __name__ == "__main__":
    main()
