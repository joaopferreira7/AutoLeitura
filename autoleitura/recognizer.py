"""Reconhecimento dos dígitos do contador com uma CRNN + CTC (etapa 5 da arquitetura).

A CRNN lê a sequência inteira de dígitos do recorte, sem precisar segmentar
cada dígito: a CNN extrai características ao longo da largura, a BiLSTM modela
a sequência e a perda CTC alinha a saída com o rótulo (ex.: "01234").
"""
import numpy as np
import torch
import torch.nn as nn

from . import config
from .preprocess import to_recognizer_input


def _block(cin, cout, pool):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class CRNN(nn.Module):
    def __init__(self, num_classes=len(config.ALPHABET) + 1):
        super().__init__()
        # Entrada 1x32x160 -> saída 256x1x40 (40 passos de tempo ao longo da largura)
        self.cnn = nn.Sequential(
            _block(1, 64, (2, 2)),        # 16x80
            _block(64, 128, (2, 2)),      # 8x40
            _block(128, 256, (2, 1)),     # 4x40
            _block(256, 256, (2, 1)),     # 2x40
            _block(256, 256, (2, 1)),     # 1x40
        )
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)   # B x T x C
        f, _ = self.rnn(f)
        return self.fc(f)                              # B x T x num_classes (logits)


def encode(text):
    return [config.ALPHABET.index(c) + 1 for c in text]


def ctc_greedy_decode(log_probs):
    """Decodificação gulosa do CTC. Retorna (texto, confiança por caractere).

    log_probs: tensor T x C (log-softmax) de uma única imagem.
    """
    probs = log_probs.exp()
    best_p, best_i = probs.max(dim=1)
    text, confs, prev = [], [], 0
    for p, i in zip(best_p.tolist(), best_i.tolist()):
        if i != 0 and i != prev:
            text.append(config.ALPHABET[i - 1])
            confs.append(p)
        prev = i
    return "".join(text), confs


class Recognizer:
    """Wrapper de inferência: recorte BGR -> (leitura, confiança, confiança por dígito)."""

    def __init__(self, weights=config.REC_WEIGHTS, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CRNN().to(self.device)
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.eval()

    @torch.no_grad()
    def read(self, crop):
        x = torch.from_numpy(to_recognizer_input(crop))[None].to(self.device)
        log_probs = self.model(x).log_softmax(2)[0].cpu()
        text, confs = ctc_greedy_decode(log_probs)
        conf = float(np.prod(confs)) if confs else 0.0
        return text, conf, confs
