"""Junta ao treino as leituras confirmadas no app (o .zip de "Desempenho do modelo" > "Baixar correções").

Os recortes entram no split de treino de data/rec (energia) ou data/agua_rec (água). Leituras já
importadas são ignoradas, então o mesmo .zip pode ser importado mais de uma vez.

  python scripts/importar_correcoes.py correcoes.zip
  python train_recognizer.py --epochs 80                 # re-treina com as fotos reais
"""
import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoleitura import config  # noqa: E402

PASTAS = {"energia": config.REC_DIR, "agua": config.AGUA_REC_DIR}
CAMPOS = ["arquivo", "leitura", "split", "origem"]


def importar(caminho_zip, pastas=PASTAS):
    """Copia os recortes para images/ e acrescenta as linhas em labels.csv. Retorna {tipo: novas}."""
    novas = {t: 0 for t in pastas}
    with zipfile.ZipFile(caminho_zip) as z:
        linhas = list(csv.DictReader(io.StringIO(z.read("labels.csv").decode("utf-8"))))
        for tipo, pasta in pastas.items():
            labels = pasta / "labels.csv"
            existentes = set()
            if labels.exists():
                with open(labels, encoding="utf-8") as f:
                    existentes = {r["arquivo"] for r in csv.DictReader(f)}
            (pasta / "images").mkdir(parents=True, exist_ok=True)
            novas_linhas = [r for r in linhas if r["tipo"] == tipo and r["arquivo"] not in existentes
                            and r["leitura"].isdigit() and len(r["leitura"]) in config.DIGITOS[tipo]]
            for r in novas_linhas:
                (pasta / "images" / r["arquivo"]).write_bytes(z.read(f"images/{r['arquivo']}"))
            if novas_linhas:
                novo = not labels.exists()
                with open(labels, "a", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, CAMPOS, extrasaction="ignore", lineterminator="\n")
                    if novo:
                        w.writeheader()
                    w.writerows(novas_linhas)
            novas[tipo] = len(novas_linhas)
    return novas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", help="arquivo baixado do app")
    args = ap.parse_args()
    for tipo, n in importar(args.zip).items():
        print(f"{tipo}: {n} recortes novos no treino ({PASTAS[tipo] / 'labels.csv'})")


if __name__ == "__main__":
    main()
