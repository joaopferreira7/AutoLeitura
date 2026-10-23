"""Baixa os pesos treinados do Release do GitHub quando eles não estão em weights/.

Os pesos não ficam no repositório (são arquivos grandes); assim o app funciona logo
depois de um `git clone` e na nuvem do Streamlit, que publica direto do GitHub.
"""
import urllib.request

from . import config

RELEASE_URL = "https://github.com/joaopferreira7/AutoLeitura/releases/download/modelos-v1"
PESOS = [config.DET_WEIGHTS, config.REC_WEIGHTS, config.DIG_WEIGHTS]


def faltando():
    return [p for p in PESOS if not p.exists()]


def baixar(progresso=print):
    """Baixa os pesos que faltam. Grava num arquivo temporário e só renomeia no fim,
    para um download interrompido não deixar um arquivo pela metade."""
    config.WEIGHTS_DIR.mkdir(exist_ok=True)
    for destino in faltando():
        progresso(f"Baixando {destino.name}...")
        tmp = destino.with_suffix(".part")
        urllib.request.urlretrieve(f"{RELEASE_URL}/{destino.name}", tmp)
        tmp.replace(destino)


if __name__ == "__main__":
    baixar()
    print("Pesos prontos em", config.WEIGHTS_DIR)
