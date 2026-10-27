"""Baixa os pesos treinados do Release do GitHub quando eles não estão em weights/.

Os pesos não ficam no repositório (são arquivos grandes); assim o app funciona logo
depois de um `git clone` e na nuvem do Streamlit, que publica direto do GitHub.
"""
import hashlib
import urllib.request

from . import config

RELEASE_URL = "https://github.com/joaopferreira7/AutoLeitura/releases/download/modelos-v2"
PESOS = [config.DET_WEIGHTS, config.REC_WEIGHTS, config.DIG_WEIGHTS, config.AGUA_SEG_WEIGHTS, config.AGUA_REC_WEIGHTS]
# o mínimo para ler cada tipo de medidor (o YOLO de dígitos da energia é opcional)
NECESSARIOS = {"energia": [config.DET_WEIGHTS, config.REC_WEIGHTS],
               "agua": [config.AGUA_SEG_WEIGHTS, config.AGUA_REC_WEIGHTS]}

# SHA-256 dos arquivos do Release modelos-v2: um download corrompido ou um arquivo trocado
# no Release é recusado em vez de carregar um modelo errado. Ao publicar novos pesos,
# crie um Release novo (modelos-v3) e atualize RELEASE_URL e estes valores.
SHA256 = {
    "detector.pt": "07eec4f748f1059f06f206c52810f924038706c39fd34fe2f8519d341ceefee1",
    "recognizer.pt": "82c9cbda898cc1dc03ebd0d99e8206279fd7993ac53a5615bf8d768d1903c26f",
    "digits.pt": "6136f5f3366c72d1c53fceb54f89b61c020fedd34bf38122ed759682c7de5b69",
    "segmentador_agua.pt": "96bd5d1dfd43b04be639b5daf4f3342f0db840d34897b8e750d193b895e955a0",
    "recognizer_agua.pt": "cecaf969a378ff8e3b42679814c7c66278778fe4b8201a0c21847063a16777da",
}


def faltando():
    return [p for p in PESOS if not p.exists()]


def prontos(tipo):
    return all(p.exists() for p in NECESSARIOS[tipo])


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar(progresso=print):
    """Baixa os pesos que faltam. Grava num arquivo temporário, confere o SHA-256 e só então
    renomeia, para um download interrompido ou corrompido nunca virar um peso "válido"."""
    config.WEIGHTS_DIR.mkdir(exist_ok=True)
    for destino in faltando():
        progresso(f"Baixando {destino.name}...")
        tmp = destino.with_suffix(".part")
        urllib.request.urlretrieve(f"{RELEASE_URL}/{destino.name}", tmp)
        esperado = SHA256.get(destino.name)
        if esperado and sha256(tmp) != esperado:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{destino.name} baixado não confere com o SHA-256 esperado (download corrompido?)")
        tmp.replace(destino)


if __name__ == "__main__":
    baixar()
    print("Pesos prontos em", config.WEIGHTS_DIR)
