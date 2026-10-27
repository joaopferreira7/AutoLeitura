"""Publica o app no Hugging Face Spaces (Docker, CPU gratuita).

Antes, faça login uma vez no terminal com um token de escrita do Hugging Face
(huggingface.co -> Settings -> Access Tokens -> New token, tipo "Write"):

  hf auth login

Depois:

  python scripts/publicar_hf.py                    # cria/atualiza <seu-usuario>/AutoLeitura
  python scripts/publicar_hf.py --privado          # Space visível só para você

Os pesos treinados (weights/*.pt) vão junto para o Space; os dados do UFPR-AMR, não.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoleitura import config  # noqa: E402

ROOT = config.ROOT
ARQUIVOS = ["app.py", "scripts/criar_demo.py", ".streamlit/config.toml"]
PASTAS = ["autoleitura"]
PESOS = [config.DET_WEIGHTS, config.REC_WEIGHTS, config.DIG_WEIGHTS, config.AGUA_SEG_WEIGHTS, config.AGUA_REC_WEIGHTS,
         config.AGUA_REC2_WEIGHTS]


def montar(destino):
    for arq in ARQUIVOS:
        (destino / arq).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / arq, destino / arq)
    for pasta in PASTAS:
        shutil.copytree(ROOT / pasta, destino / pasta, ignore=shutil.ignore_patterns("__pycache__"))
    (destino / "weights").mkdir()
    for peso in PESOS:
        if peso.exists():
            shutil.copy(peso, destino / "weights" / peso.name)
    for arq in (ROOT / "deploy" / "huggingface").iterdir():
        # o container é Linux: CRLF quebraria as continuações de linha do Dockerfile
        (destino / arq.name).write_bytes(arq.read_bytes().replace(b"\r\n", b"\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", help="usuario/nome do Space (padrão: <seu-usuario>/AutoLeitura)")
    ap.add_argument("--privado", action="store_true")
    args = ap.parse_args()

    faltando = [p.name for p in PESOS[:2] if not p.exists()]
    if faltando:
        sys.exit(f"Pesos não encontrados em {config.WEIGHTS_DIR}: {', '.join(faltando)}. Treine os modelos antes.")

    api = HfApi()
    usuario = api.whoami()["name"]          # falha com mensagem clara se não houver login
    repo_id = args.space or f"{usuario}/AutoLeitura"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=args.privado, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        montar(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="space",
                          commit_message="Publica o AutoLeitura")
    print(f"\nPublicado: https://huggingface.co/spaces/{repo_id}")
    print("O primeiro build leva alguns minutos; acompanhe a aba 'Logs' do Space.")


if __name__ == "__main__":
    main()
