"""Cria os imóveis de demonstração (ver autoleitura/demo.py), se o banco estiver vazio.

Não mexe em bancos que já têm imóveis cadastrados.

  python scripts/criar_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoleitura import config, db, demo  # noqa: E402

if __name__ == "__main__":
    if demo.criar_se_vazio(db.connect()):
        print(f"Imóveis de demonstração criados em {config.DB_PATH}")
    else:
        print(f"{config.DB_PATH} já tem imóveis; nada a fazer.")
