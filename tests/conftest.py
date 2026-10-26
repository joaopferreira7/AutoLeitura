"""Configuração dos testes.

Os testes NUNCA tocam nos dados reais: o banco aponta para um arquivo temporário (definido aqui,
antes de qualquer import do pacote) e a preparação de dados escreve em pastas temporárias.

  python -m pytest              # tudo o que for possível nesta máquina
  python -m pytest -m "not modelos"   # só o que não precisa dos pesos treinados
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_TMP = Path(tempfile.mkdtemp(prefix="autoleitura-testes-"))
os.environ["AUTOLEITURA_DB"] = str(_TMP / "app.db")      # antes de importar autoleitura.config
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from autoleitura import config as cfg, db  # noqa: E402

PESOS_OK = cfg.DET_WEIGHTS.exists() and cfg.REC_WEIGHTS.exists() and cfg.DIG_WEIGHTS.exists()
TESTE_UFPR = cfg.RAW_DIR / "ufpr" / "UFPR-AMR Dataset" / "testing"


def pytest_configure(config):
    config.addinivalue_line("markers", "modelos: precisa dos pesos treinados em weights/")
    config.addinivalue_line("markers", "dataset: precisa do UFPR-AMR em data/raw/ufpr (licença acadêmica, não versionado)")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "modelos" in item.keywords and not PESOS_OK:
            item.add_marker(pytest.mark.skip(reason="pesos treinados ausentes (python -m autoleitura.pesos)"))
        if "dataset" in item.keywords and not TESTE_UFPR.exists():
            item.add_marker(pytest.mark.skip(reason="UFPR-AMR ausente em data/raw/ufpr"))


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "teste.db")
    yield c
    c.close()


@pytest.fixture(scope="session")
def sistema():
    from autoleitura.pipeline import AutoLeitura
    return AutoLeitura()


@pytest.fixture(scope="session")
def medidor_sintetico():
    """Foto sintética de um medidor (scripts/make_synthetic.py), com a leitura e a caixa do contador."""
    from make_synthetic import make_one
    img, box, leitura, _ = make_one(np.random.default_rng(7))
    return img, box, leitura


def fotos_reais(n):
    """As n primeiras fotos de teste do UFPR-AMR com a leitura anotada."""
    from prepare_data import parse_ufpr_txt
    fotos = []
    for jpg in sorted(TESTE_UFPR.glob("*.jpg"))[:n]:
        _, leitura, _ = parse_ufpr_txt(jpg.with_suffix(".txt"))
        fotos.append((jpg, leitura))
    return fotos
