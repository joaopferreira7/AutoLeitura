"""Pipeline: regra de combinação, robustez a entradas ruins, concorrência e regressão de acurácia."""
import threading

import numpy as np
import pytest

from autoleitura.pipeline import combine, draw_result
from autoleitura.preprocess import load_image

from conftest import fotos_reais


def test_combinacao_quando_concordam_a_confianca_sobe():
    texto, conf, _ = combine(("01234", 0.7, [0.9] * 5), ("01234", 0.6, [0.9] * 5))
    assert texto == "01234" and conf == pytest.approx(1 - 0.3 * 0.4)


def test_combinacao_quando_discordam_vale_o_yolo_com_confianca_baixa():
    texto, conf, _ = combine(("01235", 0.9, [0.9] * 5), ("01234", 0.95, [0.9] * 5))
    assert texto == "01234" and conf == pytest.approx(0.45)


def test_combinacao_yolo_incompleto_usa_a_crnn():
    texto, conf, _ = combine(("01234", 0.8, [0.9] * 5), ("0123", 0.9, [0.9] * 4))
    assert texto == "01234" and conf < 0.6


def test_combinacao_na_agua_aceita_6_digitos():
    # a principal leu 4 dígitos (inválido na água) e a segunda leu 6: fica a de tamanho válido
    texto, _, _ = combine(("012345", 0.8, [0.9] * 6), ("0123", 0.9, [0.9] * 4), digitos=(5, 6))
    assert texto == "012345"
    texto, conf, _ = combine(("012345", 0.8, [0.9] * 6), ("012345", 0.7, [0.9] * 6), digitos=(5, 6))
    assert texto == "012345" and conf == pytest.approx(1 - 0.2 * 0.3)


def test_combinacao_sem_leitura():
    texto, conf, _ = combine(("", 0.0, []), ("", 0.0, []))
    assert texto == "" and conf == 0


@pytest.mark.modelos
@pytest.mark.parametrize("img", [
    np.zeros((480, 640, 3), np.uint8),                     # foto preta
    np.full((480, 640, 3), 255, np.uint8),                 # foto branca
    np.zeros((8, 8, 3), np.uint8),                         # foto minúscula
    np.random.default_rng(0).integers(0, 255, (3000, 4000, 3), dtype=np.uint8),   # ruído grande
], ids=["preta", "branca", "minuscula", "ruido"])
def test_fotos_sem_medidor_nao_quebram(sistema, img):
    res = sistema.read(img)
    assert set(res) >= {"reading", "confidence", "bbox", "crop", "warning", "leitores"}
    if res["reading"] is None:
        assert res["warning"] and res["confidence"] == 0
    draw_result(img, res)                                  # a interface desenha qualquer resultado


@pytest.mark.modelos
def test_leituras_simultaneas_dao_o_mesmo_resultado(sistema, medidor_sintetico):
    img = medidor_sintetico[0]
    esperado = sistema.read(img)["reading"]
    resultados, erros = [], []

    def ler():
        try:
            resultados.append(sistema.read(img)["reading"])
        except Exception as e:
            erros.append(e)

    ts = [threading.Thread(target=ler) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert erros == [] and resultados == [esperado] * 6


@pytest.mark.modelos
@pytest.mark.dataset
def test_regressao_de_acuracia_em_fotos_reais(sistema):
    """30 fotos do teste do UFPR-AMR, lidas como no app (bytes -> load_image). Se cair muito,
    algo mudou no pré-processamento, nos pesos ou nas bibliotecas."""
    fotos = fotos_reais(30)
    acertos = sum(sistema.read(load_image(jpg.read_bytes()))["reading"] == leitura for jpg, leitura in fotos)
    assert acertos / len(fotos) >= 0.85, f"só {acertos}/{len(fotos)} leituras certas"


# ---------------------------------------------------------------- medidores de água
from autoleitura import config  # noqa: E402
from autoleitura.pipeline import para_tipo  # noqa: E402

PESOS_AGUA = config.AGUA_SEG_WEIGHTS.exists() and config.AGUA_REC_WEIGHTS.exists()
TESTE_AGUA = config.AGUA_REC_DIR / "labels.csv"
agua = pytest.mark.skipif(not PESOS_AGUA, reason="pesos de água ausentes")


@agua
def test_pipeline_de_agua_le_nos_dois_sentidos_e_aceita_6_digitos():
    sistema_agua = para_tipo("agua")
    assert sistema_agua.digitos == (5, 6) and sistema_agua.digit_reader is None
    assert sistema_agua.recognizer.nos_dois_sentidos
    if config.AGUA_REC2_WEIGHTS.exists():               # segunda opinião, também nos dois sentidos
        assert sistema_agua.segunda is not None and sistema_agua.segunda.nos_dois_sentidos
    res = sistema_agua.read(np.zeros((480, 640, 3), np.uint8))
    assert set(res) >= {"reading", "confidence", "warning"}


@agua
@pytest.mark.skipif(not TESTE_AGUA.exists(), reason="dataset de água ausente em data/agua_rec")
def test_regressao_crnn_de_agua_em_recortes_reais():
    """100 recortes do teste oficial do dataset de água (de pé e de cabeça para baixo)."""
    import csv

    import cv2

    from autoleitura.recognizer import Recognizer
    rec = Recognizer(config.AGUA_REC_WEIGHTS, nos_dois_sentidos=True)
    rows = [r for r in csv.DictReader(open(TESTE_AGUA, encoding="utf-8")) if r["split"] == "test"][::24][:100]
    acertos = sum(rec.read(cv2.imread(str(config.AGUA_REC_DIR / "images" / r["arquivo"])))[0] == r["leitura"]
                  for r in rows)
    assert acertos / len(rows) >= 0.80, f"só {acertos}/{len(rows)} leituras certas"
