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
