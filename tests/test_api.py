"""API REST (FastAPI)."""
import io

import cv2
import pytest
from fastapi.testclient import TestClient

import api
from autoleitura import db

cliente = TestClient(api.app)


def test_saude():
    r = cliente.get("/saude")
    assert r.status_code == 200 and "modelos_prontos" in r.json()


def test_arquivo_que_nao_e_imagem():
    r = cliente.post("/leitura", files={"foto": ("x.txt", b"texto qualquer", "text/plain")})
    assert r.status_code == 400


def test_medidor_inexistente():
    assert cliente.get("/medidores/999999/consumo").status_code == 404


def test_historico_de_consumo():
    con = db.connect()
    m = db.add_medidor(con, db.add_imovel(con, "API"), "energia", tarifa=1.0, leitura_inicial="00100")
    db.save_reading(con, {"reading": "00150"}, m, "2026-09", "00150")
    r = cliente.get(f"/medidores/{m}/consumo").json()
    assert r["unidade"] == "kWh" and r["leituras"] == [
        {"referencia": "2026-09", "leitura": "00150", "consumo": 50, "conta_estimada": 50.0}]


@pytest.mark.modelos
def test_leitura_de_uma_foto(medidor_sintetico):
    img = medidor_sintetico[0]
    ok, jpg = cv2.imencode(".jpg", img)
    r = cliente.post("/leitura", files={"foto": ("medidor.jpg", io.BytesIO(jpg.tobytes()), "image/jpeg")})
    assert r.status_code in (200, 422)          # 422 = contador não encontrado (foto sintética)
    if r.status_code == 200:
        corpo = r.json()
        assert set(corpo) >= {"leitura", "confianca", "leitores", "caixa_contador"}
        assert 0 <= corpo["confianca"] <= 1


def test_saude_informa_os_dois_tipos():
    prontos = cliente.get("/saude").json()["modelos_prontos"]
    assert set(prontos) == {"energia", "agua"}


def test_tipo_de_medidor_invalido():
    r = cliente.post("/leitura?tipo=gas", files={"foto": ("x.jpg", b"123", "image/jpeg")})
    assert r.status_code == 422
