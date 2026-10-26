"""App Streamlit: cada página abre sem erro e o cadastro funciona (banco temporário com a demonstração)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

AUXILIAR = str(Path(__file__).with_name("_app_na_pagina.py"))


@pytest.fixture
def abrir(monkeypatch):
    monkeypatch.setenv("AUTOLEITURA_DEMO", "1")

    def _abrir(pagina=""):
        monkeypatch.setenv("AUTOLEITURA_TESTE_PAGINA", pagina)
        at = AppTest.from_file(AUXILIAR, default_timeout=180).run()
        assert not at.exception, [e.value for e in at.exception]
        return at
    return _abrir


def textos(at):
    return " ".join(str(getattr(el, "value", "")) for el in at.main)


def test_nova_leitura(abrir):
    at = abrir("")
    assert at.title[0].value == "📷 Nova leitura"
    assert [s.label for s in at.selectbox] == ["Medidor", "Mês de referência"]


def test_consumo_abre_no_medidor_com_historico(abrir):
    at = abrir("page_consumo")
    assert at.title[0].value == "📊 Consumo"
    rotulos = [m.label for m in at.metric]
    assert "Última leitura" in rotulos and "Conta estimada do mês" in rotulos
    assert not any("Nenhuma leitura" in i.value for i in at.info)


def test_imoveis_cadastro(abrir):
    at = abrir("page_imoveis")
    assert at.title[0].value == "🏠 Imóveis e medidores"
    next(t for t in at.text_input if t.label == "Nome").set_value("Casa do teste")
    next(b for b in at.button if b.label == "Cadastrar imóvel").click()
    at.run()
    assert not at.exception
    assert "Casa do teste" in [s.value for s in at.subheader]


def test_desempenho_do_modelo(abrir):
    at = abrir("page_modelo")
    assert at.title[0].value == "🧠 Desempenho do modelo"
