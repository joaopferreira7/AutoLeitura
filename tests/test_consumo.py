"""Regras de negócio: consumo, virada do contador, conta estimada e alertas."""
import pytest

from autoleitura import consumo, tempo
from autoleitura.metrics import evaluate_readings, levenshtein


def niveis(alertas):
    return [a.nivel for a in alertas]


@pytest.mark.parametrize("anterior, atual, esperado", [
    ("01200", "01350", 150),
    ("01200", "01200", 0),
    ("99950", "00030", 80),          # virada do contador
    ("01350", "01200", None),        # menor que a anterior: inválida
    ("50000", "00010", None),        # queda grande demais para ser virada
    (None, "01200", None),           # primeira leitura
])
def test_consumo(anterior, atual, esperado):
    assert consumo.consumo(anterior, atual) == esperado


def test_conta_estimada():
    assert consumo.conta_estimada(150, 0.9, 10) == pytest.approx(145.0)
    assert consumo.conta_estimada(None, 0.9, 10) is None
    assert consumo.conta_estimada(10, None, None) == 0


def test_leitura_normal_sem_alertas():
    assert consumo.validar("01350", "energia", confianca=0.95, anterior="01200",
                           consumos_anteriores=[140, 160, 150]) == []


def test_leitura_menor_que_anterior_bloqueia():
    assert "erro" in niveis(consumo.validar("01100", "energia", anterior="01200"))


def test_virada_do_contador_e_informada():
    alertas = consumo.validar("00030", "energia", anterior="99950")
    assert niveis(alertas) == ["info"] and "80" in alertas[0].mensagem


def test_consumo_alto_na_agua_sugere_vazamento():
    alertas = consumo.validar("00180", "agua", anterior="00100", consumos_anteriores=[20, 25, 22])
    assert niveis(alertas) == ["aviso"] and "vazamento" in alertas[0].mensagem


def test_media_ignora_meses_sem_consumo():
    # só um mês com consumo válido: média insuficiente, nenhum alerta de consumo fora do padrão
    assert consumo.validar("00180", "agua", anterior="00100", consumos_anteriores=[None, None, 20]) == []


@pytest.mark.parametrize("leitura, mensagem", [("1350", "4 dígitos"), ("01a50", "apenas números"), ("", "apenas números")])
def test_leitura_mal_formada(leitura, mensagem):
    alertas = consumo.validar(leitura, "energia", anterior="01200")
    assert "erro" in niveis(alertas) and any(mensagem in a.mensagem for a in alertas)


def test_confianca_baixa_pede_conferencia():
    assert "aviso" in niveis(consumo.validar("01350", "energia", confianca=0.4, anterior="01200"))


def test_lancamento_retroativo_maior_que_o_mes_seguinte_bloqueia():
    alertas = consumo.validar("01500", "energia", anterior="01200", posterior="01400")
    assert any(a.nivel == "erro" and "mês seguinte" in a.mensagem for a in alertas)


def test_lancamento_retroativo_coerente_passa():
    assert consumo.validar("01300", "energia", anterior="01200", posterior="01400") == []


def test_mes_repetido_avisa_que_substitui():
    alertas = consumo.validar("01300", "energia", anterior="01200", referencia_repetida=True)
    assert any("substituída" in a.mensagem for a in alertas)


def test_historico_usa_a_leitura_corrigida():
    medidor = {"leitura_inicial": "01000", "tarifa": 0.9, "taxa_fixa": 10}
    leituras = [{"referencia": "2026-08", "leitura": "01150", "valor_real": "01150"},
                {"referencia": "2026-09", "leitura": "01390", "valor_real": "01300"}]   # modelo errou, usuário corrigiu
    h = consumo.historico(medidor, leituras)
    assert [(x["valor"], x["consumo"]) for x in h] == [("01150", 150), ("01300", 150)]
    assert h[-1]["conta"] == pytest.approx(145.0)


def test_meses_no_horario_de_brasilia():
    meses = tempo.meses_ate(3)
    assert len(meses) == 3 and meses[0] == tempo.hoje().strftime("%Y-%m") and meses[0] > meses[1] > meses[2]
    assert tempo.agora().utcoffset().total_seconds() == -3 * 3600


@pytest.mark.parametrize("a, b, d", [("01234", "01234", 0), ("01234", "01235", 1), ("0123", "01234", 1), ("", "123", 3)])
def test_levenshtein(a, b, d):
    assert levenshtein(a, b) == d


def test_metricas_de_leitura():
    m = evaluate_readings(["01234", "01235", "0123"], ["01234", "01234", "01234"])
    assert m["n"] == 3
    assert m["reading_acc"] == pytest.approx(1 / 3)
    assert m["digit_acc"] == pytest.approx(9 / 15)          # 5 + 4 + 0 (tamanho errado conta tudo como erro)
    assert m["mean_lev"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------- medidores de água (5 ou 6 dígitos)
def test_virada_do_contador_de_6_digitos():
    assert consumo.consumo("999950", "000030") == 80
    assert consumo.consumo("099950", "000030") is None        # 6 dígitos, longe do fim: não é virada


def test_quantidade_de_digitos_depende_do_tipo():
    assert consumo.validar("001350", "agua", anterior="001200") == []            # 6 dígitos vale na água
    assert consumo.validar("01350", "agua", anterior="01200") == []              # 5 também
    assert "erro" in niveis(consumo.validar("001350", "energia", anterior="01200"))   # na energia, não
    assert "erro" in niveis(consumo.validar("0001350", "agua", anterior="001200"))    # 7 nunca
