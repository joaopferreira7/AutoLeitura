"""Regras de negócio da autoleitura: consumo do mês, conta estimada e validação da leitura.

É aqui que a leitura do modelo vira informação útil para o consumidor e para a
concessionária: detecção de leituras impossíveis (menores que a anterior), virada do
contador, consumo fora do padrão e leituras de baixa confiança.
"""
from dataclasses import dataclass

from . import config

UNIDADE = {"energia": "kWh", "agua": "m³"}
CONFIANCA_MINIMA = 0.6


@dataclass
class Alerta:
    nivel: str          # "erro" (impede salvar sem confirmação), "aviso" ou "info"
    mensagem: str


def consumo(anterior, atual):
    """Consumo entre duas leituras. Trata a virada do contador (ex.: 99950 -> 00030 = 80).

    O contador volta a zero conforme o número de dígitos da leitura anterior:
    depois de 99999 (5 dígitos) ou de 999999 (6 dígitos, comum em medidores de água).
    Retorna None quando não há leitura anterior ou quando a leitura atual é menor
    que a anterior sem ser uma virada plausível (leitura inválida).
    """
    if anterior is None or atual is None:
        return None
    a, b = int(anterior), int(atual)
    if b >= a:
        return b - a
    limite = 10 ** len(str(anterior))
    if a > 0.9 * limite and b < 0.1 * limite:
        return b + limite - a
    return None


def houve_virada(anterior, atual):
    return anterior is not None and int(atual) < int(anterior) and consumo(anterior, atual) is not None


def conta_estimada(consumo_mes, tarifa, taxa_fixa=0.0):
    if consumo_mes is None:
        return None
    return consumo_mes * (tarifa or 0.0) + (taxa_fixa or 0.0)


def media(valores):
    valores = [v for v in valores if v is not None]
    return sum(valores) / len(valores) if valores else None


def historico(medidor, leituras):
    """Leituras salvas (em ordem) -> linhas com o consumo do período e a conta estimada.

    A leitura confirmada pelo usuário (valor_real) prevalece sobre a do modelo.
    """
    linhas, anterior = [], medidor.get("leitura_inicial")
    for lt in leituras:
        valor = lt.get("valor_real") or lt.get("leitura")
        c = consumo(anterior, valor)
        linhas.append({**lt, "valor": valor, "consumo": c,
                       "conta": conta_estimada(c, medidor.get("tarifa"), medidor.get("taxa_fixa"))})
        anterior = valor
    return linhas


def validar(leitura, tipo, confianca=None, anterior=None, consumos_anteriores=(), referencia_repetida=False,
            posterior=None):
    """Confere uma leitura antes de salvar. Retorna a lista de alertas (vazia = tudo certo).

    `posterior` = a leitura do mês seguinte, quando a pessoa lança um mês que ficou para trás:
    o valor novo não pode ser maior que ela, senão o consumo daquele mês ficaria negativo.
    """
    alertas = []
    un = UNIDADE.get(tipo, "")

    if not leitura or not leitura.isdigit():
        return [Alerta("erro", "A leitura deve conter apenas números.")]
    aceitos = config.DIGITOS.get(tipo, (config.NUM_DIGITS,))
    if len(leitura) not in aceitos:
        esperado = " ou ".join(str(n) for n in aceitos)
        alertas.append(Alerta("erro", f"A leitura tem {len(leitura)} dígitos; o contador tem {esperado}."))

    if confianca is not None and confianca < CONFIANCA_MINIMA:
        alertas.append(Alerta("aviso", f"Confiança baixa ({confianca * 100:.0f}%). Confira os números com o medidor."))

    if referencia_repetida:
        alertas.append(Alerta("aviso", "Já existe uma leitura deste medidor neste mês: ela será substituída ao salvar."))

    if posterior is not None and consumo(leitura, posterior) is None:
        alertas.append(Alerta("erro", f"A leitura é maior que a do mês seguinte ({posterior}). "
                                      "Confira o mês de referência e os números."))

    if anterior is None:
        alertas.append(Alerta("info", "Primeira leitura deste medidor: o consumo será calculado a partir da próxima."))
        return alertas

    c = consumo(anterior, leitura)
    if c is None:
        alertas.append(Alerta("erro", f"A leitura é menor que a anterior ({anterior}). "
                                      "Pode ser erro de leitura ou indício de fraude."))
        return alertas
    if houve_virada(anterior, leitura):
        alertas.append(Alerta("info", f"O contador deu a volta (passou de 99999 para 00000). Consumo: {c} {un}."))
    if c == 0:
        alertas.append(Alerta("aviso", "Consumo zero no período: imóvel desocupado ou medidor parado?"))

    recentes = [v for v in consumos_anteriores if v is not None][-6:]     # até 6 meses válidos
    m = media(recentes)
    if m and len(recentes) >= 2:
        if c > 1.5 * m:
            extra = " Possível vazamento." if tipo == "agua" else ""
            alertas.append(Alerta("aviso", f"Consumo {c / m * 100 - 100:.0f}% acima da média dos meses anteriores ({m:.0f} {un}).{extra}"))
        elif c < 0.3 * m:
            alertas.append(Alerta("aviso", f"Consumo muito abaixo da média dos meses anteriores ({m:.0f} {un}). Confira a leitura."))
    return alertas
