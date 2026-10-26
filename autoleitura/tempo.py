"""Data e hora no horário de Brasília, independente do relógio do servidor.

Na nuvem o servidor roda em UTC: sem isto, entre 21h e meia-noite do último dia do mês
o app já sugeriria o mês seguinte como referência. O Brasil não tem horário de verão
desde 2019, então um fuso fixo UTC-3 basta e não depende do pacote tzdata (ausente no Windows).
"""
from datetime import datetime, timedelta, timezone

BRASILIA = timezone(timedelta(hours=-3), "BRT")


def agora():
    return datetime.now(BRASILIA)


def hoje():
    return agora().date()


def meses_ate(n, incluir_atual=True):
    """Os últimos n meses no formato AAAA-MM, do mais recente para o mais antigo."""
    d = hoje()
    base = d.year * 12 + d.month - 1 - (0 if incluir_atual else 1)
    return [f"{(base - i) // 12:04d}-{(base - i) % 12 + 1:02d}" for i in range(n)]
