"""Cria os imóveis de demonstração, se o banco estiver vazio.

Serve para a apresentação: "Apresentação (fotos ao vivo)" tem medidores sem histórico, para
fotografar um medidor real; "Casa de demonstração" tem seis meses de leituras, para mostrar o
gráfico de consumo, a conta estimada e o alerta de vazamento.
Não mexe em bancos que já têm imóveis cadastrados.

  python scripts/criar_demo.py
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoleitura import config, db  # noqa: E402


def meses_anteriores(n):
    """Os n meses antes do mês atual, do mais antigo para o mais recente, no formato AAAA-MM."""
    hoje = date.today()
    out = []
    for i in range(n, 0, -1):
        y, m = divmod(hoje.year * 12 + hoje.month - 1 - i, 12)
        out.append(f"{y:04d}-{m + 1:02d}")
    return out


def criar(con):
    # medidores sem histórico, para fotografar qualquer medidor real ao vivo sem conflitar com
    # leituras inventadas (o nome com "A" faz este imóvel aparecer primeiro na lista)
    ao_vivo = db.add_imovel(con, "Apresentação (fotos ao vivo)")
    db.add_medidor(con, ao_vivo, "energia", tarifa=0.90, taxa_fixa=15.0)
    db.add_medidor(con, ao_vivo, "agua", tarifa=6.50, taxa_fixa=30.0)

    imovel = db.add_imovel(con, "Casa de demonstração", "Bauru/SP")
    # tarifas apenas de exemplo
    energia = db.add_medidor(con, imovel, "energia", "relógio da frente", tarifa=0.90, taxa_fixa=15.0,
                             leitura_inicial="04210")
    agua = db.add_medidor(con, imovel, "agua", None, tarifa=6.50, taxa_fixa=30.0, leitura_inicial="00512")

    consumos = {energia: [182, 175, 190, 168, 201, 186], agua: [17, 19, 16, 18, 17, 34]}   # água: vazamento no último mês
    inicial = {energia: 4210, agua: 512}
    for medidor, lista in consumos.items():
        valor = inicial[medidor]
        for ref, c in zip(meses_anteriores(len(lista)), lista):
            valor += c
            leitura = f"{valor:05d}"
            db.save_reading(con, {"reading": leitura, "confidence": None, "det_confidence": None,
                                  "bbox": None, "time_ms": None},
                            medidor_id=medidor, referencia=ref, valor_real=leitura, arquivo="demonstração",
                            tipo_medidor="energia" if medidor == energia else "agua")


if __name__ == "__main__":
    con = db.connect()
    if db.list_imoveis(con):
        print(f"{config.DB_PATH} já tem imóveis; nada a fazer.")
    else:
        criar(con)
        print(f"Imóvel de demonstração criado em {config.DB_PATH}")
