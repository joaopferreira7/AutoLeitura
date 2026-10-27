"""Cópia de segurança do banco e exportação das leituras confirmadas para re-treinar os modelos.

Na nuvem gratuita do Streamlit o disco é apagado quando o app reinicia: o usuário baixa um
backup (.db) e o restaura depois. As leituras confirmadas (com o recorte salvo) viram um .zip
no mesmo formato de data/rec, que scripts/importar_correcoes.py junta ao treino.
"""
import csv
import io
import sqlite3
import tempfile
import zipfile
from pathlib import Path

TABELAS = {"imoveis", "medidores", "leituras"}


def gerar(con):
    """Bytes de um arquivo SQLite com todo o conteúdo do banco (feito pela API de backup do SQLite,
    que gera uma cópia consistente mesmo com outras pessoas gravando)."""
    with tempfile.TemporaryDirectory() as d:
        caminho = Path(d) / "backup.db"
        destino = sqlite3.connect(caminho)
        con.backup(destino)
        destino.close()
        return caminho.read_bytes()


def restaurar(con, dados):
    """Substitui o conteúdo do banco pelo de um backup. Recusa arquivos que não sejam um banco
    do AutoLeitura (ValueError) sem mexer no banco atual."""
    with tempfile.TemporaryDirectory() as d:
        caminho = Path(d) / "restaurar.db"
        caminho.write_bytes(dados)
        origem = sqlite3.connect(caminho)
        try:
            try:
                tabelas = {r[0] for r in origem.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            except sqlite3.DatabaseError:
                tabelas = set()
            if not TABELAS <= tabelas:
                raise ValueError("O arquivo não é um backup do AutoLeitura.")
            origem.backup(con)
        finally:
            origem.close()      # no Windows, o arquivo temporário só pode ser apagado depois de fechado
    # bancos de versões antigas: cria tabelas e colunas que faltarem
    from . import db
    db.migrar(con)


def correcoes(con, tipo=None):
    """Leituras feitas pelo modelo e confirmadas pelo usuário, com o recorte salvo."""
    sql = ("SELECT l.id, l.leitura, l.valor_real, l.recorte, COALESCE(m.tipo, l.tipo_medidor) AS tipo"
           " FROM leituras l LEFT JOIN medidores m ON m.id = l.medidor_id"
           " WHERE l.recorte IS NOT NULL AND l.valor_real IS NOT NULL AND l.confianca IS NOT NULL")
    linhas = [dict(r) for r in con.execute(sql)]
    return [r for r in linhas if tipo is None or r["tipo"] == tipo]


def zip_correcoes(con, tipo=None):
    """.zip com images/ e labels.csv (arquivo, leitura, split, origem, tipo, modelo) no formato de data/rec."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        tabela = io.StringIO()
        w = csv.writer(tabela, lineterminator="\n")
        w.writerow(["arquivo", "leitura", "split", "origem", "tipo", "modelo"])
        for r in correcoes(con, tipo):
            nome = f"app_{r['tipo']}_{r['id']:06d}.jpg"
            z.writestr(f"images/{nome}", r["recorte"])
            w.writerow([nome, r["valor_real"], "train", f"app:{r['id']}", r["tipo"], r["leitura"]])
        z.writestr("labels.csv", tabela.getvalue())
    return buf.getvalue()
