"""Armazenamento em SQLite (etapa 7 da arquitetura): imóveis, medidores, leituras e métricas."""
import json
import sqlite3
from datetime import datetime

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS imoveis (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    nome        TEXT NOT NULL,
    endereco    TEXT,
    criado_em   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS medidores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    imovel_id       INTEGER NOT NULL REFERENCES imoveis(id) ON DELETE CASCADE,
    tipo            TEXT NOT NULL,          -- 'energia' | 'agua'
    descricao       TEXT,
    tarifa          REAL DEFAULT 0,         -- R$ por kWh ou por m³
    taxa_fixa       REAL DEFAULT 0,         -- R$ fixos por mês (iluminação pública, esgoto...)
    leitura_inicial TEXT,                   -- usada como "anterior" na primeira leitura
    criado_em       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leituras (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em     TEXT NOT NULL,
    medidor_id    INTEGER REFERENCES medidores(id) ON DELETE CASCADE,
    referencia    TEXT,                     -- mês de referência, 'AAAA-MM'
    arquivo       TEXT,
    tipo_medidor  TEXT,
    leitura       TEXT,                     -- valor lido pelo modelo
    valor_real    TEXT,                     -- valor confirmado pelo usuário (pode ter sido corrigido)
    confianca     REAL,
    conf_deteccao REAL,
    bbox          TEXT,                     -- JSON [x1, y1, x2, y2]
    tempo_ms      REAL,
    recorte       BLOB                      -- JPEG do recorte do contador, para conferência/auditoria
);
CREATE TABLE IF NOT EXISTS avaliacoes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em   TEXT NOT NULL,
    conjunto    TEXT,
    n           INTEGER,
    reading_acc REAL,
    digit_acc   REAL,
    mean_lev    REAL,
    det_map50   REAL,
    detalhes    TEXT                        -- JSON com métricas por dígito
);
"""

# colunas adicionadas depois da primeira versão do banco (migração simples)
_NOVAS_COLUNAS = {"leituras": {"medidor_id": "INTEGER", "referencia": "TEXT", "recorte": "BLOB"}}


def connect(path=config.DB_PATH):
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    for tabela, colunas in _NOVAS_COLUNAS.items():
        existentes = {r["name"] for r in con.execute(f"PRAGMA table_info({tabela})")}
        for nome, tipo in colunas.items():
            if nome not in existentes:
                con.execute(f"ALTER TABLE {tabela} ADD COLUMN {nome} {tipo}")
    con.commit()
    return con


def _now():
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- imóveis e medidores
def add_imovel(con, nome, endereco=None):
    cur = con.execute("INSERT INTO imoveis (nome, endereco, criado_em) VALUES (?, ?, ?)", (nome, endereco, _now()))
    con.commit()
    return cur.lastrowid


def list_imoveis(con):
    return [dict(r) for r in con.execute("SELECT * FROM imoveis ORDER BY nome")]


def delete_imovel(con, imovel_id):
    con.execute("DELETE FROM imoveis WHERE id = ?", (imovel_id,))
    con.commit()


def add_medidor(con, imovel_id, tipo, descricao=None, tarifa=0.0, taxa_fixa=0.0, leitura_inicial=None):
    cur = con.execute(
        "INSERT INTO medidores (imovel_id, tipo, descricao, tarifa, taxa_fixa, leitura_inicial, criado_em)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (imovel_id, tipo, descricao, tarifa, taxa_fixa, leitura_inicial or None, _now()))
    con.commit()
    return cur.lastrowid


def update_medidor(con, medidor_id, **campos):
    permitidos = {"descricao", "tarifa", "taxa_fixa", "leitura_inicial"}
    campos = {k: v for k, v in campos.items() if k in permitidos}
    if campos:
        sets = ", ".join(f"{k} = ?" for k in campos)
        con.execute(f"UPDATE medidores SET {sets} WHERE id = ?", (*campos.values(), medidor_id))
        con.commit()


def delete_medidor(con, medidor_id):
    con.execute("DELETE FROM medidores WHERE id = ?", (medidor_id,))
    con.commit()


def list_medidores(con, imovel_id=None):
    sql = ("SELECT m.*, i.nome AS imovel FROM medidores m JOIN imoveis i ON i.id = m.imovel_id"
           + (" WHERE m.imovel_id = ?" if imovel_id else "") + " ORDER BY i.nome, m.tipo = 'agua', m.id")
    return [dict(r) for r in con.execute(sql, (imovel_id,) if imovel_id else ())]


def get_medidor(con, medidor_id):
    r = con.execute("SELECT m.*, i.nome AS imovel FROM medidores m JOIN imoveis i ON i.id = m.imovel_id"
                    " WHERE m.id = ?", (medidor_id,)).fetchone()
    return dict(r) if r else None


# ---------------------------------------------------------------- leituras
def save_reading(con, result, medidor_id=None, referencia=None, valor_real=None,
                 arquivo=None, tipo_medidor=None, recorte_jpg=None):
    cur = con.execute(
        "INSERT INTO leituras (criado_em, medidor_id, referencia, arquivo, tipo_medidor, leitura, valor_real,"
        " confianca, conf_deteccao, bbox, tempo_ms, recorte) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (_now(), medidor_id, referencia, arquivo, tipo_medidor, result.get("reading"), valor_real,
         result.get("confidence"), result.get("det_confidence"), json.dumps(result.get("bbox")),
         result.get("time_ms"), recorte_jpg),
    )
    con.commit()
    return cur.lastrowid


def list_readings(con, medidor_id=None, limit=500):
    """Leituras confirmadas, da mais antiga para a mais recente (ordem do histórico de consumo)."""
    sql = ("SELECT id, criado_em, medidor_id, referencia, arquivo, tipo_medidor, leitura, valor_real, confianca,"
           " conf_deteccao, tempo_ms FROM leituras"
           + (" WHERE medidor_id = ?" if medidor_id else "") + " ORDER BY referencia, criado_em LIMIT ?")
    return [dict(r) for r in con.execute(sql, (medidor_id, limit) if medidor_id else (limit,))]


def get_recorte(con, leitura_id):
    r = con.execute("SELECT recorte FROM leituras WHERE id = ?", (leitura_id,)).fetchone()
    return r["recorte"] if r else None


def delete_reading(con, leitura_id):
    con.execute("DELETE FROM leituras WHERE id = ?", (leitura_id,))
    con.commit()


# ---------------------------------------------------------------- métricas do modelo
def save_evaluation(con, conjunto, m, det_map50=None):
    con.execute(
        "INSERT INTO avaliacoes (criado_em, conjunto, n, reading_acc, digit_acc, mean_lev, det_map50, detalhes)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (_now(), conjunto, m["n"], m["reading_acc"], m["digit_acc"], m["mean_lev"], det_map50,
         json.dumps(m["per_digit"])),
    )
    con.commit()


def list_evaluations(con):
    return [dict(r) for r in con.execute("SELECT * FROM avaliacoes ORDER BY id DESC")]
