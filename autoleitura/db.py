"""Armazenamento das leituras e métricas em SQLite (etapa 7 da arquitetura)."""
import json
import sqlite3
from datetime import datetime

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS leituras (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em     TEXT NOT NULL,
    arquivo       TEXT,
    tipo_medidor  TEXT,               -- 'energia' | 'agua' | NULL
    leitura       TEXT,
    confianca     REAL,
    conf_deteccao REAL,
    bbox          TEXT,               -- JSON [x1, y1, x2, y2]
    valor_real    TEXT,               -- preenchido quando o usuário corrige/confirma
    tempo_ms      REAL
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
    detalhes    TEXT                  -- JSON com métricas por dígito
);
"""


def connect(path=config.DB_PATH):
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _now():
    return datetime.now().isoformat(timespec="seconds")


def save_reading(con, result, arquivo=None, tipo_medidor=None):
    cur = con.execute(
        "INSERT INTO leituras (criado_em, arquivo, tipo_medidor, leitura, confianca, conf_deteccao, bbox, tempo_ms)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (_now(), arquivo, tipo_medidor, result["reading"], result["confidence"],
         result["det_confidence"], json.dumps(result["bbox"]), result["time_ms"]),
    )
    con.commit()
    return cur.lastrowid


def set_ground_truth(con, reading_id, valor_real):
    con.execute("UPDATE leituras SET valor_real = ? WHERE id = ?", (valor_real, reading_id))
    con.commit()


def save_evaluation(con, conjunto, m, det_map50=None):
    con.execute(
        "INSERT INTO avaliacoes (criado_em, conjunto, n, reading_acc, digit_acc, mean_lev, det_map50, detalhes)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (_now(), conjunto, m["n"], m["reading_acc"], m["digit_acc"], m["mean_lev"], det_map50,
         json.dumps(m["per_digit"])),
    )
    con.commit()


def list_readings(con, limit=200):
    return [dict(r) for r in con.execute("SELECT * FROM leituras ORDER BY id DESC LIMIT ?", (limit,))]


def list_evaluations(con):
    return [dict(r) for r in con.execute("SELECT * FROM avaliacoes ORDER BY id DESC")]
