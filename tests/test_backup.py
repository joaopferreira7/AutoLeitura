"""Backup/restauração do banco e exportação das leituras confirmadas para o re-treino."""
import csv
import zipfile

import pytest

from autoleitura import backup, db
from importar_correcoes import importar


def leitura(valor, conf=0.9):
    return {"reading": valor, "confidence": conf, "det_confidence": 0.95, "bbox": [1, 2, 3, 4], "time_ms": 10}


def test_backup_e_restauracao(con, tmp_path):
    m = db.add_medidor(con, db.add_imovel(con, "Casa"), "energia")
    db.save_reading(con, leitura("01150"), m, "2026-09", "01150", recorte_jpg=b"jpg")
    dados = backup.gerar(con)

    outro = db.connect(tmp_path / "outro.db")
    db.add_imovel(outro, "Vai sumir")
    backup.restaurar(outro, dados)
    assert [i["nome"] for i in db.list_imoveis(outro)] == ["Casa"]
    assert db.list_readings(outro)[0]["valor_real"] == "01150"
    outro.close()


def test_restaurar_recusa_arquivo_invalido(con):
    db.add_imovel(con, "Casa")
    for lixo in (b"isso nao e um banco", b""):
        with pytest.raises(ValueError):
            backup.restaurar(con, lixo)
    assert [i["nome"] for i in db.list_imoveis(con)] == ["Casa"]       # banco atual intacto


def test_correcoes_viram_dados_de_treino(con, tmp_path):
    im = db.add_imovel(con, "Casa")
    e = db.add_medidor(con, im, "energia")
    a = db.add_medidor(con, im, "agua")
    db.save_reading(con, leitura("01150"), e, "2026-08", "01160", recorte_jpg=b"e1")     # corrigida
    db.save_reading(con, leitura("012345"), a, "2026-08", "012345", recorte_jpg=b"a1")
    db.save_reading(con, leitura("01170"), e, "2026-09", "01170")                        # sem recorte: fica de fora
    db.save_reading(con, leitura("01180", conf=None), e, "2026-10", "01180", recorte_jpg=b"d")   # demonstração

    caminho = tmp_path / "correcoes.zip"
    caminho.write_bytes(backup.zip_correcoes(con))
    with zipfile.ZipFile(caminho) as z:
        assert len([n for n in z.namelist() if n.startswith("images/")]) == 2

    pastas = {"energia": tmp_path / "rec", "agua": tmp_path / "agua_rec"}
    assert importar(caminho, pastas) == {"energia": 1, "agua": 1}
    assert importar(caminho, pastas) == {"energia": 0, "agua": 0}        # importar de novo não duplica
    with open(pastas["energia"] / "labels.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    assert [(r["leitura"], r["split"]) for r in linhas] == [("01160", "train")]
    assert (pastas["energia"] / "images" / linhas[0]["arquivo"]).read_bytes() == b"e1"
