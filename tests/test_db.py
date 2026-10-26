"""Banco de dados: cadastro, leituras, substituição do mês, exclusão em cascata, migração e concorrência."""
import sqlite3
import threading

from autoleitura import consumo, db, demo


def leitura(valor, conf=0.9):
    return {"reading": valor, "confidence": conf, "det_confidence": 0.95, "bbox": [1, 2, 3, 4], "time_ms": 10}


def test_cadastro_e_leituras(con):
    im = db.add_imovel(con, "Casa", "Rua A, 1")
    m = db.add_medidor(con, im, "energia", "frente", 0.9, 12.0, "01000")
    db.save_reading(con, leitura("01150"), m, "2026-08", "01150")
    db.save_reading(con, leitura("01300"), m, "2026-09", "01300", recorte_jpg=b"\xff\xd8jpg")
    lst = db.list_readings(con, m)
    assert [r["referencia"] for r in lst] == ["2026-08", "2026-09"]
    assert db.get_recorte(con, lst[-1]["id"]) == b"\xff\xd8jpg"
    assert db.get_medidor(con, m)["imovel"] == "Casa"


def test_mesmo_mes_substitui_em_vez_de_duplicar(con):
    m = db.add_medidor(con, db.add_imovel(con, "Casa"), "energia")
    db.save_reading(con, leitura("01150"), m, "2026-09", "01150")
    db.save_reading(con, leitura("01160"), m, "2026-09", "01160")
    lst = db.list_readings(con, m)
    assert len(lst) == 1 and lst[0]["valor_real"] == "01160"


def test_listagem_traz_as_mais_recentes(con):
    m = db.add_medidor(con, db.add_imovel(con, "Casa"), "energia")
    for i in range(30):
        y, mm = divmod(2020 * 12 + i, 12)
        db.save_reading(con, leitura(f"{i:05d}"), m, f"{y}-{mm + 1:02d}", f"{i:05d}")
    lst = db.list_readings(con, m, limit=5)
    assert [r["valor_real"] for r in lst] == ["00025", "00026", "00027", "00028", "00029"]   # as 5 últimas, em ordem


def test_excluir_imovel_apaga_medidores_e_leituras(con):
    im = db.add_imovel(con, "Casa")
    m = db.add_medidor(con, im, "agua")
    db.save_reading(con, leitura("00010"), m, "2026-09", "00010")
    db.delete_imovel(con, im)
    assert db.list_medidores(con) == [] and db.list_readings(con) == []


def test_energia_aparece_antes_da_agua(con):
    im = db.add_imovel(con, "Casa")
    db.add_medidor(con, im, "agua")
    db.add_medidor(con, im, "energia")
    assert [m["tipo"] for m in db.list_medidores(con)] == ["energia", "agua"]


def test_update_medidor_ignora_campos_nao_permitidos(con):
    m = db.add_medidor(con, db.add_imovel(con, "Casa"), "energia", tarifa=0.5)
    db.update_medidor(con, m, tarifa=1.1, tipo="agua")
    md = db.get_medidor(con, m)
    assert md["tarifa"] == 1.1 and md["tipo"] == "energia"


def test_migracao_de_banco_antigo(tmp_path):
    # banco da primeira versão do app: tabela leituras sem medidor, mês nem recorte
    antigo = tmp_path / "antigo.db"
    c = sqlite3.connect(antigo)
    c.execute("CREATE TABLE leituras (id INTEGER PRIMARY KEY AUTOINCREMENT, criado_em TEXT NOT NULL, arquivo TEXT,"
              " tipo_medidor TEXT, leitura TEXT, confianca REAL, conf_deteccao REAL, bbox TEXT, valor_real TEXT, tempo_ms REAL)")
    c.execute("INSERT INTO leituras (criado_em, leitura) VALUES ('2026-01-01', '01234')")
    c.commit()
    c.close()
    con = db.connect(antigo)
    assert db.list_readings(con)[0]["leitura"] == "01234"
    db.save_reading(con, leitura("01300"), None, None, "01300")      # colunas novas já existem
    con.close()


def test_demo_e_idempotente_e_coerente(con):
    assert demo.criar_se_vazio(con) is True
    assert demo.criar_se_vazio(con) is False
    imoveis = db.list_imoveis(con)
    assert [i["nome"] for i in imoveis] == ["Apresentação (fotos ao vivo)", "Casa de demonstração"]
    casa = next(m for m in db.list_medidores(con) if m["imovel"] == "Casa de demonstração" and m["tipo"] == "agua")
    hist = consumo.historico(casa, db.list_readings(con, casa["id"]))
    assert len(hist) == 6 and all(h["consumo"] for h in hist)
    # o último mês da demonstração dispara o alerta de vazamento
    alertas = consumo.validar(hist[-1]["valor"], "agua", anterior=hist[-2]["valor"],
                              consumos_anteriores=[h["consumo"] for h in hist[:-1]])
    assert any("vazamento" in a.mensagem for a in alertas)


def test_escritas_simultaneas_de_varios_usuarios(tmp_path):
    caminho = tmp_path / "concorrente.db"
    m = db.add_medidor(db.connect(caminho), db.add_imovel(db.connect(caminho), "Casa"), "energia")
    erros = []

    def usuario(n):
        c = db.connect(caminho)                 # como no app: uma conexão por sessão
        try:
            for i in range(20):
                db.save_reading(c, leitura("00001"), m, f"30{n:02d}-{i % 12 + 1:02d}", "00001")
        except Exception as e:                  # "database is locked" seria um erro aqui
            erros.append(e)
        finally:
            c.close()

    ts = [threading.Thread(target=usuario, args=(n,)) for n in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert erros == []
    assert len(db.list_readings(db.connect(caminho), m, limit=1000)) == 8 * 12   # 12 meses distintos por usuário
