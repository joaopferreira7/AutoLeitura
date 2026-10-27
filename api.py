"""API REST do AutoLeitura: mostra como uma concessionária integraria a leitura ao app dela.

  uvicorn api:app --host 0.0.0.0 --port 8000
  documentação interativa: http://localhost:8000/docs

Exemplo:
  curl -F "foto=@medidor.jpg" http://localhost:8000/leitura
  curl -F "foto=@hidrometro.jpg" "http://localhost:8000/leitura?tipo=agua"
"""
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from autoleitura import consumo, db, pesos
from autoleitura.preprocess import load_image

app = FastAPI(title="AutoLeitura API", version="1.0",
              description="Leitura automática de medidores de água e energia por visão computacional.")


@lru_cache(maxsize=2)
def get_system(tipo):
    from autoleitura.pipeline import para_tipo
    return para_tipo(tipo)


@app.get("/saude")
def saude():
    return {"status": "ok", "modelos_prontos": {t: pesos.prontos(t) for t in ("energia", "agua")}}


@app.post("/leitura")
async def ler(foto: UploadFile = File(..., description="Foto do medidor (JPG/PNG)"),
              tipo: Literal["energia", "agua"] = Query("energia", description="Tipo de medidor")):
    """Lê o contador da foto e devolve o valor com o índice de confiança."""
    if not pesos.prontos(tipo):
        raise HTTPException(503, f"Modelos de {tipo} indisponíveis.")
    try:
        img = load_image(await foto.read())
    except ValueError:
        raise HTTPException(400, "Arquivo não é uma imagem válida.")
    res = get_system(tipo).read(img)
    if res["reading"] is None:
        raise HTTPException(422, res["warning"])
    return {
        "leitura": res["reading"],
        "confianca": round(res["confidence"], 4),
        "confianca_deteccao": round(res["det_confidence"], 4),
        "confianca_por_digito": [round(c, 4) for c in res["digit_confidences"]],
        "leitores": res["leitores"],
        "caixa_contador": res["bbox"],
        "aviso": res["warning"],
        "tempo_ms": round(res["time_ms"], 1),
    }


@app.get("/medidores/{medidor_id}/consumo")
def historico_consumo(medidor_id: int):
    """Histórico de leituras e consumo mensal de um medidor cadastrado no app."""
    con = db.connect()
    medidor = db.get_medidor(con, medidor_id)
    if medidor is None:
        raise HTTPException(404, "Medidor não encontrado.")
    hist = consumo.historico(medidor, db.list_readings(con, medidor_id))
    return {
        "medidor": {k: medidor[k] for k in ("id", "imovel", "tipo", "descricao", "tarifa", "taxa_fixa")},
        "unidade": consumo.UNIDADE[medidor["tipo"]],
        "leituras": [{"referencia": h["referencia"], "leitura": h["valor"], "consumo": h["consumo"],
                      "conta_estimada": None if h["conta"] is None else round(h["conta"], 2)} for h in hist],
    }
