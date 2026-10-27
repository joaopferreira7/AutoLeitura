"""Avaliação ponta a ponta no conjunto de teste (seção 11 da proposta).

Roda o pipeline completo (detector + reconhecedor) em cada imagem de teste e compara
com o ground truth. Também mede o reconhecedor isolado, sobre os recortes anotados,
para separar o erro de detecção do erro de reconhecimento.

  python evaluate.py
  python evaluate.py --errors erros.csv     # salva as leituras erradas para análise
  python evaluate.py --tipo agua            # medidores de água: detector e CRNN medidos separadamente
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from autoleitura import config, db
from autoleitura.consumo import CONFIANCA_MINIMA
from autoleitura.metrics import evaluate_readings
from autoleitura.pipeline import AutoLeitura
from autoleitura.recognizer import Recognizer


def concordancia_agua(split, n=1000):
    """O dataset de água não liga as fotos às leituras, então a acurácia de ponta a ponta não pode ser
    medida. Como aproximação: em n fotos de teste, compara a leitura do pipeline completo (janela
    prevista pelo segmentador) com a da CRNN na janela ANOTADA à mão. Mede o quanto o segmentador
    piora a leitura em relação a um recorte perfeito."""
    import sys
    sys.path.insert(0, str(config.ROOT / "scripts"))
    from prepare_agua import poligono_da_mascara

    from autoleitura.pipeline import para_tipo
    from autoleitura.preprocess import endireitar

    raiz = config.RAW_DIR / "agua" / "Word-Wheel_Water_Meter_Dataset" / "detection" / split
    if not raiz.exists():
        return None
    fotos = sorted((raiz / f"{split}_img").glob("*.png"))
    fotos = fotos[:: max(1, len(fotos) // n)][:n]
    sistema = para_tipo("agua")
    iguais = validas = 0
    for p in fotos:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        poligono = poligono_da_mascara(raiz / f"{split}_seg_label" / p.name)
        referencia = sistema.recognizer.read(endireitar(img, [(x * w, y * h) for x, y in poligono]))[0]
        res = sistema.read(img)
        validas += res["reading"] is not None and len(res["reading"]) in sistema.digitos
        iguais += res["reading"] == referencia
    return {"n": len(fotos), "concordancia": iguais / len(fotos), "leituras_validas": validas / len(fotos)}


def avaliar_agua(split):
    """O dataset de água não liga as fotos de detecção às leituras: o segmentador é medido nas fotos
    (mAP da caixa e da máscara) e a CRNN nos recortes anotados, em um sentido e nos dois sentidos
    (como no app). A concordância (concordancia_agua) aproxima o pipeline completo."""
    det = YOLO(str(config.AGUA_SEG_WEIGHTS)).val(data=str(config.AGUA_DET_DIR / "data.yaml"), split=split,
                                                 imgsz=config.DET_IMG_SIZE, verbose=False,
                                                 project=str(config.ROOT / "runs" / "eval"), name="det_agua",
                                                 exist_ok=True)
    conc = concordancia_agua(split)
    with open(config.AGUA_REC_DIR / "labels.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    crops = [cv2.imread(str(config.AGUA_REC_DIR / "images" / r["arquivo"])) for r in rows]
    gts = [r["leitura"] for r in rows]
    um_sentido = Recognizer(config.AGUA_REC_WEIGHTS)
    dois_sentidos = Recognizer(config.AGUA_REC_WEIGHTS, nos_dois_sentidos=True)
    m1 = evaluate_readings([um_sentido.read(c)[0] for c in crops], gts)
    lidos = [dois_sentidos.read(c) for c in crops]
    m2 = evaluate_readings([t for t, _, _ in lidos], gts)
    print(f"\n=========== RESULTADOS ÁGUA ({split}) ===========")
    print(f"Segmentador da janela  caixa mAP@0.5 = {det.box.map50:.4f}  máscara mAP@0.5 = {det.seg.map50:.4f}"
          f"  ({split}, {len(list((config.AGUA_DET_DIR / 'images' / split).glob('*.png')))} fotos)")
    if conc:
        print(f"Pipeline completo: em {conc['n']} fotos, {conc['concordancia']:.1%} das leituras iguais às da janela"
              f" anotada; {conc['leituras_validas']:.1%} com 5 ou 6 dígitos")
    print(f"CRNN, um sentido      leitura completa = {m1['reading_acc']:.4f}")
    print(f"CRNN, dois sentidos   leitura completa = {m2['reading_acc']:.4f}  <- como no app  (n={m2['n']})")
    print(f"CRNN, dois sentidos   acurácia por dígito = {m2['digit_acc']:.4f}  Levenshtein = {m2['mean_lev']:.4f}")
    for n in (5, 6):
        idx = [i for i, g in enumerate(gts) if len(g) == n]
        mn = evaluate_readings([lidos[i][0] for i in idx], [gts[i] for i in idx])
        print(f"  contadores de {n} dígitos: {mn['reading_acc']:.4f}  (n={mn['n']})")
    aceitas = [t == g for (t, c, _), g in zip(lidos, gts) if c >= CONFIANCA_MINIMA]
    print(f"Confiança da CRNN >= {CONFIANCA_MINIMA:.0%}: {len(aceitas) / len(gts):.1%} dos recortes,"
          f" com {sum(aceitas) / max(1, len(aceitas)):.2%} de acerto")
    con = db.connect()
    db.save_evaluation(con, f"{split} água (CRNN, dois sentidos)", m2, det_map50=float(det.box.map50))
    print(f"\nMétricas gravadas em {config.DB_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", choices=["energia", "agua"], default="energia")
    ap.add_argument("--split", default="test")
    ap.add_argument("--errors", default=None, help="CSV para salvar as leituras erradas")
    args = ap.parse_args()
    if args.tipo == "agua":
        return avaliar_agua(args.split)

    with open(config.REC_DIR / "labels.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split]
    gt = {Path(r["arquivo"]).stem: r["leitura"] for r in rows}

    # 1) detector: mAP@0.5
    det = YOLO(str(config.DET_WEIGHTS)).val(data=str(config.DET_DIR / "data.yaml"), split=args.split,
                                            imgsz=config.DET_IMG_SIZE, verbose=False,
                                            project=str(config.ROOT / "runs" / "eval"), name="det", exist_ok=True)
    map50 = float(det.box.map50)

    # 2) leitores isolados (recortes do ground truth): separa o erro de detecção do erro de leitura
    system = AutoLeitura()
    crops = [cv2.imread(str(config.REC_DIR / "images" / r["arquivo"])) for r in rows]
    gts_rec = [r["leitura"] for r in rows]
    rec_m = evaluate_readings([system.recognizer.read(c)[0] for c in crops], gts_rec)
    dig_m = (evaluate_readings([system.digit_reader.read(c)[0] for c in crops], gts_rec)
             if system.digit_reader else None)

    # 3) pipeline completo
    preds, gts, errors, times, confs = [], [], [], [], []
    for img_path in sorted((config.DET_DIR / "images" / args.split).iterdir()):
        if img_path.stem not in gt:
            continue
        res = system.read(cv2.imread(str(img_path)))
        p = res["reading"] or ""
        preds.append(p)
        gts.append(gt[img_path.stem])
        times.append(res["time_ms"])
        confs.append(res["confidence"])
        if p != gt[img_path.stem]:
            errors.append([img_path.name, gt[img_path.stem], p, f"{res['confidence']:.3f}"])
    m = evaluate_readings(preds, gts)

    print("\n=========== RESULTADOS (%s, n=%d) ===========" % (args.split, m["n"]))
    print(f"Detecção          mAP@0.5               : {map50:.4f}")
    print(f"CRNN isolada      acurácia de leitura   : {rec_m['reading_acc']:.4f}  (recortes anotados)")
    if dig_m:
        print(f"YOLO de dígitos   acurácia de leitura   : {dig_m['reading_acc']:.4f}  (recortes anotados)")
    print(f"Pipeline completo acurácia de leitura   : {m['reading_acc']:.4f}  <- métrica principal")
    print(f"Pipeline completo acurácia por dígito   : {m['digit_acc']:.4f}")
    print(f"Pipeline completo Levenshtein médio     : {m['mean_lev']:.4f}")
    print(f"Tempo médio por imagem                  : {sum(times) / len(times):.0f} ms")
    aceitas = [p == g for p, g, c in zip(preds, gts, confs) if c >= CONFIANCA_MINIMA]
    print(f"Aceitas direto no app (confiança >= {CONFIANCA_MINIMA:.0%}): {len(aceitas) / len(preds):.1%} das fotos,"
          f" com {sum(aceitas) / max(1, len(aceitas)):.2%} de acerto")
    print("Acurácia por dígito (0–9):", json.dumps({k: round(v, 3) if v is not None else None
                                                  for k, v in m["per_digit"].items()}))

    con = db.connect()
    db.save_evaluation(con, f"{args.split} (pipeline)", m, det_map50=map50)
    db.save_evaluation(con, f"{args.split} (CRNN isolada)", rec_m)
    if dig_m:
        db.save_evaluation(con, f"{args.split} (YOLO de dígitos isolado)", dig_m)
    print(f"\nMétricas gravadas em {config.DB_PATH}")

    if args.errors:
        with open(args.errors, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["imagem", "real", "previsto", "confianca"])
            w.writerows(errors)
        print(f"{len(errors)} erros salvos em {args.errors}")


if __name__ == "__main__":
    main()
