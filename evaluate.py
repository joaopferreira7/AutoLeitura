"""Avaliação ponta a ponta no conjunto de teste (seção 11 da proposta).

Roda o pipeline completo (detector + reconhecedor) em cada imagem de teste e compara
com o ground truth. Também mede o reconhecedor isolado, sobre os recortes anotados,
para separar o erro de detecção do erro de reconhecimento.

  python evaluate.py
  python evaluate.py --errors erros.csv     # salva as leituras erradas para análise
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--errors", default=None, help="CSV para salvar as leituras erradas")
    args = ap.parse_args()

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
