"""Treina o YOLOv8 para detectar a região do contador (transfer learning a partir do COCO).

  python train_detector.py --epochs 100 --model yolov8s.pt     # energia (UFPR-AMR), Colab / GPU
  python train_detector.py --epochs 5 --device cpu             # teste rápido
  python train_detector.py --tipo agua --epochs 25 --fracao 0.15 --device 0   # água: segmentador (yolov8s-seg)

Na água, o modelo é de SEGMENTAÇÃO da janela dos dígitos (polígonos em data/agua_det): a janela é
endireitada antes da leitura, como os recortes de treino da CRNN de água.
"""
import argparse
import shutil

from ultralytics import YOLO

from autoleitura import config

DADOS = {"energia": (config.DET_DIR, config.DET_WEIGHTS), "agua": (config.AGUA_DET_DIR, config.AGUA_SEG_WEIGHTS)}
MODELO_BASE = {"energia": config.DET_BASE_MODEL, "agua": "yolov8s-seg.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", choices=list(DADOS), default="energia")
    ap.add_argument("--model", default=None, help="padrão: yolov8n.pt (energia) ou yolov8s-seg.pt (água)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fracao", type=float, default=1.0, help="fração do treino usada por época (datasets grandes)")
    ap.add_argument("--device", default=None, help="'0' para GPU, 'cpu' para CPU")
    args = ap.parse_args()
    pasta, saida = DADOS[args.tipo]
    projeto = str(config.ROOT / "runs" / ("detector" if args.tipo == "energia" else f"detector_{args.tipo}"))

    model = YOLO(args.model or MODELO_BASE[args.tipo])
    model.train(
        data=str(pasta / "data.yaml"),
        imgsz=config.DET_IMG_SIZE,
        epochs=args.epochs,
        batch=args.batch,
        fraction=args.fracao,
        device=args.device,
        project=projeto,
        name="train",
        exist_ok=True,
        patience=20,
        # data augmentation simulando fotos reais (seção 6 da proposta)
        degrees=8, perspective=0.0005, translate=0.1, scale=0.4,
        hsv_v=0.4, hsv_s=0.5, fliplr=0.0,     # sem espelhar: dígitos espelhados não existem
        mosaic=1.0,
    )
    best = model.trainer.best
    config.WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(best, saida)

    metrics = YOLO(str(saida)).val(data=str(pasta / "data.yaml"), split="test", imgsz=config.DET_IMG_SIZE,
                                   device=args.device, project=projeto, name="test", exist_ok=True)
    print(f"\nPesos salvos em {saida}")
    print(f"Teste  mAP@0.5 = {metrics.box.map50:.4f}   mAP@0.5:0.95 = {metrics.box.map:.4f}")
    if hasattr(metrics, "seg"):
        print(f"Teste  máscara mAP@0.5 = {metrics.seg.map50:.4f}   mAP@0.5:0.95 = {metrics.seg.map:.4f}")


if __name__ == "__main__":
    main()
