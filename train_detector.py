"""Treina o YOLOv8 para detectar a região do contador (transfer learning a partir do COCO).

  python train_detector.py --epochs 100 --model yolov8s.pt     # Colab / GPU
  python train_detector.py --epochs 5 --device cpu             # teste rápido
"""
import argparse
import shutil

from ultralytics import YOLO

from autoleitura import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DET_BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="'0' para GPU, 'cpu' para CPU")
    ap.add_argument("--project", default=str(config.ROOT / "runs" / "detector"))
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(config.DET_DIR / "data.yaml"),
        imgsz=config.DET_IMG_SIZE,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
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
    shutil.copy(best, config.DET_WEIGHTS)

    metrics = YOLO(str(config.DET_WEIGHTS)).val(data=str(config.DET_DIR / "data.yaml"), split="test",
                                               imgsz=config.DET_IMG_SIZE, device=args.device,
                                               project=args.project, name="test", exist_ok=True)
    print(f"\nPesos salvos em {config.DET_WEIGHTS}")
    print(f"Teste  mAP@0.5 = {metrics.box.map50:.4f}   mAP@0.5:0.95 = {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
