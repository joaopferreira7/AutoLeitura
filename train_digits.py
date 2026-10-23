"""Treina o YOLOv8 que detecta e classifica cada dígito (0–9) dentro do recorte do contador.

É a "alternativa de estágio único" da seção 7 da proposta, aplicada sobre o recorte:
cada dígito vira uma caixa com classe, e a leitura é a sequência das classes da esquerda
para a direita. Usa as caixas de dígito anotadas no UFPR-AMR (data/dig).

  python train_digits.py --epochs 150 --device 0
"""
import argparse
import shutil

from ultralytics import YOLO

from autoleitura import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, help="'0' para GPU, 'cpu' para CPU")
    ap.add_argument("--project", default=str(config.ROOT / "runs" / "digitos"))
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(config.DIG_DIR / "data.yaml"),
        imgsz=config.DIG_IMG_SIZE,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name="train",
        exist_ok=True,
        patience=30,
        degrees=4, perspective=0.0005, translate=0.08, scale=0.3, shear=2,
        hsv_v=0.4, hsv_s=0.5,
        fliplr=0.0,          # dígitos espelhados não existem
        mosaic=1.0, close_mosaic=15,
    )
    config.WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, config.DIG_WEIGHTS)

    metrics = YOLO(str(config.DIG_WEIGHTS)).val(data=str(config.DIG_DIR / "data.yaml"), split="test",
                                               imgsz=config.DIG_IMG_SIZE, device=args.device,
                                               project=args.project, name="test", exist_ok=True)
    print(f"\nPesos salvos em {config.DIG_WEIGHTS}")
    print(f"Teste  mAP@0.5 = {metrics.box.map50:.4f}   mAP@0.5:0.95 = {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
