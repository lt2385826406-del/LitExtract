"""
YOLOv11 Training Script — PDF Figure Detection
"""

import warnings
import argparse
import os
import sys

warnings.filterwarnings('ignore')

#
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics YOLO is not installed. Install via: pip install ultralytics"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for PDF figure detection")

    # # modelArgs
    parser.add_argument("--model", type=str, default="yolo11n",
                        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
                        help="YOLOv11 model variant")
    parser.add_argument("--weights", type=str, default=None,
                        help="Pretrained weights path (None = use Ultralytics default)")

    #
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset YAML config path")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader workers")

    # # trainArgs
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["SGD", "Adam", "AdamW", "RMSProp"],
                        help="Optimizer")
    parser.add_argument("--lr0", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final learning rate factor (lr0 * lrf)")
    parser.add_argument("--momentum", type=float, default=0.937,
                        help="Momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="Warmup epochs")
    parser.add_argument("--close_mosaic", type=int, default=10,
                        help="Close mosaic augmentation N epochs before end")

    #
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate")

    #
    parser.add_argument("--device", type=str, default="",
                        help="Device (0, 1, 'cpu', or '' for auto)")
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Project directory")
    parser.add_argument("--name", type=str, default="exp",
                        help="Experiment name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--single_cls", action="store_true",
                        help="Train as single-class")

    return parser.parse_args()


def train(args):
    """
    """
    #
    model_cfg = f"{args.model}.yaml" if not args.weights else args.weights

    print(f"[Train] Loading model: {model_cfg}")
    model = YOLO(model=model_cfg)

    # # trainArgs
    train_args = {
        "data": args.data,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "close_mosaic": args.close_mosaic,
        "label_smoothing": args.label_smoothing,
        "dropout": args.dropout,
        "resume": args.resume,
        "project": args.project,
        "name": args.name,
        "single_cls": args.single_cls,
        "cache": False,
    }

    print(f"[Train] Starting training with config: {train_args}")
    model.train(**train_args)

    print(f"[Train] Training complete. Results saved to: {args.project}/{args.name}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
