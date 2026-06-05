"""
"""

import argparse
import warnings

warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Ultralytics YOLO is not installed. Install via: pip install ultralytics")


SUPPORTED_FORMATS = ["torchscript", "onnx", "openvino", "engine",
                     "coreml", "saved_model", "pb", "tflite",
                     "edgetpu", "tfjs", "paddle", "ncnn", "all"]


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv11 model to various formats")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained weights (.pt)")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=SUPPORTED_FORMATS,
                        help="Export format (default: onnx)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Export image size")
    parser.add_argument("--half", action="store_true",
                        help="FP16 half-precision export")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 quantization")
    parser.add_argument("--dynamic", action="store_true",
                        help="Dynamic batch size (ONNX)")
    parser.add_argument("--simplify", action="store_true",
                        help="Simplify ONNX model")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset version")
    parser.add_argument("--workspace", type=float, default=4.0,
                        help="TensorRT workspace size (GB)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: same as weights)")
    return parser.parse_args()


def export_single_format(model, fmt: str, args) -> str:
    """Module functionality."""
    print(f"[Export] Exporting to {fmt}...")

    export_kwargs = {
        "format": fmt,
        "imgsz": args.imgsz,
        "half": args.half,
        "int8": args.int8,
    }

    if fmt == "onnx":
        export_kwargs["dynamic"] = args.dynamic
        export_kwargs["simplify"] = args.simplify
        export_kwargs["opset"] = args.opset
    elif fmt == "engine":
        export_kwargs["workspace"] = args.workspace

    try:
        path = model.export(**export_kwargs)
        print(f"[Export] ✓ {fmt} exported to: {path}")
        return path
    except Exception as e:
        print(f"[Export] ✗ {fmt} export failed: {e}")
        return None


ALL_INFERENCE_FORMATS = [
    "torchscript", "onnx", "openvino", "engine",
    "saved_model", "tflite", "tfjs",
]


def export(args):
    """Module functionality."""
    print(f"[Export] Loading model: {args.weights}")
    model = YOLO(args.weights)

    if args.format == "all":
        print(f"[Export] Exporting to ALL formats...")
        results = {}
        for fmt in ALL_INFERENCE_FORMATS:
            results[fmt] = export_single_format(model, fmt, args)
        print(f"\n[Export] Summary:")
        for fmt, path in results.items():
            status = "✓" if path else "✗"
            print(f"  {status} {fmt}: {path}")
    else:
        export_single_format(model, args.format, args)


if __name__ == "__main__":
    args = parse_args()
    export(args)
