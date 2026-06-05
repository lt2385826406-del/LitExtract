"""
YOLO Evaluation Script — PDF Figure Detection evaluate
"""

import os
import json
import argparse
import warnings
from typing import List, Dict, Any, Tuple

import numpy as np

warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Ultralytics YOLO not installed. Install via: pip install ultralytics")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 PDF figure detection")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained weights (e.g., best.pt)")
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset YAML config path")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="",
                        help="Device (0, 'cpu', or '' for auto)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output JSON path")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save per-image predictions")
    parser.add_argument("--pred_dir", type=str, default="predictions",
                        help="Directory for prediction outputs")
    return parser.parse_args()


def run_standard_eval(model, data: str, args) -> Dict[str, Any]:
    """
    """
    print(f"[Eval] Running standard validation on {data}...")
    results = model.val(
        data=data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_json=False,
    )

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    #
    if hasattr(results.box, "ap_class_index"):
        class_indices = results.box.ap_class_index.tolist()
        ap50_per_class = results.box.ap50.tolist()
        names = getattr(results, "names", {})
        metrics["per_class"] = {
            names.get(i, f"class_{i}"): float(ap)
            for i, ap in zip(class_indices, ap50_per_class)
        }

    print(f"[Eval] Standard metrics: mAP50={metrics['mAP50']:.4f}, "
          f"mAP50-95={metrics['mAP50_95']:.4f}")
    return metrics


def match_captions_to_images(
    images: List[Dict], captions: List[Dict],
    iou_x_thresh: float = 0.3,
) -> List[Dict]:
    """
    """
    def iou_x(a, b):
        x1 = max(a["x1"], b["x1"])
        x2 = min(a["x2"], b["x2"])
        inter = max(0.0, x2 - x1)
        union = max(a["x2"], b["x2"]) - min(a["x1"], b["x1"])
        return inter / union if union > 0 else 0.0

    matches = []
    used_caption_ids = set()

    for img in images:
        img_h = img["bbox"]["y2"] - img["bbox"]["y1"]
        margin = 1.5 * img_h

        best_cap, best_score = None, -float("inf")
        for cap in captions:
            if cap["id"] in used_caption_ids:
                continue
            if iou_x(img["bbox"], cap["bbox"]) < iou_x_thresh:
                continue

            cap_cy = (cap["bbox"]["y1"] + cap["bbox"]["y2"]) / 2.0
            v_low = img["bbox"]["y1"] - margin
            v_high = img["bbox"]["y2"] + margin
            if not (v_low <= cap_cy <= v_high):
                continue

            score = iou_x(img["bbox"], cap["bbox"])
            if score > best_score:
                best_score = score
                best_cap = cap

        if best_cap:
            matches.append({"image_id": img["id"], "caption_id": best_cap["id"]})
            used_caption_ids.add(best_cap["id"])

    return matches


def match_subgraphs_to_labels(
    subgraphs: List[Dict], labels: List[Dict],
    page_w: float, page_h: float,
    proximity_thresh: float = 0.35,
) -> List[Dict]:
    """
    Subgraph-Label matchevaluate。
    """
    def bbox_center(box):
        return ((box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0)

    def point_in_box(pt, box):
        return box["x1"] <= pt[0] <= box["x2"] and box["y1"] <= pt[1] <= box["y2"]

    def norm_dist(p1, p2):
        diag = (page_w ** 2 + page_h ** 2) ** 0.5 + 1e-6
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 / diag

    matches = []
    used_label_ids = set()

    #
    for sg in subgraphs:
        sgc = bbox_center(sg["bbox"])
        for lbl in labels:
            if lbl["id"] in used_label_ids:
                continue
            if point_in_box(bbox_center(lbl["bbox"]), sg["bbox"]):
                matches.append({"subgraph_id": sg["id"], "label_id": lbl["id"]})
                used_label_ids.add(lbl["id"])
                break

    #
    matched_sg = {m["subgraph_id"] for m in matches}
    for sg in subgraphs:
        if sg["id"] in matched_sg:
            continue
        sgc = bbox_center(sg["bbox"])
        best_lbl, best_dist = None, float("inf")
        for lbl in labels:
            if lbl["id"] in used_label_ids:
                continue
            nd = norm_dist(bbox_center(lbl["bbox"]), sgc)
            if nd < proximity_thresh and nd < best_dist:
                best_lbl, best_dist = lbl, nd
        if best_lbl:
            matches.append({"subgraph_id": sg["id"], "label_id": best_lbl["id"]})
            used_label_ids.add(best_lbl["id"])

    return matches


def compute_matching_metrics(
    pred_matches: List[Dict],
    gt_matches: List[Dict],
) -> Dict[str, float]:
    """Module functionality."""
    if not gt_matches:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_pairs = set((m["image_id"], m.get("caption_id", m.get("label_id")))
                      for m in pred_matches)
    gt_pairs = set((m["image_id"], m.get("caption_id", m.get("label_id")))
                    for m in gt_matches)

    tp = len(pred_pairs & gt_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = tp / len(gt_pairs) if gt_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(args):
    """Module functionality."""
    print(f"[Eval] Loading model: {args.weights}")
    model = YOLO(args.weights)

    #
    results = {}
    results["standard"] = run_standard_eval(model, args.data, args)

    #
    print(f"\n[Eval] Summary:")
    print(f"  mAP@0.5:        {results['standard']['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95:   {results['standard']['mAP50_95']:.4f}")
    print(f"  Precision:      {results['standard']['precision']:.4f}")
    print(f"  Recall:         {results['standard']['recall']:.4f}")

    if "per_class" in results["standard"]:
        print(f"  Per-class AP50: {json.dumps(results['standard']['per_class'], indent=2)}")

    # # 3. saveresult
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Eval] Results saved to: {args.output}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
