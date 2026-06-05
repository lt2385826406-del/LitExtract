import os
import tempfile
import shutil
import logging
from typing import List, Dict, Any
import re

import numpy as np
from PIL import Image

# Make dependencies optional
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from utils.utils import extract_text
except ImportError:
    # Fallback: return empty string for extract_text
    def extract_text(path):
        logger.warning("utils.utils not available, returning empty text")
        return ""

logger = logging.getLogger(__name__)


def bbox_to_dict(x1, y1, x2, y2):
    return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}


def bbox_center(box):
    return (
        (box["x1"] + box["x2"]) / 2.0,
        (box["y1"] + box["y2"]) / 2.0,
    )


def bbox_iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return float(inter / max(area_a + area_b - inter, 1e-6))


class VisionAgent:

    def __init__(
            self,
            dpi: int = 300,
            model_relpath: str = "processing/yolov11.pt",
            poppler_relpath: str = "poppler-23.11.0/bin",
            detect_only: bool = False,
    ):
        self.dpi = dpi
        self.detect_only = detect_only

        #
        self.base_dir = os.getcwd()
        logger.info(f"[VisionAgent] Base directory: {self.base_dir}")

        # Check if dependencies are available
        if YOLO is None:
            raise ImportError("YOLO is not available. Please install ultralytics.")
        if convert_from_path is None:
            raise ImportError("pdf2image is not available. Please install pdf2image.")

        #
        model_path = os.path.join(self.base_dir, model_relpath)
        
        #
        if not os.path.exists(model_path):
            #
            absolute_model_path = "D:/python/LitExtract_Agent/processing/yolov11.pt"
            if os.path.exists(absolute_model_path):
                model_path = absolute_model_path
                logger.info(f"[VisionAgent] Using absolute model path: {model_path}")
            else:
                #
                logger.error(f"[VisionAgent] Model file not found, current dir: {self.base_dir}")
                logger.error(f"[VisionAgent] Processing dir contents: {os.listdir(os.path.join(self.base_dir, 'processing'))}")
                raise FileNotFoundError(f"YOLO model not found: {model_path}, not found at absolute path: {absolute_model_path}")

        logger.info(f"[VisionAgent] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        self.poppler_path = os.path.join(self.base_dir, poppler_relpath)
        logger.info(f"[VisionAgent] Poppler path: {self.poppler_path}")

        self.output_root = os.path.join(self.base_dir, "output_bboxes")
        os.makedirs(self.output_root, exist_ok=True)

    # --------------------------------------------------------------------

    def _detect_page(self, image_path: str, output_dir: str, page_idx: int) -> Dict[str, Any]:
        # # ★ YOLO inferenceresult
        results = self.model(image_path)[0]
        boxes = results.boxes

        #
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)
        H, W = img_np.shape[:2]

        images, captions, subgraphs, subgraph_labels = [], [], [], []

        #
        for i, b in enumerate(boxes):
            #
            x1, y1, x2, y2 = b.xyxy[0].tolist()

            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            #
            raw_label = str(self.model.names.get(cls_id, cls_id)).lower().strip()

            #
            if raw_label in ["class0", "0", "caption"]:
                label = "caption"
            elif raw_label in ["class1", "1", "image"]:
                label = "image"
            elif raw_label in ["class2", "2", "subgraph"]:
                label = "subgraph"
            elif raw_label in ["class3", "3", "subgraph_label"]:
                label = "subgraph_label"
            else:
                label = "unknown"

            print(f"[DEBUG] YOLO: cls={cls_id}, raw='{raw_label}' → label='{label}'")

            #
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(W, int(x2)), min(H, int(y2))

            crop = img_np[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                continue

            crop_img = Image.fromarray(crop)
            crop_path = os.path.join(output_dir, f"page_{page_idx}_det_{i}_{label}.png")
            os.makedirs(os.path.dirname(crop_path), exist_ok=True)
            crop_img.save(crop_path)

            bbox = bbox_to_dict(x1, y1, x2, y2)

            if label == "image":
                images.append({
                    "id": f"img_{page_idx}_{i}",
                    "bbox": bbox,
                    "conf": conf,
                    "path": crop_path,
                })

            elif label == "caption":
                try:
                    text = extract_text(crop_path)
                except:
                    text = ""
                captions.append({
                    "id": f"cap_{page_idx}_{i}",
                    "bbox": bbox,
                    "conf": conf,
                    "path": crop_path,
                    "text": text,
                })

            elif label == "subgraph":
                subgraphs.append({
                    "id": f"sub_{page_idx}_{i}",
                    "bbox": bbox,
                    "conf": conf,
                    "path": crop_path,
                })

            elif label == "subgraph_label":
                try:
                    text = extract_text(crop_path)
                except:
                    text = ""
                subgraph_labels.append({
                    "id": f"lbl_{page_idx}_{i}",
                    "bbox": bbox,
                    "conf": conf,
                    "path": crop_path,
                    "text": text,
                })

        # ──────────────────────────────────────────────────────────────────
        #
        # ──────────────────────────────────────────────────────────────────
        relations = {
            "image_captions": [],
            "image_subgraphs": [],
            "subgraph_labels": [],
        }

        if not self.detect_only:
            #
            def _iou_x(a, b):
                x1 = max(a["x1"], b["x1"])
                x2 = min(a["x2"], b["x2"])
                inter_w = max(0.0, x2 - x1)
                union_w = max(a["x2"], b["x2"]) - min(a["x1"], b["x1"])
                return inter_w / union_w if union_w > 0 else 0.0

            def _caption_ocr_bonus(caption_text):
                t = (caption_text or "").lower()
                score = 0.0
                if "fig" in t or "figure" in t or "\u56fe" in t:
                    score += 2.0
                if re.search(r"\([a-z]\)", t):
                    score += 1.0
                if len(t) < 5:
                    score -= 2.0
                return score

            used_caption_ids = set()

            for img in images:
                img_h = img["bbox"]["y2"] - img["bbox"]["y1"]
                img_cy_bottom = img["bbox"]["y2"]
                vertical_margin = 1.5 * img_h

                candidates = []
                for cap in captions:
                    if cap["id"] in used_caption_ids:
                        continue
                    iou_x_val = _iou_x(img["bbox"], cap["bbox"])
                    if iou_x_val < 0.3:
                        continue
                    cap_cy = (cap["bbox"]["y1"] + cap["bbox"]["y2"]) / 2.0
                    v_low = img["bbox"]["y1"] - vertical_margin
                    v_high = img_cy_bottom + vertical_margin
                    if not (v_low <= cap_cy <= v_high):
                        continue
                    vert_dist = abs(cap_cy - img_cy_bottom)
                    if cap_cy < img["bbox"]["y1"]:
                        vert_dist += img_h * 0.5
                    score = (
                        iou_x_val * 3.0
                        - vert_dist / (img_h + 1e-6)
                        + _caption_ocr_bonus(cap.get("text", ""))
                    )
                    candidates.append((score, cap))

                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    best_cap = candidates[0][1]
                    relations["image_captions"].append(
                        {"image_id": img["id"], "caption_id": best_cap["id"]}
                    )
                    used_caption_ids.add(best_cap["id"])

            #
            for img in images:
                for sg in subgraphs:
                    if bbox_iou(img["bbox"], sg["bbox"]) > 0.3:
                        relations["image_subgraphs"].append(
                            {"image_id": img["id"], "subgraph_id": sg["id"]}
                        )

            #
            def _point_in_box(pt, box):
                return box["x1"] <= pt[0] <= box["x2"] and box["y1"] <= pt[1] <= box["y2"]

            def _norm_dist(p1, p2, page_w, page_h):
                diag = (page_w ** 2 + page_h ** 2) ** 0.5 + 1e-6
                return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 / diag

            PROXIMITY_THRESH = 0.35
            label_pattern = re.compile(r"^[\(\[]?[a-zA-Z0-9][\)\]]?$")

            def _valid_label(t):
                return t and label_pattern.match(t.strip())

            used_label_ids = set()

            for sg in subgraphs:
                sgc = bbox_center(sg["bbox"])
                for lbl in subgraph_labels:
                    if lbl["id"] in used_label_ids:
                        continue
                    if not _valid_label(lbl.get("text", "")):
                        continue
                    lbc = bbox_center(lbl["bbox"])
                    if _point_in_box(lbc, sg["bbox"]):
                        relations["subgraph_labels"].append(
                            {"subgraph_id": sg["id"], "label_id": lbl["id"]}
                        )
                        used_label_ids.add(lbl["id"])
                        break

            matched_sg_ids = {r["subgraph_id"] for r in relations["subgraph_labels"]}
            for sg in subgraphs:
                if sg["id"] in matched_sg_ids:
                    continue
                sgc = bbox_center(sg["bbox"])
                best_lbl, best_dist = None, float("inf")
                for lbl in subgraph_labels:
                    if lbl["id"] in used_label_ids:
                        continue
                    if not _valid_label(lbl.get("text", "")):
                        continue
                    lbc = bbox_center(lbl["bbox"])
                    nd = _norm_dist(lbc, sgc, W, H)
                    if nd < PROXIMITY_THRESH and nd < best_dist:
                        best_lbl, best_dist = lbl, nd
                if best_lbl:
                    relations["subgraph_labels"].append(
                        {"subgraph_id": sg["id"], "label_id": best_lbl["id"]}
                    )
                    used_label_ids.add(best_lbl["id"])

        return {
            "page_index": page_idx,
            "width": W,
            "height": H,
            "img": img_np,
            "images": images,
            "captions": captions,
            "subgraphs": subgraphs,
            "subgraph_labels": subgraph_labels,
            "relations": relations,
        }

    # --------------------------------------------------------------------

    def run(self, pdf_path: str, output_dir: str = None):
        if output_dir is None:
            output_dir = self.output_root

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        temp = tempfile.mkdtemp()
        pages = []

        try:
            pdf_imgs = convert_from_path(
                pdf_path, dpi=self.dpi, poppler_path=self.poppler_path
            )

            paths = []
            for i, img in enumerate(pdf_imgs):
                p = os.path.join(temp, f"page_{i}.png")
                img.save(p)
                paths.append(p)

            for idx, p in enumerate(paths):
                pages.append(self._detect_page(p, output_dir, idx))

            return pages

        finally:
            shutil.rmtree(temp, ignore_errors=True)
