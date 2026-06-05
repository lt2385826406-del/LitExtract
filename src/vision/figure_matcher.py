"""
"""

import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def _bbox_center(box: Dict[str, float]):
    return ((box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0)


def _bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
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


def _iou_x(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Module functionality."""
    x1 = max(a["x1"], b["x1"])
    x2 = min(a["x2"], b["x2"])
    inter_w = max(0.0, x2 - x1)
    union_w = max(a["x2"], b["x2"]) - min(a["x1"], b["x1"])
    return inter_w / union_w if union_w > 0 else 0.0


def _point_in_box(pt, box: Dict[str, float]) -> bool:
    return (box["x1"] <= pt[0] <= box["x2"] and
            box["y1"] <= pt[1] <= box["y2"])


def _norm_dist(p1, p2, page_w: float, page_h: float) -> float:
    """Module functionality."""
    diag = (page_w ** 2 + page_h ** 2) ** 0.5 + 1e-6
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 / diag


def _caption_ocr_bonus(caption_text: str) -> float:
    t = (caption_text or "").lower()
    score = 0.0
    if "fig" in t or "figure" in t or "\u56fe" in t:
        score += 2.0
    if re.search(r"\([a-z]\)", t):
        score += 1.0
    if len(t) < 5:
        score -= 2.0
    return score


LABEL_PATTERN = re.compile(r"^[\(\[]?[a-zA-Z0-9][\)\]]?$")
PROXIMITY_THRESH = 0.35


class FigureMatcher:
    """
    """

    def __init__(self):
        pass

    def match_one_page(self, page: Dict[str, Any]) -> Dict[str, Any]:
        """Module functionality."""
        images = page.get("images", [])
        captions = page.get("captions", [])
        subgraphs = page.get("subgraphs", [])
        subgraph_labels = page.get("subgraph_labels", [])
        W = page.get("width", 1000)
        H = page.get("height", 1000)

        relations = {
            "image_captions": [],
            "image_subgraphs": [],
            "subgraph_labels": [],
        }

        if not images:
            page["relations"] = relations
            return page

        #
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
                relations["image_captions"].append({
                    "image_id": img["id"],
                    "caption_id": best_cap["id"],
                    "score": round(candidates[0][0], 3),
                })
                used_caption_ids.add(best_cap["id"])

        #
        for img in images:
            for sg in subgraphs:
                if _bbox_iou(img["bbox"], sg["bbox"]) > 0.3:
                    relations["image_subgraphs"].append({
                        "image_id": img["id"],
                        "subgraph_id": sg["id"],
                    })

        #
        def _valid_label(t):
            return t and LABEL_PATTERN.match(t.strip())

        used_label_ids = set()

        #
        for sg in subgraphs:
            sgc = _bbox_center(sg["bbox"])
            for lbl in subgraph_labels:
                if lbl["id"] in used_label_ids:
                    continue
                if not _valid_label(lbl.get("text", "")):
                    continue
                if _point_in_box(_bbox_center(lbl["bbox"]), sg["bbox"]):
                    relations["subgraph_labels"].append({
                        "subgraph_id": sg["id"],
                        "label_id": lbl["id"],
                    })
                    used_label_ids.add(lbl["id"])
                    break

        #
        matched_sg_ids = {r["subgraph_id"] for r in relations["subgraph_labels"]}
        for sg in subgraphs:
            if sg["id"] in matched_sg_ids:
                continue
            sgc = _bbox_center(sg["bbox"])
            best_lbl, best_dist = None, float("inf")
            for lbl in subgraph_labels:
                if lbl["id"] in used_label_ids:
                    continue
                if not _valid_label(lbl.get("text", "")):
                    continue
                nd = _norm_dist(_bbox_center(lbl["bbox"]), sgc, W, H)
                if nd < PROXIMITY_THRESH and nd < best_dist:
                    best_lbl, best_dist = lbl, nd
            if best_lbl:
                relations["subgraph_labels"].append({
                    "subgraph_id": sg["id"],
                    "label_id": best_lbl["id"],
                })
                used_label_ids.add(best_lbl["id"])

        page["relations"] = relations
        return page

    def match(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Module functionality."""
        logger.info(f"[FigureMatcher] Matching figure-caption pairs across {len(pages)} pages")
        matched = [self.match_one_page(p) for p in pages]

        # # statistics
        total_ic = sum(len(p.get("relations", {}).get("image_captions", []))
                       for p in matched)
        total_is = sum(len(p.get("relations", {}).get("image_subgraphs", []))
                       for p in matched)
        total_sl = sum(len(p.get("relations", {}).get("subgraph_labels", []))
                       for p in matched)
        logger.info(
            f"[FigureMatcher] matchcompleted: "
            f"image↔caption={total_ic}, image↔subgraph={total_is}, "
            f"subgraph↔label={total_sl}"
        )
        return matched
