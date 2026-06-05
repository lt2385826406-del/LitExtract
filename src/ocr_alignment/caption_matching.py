"""
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional

from src.vision.image_preprocessing import iou_x, bbox_center, bbox_height

logger = logging.getLogger(__name__)

#
CAPTION_KEYWORDS = {
    r"(?i)\bfig\b": 2.0,
    r"(?i)\bfigure\b": 2.0,
    r"图": 2.0,
    r"\([a-z]\)": 1.0,  
}


def _caption_ocr_bonus(caption_text: str) -> float:
    """
    """
    if not caption_text:
        return -1.0

    t = caption_text.lower()
    score = 0.0

    for pattern, weight in CAPTION_KEYWORDS.items():
        if re.search(pattern, t):
            score += weight

    #
    if len(t) < 5:
        score -= 2.0

    return score


def match_captions_to_images(
    images: List[Dict[str, Any]],
    captions: List[Dict[str, Any]],
    iou_x_threshold: float = 0.3,
    vertical_margin_ratio: float = 1.5,
) -> List[Dict[str, str]]:
    """
    """
    relations: List[Dict[str, str]] = []
    used_caption_ids: set = set()

    for img in images:
        img_bbox = img["bbox"]
        img_h = bbox_height(img_bbox)
        img_cy_bottom = img_bbox["y2"]
        vertical_margin = vertical_margin_ratio * img_h

        candidates = []

        for cap in captions:
            if cap["id"] in used_caption_ids:
                continue

            #
            iou_x_val = iou_x(img_bbox, cap["bbox"])
            if iou_x_val < iou_x_threshold:
                continue

            #
            cap_cy, _ = bbox_center(cap["bbox"])
            v_low = img_bbox["y1"] - vertical_margin
            v_high = img_cy_bottom + vertical_margin
            if not (v_low <= cap_cy <= v_high):
                continue

            #
            vert_dist = abs(cap_cy - img_cy_bottom)
            #
            if cap_cy < img_bbox["y1"]:
                vert_dist += img_h * 0.5

            score = (
                iou_x_val * 3.0
                - vert_dist / (img_h + 1e-6)
                + _caption_ocr_bonus(cap.get("text", ""))
            )
            candidates.append((score, cap))

        if candidates:
            #
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_cap = candidates[0][1]
            relations.append({
                "image_id": img["id"],
                "caption_id": best_cap["id"],
            })
            used_caption_ids.add(best_cap["id"])
            logger.debug(
                f"Matched caption {best_cap['id']} → image {img['id']} "
                f"(score={candidates[0][0]:.2f})"
            )

    logger.info(
        f"Caption matching: {len(relations)} pairs from "
        f"{len(images)} images × {len(captions)} captions"
    )
    return relations


def match_captions_to_images_strict(
    images: List[Dict[str, Any]],
    captions: List[Dict[str, Any]],
    iou_x_threshold: float = 0.5,
    vertical_range: Tuple[float, float] = (0.0, 1.0),
) -> List[Dict[str, str]]:
    """
    """
    relations = []
    used_caption_ids = set()

    for img in images:
        img_bbox = img["bbox"]
        img_h = bbox_height(img_bbox)
        img_bottom = img_bbox["y2"]

        v_low = img_bottom + vertical_range[0] * img_h
        v_high = img_bottom + vertical_range[1] * img_h

        best_cap = None
        best_score = -float("inf")

        for cap in captions:
            if cap["id"] in used_caption_ids:
                continue

            if iou_x(img_bbox, cap["bbox"]) < iou_x_threshold:
                continue

            cap_cy, _ = bbox_center(cap["bbox"])
            if not (v_low <= cap_cy <= v_high):
                continue

            #
            dist_score = 1.0 - abs(cap_cy - img_bottom) / (img_h + 1e-6)
            score = iou_x(img_bbox, cap["bbox"]) + dist_score

            if score > best_score:
                best_score = score
                best_cap = cap

        if best_cap:
            relations.append({
                "image_id": img["id"],
                "caption_id": best_cap["id"],
            })
            used_caption_ids.add(best_cap["id"])

    return relations
