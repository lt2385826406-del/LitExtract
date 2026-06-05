"""
"""

import re
import logging
from typing import List, Dict, Any

from src.vision.image_preprocessing import (
    bbox_center,
    point_in_box,
    norm_distance,
)

logger = logging.getLogger(__name__)

#
LABEL_PATTERN = re.compile(r"^[\(\[]?[a-zA-Z0-9][\)\]]?$")


def is_valid_subgraph_label(text: str) -> bool:
    """
    """
    return bool(text) and bool(LABEL_PATTERN.match(text.strip()))


def align_subgraph_labels(
    subgraphs: List[Dict[str, Any]],
    labels: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    proximity_threshold: float = 0.35,
) -> List[Dict[str, str]]:
    """
    """
    relations: List[Dict[str, str]] = []
    used_label_ids: set = set()

    #
    valid_labels = [lbl for lbl in labels if is_valid_subgraph_label(lbl.get("text", ""))]

    #
    for sg in subgraphs:
        sgc = bbox_center(sg["bbox"])
        for lbl in valid_labels:
            if lbl["id"] in used_label_ids:
                continue
            lbc = bbox_center(lbl["bbox"])
            if point_in_box(lbc, sg["bbox"]):
                relations.append({
                    "subgraph_id": sg["id"],
                    "label_id": lbl["id"],
                })
                used_label_ids.add(lbl["id"])
                logger.debug(f"Contained match: label {lbl['id']} → subgraph {sg['id']}")
                break  

    #
    matched_sg_ids = {r["subgraph_id"] for r in relations}
    for sg in subgraphs:
        if sg["id"] in matched_sg_ids:
            continue

        sgc = bbox_center(sg["bbox"])
        best_lbl = None
        best_dist = float("inf")

        for lbl in valid_labels:
            if lbl["id"] in used_label_ids:
                continue
            lbc = bbox_center(lbl["bbox"])
            nd = norm_distance(lbc, sgc, page_width, page_height)
            if nd < proximity_threshold and nd < best_dist:
                best_lbl = lbl
                best_dist = nd

        if best_lbl:
            relations.append({
                "subgraph_id": sg["id"],
                "label_id": best_lbl["id"],
            })
            used_label_ids.add(best_lbl["id"])
            logger.debug(
                f"Proximity match: label {best_lbl['id']} → subgraph {sg['id']} "
                f"(dist={best_dist:.3f})"
            )

    logger.info(
        f"Subgraph alignment: {len(relations)} pairs from "
        f"{len(subgraphs)} subgraphs × {len(valid_labels)} valid labels"
    )
    return relations


def compute_subgraph_grid(
    subgraphs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    """
    if not subgraphs:
        return {"rows": 0, "cols": 0, "total": 0, "grid": []}

    #
    sorted_by_y = sorted(subgraphs, key=lambda s: s["bbox"]["y1"])
    
    #
    rows = []
    current_row = [sorted_by_y[0]]
    y_threshold = 20  

    for sg in sorted_by_y[1:]:
        last_y = current_row[-1]["bbox"]["y1"]
        if abs(sg["bbox"]["y1"] - last_y) < y_threshold:
            current_row.append(sg)
        else:
            #
            current_row.sort(key=lambda s: s["bbox"]["x1"])
            rows.append(current_row)
            current_row = [sg]

    #
    current_row.sort(key=lambda s: s["bbox"]["x1"])
    rows.append(current_row)

    num_rows = len(rows)
    num_cols = max(len(r) for r in rows) if rows else 0
    grid = [[sg["id"] for sg in row] for row in rows]

    return {
        "rows": num_rows,
        "cols": num_cols,
        "total": len(subgraphs),
        "grid": grid,
    }
