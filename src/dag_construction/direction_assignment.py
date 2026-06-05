"""
"""

from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================
#
# ============================================================

NODE_ORDER = ["Element", "Alloy", "Process", "Microstructure", "Property"]

#
TYPE_ALIASES = {
    "Processing": "Process",
    "processing": "Process",
    "element": "Element",
    "elements": "Element",
    "composition": "Element",
    "alloy": "Alloy",
    "microstructure": "Microstructure",
    "ms": "Microstructure",
    "property": "Property",
    "mechanical_property": "Property",
    "physical_property": "Property",
}


# ============================================================
#
# ============================================================

#
ALLOWED_EDGE_DIRECTIONS = {
    "element_to_process":        ("Element", "Process"),
    "element_to_microstructure":  ("Element", "Microstructure"),
    "element_to_property":        ("Element", "Property"),
    "alloy_to_process":           ("Alloy", "Process"),
    "alloy_to_microstructure":    ("Alloy", "Microstructure"),
    "alloy_to_property":          ("Alloy", "Property"),
    "process_to_microstructure":  ("Process", "Microstructure"),
    "process_to_property":        ("Process", "Property"),
    "microstructure_to_property": ("Microstructure", "Property"),
}

#
FORBIDDEN_DIRECTIONS: List[Tuple[str, str]] = [
    ("Property", "Microstructure"),
    ("Property", "Process"),
    ("Property", "Alloy"),
    ("Property", "Element"),
    ("Microstructure", "Process"),
    ("Microstructure", "Alloy"),
    ("Microstructure", "Element"),
    ("Process", "Alloy"),
    ("Process", "Element"),
]


# ============================================================
#
# ============================================================

def normalize_type(node_type: str) -> str:
    """
    """
    return TYPE_ALIASES.get(node_type.strip(), node_type.strip())


def get_type_order(node_type: str) -> int:
    """
    """
    nt = normalize_type(node_type)
    if nt in NODE_ORDER:
        return NODE_ORDER.index(nt)
    return -1


def is_valid_direction(
    src_type: str,
    dst_type: str,
) -> bool:
    """
    """
    src = normalize_type(src_type)
    dst = normalize_type(dst_type)

    #
    if src == dst:
        return True

    src_order = get_type_order(src)
    dst_order = get_type_order(dst)

    #
    if src_order == -1 or dst_order == -1:
        logger.debug(f"Unknown type(s) — allowing: {src_type}({src}) → {dst_type}({dst})")
        return True

    #
    if (src, dst) in FORBIDDEN_DIRECTIONS:
        return False

    #
    return src_order <= dst_order


def assign_edge_direction(
    src_type: str,
    dst_type: str,
    confidence: float = 0.5,
) -> Tuple[str, str, bool]:
    """
    """
    if is_valid_direction(src_type, dst_type):
        return (src_type, dst_type, False)

    #
    if is_valid_direction(dst_type, src_type):
        logger.debug(f"Flipped edge direction: {src_type} → {dst_type}")
        return (dst_type, src_type, True)

    #
    logger.warning(
        f"Both directions invalid for {src_type} ↔ {dst_type} — keeping original"
    )
    return (src_type, dst_type, False)


def filter_edges_by_direction(
    edges: List[Dict],
    min_confidence: float = 0.3,
) -> Tuple[List[Dict], List[Dict]]:
    """
    """
    kept = []
    removed = []

    for edge in edges:
        src_type = edge.get("src_type", "")
        dst_type = edge.get("dst_type", "")
        conf = edge.get("confidence", 0.0)

        #
        if conf < min_confidence:
            removed.append({**edge, "remove_reason": "low_confidence"})
            continue

        #
        if not is_valid_direction(src_type, dst_type):
            removed.append({**edge, "remove_reason": "invalid_direction"})
            continue

        kept.append(edge)

    logger.info(
        f"Direction filter: {len(kept)} kept, {len(removed)} removed "
        f"({len(edges)} total)"
    )
    return kept, removed


def validate_dag_edges(
    edges: List[Tuple[str, str, str, str]],
    node_types: Dict[str, str],
) -> Dict[str, Any]:
    """
    """
    valid = []
    invalid = []
    flipped = 0

    for src_id, dst_id, src_type, dst_type in edges:
        src_norm = normalize_type(src_type)
        dst_norm = normalize_type(dst_type)

        if is_valid_direction(src_norm, dst_norm):
            valid.append((src_id, dst_id, src_norm, dst_norm))
        else:
            #
            if is_valid_direction(dst_norm, src_norm):
                valid.append((dst_id, src_id, dst_norm, src_norm))
                flipped += 1
            else:
                invalid.append({
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "src_type": src_norm,
                    "dst_type": dst_norm,
                    "reason": "invalid_direction",
                })

    return {
        "valid_edges": valid,
        "invalid_edges": invalid,
        "stats": {
            "total": len(edges),
            "valid": len(valid),
            "invalid": len(invalid),
            "flipped": flipped,
        },
    }
