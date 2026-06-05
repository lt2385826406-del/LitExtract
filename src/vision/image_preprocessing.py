"""
"""

from typing import Dict, Any, Tuple, Optional
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================
#
# ============================================================

def bbox_to_dict(x1: float, y1: float, x2: float, y2: float) -> Dict[str, float]:
    """
    """
    return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}


def bbox_center(box: Dict[str, float]) -> Tuple[float, float]:
    """
    """
    return (
        (box["x1"] + box["x2"]) / 2.0,
        (box["y1"] + box["y2"]) / 2.0,
    )


def bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    """
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


def bbox_area(box: Dict[str, float]) -> float:
    """Module functionality."""
    return (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])


def bbox_width(box: Dict[str, float]) -> float:
    """Module functionality."""
    return box["x2"] - box["x1"]


def bbox_height(box: Dict[str, float]) -> float:
    """Module functionality."""
    return box["y2"] - box["y1"]


def clip_bbox(
    box: Dict[str, float], W: int, H: int
) -> Dict[str, float]:
    """
    """
    return {
        "x1": max(0.0, box["x1"]),
        "y1": max(0.0, box["y1"]),
        "x2": min(float(W), box["x2"]),
        "y2": min(float(H), box["y2"]),
    }


def iou_x(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    """
    x1 = max(a["x1"], b["x1"])
    x2 = min(a["x2"], b["x2"])
    inter_w = max(0.0, x2 - x1)
    union_w = (max(a["x2"], b["x2"]) - min(a["x1"], b["x1"]))
    return inter_w / union_w if union_w > 0 else 0.0


def point_in_box(pt: Tuple[float, float], box: Dict[str, float]) -> bool:
    """Module functionality."""
    return (box["x1"] <= pt[0] <= box["x2"] and
            box["y1"] <= pt[1] <= box["y2"])


def norm_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    W: float, H: float
) -> float:
    """
    """
    diag = (W ** 2 + H ** 2) ** 0.5 + 1e-6
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 / diag


# ============================================================
#
# ============================================================

def crop_region(
    img: np.ndarray,
    box: Dict[str, float],
    W: Optional[int] = None,
    H: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    """
    if W is None:
        W = img.shape[1]
    if H is None:
        H = img.shape[0]
    
    ix1 = max(0, int(box["x1"]))
    iy1 = max(0, int(box["y1"]))
    ix2 = min(W, int(box["x2"]))
    iy2 = min(H, int(box["y2"]))
    
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    
    return img[iy1:iy2, ix1:ix2]


def crop_and_save(
    img: np.ndarray,
    box: Dict[str, float],
    output_path: str,
    W: Optional[int] = None,
    H: Optional[int] = None,
) -> Optional[str]:
    """
    """
    crop = crop_region(img, box, W, H)
    if crop is None or crop.size == 0:
        logger.warning(f"Empty crop for box {box}")
        return None
    
    os_import = __import__('os')
    os_import.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    crop_pil = Image.fromarray(crop)
    crop_pil.save(output_path)
    return output_path


# ============================================================
#
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_for_model(
    img: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> np.ndarray:
    """
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    img = img.resize(target_size, Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0

    # # standardize
    for c in range(3):
        img_np[:, :, c] = (img_np[:, :, c] - mean[c]) / std[c]

    # HWC → CHW
    return img_np.transpose(2, 0, 1)
