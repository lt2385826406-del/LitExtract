"""
"""

import logging
from typing import Optional

from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None

logger = logging.getLogger(__name__)


def extract_text(
    image_path: str,
    lang: str = "eng",
    config: str = "--psm 6",
    tesseract_cmd: Optional[str] = None,
) -> str:
    """
    """
    if pytesseract is None:
        logger.warning("pytesseract not installed — returning empty string")
        return ""

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        return text.strip()
    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {e}")
        return ""


def extract_text_with_confidence(
    image_path: str,
    lang: str = "eng",
    config: str = "--psm 6",
    tesseract_cmd: Optional[str] = None,
) -> dict:
    """
    """
    if pytesseract is None:
        return {"text": "", "confidence": 0.0, "words": []}

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        img = Image.open(image_path)
        data = pytesseract.image_to_data(
            img, lang=lang, config=config, output_type=pytesseract.Output.DICT
        )

        words = []
        confidences = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if text and conf > 0:
                words.append({
                    "text": text,
                    "confidence": conf,
                    "bbox": (
                        data["left"][i],
                        data["top"][i],
                        data["width"][i],
                        data["height"][i],
                    ),
                })
                confidences.append(conf)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        full_text = " ".join(w["text"] for w in words)

        return {
            "text": full_text,
            "confidence": avg_conf,
            "words": words,
        }
    except Exception as e:
        logger.error(f"OCR with confidence failed for {image_path}: {e}")
        return {"text": "", "confidence": 0.0, "words": []}


def is_valid_caption(text: str, min_length: int = 3) -> bool:
    """
    """
    t = text.lower()
    if len(t.strip()) < min_length:
        return False
    if any(kw in t for kw in ["fig", "figure", "図", "图"]):
        return True
    return False


def is_valid_label(text: str) -> bool:
    """
    """
    import re
    t = text.strip()
    if not t:
        return False
    #
    pattern = re.compile(r"^[\(\[]?[a-zA-Z0-9][\)\]]?$")
    return bool(pattern.match(t))
