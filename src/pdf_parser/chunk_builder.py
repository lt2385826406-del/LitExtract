"""
"""

import os
import tempfile
import logging
from typing import List, Optional

from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

logger = logging.getLogger(__name__)


def pdf_to_page_images(
    pdf_path: str,
    dpi: int = 300,
    poppler_path: Optional[str] = None,
) -> List[str]:
    """
    """
    if convert_from_path is None:
        raise ImportError(
            "pdf2image is not available. Install via: pip install pdf2image"
        )

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    temp_dir = tempfile.mkdtemp()
    logger.info(f"Converting PDF to images: {pdf_path} (DPI={dpi})")

    #
    kwargs = {"dpi": dpi}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    pdf_images = convert_from_path(pdf_path, **kwargs)

    paths = []
    for i, img in enumerate(pdf_images):
        p = os.path.join(temp_dir, f"page_{i}.png")
        img.save(p, "PNG")
        paths.append(p)
        logger.debug(f"Saved page {i} → {p} ({img.size[0]}x{img.size[1]})")

    logger.info(f"PDF conversion done: {len(paths)} pages")
    return paths


def save_processed_pages(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    poppler_path: Optional[str] = None,
) -> List[str]:
    """
    """
    os.makedirs(output_dir, exist_ok=True)

    if convert_from_path is None:
        raise ImportError("pdf2image is not available.")

    kwargs = {"dpi": dpi}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    pdf_images = convert_from_path(pdf_path, **kwargs)

    paths = []
    for i, img in enumerate(pdf_images):
        p = os.path.join(output_dir, f"page_{i}.png")
        img.save(p, "PNG")
        paths.append(p)

    logger.info(f"Saved {len(paths)} pages to {output_dir}")
    return paths


def split_pdf_pages(
    pdf_path: str,
    start_page: int = 0,
    end_page: Optional[int] = None,
    dpi: int = 300,
    poppler_path: Optional[str] = None,
) -> List[str]:
    """
    """
    all_paths = pdf_to_page_images(pdf_path, dpi=dpi, poppler_path=poppler_path)

    if end_page is None:
        end_page = len(all_paths)

    valid_range = all_paths[start_page:end_page]
    logger.info(f"Split pages {start_page}–{end_page-1} from {len(all_paths)} total")
    return valid_range
