"""
PDF processing pipeline for materials science literature.

Modules:
    pdf_to_images: VisionAgent — YOLOv11-based detection of figures/tables/captions
                   from rendered PDF pages, with optional OCR and bbox matching.
"""

__all__ = ["pdf_to_images"]
