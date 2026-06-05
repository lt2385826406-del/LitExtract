"""
Vision modules for microstructure image analysis.

Modules:
    microstructure_classifier: ResNet/VGG-based microstructure type classifier
                               (equiaxed, lamellar, bimodal, etc.).
    figure_matcher:            Figure↔caption spatial matching via IoU + OCR scoring.
"""

__all__ = ["microstructure_classifier", "figure_matcher"]
