"""
Vision modules for microstructure image analysis.

Modules:
    microstructure_classifier: ResNet/VGG-based binary classifier
                               (microstructure vs. other).
    figure_matcher:            Figure↔caption spatial matching via IoU + OCR scoring.
"""

__all__ = ["microstructure_classifier", "figure_matcher"]
