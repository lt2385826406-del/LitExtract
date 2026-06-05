"""
LLM-based semantic extraction from materials science literature.

Modules:
    run_extraction: SemanticAgent — Mistral-7B (+LoRA) extraction of composition,
                    processing, microstructure, and property data from structured
                    page outputs (detections + classifications + figure matches).
"""

__all__ = ["run_extraction"]
