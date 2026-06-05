# Model Checkpoints

This file records all trained model checkpoints for reproducibility.

## YOLOv11 — PDF Figure Detection
| Variant | Dataset | mAP@0.5 | Path |
|---------|---------|---------|------|
| yolov11n | 98 Ti-alloy PDFs | — | models/yolov11_pdf.pt |

## ResNet-50 — Microstructure Classification
| Task | Accuracy | Path |
|------|----------|------|
| Binary (microstructure vs. other) | — | models/resnet50_micro.pth |

## Qwen2.5-7B-Instruct + LoRA — Semantic Extraction
| Base Model | LoRA Rank | Dataset | Path |
|-----------|-----------|---------|------|
| Qwen2.5-7B | r=16 | 98+ Ti papers | — |

> Note: Checkpoint files are NOT included in this repository.
> Download links are provided in `hf_links.md`.
