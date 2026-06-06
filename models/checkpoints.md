# Model Checkpoints

This file records all trained model checkpoints for reproducibility.

> **Note**: Pre-trained checkpoint files are NOT included in this repository due to file size limits.
> For YOLOv11 and ResNet-50 weights, see `docs/data_availability.md` for access instructions.
> For LLM LoRA adapters, see `training/llm/README.md`.

---

## YOLOv11 — PDF Graphic Element Detection

| Variant | Dataset | Classes | mAP@0.5 | Weights Path |
|---------|---------|---------|---------|--------------|
| yolov11n | 219 annotated PDF page images (Ti + Ni alloys) | 4 (`caption`, `image`, `subgraph`, `subgraph_label`) | 0.9175 (validation set) | `models/yolov11_pdf.pt` |

Per-class Precision / Recall / F1 (validation set):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| caption (0) | 0.9906 | 0.9814 | 0.9860 |
| image (1) | 0.9846 | 0.9938 | 0.9891 |
| subgraph (2) | 0.8866 | 0.9339 | 0.9097 |
| subgraph_label (3) | 0.9054 | 0.8996 | 0.9025 |
| **Overall** | **0.9326** | **0.9456** | **0.9390** |

**Training config**: `configs/yolo/hyperparams.yaml`
- Epochs: 58 (early stopped, patience=100), batch size: 4–6, image size: 640
- Optimizer: SGD (lr0=0.01, momentum=0.937, weight_decay=0.0005)
- Data augmentation: Mosaic, HSV jitter, horizontal flip
- See `training/yolo/evaluate_yolo.py` for evaluation script
- See `training_process_paper.md` (training folder) for full training documentation

---

## ResNet-50 — Microstructure Binary Classification

| Task | Dataset | Accuracy | Weights Path |
|------|---------|----------|--------------|
| Binary (microstructure vs. other) | Ti-alloy figure crops, 3,200 images | 0.91 (validation) | `models/resnet50_micro.pth` |

**Training config**: `configs/classification/resnet50.yaml`
- Pre-trained on ImageNet (`IMAGENET1K_V2`), fine-tuned for 100 epochs
- Input: 224×224 RGB, batch size: 32
- See `training/classification/train_resnet.py` and `evaluate_classifier.py`

---

## Qwen2.5-7B-Instruct + QLoRA — Semantic Extraction

| Base Model | LoRA Rank (r) | LoRA Alpha | Max Seq Length | Dataset | LoRA Path |
|-----------|---------------|------------|----------------|---------|-----------|
| `Qwen/Qwen2.5-7B-Instruct` | 16 | 32 | 8,192 | 201 papers (Ti + Ni), ~5,500 samples | `models/qwen25_lora/` (not included) |

**Training config**: `configs/llm/qlora_config.yaml`
- 4-bit QLoRA (NF4, double quant, bfloat16 compute)
- Epochs: 3, effective batch size: 16 (4 × 4 gradient accumulation)
- Learning rate: 2×10⁻⁴, cosine scheduler, warmup 3%
- Target modules: all linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Prompt format: ChatML; see `configs/llm/inference_config.yaml`

---

## Mistral-7B-Instruct-v0.3 + QLoRA — Semantic Extraction (Alternative)

| Base Model | LoRA Rank (r) | LoRA Alpha | Max Seq Length | Dataset | LoRA Path |
|-----------|---------------|------------|----------------|---------|-----------|
| `mistralai/Mistral-7B-Instruct-v0.3` | 16 | 32 | 8,192 | 201 papers (Ti + Ni), ~5,500 samples | `models/mistral7b_lora/` (not included) |

Same QLoRA configuration as Qwen2.5-7B (see `configs/llm/qlora_config.yaml`).
Mistral variant is provided as a drop-in alternative; performance is comparable (see paper Section 4.3).

---

## Tesseract OCR

| Component | Version | Notes |
|-----------|---------|-------|
| Tesseract (system) | 4.x or 5.x (system-dependent) | Installed via `apt install tesseract-ocr` (Linux) or installer (Windows) |
| pytesseract (Python wrapper) | ≥0.3.10 | Optional; required only if `ocr_alignment` module is used |

Tesseract is used for OCR text extraction from detected figure regions. The language data pack (`eng`) is required. See `docs/usage.md` for installation instructions.

---

## Model Selection Rationale

See `models/model_selection.md` for the full discussion of model choices and alternatives.
