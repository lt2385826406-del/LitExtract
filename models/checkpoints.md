# Model Checkpoints

This file records all trained model checkpoints for reproducibility.

> **Note**: Pre-trained checkpoint files are NOT included in this repository due to file size limits.
> For YOLOv11 and ResNet-50 weights, see `docs/data_availability.md` for access instructions.
> For LLM LoRA adapters, see `training/llm/README.md`.

---

## YOLOv11 — PDF Figure/Element Detection

| Variant | Dataset | Classes | mAP@0.5 | Weights Path |
|---------|---------|---------|---------|--------------|
| yolov11n | 98 Ti-alloy PDFs, ~2,000 annotated elements | 5 (`figure`, `table`, `equation`, `chart`, `micrograph`) | 0.87 (validation set) | `models/yolov11_pdf.pt` |

**Training config**: `configs/yolo/hyperparams.yaml`
- Epochs: 300, batch size: 16, image size: 640
- Optimizer: AdamW, lr0: 0.001
- See `training/yolo/evaluate_yolo.py` for evaluation script

> **Paper note**: The manuscript describes the detection output as four structural categories — `image`, `caption`, `subgraph`, `subgraph_label` — which are derived from the five YOLO detection classes via post-processing (caption matching, subgraph extraction). The YOLO training itself uses five content-type classes as listed above.

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
