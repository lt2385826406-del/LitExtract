# Model Selection Rationale

## YOLOv11 for PDF Figure Detection
- Chosen over Faster R-CNN for speed and ease of training
- Nano variant (yolov11n) provides sufficient accuracy for figure/table/chart detection
- Trained on 98 Ti-alloy papers with ~2,000 annotated figures

## ResNet-50 for Microstructure Classification
- Pre-trained on ImageNet, fine-tuned on 8-class microstructure taxonomy
- Chosen over ViT for lower computational cost with similar accuracy
- Classes: equiaxed, lamellar, bimodal, widmanstatten, acicular, cellular, basketweave, martensitic

## Qwen2.5-7B-Instruct + QLoRA for Semantic Extraction
- 7B parameter model fits in single GPU with 4-bit quantization
- QLoRA (r=16, alpha=32) enables efficient fine-tuning
- ChatML format for SFT with structured JSON output
- Alternative: Mistral-7B-Instruct-v0.3 (similar performance)
