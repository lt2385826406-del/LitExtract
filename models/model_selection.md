# Model Selection Rationale

## YOLOv11 for PDF Graphic Element Detection
- Chosen over Faster R-CNN for speed and ease of training
- Nano variant (yolov11n) provides sufficient accuracy for detecting four structural categories: `caption`, `image`, `subgraph`, `subgraph_label`
- Trained on 1,091 annotated PDF page images (763 train / 109 val / 219 test) from Ti/Ni-alloy papers
- See `training_process_paper.md` (training folder) for full training documentation

## ResNet-50 for Microstructure Classification
- Pre-trained on ImageNet, fine-tuned for binary classification (microstructure vs. other)
- Chosen over ViT for lower computational cost with similar accuracy
- A downstream step filters detected figures into microstructure images and non-microstructure content

## LLM Benchmark and Final Model Selection

Three open-weight LLMs were benchmarked under the same QLoRA configuration (r=16, α=32, 4-bit NF4, 3 epochs):

| Model | Selected as Final | Notes |
|-------|-------------------|-------|
| Mistral-7B-Instruct-v0.2 | ✅ **Yes** | Best overall performance on materials NER and relation extraction (see paper Section 4.3) |
| Qwen2.5-7B-Instruct | Benchmarked | Strong baseline, comparable performance |
| LLaMA-3-8B-Instruct | Benchmarked | Included for open-weight benchmark completeness |

### Mistral-7B-Instruct-v0.2 — Final Selected Model
- 7B parameters, fits in single GPU with 4-bit quantization
- QLoRA (r=16, α=32) enables efficient fine-tuning
- ChatML format for SFT with structured JSON output
- Selected as the final model for all reported results in the manuscript

### Qwen2.5-7B-Instruct and LLaMA-3-8B-Instruct — Benchmarked Alternatives
- Same QLoRA configuration as Mistral (r=16, α=32)
- Provided as drop-in alternatives for users who prefer different model families
- Performance comparison reported in paper Section 4.3
