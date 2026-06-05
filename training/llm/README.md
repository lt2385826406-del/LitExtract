# LLM Fine-tuning — Training Data

## Data Source

The LLM is fine-tuned on semantic extraction data derived from 201 materials science papers (Ti alloys + Ni-based superalloys).

### DOI Index

The complete paper list with DOIs is at `data/metadata/paper_dois.csv` (also available as `paper_dois.json`).

| Field | Description |
|-------|-------------|
| `paper` | Paper title |
| `doi` | Digital Object Identifier |
| `doi_url` | Resolvable DOI link |

### Copyright Notice

**The original PDFs and images are NOT included in this repository** due to publisher copyright restrictions. Users who wish to reproduce the training pipeline must:

1. Obtain the original PDFs from publishers using the DOIs in `paper_dois.csv`
2. Run the LitExtract pipeline (YOLO detection → OCR → semantic extraction) to produce training data
3. Use the annotation schemas under `configs/` and `data/annotations/` for label formatting

## Training Scripts

| Script | Description |
|--------|-------------|
| `train_qlora.py` | QLoRA fine-tuning (Mistral-7B / Qwen2.5-7B / DeepSeek-7B) |
| `evaluate_json.py` | JSON output evaluation metrics |
| `context_ablation.py` | Context window ablation (4K vs 8K) |

## Model Outputs

Trained LoRA adapters and checkpoints are available upon request (>500MB each). See `docs/data_availability.md`.
