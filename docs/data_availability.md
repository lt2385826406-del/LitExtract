# Data Availability

## Copyright Notice

**PDF files and images are NOT included in this repository.** All source documents (academic papers in PDF format) are copyright-protected by their respective publishers (Elsevier, Springer Nature, MDPI, etc.). Under publisher copyright policies, we cannot publicly redistribute the full-text PDFs or raw images extracted from those PDFs.

Users who wish to reproduce the experiments must obtain the original PDFs directly from the publishers via the DOIs listed in `data/metadata/paper_dois.csv`.

The extracted structured data (compositions, processing parameters, property values, etc.) consists of factual information and is not subject to the original publications' copyright. This data is included in the repository.

---

## Assets Publicly Available in This Repository

| Asset | Path | Description |
|-------|------|-------------|
| Unified DOI index | `data/metadata/paper_dois.csv` | 199 papers (Ti + Ni alloys) with titles and DOIs |
| LLM training DOI index | `data/metadata/llm_training_dois.csv` | 199 training papers — SFT corpus |
| Ti-MMKG DOI index | `data/metadata/ti_mmkg_dois.csv` | 98 Ti-alloy papers — KG case study |
| Ni-DAG DOI index | `data/metadata/ni_dag_dois.csv` | 68 Ni-alloy papers — CHG case study |
| Corpus metadata | `data/metadata/paper_metadata.csv` | Paper-level metadata (journal, year, alloy type) |
| Extracted JSON data | `data/extracted_json/` | LLM-extracted structured data (Ti + Ni alloys) |
| KG outputs | `data/kg_outputs/` | Knowledge graph edge/node lists (JSON, HTML) |
| Causal Hypothesis Graph outputs | `data/dag_outputs/` | Candidate causal edges with confidence (JSON) |
| Annotation schemas | `data/annotations/` | YOLO label format, classification label definitions |
| Ground truth | `data/validation_ground_truth/` | Expert-annotated validation set |
| Train/val/test splits | `data/splits/` | Dataset splits (where applicable) |
| Configuration files | `configs/` | All model, training, and inference configs |
| Training scripts | `training/` | YOLO, ResNet, QLoRA training scripts |
| Inference scripts | `scripts/` | End-to-end pipeline and KG/DAG construction |
| Prompt templates | `prompts/` | LLM prompt templates (ChatML format) |

---

## Assets Available Upon Reasonable Request

Due to copyright restrictions or file size limits, the following are **not** included in the repository but may be made available upon reasonable request to the corresponding author:

- Raw PDF files (users should obtain via publisher DOIs)
- Trained model checkpoint files (>500 MB each): YOLOv11 weights, ResNet-50 weights, QLoRA adapters
- Full YOLO annotation label files (bounding box coordinates for ~2,000 figure regions)

To request access, please contact the corresponding author with a brief description of your intended use. We may provide download links or arrange data transfer via a secure channel.

---

## Reproduction Instructions

To fully reproduce the LitExtract pipeline from scratch:

1. Obtain the 199 paper PDFs from publishers using DOIs in `data/metadata/paper_dois.csv` (or the per-dataset indices `ti_mmkg_dois.csv` / `ni_dag_dois.csv`)
2. Place PDFs in `data/pdfs/` (not included in repo)
3. Run the full pipeline: `python scripts/run_pipeline.py --pdf_dir data/pdfs/ --output outputs/`
4. For training YOLO/ResNet: use scripts under `training/` with your own annotated data, or request our pre-trained weights
5. For LLM fine-tuning: use `training/llm/train_qlora.py` with configs in `configs/llm/`

---

## Contact

For data requests or questions about reproduction, please open a GitHub Issue or contact the corresponding author.
