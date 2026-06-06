# LitExtract

**Automated Literature Knowledge Extraction for Materials Science**

LitExtract is a modular pipeline that extracts structured materials knowledge — composition, processing, microstructure, and properties — from scientific literature using a combination of computer vision and large language models.

## Architecture

```
PDF → [YOLO Element Detection] → [OCR/Alignment] → [LLM Extraction] → [KG + Causal Hypothesis Graph]
```

### Key Components

| Module | Description | Technology |
|--------|-------------|------------|
| `pdf_parser` | PDF-to-image conversion, page parsing | PyMuPDF, pdf2image |
| `vision` | Image/caption/subgraph detection & microstructure classification | YOLOv11, ResNet-50 |
| `ocr_alignment` | OCR text extraction, figure-caption matching | Tesseract |
| `semantic_extraction` | Structured entity/relation extraction | DeepSeek / Qwen2.5-7B + LoRA |
| `kg_construction` | Knowledge graph construction | NetworkX, pyvis |
| `dag_construction` | Causality-aware DAG construction from co-occurrence patterns | Hypothesis graph construction |

### Output

- **Knowledge Graph (KG)**: Element → Microstructure → Property triples
- **Causal Hypothesis Graph (CHG)**: Cross-paper candidate causal relations (hypothesis graph, not validated causal claims)

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run pipeline on a PDF
python scripts/run_pipeline.py paper.pdf --api-key $DEEPSEEK_API_KEY

# Build knowledge graph
python src/kg_construction/build_graph.py --semantic_results outputs/ --output kg.json

# Build Causal Hypothesis Graph (CHG)
python src/dag_construction/build_dag.py --kg kg.json --output chg.json
```

## Repository Structure

```
LitExtract/
├── configs/      # Model & pipeline configurations
├── prompts/      # LLM prompt templates
├── src/          # Source code (modular)
├── training/     # Model training scripts
├── scripts/      # End-to-end scripts & reproducibility
├── data/         # Metadata, annotations, outputs
├── models/       # Model references (no weights)
├── supplementary/ # Supplementary materials
├── docs/         # Documentation
└── examples/     # Minimal usage examples
```

## Citation

If you use LitExtract in your research, please cite:

```
@software{litExtract2025,
  title = {LitExtract: Automated Literature Knowledge Extraction for Materials Science},
  year = {2025},
}
```

## Datasets

This repository involves three distinct paper collections. Each has its own DOI index and serves a different purpose:

| Dataset | Papers | DOI Index | Purpose | Example Output |
|---------|--------|-----------|---------|---------------|
| **LLM Training** | 199 (Ti + Ni alloys) | `data/metadata/llm_training_dois.csv` | Fine-tune Qwen2.5-7B / Mistral-7B for semantic extraction (SFT) | `data/extracted_json/` |
| **Ti-MMKG** | 98 Ti-alloy papers | `data/metadata/ti_mmkg_dois.csv` | Multi-modal knowledge graph construction case study | `data/kg_outputs/ti_mmkg_data.json` |
| **Ni-DAG** | 68 Ni-alloy papers | `data/metadata/ni_dag_dois.csv` | Causal Hypothesis Graph construction case study | `data/dag_outputs/ni/` |

**Key distinctions:**
- **LLM Training (199)**: The full fine-tuning corpus used for SFT of Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3. Covers Ti-based and Ni-based alloys for broad domain coverage. Each paper has manually verified structured JSON annotations in `output_0205/`.
- **Ti-MMKG (98)**: A focused subset of Ti-alloy papers used to demonstrate the full multi-modal KG workflow — YOLO detection → OCR → LLM extraction → KG construction. The output `ti_mmkg_data.json` is a published example KG with `Element → Microstructure → Property` triples.
- **Ni-DAG (68)**: A focused subset of Ni-alloy papers used to demonstrate the CHG construction pipeline. The output `dag_outputs/ni/` contains candidate causal edge lists with confidence scores derived from cross-paper co-occurrence patterns. **These are hypotheses, not validated causal claims.**

## Data Availability

**PDF files and images are NOT included in this repository** due to publisher copyright restrictions.

The repository provides the following publicly redistributable assets:

- **DOI indices**: `data/metadata/` — three per-dataset DOI indices (`ti_mmkg_dois.csv`, `ni_dag_dois.csv`, `llm_training_dois.csv`), plus the unified `paper_dois.csv` and `paper_dois.json`
- **Extracted structured data**: `data/extracted_json/` — factual information (compositions, processing, properties, heat treatment) extracted by LLM, in structured JSON format
- **Knowledge graph outputs**: `data/kg_outputs/` — constructed KGs in JSON/HTML format (node/edge lists, not raw images)
- **Causal Hypothesis Graph outputs**: `data/dag_outputs/` — candidate causal edge lists with confidence scores (hypothesis only, not validated claims)
- **Annotation schemas**: `data/annotations/` — label formats, YOLO label README, classification label definitions (no underlying images)
- **Ground truth**: `data/validation_ground_truth/` — domain expert-annotated metallurgical mechanism validation set
- **Configuration files**: `configs/` — all model, training, and inference configs required to reproduce the pipeline
- **Training & inference scripts**: `training/` and `scripts/` — full reproducibility scripts
- **Metadata & splits**: `data/metadata/` — paper indices, corpus statistics, train/val/test splits (where applicable)

**What is NOT redistributed:**
- Original publisher PDFs
- Raw figure images extracted from PDFs
- Trained model checkpoint files (available upon reasonable request)

See `docs/data_availability.md` for the complete data access policy and request process.

## License

MIT License. See [LICENSE](LICENSE) for details.
