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

## Data Availability

**PDF files and images are NOT included in this repository** due to publisher copyright restrictions.

The repository provides the following publicly redistributable assets:

- **DOI index**: `data/metadata/paper_dois.csv` — 201 papers (Ti + Ni alloys) with DOIs, titles, and DOIs so users can obtain the original PDFs from publishers
- **Extracted structured data**: `data/extracted_json/` — factual information (compositions, processing, properties, heat treatment) extracted by LLM, in structured JSON format
- **Knowledge graph outputs**: `data/kg_outputs/` — constructed KGs in JSON/HTML format (node/edge lists, not raw images)
- **Causal Hypothesis Graph outputs**: `data/dag_outputs/` — candidate causal edge lists with confidence scores (hypothesis only, not validated claims)
- **Annotation schemas**: `data/annotations/` — label formats, YOLO label README, classification label definitions (no underlying images)
- **Ground truth**: `data/validation_ground_truth/` — domain expert-annotated metallurgical mechanism validation set
- **Configuration files**: `configs/` — all model, training, and inference configs required to reproduce the pipeline
- **Training & inference scripts**: `training/` and `scripts/` — full reproducibility scripts
- **Metadata & splits**: `data/metadata/` — paper index, train/val/test splits (where applicable)

**What is NOT redistributed:**
- Original publisher PDFs
- Raw figure images extracted from PDFs
- Trained model checkpoint files (available upon reasonable request)

See `docs/data_availability.md` for the complete data access policy and request process.

## License

MIT License. See [LICENSE](LICENSE) for details.
