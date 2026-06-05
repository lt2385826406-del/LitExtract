# LitExtract

**Automated Literature Knowledge Extraction for Materials Science**

LitExtract is a modular pipeline that extracts structured materials knowledge — composition, processing, microstructure, and properties — from scientific literature using a combination of computer vision and large language models.

## Architecture

```
PDF → [YOLO Figure Detection] → [OCR/Alignment] → [LLM Extraction] → [KG + Causal DAG]
```

### Key Components

| Module | Description | Technology |
|--------|-------------|------------|
| `pdf_parser` | PDF-to-image conversion, page parsing | PyMuPDF, pdf2image |
| `vision` | Figure/table/micrograph detection | YOLOv11 |
| `ocr_alignment` | OCR text extraction, figure-caption matching | Tesseract |
| `semantic_extraction` | Structured entity/relation extraction | DeepSeek / Qwen2.5-7B + LoRA |
| `kg_construction` | Knowledge graph construction | NetworkX, pyvis |
| `dag_construction` | Causal DAG from co-occurrence patterns | Causal discovery pipeline |

### Output

- **Knowledge Graph (KG)**: Element → Microstructure → Property triples
- **Causal Hypothesis Graph (CHG)**: Cross-paper causal inferences

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run pipeline on a PDF
python scripts/run_pipeline.py paper.pdf --api-key $DEEPSEEK_API_KEY

# Build knowledge graph
python src/kg_construction/build_graph.py --semantic_results outputs/ --output kg.json

# Build causal DAG
python src/dag_construction/build_dag.py --kg kg.json --output dag.json
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

**PDF files and images are NOT included in this repository** due to publisher copyright restrictions. The repository provides:

- **DOI index**: `data/metadata/paper_dois.csv` — 201 papers (Ti + Ni alloys) with DOIs so users can obtain the original PDFs from publishers
- **Extracted structured data**: `data/extracted_json/` — factual information (compositions, processing, properties) extracted by LLM
- **Annotation templates**: `data/annotations/` — label formats and schema, without the underlying images
- **Ground truth**: `data/validation_ground_truth/` — domain expert-annotated metallurgical mechanisms

See `docs/data_availability.md` for the complete data access policy.

## License

MIT License. See [LICENSE](LICENSE) for details.
