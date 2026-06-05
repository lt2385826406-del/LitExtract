# Usage Guide

## Installation

```bash
pip install -r requirements.txt
```

For optional dependencies (vision, LLM fine-tuning):
```bash
pip install torch ultralytics transformers peft bitsandbytes
```

System dependencies:
- Poppler (for PDF-to-image): `apt install poppler-utils` or download from poppler releases
- Tesseract (for OCR): `apt install tesseract-ocr`

## Quick Start

### 1. PDF Figure Detection (YOLO)
```bash
python src/vision/yolo_detection.py --pdf_dir pdf/ --output_dir output_bboxes/
```

### 2. Semantic Extraction (DeepSeek API)
```bash
python scripts/run_pipeline.py paper.pdf --api-key $DEEPSEEK_API_KEY
```

### 3. Knowledge Graph Construction
```bash
python src/kg_construction/build_graph.py --semantic_results outputs/ --output kg.json
```

### 4. Causal DAG Construction
```bash
python src/dag_construction/build_dag.py --kg kg.json --output dag.json
```
