# LitExtract Data Directory

## Directory Structure

```
data/
├── README.md                          # This file
├── metadata/                          # Paper metadata and DOI indices
│   ├── paper_dois.json               # Unified DOI index (JSON) — 199 papers (Ti + Ni)
│   ├── paper_dois.csv                # Unified DOI index (CSV) — same data, spreadsheet-compatible
│   ├── llm_training_dois.csv         # LLM training set only (199 papers)
│   ├── ti_mmkg_dois.csv              # Ti-MMKG case study subset (98 papers)
│   ├── ni_dag_dois.csv               # Ni-DAG case study subset (68 papers)
│   ├── paper_metadata.csv            # Paper metadata template
│   └── corpus_statistics.csv         # Corpus statistics
├── extracted_json/                    # LLM semantic extraction results
│   ├── ti_semantic_data.json         # Titanium alloys
│   └── nickel_semantic_data.json     # Nickel-based alloys (68 papers)
├── kg_outputs/                        # Knowledge graph outputs (standardized format)
│   ├── nodes.csv                     # Node table (301 nodes)
│   ├── edges.csv                     # Edge table (3,747 edges)
│   ├── kg.graphml                    # GraphML network graph
│   ├── ti_mmkg_data.json             # Ti-MMKG case study extracted data
│   └── example_subgraphs/           # 16 key node subgraphs (JSON + INDEX)
├── dag_outputs/                       # Causal Hypothesis Graph (CHG) outputs (Ti/Ni separated)
│   ├── ti/                           # Titanium alloy DAG
│   │   ├── dag_edges.csv             # Candidate hypothesis edge table (8,625 edges)
│   │   ├── node_metadata.csv         # Node metadata (647 nodes)
│   │   ├── representative_paths.md   # Representative candidate paths (Top-10)
│   │   └── dag.graphml               # GraphML (647 nodes, 8,625 edges)
│   └── ni/                           # Nickel-based alloy CHG
│       ├── dag_edges.csv             # Candidate hypothesis edge table
│       ├── node_metadata.csv         # Node metadata (195 nodes)
│       ├── validation_results.csv    # Validation metrics (7 metrics)
│       ├── representative_paths.md   # Representative candidate paths (Top-20)
│       └── dag.graphml               # GraphML (195 nodes, 1,010 edges)
├── validation_ground_truth/           # Ground Truth validation data
│   └── ground_truth_mechanisms.json  # 10 classical metallurgical mechanisms for titanium alloys
├── annotations/                       # Annotation data (templates)
│   ├── extraction_annotations/       # Information extraction annotations
│   ├── schema_examples/              # Schema examples
│   ├── yolo_labels/                  # YOLO detection annotations (no images, see copyright notice)
│   └── classification_labels/        # Microstructure classification annotations (no images, see copyright notice)
└── splits/                            # Dataset splits (templates)
    └── (extraction_split.csv generated at runtime)
```

## Copyright Notice

### Image Data (YOLO / Classification)

**YOLO detection** and **microstructure classification** training/validation image datasets **are NOT included in this repository**, because:

1. All images are sourced from copyright-protected academic paper PDFs
2. Paper publishers impose restrictions on large-scale image redistribution
3. Annotation files (bounding boxes / class labels) can be obtained by requesting academic collaboration from the authors

To reproduce training, please:
- Collect relevant paper PDFs from publicly available literature yourself
- Use this project's `VisionAgent` (YOLO detection) and `MicrostructureClassifier` for preprocessing
- Annotation format is defined in the Schema files under `configs/`

### Text Data (LLM Extraction)

The semantic extraction results in `extracted_json/` are machine-extracted structured data, consisting of **factual information** (alloy composition, processing parameters, property values, etc.), which is not subject to copyright restrictions of the original publications. Original DOI information is available in `metadata/paper_dois.json` (JSON) and `metadata/paper_dois.csv` (CSV). **The underlying full-text PDFs are NOT included** — users must obtain them from the respective publishers via the DOIs provided.

## Data Sources

| Dataset | # Papers | Alloy System | Source |
|---------|----------|--------------|--------|
| Ti-MMKG | 98 | Titanium alloys (Ti-6Al-4V, Ti-17, etc.) | Automated extraction from public literature |
| Ni-DAG | 68 | Nickel-based superalloys (IN718, CMSX-4, etc.) | Automated extraction from public literature |
| Ground Truth | 10 mechanisms | Classical titanium alloy metallurgical theory | Domain expert annotation |
| DOI Index | 199 | Ti + Ni alloys | CrossRef + automated extraction |
