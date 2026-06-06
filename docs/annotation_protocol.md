# Annotation Protocol

## 1. YOLO Task — PDF Graphic Element Detection

### 1.1 Task Definition

Detect and localize four classes of graphic elements on PDF pages of scientific literature:

| Class ID | Class Name | Description |
|----------|-----------|-------------|
| 0 | `caption` | Figure caption text block |
| 1 | `image` | Full figure / image region |
| 2 | `subgraph` | Individual sub-figure panel (e.g., (a), (b)) |
| 3 | `subgraph_label` | Sub-figure label marker (e.g., "a", "b", "c") |

### 1.2 Annotation Procedure

1. Convert PDF pages to PNG images (300 DPI, PyMuPDF)
2. Open images in LabelImg or Label Studio
3. Draw tight bounding boxes around each visible element
4. Assign the appropriate class ID (0–3)
5. Export in YOLO format: `class x_center y_center width height` (normalized to [0, 1])

### 1.3 Dataset Summary

| Split | Pages |
|-------|-------|
| Train | 763 |
| Val | 109 |
| Test | 219 |
| **Total** | **1,091** |

### 1.4 Quality Control

- All bounding boxes were manually drawn by a single annotator
- DeepSeek-assisted pre-annotation was used to suggest initial bounding box candidates, followed by manual verification and correction
- Annotation consistency was checked by reviewing edge cases (overlapping elements, dense layouts)
- The validation split was used to monitor training convergence; final metrics are reported on the held-out test set

---

## 2. Classification Task — Microstructure Binary Classification

### 2.1 Task Definition

Binary image classification: determine whether a YOLO-detected figure region (class 1: image) contains a material microstructure.

| Label | Class Name | Description |
|-------|-----------|-------------|
| 0 | `microstructure` | SEM/TEM/optical micrographs showing grain structure, phases, etc. |
| 1 | `other` | All other figure types (schematics, plots, tables, flowcharts, etc.) |

### 2.2 Annotation Procedure

1. Extract figure crops using YOLO bounding boxes (class 1: image)
2. DeepSeek-assisted pre-annotation: assign preliminary binary labels
3. Manual verification: review and correct all pre-annotated labels
4. Dataset organization: images placed in class subfolders (`microstructure/`, `other/`) for PyTorch `ImageFolder` loading

### 2.3 Dataset Summary

| Dataset | Images | Notes |
|---------|--------|-------|
| Paper-reported | 1,684 | Ti-alloy figure crops used in paper results |
| Expanded pool | 3,200 | Larger pool available in repository |

### 2.4 Quality Control

- Every pre-annotated label was manually verified and corrected
- Edge cases (partial micrographs, composite figures) were flagged for additional review
- 10-fold stratified cross-validation was used for model evaluation (see `training/classification/train_resnet.py`)

---

## 3. Semantic Extraction Task — LLM-Based Information Extraction

### 3.1 Task Definition

Extract structured materials science information from full-text papers into a standardized JSON schema with four top-level categories:

| Category | Example Fields |
|----------|---------------|
| `composition` | Alloy name, nominal composition (wt.%), element ranges |
| `processes` | Manufacturing method, process parameters, heat treatment |
| `microstructures` | Phase types, grain size, morphology descriptions |
| `properties` | UTS, YS, elongation, hardness, test conditions |

### 3.2 Annotation Procedure

1. Read the full paper text (abstract, body, conclusions)
2. Identify and extract relevant information for each schema category
3. Populate the JSON schema fields with extracted values and units
4. Mark causal/associative relations between fields (with confidence scores)
5. DeepSeek-assisted pre-annotation followed by full manual verification

### 3.3 Dataset Summary

| Dataset | Papers | Alloy Systems |
|---------|--------|---------------|
| LLM Training (SFT) | 199 | Ti-based + Ni-based alloys |
| Ti-MMKG case study | 98 | Ti-based alloys |
| Ni-DAG case study | 68 | Ni-based superalloys |

### 3.4 Quality Control

- All 199 papers underwent DeepSeek-assisted pre-annotation
- Every JSON output was manually reviewed and corrected for factual accuracy
- Schema completeness was verified (all required fields populated)
- Extraction consistency was cross-checked across papers discussing similar alloys/processes

---

## 4. General QC Procedure

### 4.1 Inter-Annotator Agreement

Due to the specialized domain nature, annotation was performed by a single materials science domain expert, with DeepSeek-assisted pre-annotation for efficiency. All pre-annotated labels were manually verified.

### 4.2 Annotation Guidelines

- Prioritize factual accuracy over completeness — mark ambiguous fields as "unknown" rather than guessing
- Use consistent terminology (e.g., "wt.%" not "weight percent"; "solution treatment" not "solutionizing")
- Record extraction source (figure/table/text section) when possible
- Schema definitions are documented in `configs/` and `data/annotations/schema_examples/`
