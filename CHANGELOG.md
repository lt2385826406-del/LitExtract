# Changelog

All notable changes to the LitExtract project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `.gitignore` for clean version control
- `CONTRIBUTING.md` with contribution guidelines
- `CHANGELOG.md` (this file)
- Dockerfile for containerized deployment
- Minimum test suite (`tests/`)

### Changed
- Updated `environment.yml` with complete dependency list
- Translated all Chinese comments/strings in source code to English (Batch 1-4)

### Fixed
- Syntax errors in `kg_visualizer.py` and `export_neo4j.py`

---

## [1.0.0] - 2026-06-05

### Added
- Initial public release
- PDF parsing via PyMuPDF and pdf2image
- Figure/table detection via YOLOv11
- OCR and caption matching via Tesseract
- Semantic extraction via DeepSeek API and local LLM (Mistral-7B, Qwen2.5-7B + LoRA)
- Knowledge graph construction (NetworkX, pyvis)
- Causal DAG construction from co-occurrence patterns
- Support for titanium alloy and nickel-based superalloy literature
- Training scripts for YOLO, ResNet-50, and LLM QLoRA
- 201-paper DOI index in `data/metadata/paper_dois.csv`
- MIT License

---

[Unreleased]: https://github.com/lt2385826406-del/LitExtract/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/lt2385826406-del/LitExtract/releases/tag/v1.0.0
