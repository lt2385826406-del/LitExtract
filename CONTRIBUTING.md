# Contributing to LitExtract

Thank you for your interest in contributing to LitExtract!

## How to Contribute

### Reporting Bugs

- Open a [GitHub Issue](https://github.com/lt2385826406-del/LitExtract/issues) with the label `bug`
- Include: Python version, OS, steps to reproduce, expected vs actual behavior
- If relevant, attach a minimal PDF snippet (ensure you have permission to share it)

### Suggesting Features

- Open a [GitHub Issue](https://github.com/lt2385826406-del/LitExtract/issues) with the label `enhancement`
- Describe the use case and expected behavior

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run `ruff check src/` and `ruff format src/` to ensure code style
5. Add tests for new functionality in `tests/`
6. Submit a pull request with a clear description

## Code Style

- Python: [PEP 8](https://pep8.org/), enforced via `ruff`
- Line length: 120 characters
- Docstrings: Google style
- All user-facing strings and comments must be in English

## Adding New Modules

- Place source code under `src/<module_name>/`
- Add a `configs/<module_name>/` directory for YAML configs if needed
- Update `README.md` and `docs/architecture.md` with the new module description
- Add at least one smoke test in `tests/test_<module_name>.py`

## Dataset Contributions

- Due to publisher copyright restrictions, **do not** commit PDF files or images from papers
- You may contribute:
  - Annotation files (YOLO labels, classification labels) in `data/annotations/`
  - Extracted JSON data in `data/extracted_json/`
  - DOI lists in `data/metadata/`
- All contributed data must be accompanied by a statement that you have the right to share it

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
