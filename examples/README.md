# LitExtract Examples

This directory contains example inputs and outputs for the LitExtract pipeline.

## Sample Inputs

Sample PDF files for testing the pipeline are not included in the repository due to copyright restrictions. You can use your own materials science papers in PDF format.

### Using your own PDFs

Place your PDF files in `examples/sample_inputs/`, then run:

```bash
python scripts/run_pipeline.py --input examples/sample_inputs/your_paper.pdf --output examples/sample_outputs/
```

## Sample Outputs

After running the pipeline, the output directory will contain:

- `{paper_name}_kg.html` — Interactive Knowledge Graph visualization
- `{paper_name}_kg.json` — KG structured data
- `{paper_name}_chg.html` — Causal Hypothesis Graph visualization
- `{paper_name}_chg.json` — CHG structured data

## Demo Pipeline

For a quick demo using pre-processed example data:

```bash
python examples/demo_pipeline/run_demo.py
```

See `docs/usage.md` for detailed usage instructions.
