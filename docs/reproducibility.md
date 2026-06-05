# Reproducibility

## Environment

- Python 3.10+
- Key packages pinned in `requirements.txt`

## Data

- Corpus metadata: `data/metadata/paper_metadata.csv`
- Annotations: `data/annotations/`
- Train/val/test splits: `data/splits/`

## Experiments

### Ti-alloy Knowledge Graph (98 papers)
```bash
python scripts/build_ti_case_study.py
```

### Nickel-alloy Causal DAG Demo (69 papers)
```bash
python scripts/build_nickel_demo.py
```

### Robustness Testing
```bash
python src/validation/robustness_test.py --dag dag.json --remove_ratios 0.1,0.2,0.3
```

## Figures & Tables

```bash
python scripts/reproduce_figures.py
python scripts/reproduce_tables.py
```
