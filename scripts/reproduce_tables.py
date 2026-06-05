"""
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

#
ALL_TABLES = {
    1: {
        "name": "Corpus Statistics",
        "description": "Corpus statistics (paper count, page count, annotation volume)",
        "source": "data/metadata/corpus_statistics.csv",
    },
    2: {
        "name": "YOLO Detection Performance",
        "description": "YOLOv11 four-class detection mAP comparison table",
        "source": "evaluation_results.json → standard metrics",
    },
    3: {
        "name": "Classification Model Comparison",
        "description": "ResNet18/ResNet50/VGG19 classification accuracy/F1 comparison",
        "source": "results/classification/*_summary.json",
    },
    4: {
        "name": "LLM Extraction F1 by Field",
        "description": "Three models × four fields F1 comparison table",
        "source": "eval_correct_results_v2_all/*/summary.json",
    },
    5: {
        "name": "Context Ablation Results",
        "description": "4k vs 8k context window ablation comparison",
        "source": "charts/ablation_summary.csv",
    },
    6: {
        "name": "DAG Statistics",
        "description": "DAG node count, edge count, causal chain statistics",
        "source": "dag_outputs/*_stats.json",
    },
    7: {
        "name": "Robustness Test",
        "description": "Literature removal robustness test (10%/20%/30% retention rates)",
        "source": "supplementary/robustness_results/",
    },
    8: {
        "name": "Model Training Hyperparameters",
        "description": "All model training hyperparameter summary table",
        "source": "configs/",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce all paper tables")
    parser.add_argument("--tabs", type=str, default=None,
                        help="Comma-separated table numbers (e.g., 1,3,5). Default: all.")
    parser.add_argument("--output", type=str, default="outputs/tables",
                        help="Output directory")
    parser.add_argument("--format", type=str, default="csv",
                        choices=["csv", "latex", "json", "all"],
                        help="Output format")
    parser.add_argument("--list", action="store_true",
                        help="List all tables without generating")
    return parser.parse_args()


def list_tables():
    """Module functionality."""
    print(f"\n{'Tab':<6} {'Name':<45} {'Source'}")
    print("-" * 85)
    for num, info in ALL_TABLES.items():
        print(f"{num:<6} {info['name']:<45} {info['source']}")


def export_to_latex(data: list, headers: list, caption: str, label: str) -> str:
    """
    """
    n_cols = len(headers)
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append(f"  \\caption{{{caption}}}")
    latex.append(f"  \\label{{tab:{label}}}")
    latex.append(f"  \\begin{{tabular}}{{{'l' + 'r' * (n_cols - 1)}}}")
    latex.append("    \\toprule")
    latex.append("    " + " & ".join(headers) + " \\\\")
    latex.append("    \\midrule")
    for row in data:
        latex.append("    " + " & ".join(str(c) for c in row) + " \\\\")
    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")
    return "\n".join(latex)


def generate_table(tab_num: int, info: dict, output_dir: str, fmt: str) -> bool:
    """
    """
    print(f"\n{'='*60}")
    print(f"Table {tab_num}: {info['name']}")
    print(f"  Source: {info['source']}")

    source_path = os.path.join(PROJECT_ROOT, info["source"])
    if not os.path.exists(source_path) and not "*" in info["source"]:
        print(f"  [WARN] Source not found: {source_path}")
        print(f"  [TODO] Implement corresponding data aggregation logic.")
        return False

    print(f"  [TODO] Full table generation requires data aggregation logic.")
    return False


def main(args):
    os.makedirs(args.output, exist_ok=True)

    if args.list:
        list_tables()
        return

    if args.tabs:
        tab_nums = [int(x.strip()) for x in args.tabs.split(",")]
    else:
        tab_nums = list(ALL_TABLES.keys())

    print(f"Reproducing {len(tab_nums)} tables...")
    print(f"Output format: {args.format}")

    generated = 0
    for num in tab_nums:
        if num not in ALL_TABLES:
            print(f"  [WARN] Unknown table {num}, skipping")
            continue
        if generate_table(num, ALL_TABLES[num], args.output, args.format):
            generated += 1

    print(f"\nDone: {generated}/{len(tab_nums)} tables generated")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    main(args)
