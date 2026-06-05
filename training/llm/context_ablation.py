"""
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
#
# ============================================================

def load_ablation_results(results_dir: str) -> Dict[str, Dict[str, float]]:
    """
    """
    results = {}

    for entry in os.listdir(results_dir):
        entry_path = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        summary_file = os.path.join(entry_path, "summary.json")
        if not os.path.exists(summary_file):
            summary_file = os.path.join(entry_path, "evaluation_results.json")

        if os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            results[entry] = data

    return results


def parse_experiment_name(name: str) -> Dict[str, str]:
    """
    """
    parts = name.replace("lora_out_", "").split("_")
    info = {"raw": name}

    #
    models = ["qwen", "mistral", "deepseek", "llama3", "gemma"]
    for m in models:
        if m in parts:
            info["model"] = m
            break

    #
    if "8k" in name:
        info["context"] = "8k"
    elif "4k" in name:
        info["context"] = "4k"

    #
    if "v2" in name:
        info["data_version"] = "v2"
    else:
        info["data_version"] = "v1"

    return info


# ============================================================
#
# ============================================================

def context_length_ablation(
    results: Dict[str, Dict],
) -> pd.DataFrame:
    """
    """
    rows = []
    for exp_name, data in results.items():
        info = parse_experiment_name(exp_name)
        if "context" not in info:
            continue

        rows.append({
            "experiment": exp_name,
            "model": info.get("model", "unknown"),
            "context": info["context"],
            "f1": data.get("mean_f1", data.get("f1", 0)),
            "json_valid_rate": data.get("json_valid_rate", data.get("json_valid_rate", 0)),
            "precision": data.get("mean_precision", 0),
            "recall": data.get("mean_recall", 0),
        })

    return pd.DataFrame(rows)


def data_quality_ablation(
    results: Dict[str, Dict],
) -> pd.DataFrame:
    """
    """
    rows = []
    for exp_name, data in results.items():
        info = parse_experiment_name(exp_name)

        rows.append({
            "experiment": exp_name,
            "model": info.get("model", "unknown"),
            "context": info.get("context", "unknown"),
            "data_version": info.get("data_version", "v1"),
            "f1": data.get("mean_f1", data.get("f1", 0)),
            "n_samples": data.get("n_samples", 0),
        })

    return pd.DataFrame(rows)


def field_level_heatmap(
    per_sample_results: List[Dict],
    field_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    """
    #
    field_scores = defaultdict(list)

    for sample in per_sample_results:
        for key, value in sample.items():
            if key.startswith("f1_") or key.endswith("_f1"):
                field = key.replace("f1_", "").replace("_f1", "")
                if isinstance(value, (int, float)):
                    field_scores[field].append(value)

    rows = []
    for field, scores in field_scores.items():
        display_name = field_mapping.get(field, field) if field_mapping else field
        rows.append({
            "field": display_name,
            "mean_f1": float(np.mean(scores)),
            "std_f1": float(np.std(scores)),
            "n": len(scores),
        })

    return pd.DataFrame(rows)


# ============================================================
#
# ============================================================

def plot_context_ablation(
    df: pd.DataFrame, output_path: str, title: str = "Context Length Ablation"
):
    """
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    #
    models = df["model"].unique()
    x = np.arange(len(models))
    width = 0.35

    ax1 = axes[0]
    for i, ctx in enumerate(["4k", "8k"]):
        ctx_data = df[df["context"] == ctx]
        values = []
        for m in models:
            model_data = ctx_data[ctx_data["model"] == m]
            values.append(model_data["f1"].mean() if not model_data.empty else 0)
        ax1.bar(x + i * width, values, width, label=f"Context={ctx}")

    ax1.set_xlabel("Model")
    ax1.set_ylabel("F1 Score")
    ax1.set_title(f"{title} — F1")
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    #
    ax2 = axes[1]
    for i, ctx in enumerate(["4k", "8k"]):
        ctx_data = df[df["context"] == ctx]
        values = []
        for m in models:
            model_data = ctx_data[ctx_data["model"] == m]
            values.append(
                model_data["json_valid_rate"].mean() * 100
                if not model_data.empty else 0
            )
        ax2.bar(x + i * width, values, width, label=f"Context={ctx}")

    ax2.set_xlabel("Model")
    ax2.set_ylabel("JSON Valid Rate (%)")
    ax2.set_title(f"{title} — JSON Validity")
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved context ablation chart to {output_path}")


def plot_field_heatmap(
    df: pd.DataFrame, output_path: str, title: str = "Field-Level F1 Heatmap"
):
    """
    """
    if df.empty:
        logger.warning("No field-level data for heatmap")
        return

    pivot = df.pivot_table(
        values="mean_f1", index="field", aggfunc="mean"
    ).sort_values("mean_f1", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.4)))

    im = ax.imshow(pivot.values.reshape(-1, 1), aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks([])
    ax.set_title(title)

    #
    for i, val in enumerate(pivot.values.flatten()):
        ax.text(0, i, f"{val:.3f}", ha="center", va="center",
                fontweight="bold", fontsize=10,
                color="white" if val < 0.5 else "black")

    plt.colorbar(im, ax=ax, label="F1 Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved field heatmap to {output_path}")


def plot_model_comparison(
    results: Dict[str, Dict], output_path: str, title: str = "Model Comparison"
):
    """
    """
    model_names = list(results.keys())
    metrics = {
        "F1": [results[m].get("mean_f1", 0) for m in model_names],
        "JSON Valid %": [results[m].get("json_valid_rate", 0) * 100 for m in model_names],
        "Precision": [results[m].get("mean_precision", 0) for m in model_names],
        "Recall": [results[m].get("mean_recall", 0) for m in model_names],
    }

    x = np.arange(len(model_names))
    width = 0.2
    n_metrics = len(metrics)
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (metric_name, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric_name,
                      color=colors[i])
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(model_names)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved model comparison chart to {output_path}")


# ============================================================
#
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run context ablation analysis")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing per-experiment evaluation results")
    parser.add_argument("--output", type=str, default="charts",
                        help="Output directory for charts")
    parser.add_argument("--title_prefix", type=str, default="LLM Extraction Ablation",
                        help="Chart title prefix")
    return parser.parse_args()


def main(args):
    os.makedirs(args.output, exist_ok=True)

    #
    logger.info(f"Loading results from {args.results_dir}")
    results = load_ablation_results(args.results_dir)
    logger.info(f"Loaded {len(results)} experiments")

    if not results:
        logger.error("No results found!")
        return

    #
    ctx_df = context_length_ablation(results)
    if not ctx_df.empty:
        ctx_df.to_csv(os.path.join(args.output, "context_ablation.csv"), index=False)

        #
        has_both = (ctx_df["context"].nunique() >= 2)
        if has_both:
            plot_context_ablation(
                ctx_df,
                os.path.join(args.output, "context_ablation.png"),
                title=f"{args.title_prefix}: Context Length",
            )

    #
    qual_df = data_quality_ablation(results)
    if not qual_df.empty:
        qual_df.to_csv(os.path.join(args.output, "data_quality_ablation.csv"), index=False)

    #
    if results:
        #
        model_results = {}
        for exp_name, data in results.items():
            info = parse_experiment_name(exp_name)
            model_info = info.get("model", exp_name)
            if model_info not in model_results:
                model_results[model_info] = data

        if len(model_results) > 1:
            plot_model_comparison(
                model_results,
                os.path.join(args.output, "model_comparison.png"),
                title=f"{args.title_prefix}: Model Comparison",
            )

    #
    summary_rows = []
    for exp_name, data in results.items():
        info = parse_experiment_name(exp_name)
        summary_rows.append({
            "experiment": exp_name,
            "model": info.get("model", "unknown"),
            "context": info.get("context", "unknown"),
            "data_version": info.get("data_version", "v1"),
            "f1": data.get("mean_f1", data.get("f1", 0)),
            "json_valid_rate": data.get("json_valid_rate", 0),
            "precision": data.get("mean_precision", 0),
            "recall": data.get("mean_recall", 0),
            "n_samples": data.get("n_samples", 0),
        })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(args.output, "ablation_summary.csv"), index=False)
        print(f"\nAblation Summary:")
        print(summary_df.to_string(index=False))

    print(f"\nAll outputs saved to: {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    main(args)
