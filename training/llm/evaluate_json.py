"""
"""

import os
import re
import json
import argparse
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
#
# ============================================================

def parse_json_output(text: str) -> Tuple[Optional[Dict], str]:
    """
    """
    if not text or not text.strip():
        return None, "empty_output"

    text = text.strip()

    #
    try:
        return json.loads(text), "direct"
    except json.JSONDecodeError:
        pass

    #
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip()), "markdown_block"
        except json.JSONDecodeError:
            pass

    #
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0)), "brace_extract"
        except json.JSONDecodeError as e:
            #
            fixed = _fix_braces(brace_match.group(0))
            try:
                return json.loads(fixed), "brace_fixed"
            except json.JSONDecodeError:
                return None, f"parse_failed_at_pos_{e.pos}"

    #
    bracket_match = re.search(r"\[[\s\S]*\]", text)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0)), "bracket_extract"
        except json.JSONDecodeError:
            pass

    return None, "no_json_found"


def _fix_braces(text: str) -> str:
    """Module functionality."""
    depth = 0
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

    if depth > 0:
        text += "}" * depth
    return text


def check_truncation(text: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """
    """
    text = text.strip()
    return {
        "is_truncated": False,  # Set to True when max_tokens check is needed
        "likely_truncated": (
            text.endswith(",") or
            text.endswith(":") or
            text.endswith('"') or
            "..." == text[-3:]
        ),
        "ends_with_comma": text.endswith(","),
        "has_unclosed_brace": text.count("{") != text.count("}"),
    }


# ============================================================
#
# ============================================================

def flatten_dict(d: Dict, prefix: str = "") -> Dict[str, Any]:
    """
    """
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.update(flatten_dict(item, f"{key}[{i}]"))
                else:
                    items[f"{key}[{i}]"] = item
        else:
            items[key] = v
    return items


def compute_loose_entity_f1(
    pred: Dict,
    gt: Dict,
    prefix: str = "samples",
) -> Dict[str, float]:
    """
    """
    pred_flat = flatten_dict(pred)
    gt_flat = flatten_dict(gt)

    pred_keys = set(pred_flat.keys())
    gt_keys = set(gt_flat.keys())

    #
    tp_exact = 0
    tp_loose = 0
    for key in gt_keys:
        if key in pred_keys:
            pv = pred_flat[key]
            gv = gt_flat[key]
            if pv == gv:
                tp_exact += 1
            elif _loose_match(pv, gv):
                tp_loose += 1

    tp = tp_exact + tp_loose
    precision = tp / len(pred_keys) if pred_keys else 0.0
    recall = tp / len(gt_keys) if gt_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_matches": tp_exact,
        "loose_matches": tp_loose,
        "total_gt": len(gt_keys),
        "total_pred": len(pred_keys),
    }


def _loose_match(pv, gv) -> bool:
    """Module functionality."""
    if pv is None or gv is None:
        return pv == gv
    pv_str = str(pv).strip().lower()
    gv_str = str(gv).strip().lower()
    if pv_str == gv_str:
        return True
    #
    try:
        pv_num = float(pv_str.replace(",", ""))
        gv_num = float(gv_str.replace(",", ""))
        if gv_num == 0:
            return abs(pv_num) < 0.001
        return abs(pv_num - gv_num) / abs(gv_num) < 0.01
    except (ValueError, TypeError):
        pass
    return False


# ============================================================
#
# ============================================================

def evaluate_single(
    pred_text: str,
    gt_json: Dict,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    """
    parsed, parse_method = parse_json_output(pred_text)
    trunc_info = check_truncation(pred_text, max_tokens)

    result = {
        "json_valid": parsed is not None,
        "parse_method": parse_method,
        "truncation": trunc_info,
    }

    if parsed is not None:
        f1_metrics = compute_loose_entity_f1(parsed, gt_json)
        result.update(f1_metrics)
    else:
        result.update({
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "exact_matches": 0, "loose_matches": 0,
            "total_gt": 0, "total_pred": 0,
        })

    return result


def evaluate_batch(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    """
    gt_map = {g["id"]: g["target"] for g in ground_truths}

    per_sample = []
    total_f1 = 0.0
    valid_count = 0
    truncated_count = 0

    for pred in predictions:
        pid = pred["id"]
        if pid not in gt_map:
            logger.warning(f"Missing ground truth for {pid}, skipping")
            continue

        result = evaluate_single(pred["output"], gt_map[pid], max_tokens)
        result["id"] = pid
        per_sample.append(result)

        if result["json_valid"]:
            valid_count += 1
            total_f1 += result["f1"]
        if result["truncation"]["likely_truncated"]:
            truncated_count += 1

    n = len(per_sample)
    avg_f1 = total_f1 / valid_count if valid_count > 0 else 0.0
    json_valid_rate = valid_count / n if n > 0 else 0.0

    return {
        "n_samples": n,
        "json_valid_rate": json_valid_rate,
        "json_valid_count": valid_count,
        "truncated_count": truncated_count,
        "mean_f1": avg_f1,
        "mean_precision": float(np.mean([s.get("precision", 0) for s in per_sample])),
        "mean_recall": float(np.mean([s.get("recall", 0) for s in per_sample])),
        "per_sample": per_sample,
    }


# ============================================================
#
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM extraction JSON outputs")
    parser.add_argument("--predictions", type=str,
                        help="Path to predictions JSONL file")
    parser.add_argument("--ground_truth", type=str,
                        help="Path to ground truth JSON file")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max tokens used during generation")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output JSON path")
    parser.add_argument("--compare", type=str, default=None,
                        help="Comma-separated model names for comparison (qwen,mistral,llama3)")
    parser.add_argument("--data_dir", type=str, default="eval_results",
                        help="Directory containing per-model prediction files (for --compare)")
    return parser.parse_args()


def evaluate(args):
    """Module functionality."""
    if args.compare:
        model_names = [m.strip() for m in args.compare.split(",")]
        all_results = {}

        for model_name in model_names:
            pred_file = os.path.join(args.data_dir, model_name, "predictions.jsonl")
            gt_file = os.path.join(args.data_dir, "ground_truth.json")

            if not os.path.exists(pred_file):
                logger.warning(f"Predictions not found for {model_name}: {pred_file}")
                continue

            predictions = []
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    predictions.append(json.loads(line.strip()))

            with open(gt_file, "r", encoding="utf-8") as f:
                ground_truths = json.load(f)

            all_results[model_name] = evaluate_batch(
                predictions, ground_truths, args.max_tokens
            )

        #
        print(f"\n{'Model':<15} {'JSON%':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Trunc':>8}")
        print("-" * 65)
        for name, r in all_results.items():
            print(f"{name:<15} {r['json_valid_rate']:>8.4f} {r['mean_f1']:>8.4f} "
                  f"{r['mean_precision']:>8.4f} {r['mean_recall']:>8.4f} "
                  f"{r['truncated_count']:>8}")

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    else:
        predictions = []
        with open(args.predictions, "r", encoding="utf-8") as f:
            for line in f:
                predictions.append(json.loads(line.strip()))

        with open(args.ground_truth, "r", encoding="utf-8") as f:
            ground_truths = json.load(f)

        results = evaluate_batch(predictions, ground_truths, args.max_tokens)

        print(f"\nEvaluation Results:")
        print(f"  Samples:        {results['n_samples']}")
        print(f"  JSON Valid:     {results['json_valid_rate']:.4f} ({results['json_valid_count']}/{results['n_samples']})")
        print(f"  Truncated:      {results['truncated_count']}")
        print(f"  Mean F1:        {results['mean_f1']:.4f}")
        print(f"  Mean Precision: {results['mean_precision']:.4f}")
        print(f"  Mean Recall:    {results['mean_recall']:.4f}")

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    evaluate(args)
