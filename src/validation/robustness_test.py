"""
"""

import json
import os
import sys
import argparse
import random
import copy
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

import networkx as nx

#
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)  # LitExtract_Agent/

from ..dag_construction.cooccurrence_miner import run_step1, load_semantic_data as _load_semantic_from_file
from ..dag_construction.dag_causal_builder import CausalHypothesisGraph, NODE_ORDER

#
from .mechanism_validation import (
    load_ground_truth, flatten_triads, evaluate_mechanism_recovery,
    resolve_node_id
)


# ============================================================
#
# ============================================================

def load_semantic_data(data_path: str) -> List[Dict]:
    """Module functionality."""
    if os.path.isdir(data_path):
        # # batch_processing_data directory
        papers = []
        skipped = 0
        for fname in sorted(os.listdir(data_path)):
            if not fname.endswith("_result.json"):
                continue
            fpath = os.path.join(data_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                skipped += 1
                print(f"  [skip] {fname}: {e}")
                continue
            record = _convert_batch_result_to_paper(data)
            if record:
                papers.append(record)
        if skipped:
            print(f"  Skipped {skipped} invalid files")
        return papers
    else:
        #
        return _load_semantic_from_file(data_path)


def _convert_batch_result_to_paper(data: Dict) -> Optional[Dict]:
    """Module functionality."""
    filename = data.get("filename", data.get("task_id", "unknown"))
    sem = data.get("semantic_results", {})
    samples = sem.get("samples", [])

    elements = set()
    alloys = set()
    processes = set()
    micros = set()
    properties = set()

    for s in samples:
        comp = s.get("composition") or {}
        if comp.get("canonical_name"):
            alloys.add(comp["canonical_name"])
        for el in (comp.get("elements") or []):
            if isinstance(el, dict):
                el_name = el.get("element", "")
                if el_name:
                    elements.add(el_name)
            elif isinstance(el, str):
                elements.add(el)

        proc = s.get("process") or {}
        if proc.get("method"):
            processes.add(proc["method"])

        micro = s.get("microstructure") or {}
        phase_types = micro.get("phase_type") or []
        for pt in phase_types:
            if isinstance(pt, str):
                micros.add(pt)
        morph = micro.get("morphology")
        if morph and isinstance(morph, str):
            micros.add(morph)

        prop = s.get("property") or {}
        if prop.get("type"):
            properties.add(prop["type"])

    if not (elements or processes or micros or properties):
        return None  

    return {
        "title": filename,
        "folder": filename.replace(".pdf", ""),
        "elements": sorted(elements),
        "alloy_names": sorted(alloys),
        "processes": sorted(processes),
        "microstructures": sorted(micros),
        "properties": sorted(properties),
        "beta_transus": None,
        "proc_temps": [],
        "mech_props": {},
        "causal_pairs": [],
        "chunk_count": 0,
    }


# ============================================================
#
# ============================================================

def _export_dag_to_json(g_nx: nx.DiGraph, output_path: str):
    """Module functionality."""
    data = {
        "nodes": [],
        "edges": []
    }
    for nid, ndata in g_nx.nodes(data=True):
        node_entry = {"id": nid}
        node_entry.update(ndata)
        data["nodes"].append(node_entry)
    
    for u, v, edata in g_nx.edges(data=True):
        edge_entry = {"source": u, "target": v}
        edge_entry.update(edata)
        #
        edge_entry.pop("_key", None)
        data["edges"].append(edge_entry)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [export] DAG saved to: {output_path}")
    print(f"         nodes: {len(data['nodes'])}, edges: {len(data['edges'])}")


# ============================================================
#
# ============================================================

def build_dag_from_papers(
    papers: List[Dict],
    min_freq: int = 2,
    apply_constraints: bool = True,
    enforce_dag: bool = True,
    verbose: bool = False,
) -> Tuple[CausalHypothesisGraph, nx.DiGraph, Dict[str, Any]]:
    """
    """
    #
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="ti_robustness_")
    sem_json = os.path.join(tmp_dir, "semantic_data.json")
    cand_json = os.path.join(tmp_dir, "candidates.json")

    with open(sem_json, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    #
    candidates = run_step1(sem_json, cand_json, min_freq=min_freq)

    #
    random.seed(42)
    chg = CausalHypothesisGraph()
    stats = chg.build_dag_from_candidates(
        candidates,
        apply_constraints=apply_constraints,
        enforce_dag=enforce_dag,
        verbose=verbose,
    )

    #
    g_simple = nx.DiGraph()
    for u, v, key, data in chg.cg.edges(data=True, keys=True):
        if g_simple.has_edge(u, v):
            existing_conf = float(g_simple[u][v].get("confidence", 0))
            new_conf = float(data.get("confidence", 0))
            if new_conf > existing_conf:
                g_simple[u][v].update(data)
                g_simple[u][v]["_key"] = key
        else:
            g_simple.add_edge(u, v, **data)
            g_simple[u][v]["_key"] = key

    #
    for nid, ndata in chg.cg.nodes(data=True):
        if nid not in g_simple.nodes:
            g_simple.add_node(nid, **ndata)

    #
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return chg, g_simple, stats


# ============================================================
#
# ============================================================

def compute_graph_metrics(g_full: nx.DiGraph, g_partial: nx.DiGraph) -> Dict[str, Any]:
    """Module functionality."""
    full_nodes = set(g_full.nodes())
    part_nodes = set(g_partial.nodes())
    full_edges = set(g_full.edges())
    part_edges = set(g_partial.edges())

    #
    shared_nodes = full_nodes & part_nodes
    node_jaccard = len(shared_nodes) / len(full_nodes | part_nodes) if (full_nodes | part_nodes) else 0
    node_retention = len(shared_nodes) / len(full_nodes) if full_nodes else 0

    #
    shared_edges = full_edges & part_edges
    edge_jaccard = len(shared_edges) / len(full_edges | part_edges) if (full_edges | part_edges) else 0
    edge_retention = len(shared_edges) / len(full_edges) if full_edges else 0

    #
    metrics = {
        "full": {
            "nodes": len(full_nodes),
            "edges": len(full_edges),
            "density": nx.density(g_full),
        },
        "partial": {
            "nodes": len(part_nodes),
            "edges": len(part_edges),
            "density": nx.density(g_partial),
        },
        "comparison": {
            "shared_nodes": len(shared_nodes),
            "node_jaccard": node_jaccard,
            "node_retention": node_retention,
            "shared_edges": len(shared_edges),
            "edge_jaccard": edge_jaccard,
            "edge_retention": edge_retention,
            "nodes_lost": len(full_nodes - part_nodes),
            "edges_lost": len(full_edges - part_edges),
            "nodes_new": len(part_nodes - full_nodes),
            "edges_new": len(part_edges - full_edges),
        },
    }

    return metrics


# ============================================================
#
# ============================================================

def run_robustness_experiment(
    semantic_data: List[Dict],
    ground_truth_path: str,
    deletion_rates: List[float] = [0.10, 0.20, 0.30],
    num_seeds: int = 3,
    min_freq: int = 2,
    output_dir: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    """
    n_total = len(semantic_data)
    print(f"\n{'=' * 70}")
    print(" Experiment 2: Literature Robustness Test")
    print(f"{'=' * 70}")
    print(f"  Total papers: {n_total}")
    print(f"  Deletion rates: {[f'{r:.0%}' for r in deletion_rates]}")
    print(f"  Seeds per rate: {num_seeds}")
    print(f"  Min co-occurrence frequency: {min_freq}")

    #
    print(f"\n[Baseline] Building full DAG ({n_total} papers)...")
    _, g_full, stats_full = build_dag_from_papers(
        semantic_data, min_freq=min_freq, verbose=verbose
    )
    print(f"  Full DAG: {g_full.number_of_nodes()} nodes, {g_full.number_of_edges()} edges")

    #
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        baseline_dag_path = os.path.join(output_dir, "baseline_dag.json")
        _export_dag_to_json(g_full, baseline_dag_path)
        print(f"  Baseline DAG exported to: {baseline_dag_path}")

    # # 2. loadGround Truth
    mechanisms = load_ground_truth(ground_truth_path)
    triads = flatten_triads(mechanisms)

    #
    baseline_eval = evaluate_mechanism_recovery(g_full, triads, verbose=False)
    baseline_score = baseline_eval["summary"]["weighted_recovery_rate"]
    print(f"  BaselineWeighted recovery rate: {baseline_score:.1%}")

    #
    all_results = {
        "config": {
            "total_papers": n_total,
            "deletion_rates": deletion_rates,
            "num_seeds": num_seeds,
            "min_freq": min_freq,
        },
        "baseline": {
            "nodes": g_full.number_of_nodes(),
            "edges": g_full.number_of_edges(),
            "is_dag": stats_full.get("is_dag", False),
            "weighted_recovery_rate": baseline_score,
            "mechanism_summary": baseline_eval["by_mechanism"],
        },
        "experiments": [],
    }

    for rate in deletion_rates:
        n_delete = int(n_total * rate)
        n_remain = n_total - n_delete
        print(f"\n{'─' * 70}")
        print(f" Deletion rates: {rate:.0%} ({n_delete} papers removed, {n_remain} papers kept)")
        print(f"{'─' * 70}")

        rate_results = {
            "rate": rate,
            "n_deleted": n_delete,
            "n_remaining": n_remain,
            "seeds": [],
        }

        for seed in range(num_seeds):
            #
            random.seed(42 + seed * 100)

            #
            indices = list(range(n_total))
            random.shuffle(indices)
            keep_indices = sorted(indices[:n_remain])
            papers_remaining = [semantic_data[i] for i in keep_indices]

            print(f"   Seed {seed+1}: kept {len(papers_remaining)} papers, building DAG...")

            try:
                _, g_partial, stats_partial = build_dag_from_papers(
                    papers_remaining, min_freq=min_freq, verbose=False
                )
            except Exception as e:
                print(f"    ✗ Build failed: {e}")
                rate_results["seeds"].append({
                    "seed": seed,
                    "error": str(e),
                    "n_papers": len(papers_remaining),
                })
                continue

            #
            graph_metrics = compute_graph_metrics(g_full, g_partial)

            #
            mech_eval = evaluate_mechanism_recovery(g_partial, triads, verbose=False)

            seed_result = {
                "seed": seed,
                "n_papers": len(papers_remaining),
                "dag_stats": stats_partial,
                "graph_metrics": graph_metrics,
                "mechanism_eval": {
                    "summary": mech_eval["summary"],
                },
            }

            print(f"    Size: {g_partial.number_of_nodes()} nodes, {g_partial.number_of_edges()} edges")
            print(f"    Node retention: {graph_metrics['comparison']['node_retention']:.1%}")
            print(f"    Edge retention:   {graph_metrics['comparison']['edge_retention']:.1%}")
            print(f"    Weighted recovery rate: {mech_eval['summary']['weighted_recovery_rate']:.1%}")

            rate_results["seeds"].append(seed_result)

        all_results["experiments"].append(rate_results)

    #
    print(f"\n{'=' * 70}")
    print(" Summary Statistics")
    print(f"{'=' * 70}")
    _print_summary(all_results)

    # # 6. saveresult
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "robustness_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_make_serializable(all_results), f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to: {output_path}")

    return all_results


def _print_summary(results: Dict[str, Any]):
    """Module functionality."""
    baseline = results["baseline"]

    print(f"\n  DAG Comparison Metrics")
    print(f"  {'Removal rate':<10s} {'Avg Node Ret':>12s} {'Avg Edge Ret':>12s} {'Node Jaccard':>12s}")
    print(f"  {'─' * 50}")

    for exp in results["experiments"]:
        rate = exp["rate"]
        node_ret = []
        edge_ret = []
        node_jac = []
        for seed in exp["seeds"]:
            if "graph_metrics" in seed:
                gm = seed["graph_metrics"]["comparison"]
                node_ret.append(gm["node_retention"])
                edge_ret.append(gm["edge_retention"])
                node_jac.append(gm["node_jaccard"])

        if node_ret:
            avg_nr = sum(node_ret) / len(node_ret)
            avg_er = sum(edge_ret) / len(edge_ret)
            avg_nj = sum(node_jac) / len(node_jac)
            print(f"  {rate:<10.0%} {avg_nr:>12.1%} {avg_er:>12.1%} {avg_nj:>12.3f}")

    print(f"\n  Mechanism Recovery Rate Comparison")
    print(f"  {'Deletion rates':<10s} {'Avg Weighted Recovery':>14s} {'Rel Change':>10s}")
    print(f"  {'─' * 40}")
    bl_score = baseline["weighted_recovery_rate"]
    print(f"  {'Baseline':<10s} {bl_score:>14.1%} {'--':>10s}")

    for exp in results["experiments"]:
        rate = exp["rate"]
        scores = []
        for seed in exp["seeds"]:
            if "mechanism_eval" in seed:
                scores.append(seed["mechanism_eval"]["summary"]["weighted_recovery_rate"])
        if scores:
            avg_score = sum(scores) / len(scores)
            change = avg_score - bl_score
            print(f"  {rate:<10.0%} {avg_score:>14.1%} {change:>+10.1%}")


def _make_serializable(obj):
    """Module functionality."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, float)):
        return obj
    elif obj is None:
        return None
    else:
        return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Literature Robustness Test: random paper removal + DAG reconstruction")
    parser.add_argument("--semantic_data", required=True,
                        help="Path to semantic data (JSON file or batch processing data directory)")
    parser.add_argument("--ground_truth", default="ground_truth_mechanisms.json",
                        help="Ground truth mechanisms file")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory")
    parser.add_argument("--deletion_rates", nargs="+", type=float,
                        default=[0.10, 0.20, 0.30],
                        help="Deletion rates (default 0.10 0.20 0.30)")
    parser.add_argument("--num_seeds", type=int, default=3,
                        help="Number of random seeds per rate (default 3)")
    parser.add_argument("--min_freq", type=int, default=2,
                        help="Step1 min co-occurrence frequency (default 2)")
    parser.add_argument("--quiet", action="store_true",
                        help="Quiet mode")
    args = parser.parse_args()

    #
    sem_path = args.semantic_data
    gt_path = args.ground_truth
    if not os.path.isabs(gt_path):
        gt_path = os.path.join(os.path.dirname(__file__), gt_path)

    #
    print(f"Loading semantic data: {sem_path}")
    papers = load_semantic_data(sem_path)
    print(f"  Valid papers: {len(papers)}")

    #
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "robustness_output"
    )
    run_robustness_experiment(
        semantic_data=papers,
        ground_truth_path=gt_path,
        deletion_rates=args.deletion_rates,
        num_seeds=args.num_seeds,
        min_freq=args.min_freq,
        output_dir=output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
