"""
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import networkx as nx


# ============================================================
# DAG load
# ============================================================

def load_dag(dag_path: str) -> nx.DiGraph:
    """Module functionality."""
    if dag_path.endswith(".graphml"):
        return _load_graphml(dag_path)
    elif dag_path.endswith(".json"):
        return _load_json(dag_path)
    else:
        raise ValueError(f"Unsupported format: {dag_path}, need .graphml or .json")


def _load_graphml(path: str) -> nx.DiGraph:
    """Module functionality."""
    g = nx.read_graphml(path)
    assert isinstance(g, (nx.DiGraph, nx.MultiDiGraph)), "GraphML must be a directed graph"

    #
    if isinstance(g, nx.MultiDiGraph):
        simple = nx.DiGraph()
        for u, v, data in g.edges(data=True):
            if simple.has_edge(u, v):
                #
                existing_conf = float(simple[u][v].get("confidence", 0))
                new_conf = float(data.get("confidence", 0))
                if new_conf > existing_conf:
                    simple[u][v].update(data)
            else:
                simple.add_edge(u, v, **data)
        g = simple

    return g


def _load_json(path: str) -> nx.DiGraph:
    """Module functionality."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    g = nx.DiGraph()

    #
    for node in data.get("nodes", []):
        nid = node["id"]
        ntype = node.get("type", "unknown")
        attrs = node.get("attributes", {})
        #
        attrs.pop("node_type", None)
        g.add_node(nid, node_type=ntype, **attrs)

    #
    for edge in data.get("edges", []):
        src = edge["source"]
        dst = edge["target"]
        if g.has_edge(src, dst):
            #
            existing_conf = float(g[src][dst].get("confidence", 0))
            new_conf = float(edge.get("confidence", 0))
            if new_conf > existing_conf:
                g[src][dst].update(edge)
        else:
            g.add_edge(src, dst, **edge)

    return g


# ============================================================
# Ground Truth load
# ============================================================

def load_ground_truth(gt_path: str) -> List[Dict]:
    """Module functionality."""
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("mechanisms", [])


def flatten_triads(mechanisms: List[Dict]) -> List[Dict]:
    """Module functionality."""
    triads = []
    for mech in mechanisms:
        for triad in mech.get("triads", []):
            triads.append({
                "mechanism_id": mech["id"],
                "mechanism_name": mech["mechanism"],
                "category": mech.get("category", ""),
                "cause_type": triad["cause_type"],
                "cause": triad["cause"],
                "effect_type": triad["effect_type"],
                "effect": triad["effect"],
                "relation": triad["relation"],
                "detail": triad.get("mechanism_detail", ""),
            })
    return triads


# ============================================================
#
# ============================================================

#
_SYNONYM_MAP = {
    "alpha": ["alpha", "α", "α-phase", "α phase"],
    "beta": ["beta", "β", "β-phase", "β phase"],
    "gamma": ["gamma", "γ", "γ-phase", "γ phase"],
    "omega": ["omega", "ω", "ω-phase", "ω phase"],
    "martensite": ["martensite", "α'", "α' martensite", "α'-martensite", "α′ martensite",
                    "α\"", "α\" martensite", "α''", "α'' phase", "α″ martensite"],
    "grain": ["grain", "grains"],
    "lath": ["lath", "laths"],
    "lamellar": ["lamellar", "lamella", "lamellae", "lamellar α"],
    "equiaxed": ["equiaxed", "equiaxed α", "equiaxed alpha"],
    "acicular": ["acicular", "acicular α", "acicular alpha", "needle-like", "needle-shaped", "fine acicular"],
    "precipitation": ["precipitation", "precipitates", "precipitate", "α precipitate"],
    "grain refinement": ["grain refinement", "grain refining", "refined grains",
                          "fine grain", "fine grains", "ultrafine grains"],
    "recrystallization": ["recrystallization", "recrystallized", "recrystallised",
                           "DRX", "dynamic recrystallization"],
    "elongation": ["elongation", "ductility", "elongation/ductility"],
    "yield strength": ["yield strength", "YS", "yield stress"],
    "ultimate tensile strength": ["ultimate tensile strength", "UTS", "tensile strength"],
    "fracture toughness": ["fracture toughness", "KIC", "K1C", "fracture"],
    "fatigue strength": ["fatigue strength", "fatigue", "fatigue life"],
    "creep resistance": ["creep resistance", "creep", "creep rupture"],
}


def _normalize_node_name(name: str) -> str:
    """Module functionality."""
    if ":" in name:
        name = name.split(":", 1)[1]
    return name.strip().lower()


def _tokenize(name: str) -> set:
    """Module functionality."""
    norm = _normalize_node_name(name)
    tokens = set(norm.split())
    #
    tokens.update(norm.replace("-", " ").replace("/", " ").split())
    return tokens


def _get_synonym_set(name: str) -> set:
    """Module functionality."""
    norm = _normalize_node_name(name)
    variants = {norm}
    #
    for key, syns in _SYNONYM_MAP.items():
        if norm in syns or any(s in norm for s in syns if len(s) > 3):
            variants.update(syns)
    return variants


def resolve_node_id(node_name: str, node_type: str, g: nx.DiGraph) -> Optional[str]:
    """
    """
    #
    candidate = f"{node_type}:{node_name}"
    if candidate in g.nodes:
        return candidate

    #
    if node_name in g.nodes:
        return node_name

    #
    node_lower = node_name.lower()
    cand_lower = candidate.lower()
    for nid in g.nodes:
        nid_lower = nid.lower()
        if nid_lower == node_lower or nid_lower == cand_lower:
            return nid

    #
    #
    gt_tokens = _tokenize(node_name)
    gt_synonyms = _get_synonym_set(node_name)

    best_match = None
    best_score = 0

    for nid, ndata in g.nodes(data=True):
        ntype = ndata.get("node_type", "")
        if ntype != node_type:
            continue

        dag_norm = _normalize_node_name(nid)
        dag_tokens = _tokenize(nid)
        dag_synonyms = _get_synonym_set(nid)

        #
        if gt_tokens and gt_tokens.issubset(dag_tokens):
            score = len(gt_tokens) / len(dag_tokens)  
            if score > best_score:
                best_score = score
                best_match = nid

        #
        if dag_tokens and dag_tokens.issubset(gt_tokens):
            score = len(dag_tokens) / len(gt_tokens)
            if score > best_score:
                best_score = score
                best_match = nid

        #
        gt_syn_norm = {_normalize_node_name(s) for s in gt_synonyms}
        dag_syn_norm = {_normalize_node_name(s) for s in dag_synonyms}
        overlap = gt_syn_norm & dag_syn_norm
        if overlap:
            score = 0.3 + 0.2 * len(overlap)  
            if score > best_score:
                best_score = score
                best_match = nid

        #
        if node_lower in dag_norm or dag_norm in node_lower:
            score = 0.25
            if score > best_score:
                best_score = score
                best_match = nid

    if best_match and best_score >= 0.25:
        return best_match

    #
    for nid, ndata in g.nodes(data=True):
        ntype = ndata.get("node_type", "")
        if ntype == node_type and node_lower in _normalize_node_name(nid):
            return nid

    return None


def find_path_between(src_id: str, dst_id: str, g: nx.DiGraph, max_hops: int = 3) -> Optional[List[str]]:
    """Module functionality."""
    try:
        for path in nx.all_simple_paths(g, src_id, dst_id, cutoff=max_hops):
            return path  
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    return None


# ============================================================
# validateevaluate
# ============================================================

def evaluate_mechanism_recovery(
    g: nx.DiGraph,
    triads: List[Dict],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    """
    results = []
    by_mechanism = defaultdict(lambda: {"total": 0, "recovered": 0, "path_recovered": 0, "triads": []})
    
    for i, triad in enumerate(triads):
        src_id = resolve_node_id(triad["cause"], triad["cause_type"], g)
        dst_id = resolve_node_id(triad["effect"], triad["effect_type"], g)

        result = {
            "index": i,
            "mechanism_id": triad["mechanism_id"],
            "mechanism_name": triad["mechanism_name"],
            "category": triad["category"],
            "cause": f"{triad['cause_type']}:{triad['cause']}",
            "effect": f"{triad['effect_type']}:{triad['effect']}",
            "relation": triad["relation"],
            "detail": triad["detail"],
        }

        if src_id is None or dst_id is None:
            result["status"] = "node_missing"
            result["score"] = 0.0
            result["path"] = None
            if verbose:
                missing = []
                if src_id is None:
                    missing.append(triad["cause"])
                if dst_id is None:
                    missing.append(triad["effect"])
                print(f"  [{triad['mechanism_id']}] ✗ {triad['cause']} → {triad['effect']}: node_missing ({', '.join(missing)})")
        else:
            #
            direct_edge = g.has_edge(src_id, dst_id)
            
            #
            path = None
            path_hops = 0
            if not direct_edge:
                path = find_path_between(src_id, dst_id, g, max_hops=2)
                if path:
                    path_hops = len(path) - 1  

            if direct_edge:
                edge_data = g[src_id][dst_id]
                result["status"] = "direct_edge"
                result["score"] = 1.0
                result["path"] = [src_id, dst_id]
                if verbose:
                    print(f"  [{triad['mechanism_id']}] ✓ {triad['cause']} → {triad['effect']}: direct_edge (conf={edge_data.get('confidence', '?')}, type={edge_data.get('claim_type', '?')})")
            elif path is not None:
                result["status"] = "indirect_path"
                result["score"] = 0.5
                result["path"] = path
                if verbose:
                    print(f"  [{triad['mechanism_id']}] ~ {triad['cause']} → {triad['effect']}: indirect_path ({path_hops} hops, {path})")
            else:
                result["status"] = "not_found"
                result["score"] = 0.0
                result["path"] = None
                if verbose:
                    print(f"  [{triad['mechanism_id']}] ✗ {triad['cause']} → {triad['effect']}: Not found")

        results.append(result)

        #
        mech_id = triad["mechanism_id"]
        by_mechanism[mech_id]["total"] += 1
        by_mechanism[mech_id]["triads"].append(result)
        if result["status"] in ("direct_edge", "indirect_path"):
            by_mechanism[mech_id]["recovered"] += 1
        if result["status"] == "direct_edge":
            by_mechanism[mech_id]["path_recovered"] += 1

    #
    total_triads = len(results)
    direct_recovered = sum(1 for r in results if r["status"] == "direct_edge")
    path_recovered = sum(1 for r in results if r["status"] == "indirect_path")
    not_recovered = sum(1 for r in results if r["status"] in ("not_found", "node_missing"))
    node_missing = sum(1 for r in results if r["status"] == "node_missing")

    #
    weighted_sum = sum(r["score"] for r in results)
    max_possible = total_triads  

    #
    mechanism_summary = []
    for mech_id in sorted(by_mechanism.keys()):
        info = by_mechanism[mech_id]
        mechanism_summary.append({
            "mechanism_id": mech_id,
            "mechanism_name": info["triads"][0]["mechanism_name"] if info["triads"] else "?",
            "category": info["triads"][0]["category"] if info["triads"] else "?",
            "total": info["total"],
            "direct": info["path_recovered"],
            "recovered_any": info["recovered"],
            "recovery_rate": info["recovered"] / info["total"] if info["total"] > 0 else 0,
            "direct_rate": info["path_recovered"] / info["total"] if info["total"] > 0 else 0,
        })

    return {
        "summary": {
            "total_mechanisms": len(by_mechanism),
            "total_triads": total_triads,
            "direct_edges": direct_recovered,
            "indirect_paths": path_recovered,
            "not_recovered": not_recovered,
            "node_missing": node_missing,
            "direct_recovery_rate": direct_recovered / total_triads if total_triads > 0 else 0,
            "any_recovery_rate": (direct_recovered + path_recovered) / total_triads if total_triads > 0 else 0,
            "weighted_recovery_rate": weighted_sum / max_possible if max_possible > 0 else 0,
        },
        "by_mechanism": mechanism_summary,
        "details": results,
    }


# ============================================================
#
# ============================================================

def print_report(report: Dict[str, Any], dag_path: str, dag_stats: Dict[str, Any]):
    """Module functionality."""
    s = report["summary"]

    print("\n" + "=" * 70)
    print(" Experiment 1: DAG Known Mechanism Recovery Validation Report")
    print("=" * 70)
    print(f"\nDAG file: {dag_path}")
    print(f"DAG size: {dag_stats['nodes']} nodes, {dag_stats['edges']} edges")
    print(f"Is DAG:  {'YES' if dag_stats.get('is_dag', False) else 'NO'}")

    print(f"\n{'─' * 70}")
    print(" Overall Evaluation")
    print(f"{'─' * 70}")
    print(f"  Ground Truth mechanisms:     {s['total_mechanisms']}")
    print(f"  Total triads:              {s['total_triads']}")
    print(f"  Direct edge hits:              {s['direct_edges']}  ({s['direct_recovery_rate']:.1%})")
    print(f"  Indirect path hits (<=2 hops):   {s['indirect_paths']}  ({s['indirect_paths']/s['total_triads']:.1%})")
    print(f"  Not found:                  {s['not_recovered']}  ({s['not_recovered']/s['total_triads']:.1%})")
    print(f"  Of which node missing:            {s['node_missing']}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Any recovery rate:          {s['any_recovery_rate']:.1%}")
    print(f"  Weighted recovery rate:              {s['weighted_recovery_rate']:.1%}  (direct=1.0, indirect=0.5)")

    print(f"\n{'─' * 70}")
    print(" Per-Mechanism Recovery Details")
    print(f"{'─' * 70}")
    print(f"  {'Mechanism':<35s} {'Triads':>6s} {'Direct':>5s} {'Any':>5s} {'Recov%':>7s}")
    print(f"  {'─' * 60}")
    for m in report["by_mechanism"]:
        bar = "█" * int(m["recovery_rate"] * 20)
        print(f"  {m['mechanism_name']:<35s} {m['total']:>6d} {m['direct']:>5d} {m['recovered_any']:>5d} {m['recovery_rate']:>6.0%} {bar}")

    print(f"\n{'─' * 70}")
    print(" Conclusion")
    print(f"{'─' * 70}")
    if s["weighted_recovery_rate"] >= 0.8:
        conclusion = "Excellent - DAG sufficiently covers most known titanium alloy mechanisms"
    elif s["weighted_recovery_rate"] >= 0.6:
        conclusion = "Good - DAG covers major known mechanisms, with room for improvement"
    elif s["weighted_recovery_rate"] >= 0.4:
        conclusion = "Fair - DAG partially recovers known mechanisms, with notable omissions"
    else:
        conclusion = "Insufficient - DAG recovery of known mechanisms is weak; consider adjusting build parameters"
    print(f"  {conclusion}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Validate whether DAG recovers known titanium alloy mechanisms")
    parser.add_argument("--dag", required=True, help="DAG file path (.graphml or .json)")
    parser.add_argument("--ground_truth", default="ground_truth_mechanisms.json",
                        help="Ground truth mechanisms file path")
    parser.add_argument("--output", default=None,
                        help="Evaluation report output path (.json)")
    parser.add_argument("--quiet", action="store_true",
                        help="Quiet mode, skip detailed match output")
    args = parser.parse_args()

    # # Loading DAG
    print(f"Loading DAG: {args.dag}")
    g = load_dag(args.dag)
    is_dag = nx.is_directed_acyclic_graph(g)
    dag_stats = {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "is_dag": is_dag,
    }
    print(f"  Size: {dag_stats['nodes']} nodes, {dag_stats['edges']} edges, DAG={is_dag}")

    # # Loading Ground Truth
    gt_path = args.ground_truth
    if not os.path.isabs(gt_path):
        gt_path = os.path.join(os.path.dirname(__file__), gt_path)
    print(f"Loading Ground Truth: {gt_path}")
    mechanisms = load_ground_truth(gt_path)
    triads = flatten_triads(mechanisms)
    print(f"  Mechanisms: {len(mechanisms)}, Total triads: {len(triads)}")

    # # evaluate
    print("\nEvaluating...\n")
    report = evaluate_mechanism_recovery(g, triads, verbose=not args.quiet)

    #
    print_report(report, args.dag, dag_stats)

    #
    if args.output:
        output_data = {
            "dag_path": args.dag,
            "dag_stats": dag_stats,
            "evaluation": report,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Detailed report saved to: {args.output}")

    return report


if __name__ == "__main__":
    main()
