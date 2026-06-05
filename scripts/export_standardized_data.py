#!/usr/bin/env python3
"""
LitExtract Standardized Data Export
Converts raw JSON blobs to open-source-friendly CSV + GraphML format.

Output structure:
  kg_outputs/
    nodes.csv               - KG node table
    edges.csv               - KG edge table
    kg.graphml              - GraphML network
    example_subgraphs/      - key-node subgraphs
  dag_outputs/
    ti/
      dag_edges.csv         - Ti causal edges
      node_metadata.csv     - Ti node metadata
      validation_results.csv
      representative_paths.md
      dag.graphml
    ni/
      dag_edges.csv         - Ni causal edges
      node_metadata.csv     - Ni node metadata
      validation_results.csv
      representative_paths.md
      dag.graphml
"""

import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # LitExtract/
AGENT_ROOT   = PROJECT_ROOT.parent                       # LitExtract_Agent/
DATA_DIR = PROJECT_ROOT / "data"

# Original source files (outside the data/ tree, in LitExtract_Agent/)
SOURCE_TI_DAG  = AGENT_ROOT / "KG_and_Causal" / "output" / "dag_result.json"
SOURCE_NI_DAG  = AGENT_ROOT / "KG_and_Causal" / "output" / "nickel_v6_dag_result.json"
SOURCE_NI_STAT = AGENT_ROOT / "KG_and_Causal" / "output" / "nickel_v6_dag_stats.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =======================================================================
# Part 1: kg_outputs/  - from ti_mmkg_data.json
# =======================================================================

def export_kg_outputs():
    kg_dir = DATA_DIR / "kg_outputs"
    kg_dir.mkdir(parents=True, exist_ok=True)

    mmkg = load_json(kg_dir / "ti_mmkg_data.json")
    nodes = mmkg["nodes"]
    edges = mmkg["edges"]

    # --- nodes.csv ---
    node_rows = []
    id_to_name = {}
    for n in nodes:
        id_to_name[n["id"]] = n["name"]
        node_rows.append({
            "node_id": n["id"],
            "type": n["type"],
            "name": n["name"],
            "count": n.get("count", n.get("paper_count", 0)),
            "alloy_system": "Ti-based",
        })

    with open(kg_dir / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["node_id", "type", "name", "count", "alloy_system"])
        w.writeheader()
        w.writerows(node_rows)
    print(f"  [OK] nodes.csv  - {len(node_rows)} nodes")

    # --- edges.csv ---
    edge_rows = []
    for e in edges:
        edge_rows.append({
            "source": e["source"],
            "target": e["target"],
            "edge_type": e["type"],
            "weight": e.get("weight", 1),
            "n_papers": len(e.get("papers", [])),
        })

    with open(kg_dir / "edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target", "edge_type", "weight", "n_papers"])
        w.writeheader()
        w.writerows(edge_rows)
    print(f"  [OK] edges.csv  - {len(edge_rows)} edges")

    # --- kg.graphml ---
    try:
        import networkx as nx
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n["id"], type=n["type"], name=n["name"][:80],
                       count=n.get("count", 0))
        for e in edges:
            G.add_edge(e["source"], e["target"], edge_type=e["type"],
                       weight=e.get("weight", 1))
        nx.write_graphml(G, str(kg_dir / "kg.graphml"))
        print(f"  [OK] kg.graphml  - {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except ImportError:
        print("  [WARN] networkx not installed, skipping GraphML")
    except Exception as exc:
        print(f"  [WARN] GraphML export failed: {exc}")

    # --- example_subgraphs/ ---
    sub_dir = kg_dir / "example_subgraphs"
    sub_dir.mkdir(exist_ok=True)

    degree = defaultdict(int)
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    type_top_nodes = defaultdict(list)
    for n in nodes:
        type_top_nodes[n["type"]].append((n["id"], degree.get(n["id"], 0)))
    for t in type_top_nodes:
        type_top_nodes[t].sort(key=lambda x: x[1], reverse=True)

    subgraph_index = []
    sub_idx = 0
    for ntype, top_list in type_top_nodes.items():
        for node_id, deg in top_list[:2]:
            if deg == 0:
                continue
            edges_in = [e for e in edges if e["source"] == node_id or e["target"] == node_id]
            edges_out = edges_in[:50]

            sub_nodes = set()
            for e in edges_out:
                sub_nodes.add(e["source"])
                sub_nodes.add(e["target"])

            sub_idx += 1
            safe = id_to_name[node_id][:30]
            for ch in '/\\:*?"<>|':
                safe = safe.replace(ch, '_')
            sub_name = f"subgraph_{sub_idx:02d}_{safe}"
            sub_file = sub_dir / f"{sub_name}.json"
            sub_data = {
                "center_node": {"id": node_id, "type": ntype, "name": id_to_name[node_id]},
                "degree": deg,
                "nodes": [
                    {"id": n["id"], "type": n.get("type", ""), "name": id_to_name.get(n["id"], "")}
                    for n in nodes if n["id"] in sub_nodes
                ],
                "edges": [{"source": e["source"], "target": e["target"],
                           "type": e["type"], "weight": e.get("weight", 1)}
                          for e in edges_out],
            }
            with open(sub_file, "w", encoding="utf-8") as f:
                json.dump(sub_data, f, ensure_ascii=False, indent=2)

            subgraph_index.append({
                "id": sub_idx,
                "file": f"{sub_name}.json",
                "center_node": node_id,
                "center_type": ntype,
                "center_name": id_to_name[node_id][:60],
                "degree": deg,
                "n_edges": len(edges_out),
            })

    with open(sub_dir / "INDEX.json", "w", encoding="utf-8") as f:
        json.dump(subgraph_index, f, ensure_ascii=False, indent=2)
    print(f"  [OK] example_subgraphs/  - {sub_idx} subgraphs")


# =======================================================================
# Part 2: dag_outputs/  - Ti/Ni split subdirectories
# =======================================================================

def export_dag_outputs():
    """Export Ti and Ni DAG data into separate subdirectories."""
    dag_root = DATA_DIR / "dag_outputs"
    dag_root.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "name": "ti",
            "alloy_system": "Ti-based",
            "source_path": SOURCE_TI_DAG,
            "stats_path": None,
        },
        {
            "name": "ni",
            "alloy_system": "Ni-based",
            "source_path": SOURCE_NI_DAG,
            "stats_path": SOURCE_NI_STAT,
        },
    ]

    for ds in datasets:
        out_dir = dag_root / ds["name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        if not ds["source_path"].exists():
            print(f"  [WARN] {ds['source_path']} not found, skipping {ds['name']}")
            continue

        print(f"\n  [{ds['name'].upper()}] {ds['alloy_system']} ({ds['source_path'].name})")
        dag = load_json(ds["source_path"])
        nodes = dag.get("nodes", [])
        edges = dag.get("edges", [])

        # -- node_metadata.csv --
        node_rows = []
        for n in nodes:
            name = n.get("name", "")
            if not name and ":" in n["id"]:
                name = n["id"].split(":", 1)[1]
            node_rows.append({
                "node_id": n["id"],
                "type": n["type"],
                "name": name,
            })
        _write_csv(out_dir / "node_metadata.csv",
                   ["node_id", "type", "name"], node_rows)
        print(f"    [OK] node_metadata.csv  - {len(node_rows)} nodes")

        # -- dag_edges.csv --
        edge_rows = []
        for e in edges:
            src_type, tgt_type = _resolve_types(e, nodes)
            edge_rows.append({
                "source": e["source"],
                "target": e["target"],
                "src_type": src_type,
                "tgt_type": tgt_type,
                "confidence": e.get("confidence", 0.5),
                "polarity": e.get("polarity", "unknown"),
                "strength": e.get("strength", "medium"),
                "claim_type": e.get("claim_type", ""),
            })
        _write_csv(out_dir / "dag_edges.csv",
                   ["source", "target", "src_type", "tgt_type",
                    "confidence", "polarity", "strength", "claim_type"],
                   edge_rows)
        print(f"    [OK] dag_edges.csv  - {len(edge_rows)} edges")

        # -- validation_results.csv --
        val_rows = _build_validation(ds)
        if val_rows:
            _write_csv(out_dir / "validation_results.csv",
                       ["metric", "value", "description"], val_rows)
            print(f"    [OK] validation_results.csv  - {len(val_rows)} metrics")

        # -- representative_paths.md --
        if ds["name"] == "ti":
            paths_md = _build_ti_paths(nodes, edges)
        else:
            paths_md = _build_ni_paths(nodes, edges, dag.get("top_causal_edges", []))
        (out_dir / "representative_paths.md").write_text(paths_md, encoding="utf-8")
        print(f"    [OK] representative_paths.md")

        # -- dag.graphml --
        try:
            import networkx as nx
            G = nx.DiGraph(name=f"{ds['alloy_system']} DAG")
            for n in nodes:
                G.add_node(n["id"], type=n["type"],
                           name=(n.get("name", "") or "")[:80])
            for e in edges:
                G.add_edge(e["source"], e["target"],
                           confidence=e.get("confidence", 0.5),
                           polarity=e.get("polarity", ""),
                           strength=e.get("strength", ""),
                           claim_type=e.get("claim_type", ""))
            nx.write_graphml(G, str(out_dir / "dag.graphml"))
            print(f"    [OK] dag.graphml  - {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except ImportError:
            print("    [WARN] networkx not installed, skipping GraphML")
        except Exception as exc:
            print(f"    [WARN] GraphML export failed: {exc}")


def _resolve_types(edge, nodes):
    src_type = ""
    tgt_type = ""
    if ":" in edge["source"]:
        src_type = edge["source"].split(":", 1)[0]
    if ":" in edge["target"]:
        tgt_type = edge["target"].split(":", 1)[0]
    if edge.get("src_type"):
        src_type = edge["src_type"]
    if edge.get("tgt_type"):
        tgt_type = edge["tgt_type"]
    if not src_type:
        for n in nodes:
            if n["id"] == edge["source"]:
                src_type = n["type"]
                break
    if not tgt_type:
        for n in nodes:
            if n["id"] == edge["target"]:
                tgt_type = n["type"]
                break
    return src_type, tgt_type


def _build_validation(ds):
    rows = []
    stats_path = ds["source_path"]
    dag = load_json(stats_path)
    stats = dag.get("statistics", {})

    if ds.get("stats_path") and Path(ds["stats_path"]).exists():
        ni_stats = load_json(ds["stats_path"])
        stats = {**stats, **ni_stats}

    is_dag = stats.get("is_dag")
    if is_dag is not None:
        rows.append({"metric": "is_dag", "value": str(is_dag),
                     "description": "Whether the graph is a valid DAG (no cycles)"})

    for key in ["final_nodes", "final_edges"]:
        if key in stats:
            rows.append({"metric": key, "value": str(stats[key]), "description": ""})

    entity_div = stats.get("entity_diversity", {})
    for ek, ev in entity_div.items():
        rows.append({"metric": f"entity_diversity_{ek}", "value": str(ev),
                     "description": f"Number of unique {ek} entities"})

    return rows


def _build_ti_paths(nodes, edges):
    lines = ["# Ti-based Alloys - Representative Causal Paths", ""]
    nodes_dict = {n["id"]: n for n in nodes}

    adj = defaultdict(list)
    for e in edges:
        adj[e["source"]].append(e)

    paths_found = []
    for e1 in edges:
        src_type = nodes_dict.get(e1["source"], {}).get("type", "")
        if src_type != "Element":
            continue
        mid_nid = e1["target"]
        for e2 in adj.get(mid_nid, []):
            tgt_type = nodes_dict.get(e2["target"], {}).get("type", "")
            if tgt_type == "Property" and e1["confidence"] >= 0.7 and e2["confidence"] >= 0.7:
                paths_found.append((e1, e2))

    paths_found.sort(key=lambda p: p[0]["confidence"] + p[1]["confidence"], reverse=True)

    lines.append("## Element -> Microstructure -> Property (Top 10)")
    lines.append("")
    if paths_found:
        for i, (e1, e2) in enumerate(paths_found[:10]):
            src = e1["source"].split(":", 1)[-1]
            mid = e1["target"].split(":", 1)[-1]
            tgt = e2["target"].split(":", 1)[-1]
            lines.append(
                f"{i+1}. **{src}** -> **{mid}** -> **{tgt}**  "
                f"(conf: {e1['confidence']:.2f} -> {e2['confidence']:.2f}, "
                f"{e1.get('polarity','?')} / {e2.get('polarity','?')})"
            )
    else:
        lines.append("_No Element->Microstructure->Property chains with conf >= 0.7 found._")
    lines.append("")

    return "\n".join(lines)


def _build_ni_paths(nodes, edges, top_edges):
    lines = ["# Ni-based Alloys - Representative Causal Paths", ""]

    lines.append("## Top-20 Causal Edges (by confidence)")
    lines.append("")
    lines.append("| Rank | Source | Target | Type Chain | Confidence | Polarity |")
    lines.append("|------|--------|--------|------------|------------|----------|")

    for te in top_edges[:20]:
        rank = te.get("rank", "-")
        src = te.get("source", "-")
        tgt = te.get("target", "-")
        chain = f"{te.get('src_type','')} -> {te.get('tgt_type','')}"
        conf = te.get("confidence", 0)
        pol = te.get("polarity", "-")
        lines.append(f"| {rank} | {src} | {tgt} | {chain} | {conf:.2f} | {pol} |")

    lines.append("")
    return "\n".join(lines)


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# =======================================================================
# Main
# =======================================================================

if __name__ == "__main__":
    print("LitExtract Standardized Data Export")
    print("=" * 50)

    print("\n[1/2] kg_outputs/ ...")
    export_kg_outputs()

    print("\n[2/2] dag_outputs/ (Ti / Ni split) ...")
    export_dag_outputs()

    print("\nDone")
    print(f"  Output: {DATA_DIR}")
