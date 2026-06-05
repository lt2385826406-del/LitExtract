# kg_construction/export_neo4j.py
"""
KG & Causal Knowledge Graph visualization module (Neo4j export version).
Outputs paper-ready:
  1. Material KG — node coloring, layered layout, with legend
  2. Causal Hypothesis Graph — directed, polarity/strength annotation,
     edge weight visualization
  3. JSON structured export — full node/edge info for external tools

Dependency: pyvis
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# ── Color schemes (paper-standard palettes) ─────────────────────────────────

NODE_COLORS = {
    "Paper":          "#6C8EBF",   # Steel Blue
    "Sample":         "#D6B656",   # Amber Yellow
    "Alloy":          "#82B366",   # Grass Green
    "Composition":    "#A9C4A0",   # Light Green
    "Element":        "#B8D4B8",   # Pale Green
    "Processing":     "#D79B00",   # Orange
    "Microstructure": "#6D8764",   # Dark Green
    "Property":       "#AE4132",   # Brick Red
    "Figure":         "#9673A6",   # Purple
    "default":        "#B0B0B0",   # Gray
}

EDGE_COLORS = {
    "HAS_ELEMENT":       "#82B366",
    "HAS_COMPOSITION":   "#A9C4A0",
    "PROCESSED_BY":      "#D79B00",
    "HAS_MICROSTRUCTURE":"#6D8764",
    "HAS_PROPERTY":      "#AE4132",
    "FORMS_MICROSTRUCTURE": "#FF8C00",
    "AFFECTS_PROPERTY":  "#CC3300",
    "SHOWN_IN":          "#9673A6",
    "HAS_FIGURE":        "#9673A6",
    "HAS_ALLOY":         "#82B366",
    "REPORTED_IN":       "#6C8EBF",
    "default":           "#888888",
}

# Causal graph colors
CHG_POLARITY_COLORS = {
    "increase":  "#FF6633",   # Orange-Red (promote)
    "promote":   "#FF6633",
    "decrease":  "#3366FF",   # Blue (suppress/decrease)
    "suppress":  "#3366FF",
    "default":   "#888888",
}

CHG_NODE_LAYER_COLORS = {
    "composition":    "#82B366",
    "process":        "#D79B00",
    "microstructure": "#6D8764",
    "property":       "#AE4132",
    "default":        "#B0B0B0",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _node_layer(node_id: str) -> str:
    """Infer node layer from controlled-path format."""
    nid = node_id.lower()
    if nid.startswith("composition"):
        return "composition"
    elif nid.startswith("process"):
        return "process"
    elif nid.startswith("microstructure") or nid.startswith("ms:"):
        return "microstructure"
    elif nid.startswith("property") or nid.startswith("prop:"):
        return "property"
    elif nid.startswith("mech:"):
        return "mechanism"
    elif nid.startswith("var:"):
        return "varied_factor"
    elif nid.startswith("ctrl:") or nid.startswith("trt:"):
        return "contrast"
    return "default"


def _truncate(s: str, max_len: int = 40) -> str:
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


# ── Knowledge Graph Visualization (KG) ──────────────────────────────────────

def build_kg_html(kg_data: Dict[str, Any], output_path: str) -> str:
    """
    Render KG data as an interactive HTML file (via pyvis).

    Nodes colored by type; edges labeled with relationship type;
    supports drag/zoom/hover tooltips.

    Args:
        kg_data:     Return value of KGBuilder.export_bundle("json")
        output_path: Output HTML file path

    Returns:
        Absolute path to the HTML file
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("Please install pyvis: pip install pyvis")

    nodes_data = kg_data.get("nodes", [])
    edges_data = kg_data.get("edges", [])

    # Build node_id -> node map
    id_to_node = {n["node_id"]: n for n in nodes_data}

    net = Network(
        height="700px",
        width="100%",
        directed=True,
        bgcolor="#FAFAFA",
        font_color="#333333",
        notebook=False,
    )
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150
        },
        "stabilization": {"iterations": 200}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      },
      "edges": {
        "smooth": {"type": "dynamic"},
        "font": {"size": 11, "align": "middle"}
      },
      "nodes": {
        "font": {"size": 13}
      }
    }
    """)

    # Add nodes
    for node in nodes_data:
        ntype = node.get("node_type", "default")
        color = NODE_COLORS.get(ntype, NODE_COLORS["default"])
        label = _truncate(node.get("name", node["node_id"]))
        props = node.get("properties", {})

        # Build tooltip (detailed on hover)
        tooltip_lines = [f"<b>{ntype}</b>: {node.get('name', '')}"]
        for k, v in props.items():
            if v is not None and v != "" and v != [] and k not in ("created_at", "updated_at"):
                tooltip_lines.append(f"{k}: {_truncate(str(v), 60)}")
        tooltip = "<br>".join(tooltip_lines)

        # Node size by type
        size_map = {
            "Paper": 30, "Sample": 25, "Alloy": 22,
            "Processing": 20, "Microstructure": 20,
            "Property": 20, "Composition": 18,
            "Element": 14, "Figure": 16,
        }
        size = size_map.get(ntype, 16)

        net.add_node(
            node["node_id"],
            label=label,
            title=tooltip,
            color=color,
            size=size,
            shape="dot",
            group=ntype,
        )

    # Add edges
    for edge in edges_data:
        src = edge.get("source_node_id")
        tgt = edge.get("target_node_id")
        if src not in id_to_node or tgt not in id_to_node:
            continue

        rel = edge.get("relationship_type", "")
        conf = edge.get("confidence", 0.7)
        color = EDGE_COLORS.get(rel, EDGE_COLORS["default"])
        value = edge.get("value")
        unit = edge.get("unit", "")

        label_parts = [rel]
        if value is not None:
            label_parts.append(f"{value}{' ' + unit if unit else ''}")
        edge_label = " | ".join(label_parts)

        tooltip = f"<b>{rel}</b><br>Confidence: {conf:.2f}"
        if value is not None:
            tooltip += f"<br>Value: {value} {unit}"

        net.add_edge(
            src, tgt,
            label=edge_label,
            title=tooltip,
            color=color,
            width=max(1.0, conf * 3.0),
            arrows="to",
        )

    # Write HTML
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    net.save_graph(output_path)

    # Inject legend into HTML head
    _inject_kg_legend(output_path)

    logger.info(f"[KGVisualizer] KG HTML saved: {output_path}")
    return os.path.abspath(output_path)


def _inject_kg_legend(html_path: str):
    """Inject node-type legend panel (top-right floating) into KG HTML."""
    legend_html = """
<div id="kg-legend" style="
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(255,255,255,0.92); border:1px solid #ccc;
    border-radius:8px; padding:12px 16px; font-family:Arial,sans-serif;
    font-size:13px; box-shadow:2px 2px 8px rgba(0,0,0,0.15);
    max-width:200px;">
  <b style="font-size:14px;">Node Type Legend</b>
  <hr style="margin:6px 0;">
"""
    for ntype, color in NODE_COLORS.items():
        if ntype == "default":
            continue
        legend_html += (
            f'  <div style="display:flex;align-items:center;margin:4px 0;">'
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'border-radius:50%;background:{color};margin-right:8px;"></span>'
            f'{ntype}</div>\n'
        )
    legend_html += "</div>\n"

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    insert_pos = content.find("</body>")
    if insert_pos != -1:
        content = content[:insert_pos] + legend_html + content[insert_pos:]
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)


# ── Causal Knowledge Graph Visualization (CHG) ──────────────────────────────

def build_chg_html(chg_data: Dict[str, Any], output_path: str) -> str:
    """
    Render Causal Hypothesis Graph data as an interactive HTML file.

    Nodes are colored by layer (composition / process / microstructure / property);
    edges colored by polarity (increase / decrease), edge width corresponds to confidence,
    edge labels contain polarity + claim type + confidence.

    Args:
        chg_data:    Return value of CausalHypothesisGraph.export_json()
        output_path: Output HTML file path

    Returns:
        Absolute path to the HTML file
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("Please install pyvis: pip install pyvis")

    nodes_data = chg_data.get("nodes", [])
    edges_data = chg_data.get("edges", [])
    stats = chg_data.get("statistics", {})

    net = Network(
        height="700px",
        width="100%",
        directed=True,
        bgcolor="#F8F8F8",
        font_color="#222222",
        notebook=False,
    )
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": false
        }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 300}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      },
      "edges": {
        "smooth": {"type": "curvedCW", "roundness": 0.2},
        "font": {"size": 11, "align": "middle"},
        "arrows": {"to": {"enabled": true, "scaleFactor": 1.2}}
      },
      "nodes": {
        "font": {"size": 13, "bold": true}
      }
    }
    """)

    # Add nodes
    for node in nodes_data:
        nid = node.get("id", "")
        layer = _node_layer(nid)
        color = CHG_NODE_LAYER_COLORS.get(layer, CHG_NODE_LAYER_COLORS["default"])
        attrs = node.get("attributes", {})

        label = _truncate(nid, 35)
        tooltip = f"<b>{nid}</b><br>Layer: {layer}"
        for k, v in attrs.items():
            if v and k != "node_type":
                tooltip += f"<br>{k}: {_truncate(str(v), 50)}"

        # Shape by layer (paper-standard conventions)
        shape_map = {
            "composition": "ellipse",
            "process":     "box",
            "microstructure": "diamond",
            "property":    "star",
            "mechanism":   "triangle",
            "default":     "dot",
        }
        shape = shape_map.get(layer, "dot")

        net.add_node(
            nid,
            label=label,
            title=tooltip,
            color=color,
            shape=shape,
            size=20,
            group=layer,
        )

    # Add edges
    for edge in edges_data:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if not src or not tgt:
            continue

        polarity = edge.get("polarity", "default")
        claim_type = edge.get("claim_type", "")
        confidence = edge.get("confidence", 0.5)
        strength = edge.get("strength", "medium")
        evidence_text = edge.get("evidence_text", "")
        evidence_ids = edge.get("evidence_ids", [])

        color = CHG_POLARITY_COLORS.get(polarity, CHG_POLARITY_COLORS["default"])

        # Edge label: polarity + abbreviated claim type
        claim_abbr = {
            "explicit_causal":    "explicit",
            "contrast_based":     "contrast",
            "mechanism_supported":"mechanism",
            "cooccurrence":       "cooccur",
        }.get(claim_type, claim_type)
        edge_label = f"{polarity}\n({claim_abbr})"

        # Edge width: higher confidence -> wider
        width = 1.5 + confidence * 4.0

        # Dashed for weak evidence
        dashes = (strength == "weak")

        tooltip = (
            f"<b>{polarity.upper()}</b><br>"
            f"Type: {claim_type}<br>"
            f"Strength: {strength}<br>"
            f"Confidence: {confidence:.2f}<br>"
            f"Source samples: {', '.join(evidence_ids[:5])}<br>"
        )
        if evidence_text:
            tooltip += f"<br><i>Evidence: {_truncate(evidence_text, 120)}</i>"

        net.add_edge(
            src, tgt,
            label=edge_label,
            title=tooltip,
            color=color,
            width=width,
            dashes=dashes,
            arrows="to",
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    net.save_graph(output_path)

    # Inject legend + stats panel
    _inject_chg_legend_and_stats(output_path, stats)

    logger.info(f"[KGVisualizer] CHG HTML saved: {output_path}")
    return os.path.abspath(output_path)


def _inject_chg_legend_and_stats(html_path: str, stats: Dict[str, Any]):
    """Inject legend panel (layer colors + polarity colors) and stats into CHG HTML."""
    # Stats
    total_nodes = stats.get("total_nodes", "N/A")
    total_edges = stats.get("total_edges", "N/A")
    pol_dist = stats.get("polarity_distribution", {})
    ct_dist = stats.get("claim_type_distribution", {})
    st_dist = stats.get("strength_distribution", {})

    pol_html = "".join(
        f'<span style="color:{CHG_POLARITY_COLORS.get(k, "#888")};font-weight:bold;">'
        f'{k}:{v}</span> '
        for k, v in pol_dist.items()
    )
    ct_html = " | ".join(f"{k}: {v}" for k, v in ct_dist.items())
    st_html = " | ".join(f"{k}: {v}" for k, v in st_dist.items())

    legend_html = f"""
<div id="chg-legend" style="
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(255,255,255,0.94); border:1px solid #ccc;
    border-radius:8px; padding:12px 16px; font-family:Arial,sans-serif;
    font-size:12px; box-shadow:2px 2px 8px rgba(0,0,0,0.15);
    max-width:230px;">
  <b style="font-size:13px;">Causal Knowledge Graph Legend</b>
  <hr style="margin:6px 0;">
  <b>Node Layers:</b><br>
"""
    for layer, color in CHG_NODE_LAYER_COLORS.items():
        if layer == "default":
            continue
        shape_symbol = {"composition": "⬭", "process": "▬", "microstructure": "◆",
                        "property": "★", "mechanism": "▲"}.get(layer, "●")
        legend_html += (
            f'  <div style="display:flex;align-items:center;margin:3px 0;">'
            f'<span style="color:{color};margin-right:6px;font-size:16px;">{shape_symbol}</span>'
            f'{layer}</div>\n'
        )

    legend_html += f"""
  <hr style="margin:6px 0;">
  <b>Edge Polarity:</b><br>
  <div><span style="color:#FF6633;font-weight:bold;">→ increase / promote</span></div>
  <div><span style="color:#3366FF;font-weight:bold;">→ decrease / suppress</span></div>
  <div><span style="color:#888;">--- weak evidence (dashed)</span></div>
  <hr style="margin:6px 0;">
  <b>Statistics:</b><br>
  Nodes: {total_nodes} | Edges: {total_edges}<br>
  Polarity: {pol_html}<br>
  Claim Type: {ct_html}<br>
  Strength: {st_html}
</div>
"""

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    insert_pos = content.find("</body>")
    if insert_pos != -1:
        content = content[:insert_pos] + legend_html + content[insert_pos:]
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)


# ── JSON Structured Export (paper appendix / external tools) ─────────────────

def export_kg_json(kg_data: Dict[str, Any], output_path: str) -> str:
    """
    Export the complete KG JSON file (nodes + edges + evidences).
    Format is suitable for paper appendix, with statistical summary.
    """
    nodes = kg_data.get("nodes", [])
    edges = kg_data.get("edges", [])
    evidences = kg_data.get("evidences", [])

    # Summary stats
    node_type_counts = defaultdict(int)
    for n in nodes:
        node_type_counts[n.get("node_type", "unknown")] += 1

    edge_type_counts = defaultdict(int)
    for e in edges:
        edge_type_counts[e.get("relationship_type", "unknown")] += 1

    export_data = {
        "metadata": {
            "description": "Material Knowledge Graph - LitExtract Agent",
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_evidences": len(evidences),
            "node_type_distribution": dict(node_type_counts),
            "edge_type_distribution": dict(edge_type_counts),
        },
        "nodes": nodes,
        "edges": edges,
        "evidences": evidences,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    logger.info(f"[KGVisualizer] KG JSON saved: {output_path}")
    return os.path.abspath(output_path)


def export_chg_json(chg_data: Dict[str, Any], output_path: str) -> str:
    """
    Export the complete Causal Hypothesis Graph JSON file.
    Includes node layer annotations and polarity distribution statistics,
    formatted for paper citation.
    """
    nodes = chg_data.get("nodes", [])
    edges = chg_data.get("edges", [])
    stats = chg_data.get("statistics", {})

    # Annotate nodes with layer
    for node in nodes:
        node["layer"] = _node_layer(node.get("id", ""))

    # Layer summary
    layer_summary = defaultdict(list)
    for node in nodes:
        layer_summary[node["layer"]].append(node["id"])

    export_data = {
        "metadata": {
            "description": "Causal Hypothesis Graph (CHG) - LitExtract Agent",
            "statistics": stats,
            "layer_summary": {k: {"count": len(v), "nodes": v}
                              for k, v in layer_summary.items()},
        },
        "nodes": nodes,
        "edges": edges,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    logger.info(f"[KGVisualizer] CHG JSON saved: {output_path}")
    return os.path.abspath(output_path)


# ── Unified entry: generate KG + CHG HTML and JSON in one pass ──────────────

def generate_all_outputs(
    kg_data: Dict[str, Any],
    chg_data: Dict[str, Any],
    output_dir: str = "outputs",
    prefix: str = "litextract",
) -> Dict[str, str]:
    """
    Generate all visualization files in one pass:
      - {prefix}_kg.html      KG visualization
      - {prefix}_kg.json      KG complete JSON
      - {prefix}_chg.html     Causal Knowledge Graph visualization
      - {prefix}_chg.json     Causal Knowledge Graph complete JSON

    Returns:
        {
            "kg_html":  absolute path,
            "kg_json":  absolute path,
            "chg_html": absolute path,
            "chg_json": absolute path,
        }
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    kg_html_path = os.path.join(output_dir, f"{prefix}_kg.html")
    kg_json_path = os.path.join(output_dir, f"{prefix}_kg.json")
    chg_html_path = os.path.join(output_dir, f"{prefix}_chg.html")
    chg_json_path = os.path.join(output_dir, f"{prefix}_chg.json")

    try:
        results["kg_html"] = build_kg_html(kg_data, kg_html_path)
    except Exception as e:
        logger.error(f"[KGVisualizer] Failed to generate KG HTML: {e}")
        results["kg_html"] = None

    try:
        results["kg_json"] = export_kg_json(kg_data, kg_json_path)
    except Exception as e:
        logger.error(f"[KGVisualizer] Failed to export KG JSON: {e}")
        results["kg_json"] = None

    try:
        results["chg_html"] = build_chg_html(chg_data, chg_html_path)
    except Exception as e:
        logger.error(f"[KGVisualizer] Failed to generate CHG HTML: {e}")
        results["chg_html"] = None

    try:
        results["chg_json"] = export_chg_json(chg_data, chg_json_path)
    except Exception as e:
        logger.error(f"[KGVisualizer] Failed to export CHG JSON: {e}")
        results["chg_json"] = None

    return results
