"""
Ti-alloy Knowledge Graph Builder
=================================
"""

import os, json, re, glob
from pathlib import Path
from collections import defaultdict, Counter
import traceback

#
try:
    from pyvis.network import Network
    PYVIS_OK = True
except ImportError:
    PYVIS_OK = False
    print("[WARN] pyvis not installed, skipping interactive HTML generation")

try:
    import networkx as nx
    NX_OK = True
except ImportError:
    NX_OK = False
    print("[WARN] networkx not installed, some statistics functions unavailable")

# ─── pathconfig ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
SEMANTIC_DIR  = BASE_DIR / "semantic_outputs"
OUTPUT_DIR    = BASE_DIR / "Ti_graphs"
OUTPUT_DIR.mkdir(exist_ok=True)

#
#
ALLOY_ELEMENTS = [
    "Al", "V", "Mo", "Nb", "Fe", "Cr", "Zr", "Sn", "Si", "Cu",
    "Ta", "W", "Mn", "Co", "Ni", "Hf", "Pd", "Ru", "B", "C", "N", "O",
    "Re", "Y", "Ce", "La", "Sc", "Ge"
]
ELEMENT_SET = set(ALLOY_ELEMENTS)

#
PROCESS_KEYWORDS = {
    #
    r"solution\s*treat": "Solution Treatment",
    r"SHT|solution\s+heat\s+treat": "Solution Heat Treatment",
    r"anneal": "Annealing",
    r"aging|ageing|time?ing": "Aging",
    r"quench": "Quenching",
    r"furnace\s*cool": "Furnace Cooling",
    r"air\s*cool": "Air Cooling",
    r"water\s*quench": "Water Quenching",
    r"HIP|hot\s*isostatic\s*press": "HIP",
    r"双步.*aging|two.?step.*age": "Two-step Aging",
    r"aging|aging": "Aging",
    r"固溶": "Solution Treatment",
    r"annealing": "Annealing",
    r"quenching": "Quenching",
    #
    r"hot\s*roll": "Hot Rolling",
    r"cold\s*roll": "Cold Rolling",
    r"hot\s*forg": "Hot Forging",
    r"hot\s*compress": "Hot Compression",
    r"cross.?roll": "Cross-Rolling",
    r"热轧": "Hot Rolling",
    r"热锻": "Hot Forging",
    r"热压": "Hot Compression",
    # # additive manufacturing
    r"LPBF|laser\s*powder\s*bed\s*fus": "LPBF",
    r"SLM|selective\s*laser\s*melt": "SLM",
    r"DED|directed\s*energy\s*dep": "DED",
    r"WAAM|wire\s*arc\s*add": "WAAM",
    r"EBM|electron\s*beam\s*melt": "EBM",
    r"additive\s*manufactur": "Additive Manufacturing",
    r"additive manufacturing|激光增材": "Additive Manufacturing",
    #
    r"sintering|spark\s*plasma\s*sint": "Sintering/SPS",
    r"powder\s*metallurgy|PM": "Powder Metallurgy",
    r"粉末冶金": "Powder Metallurgy",
    #
    r"welding|friction\s*weld": "Welding",
    r"焊接": "Welding",
}

#
MICRO_KEYWORDS = {
    r"equiaxed\s*α|globulariz|球化|equiaxed.*α": "Equiaxed α",
    r"lamellar\s*α|plate.?let|lamellar.*α|层片": "Lamellar α",
    r"basketweave|魏氏体|widmanstätten": "Widmanstätten/Basketweave",
    r"bimodal|双态|双模": "Bimodal Microstructure",
    r"trimodal|三态": "Trimodal Microstructure",
    r"martensite|martensite": "Martensite (α')",
    r"ω.?phase|omega\s*phase|ω相": "ω Phase",
    r"grain\s*refin|细晶|晶粒细化": "Grain Refinement",
    r"recrystal|recrystallization": "Recrystallization",
    r"twinning|孪晶|twin": "Twinning (TWIP)",
    r"precipitation|析出": "Precipitation",
    r"phase\s*transform|phase transformation": "Phase Transformation",
    r"columnar\s*grain|columnar": "Columnar Grains",
    r"α\+β|duplex|双相": "α+β Dual Phase",
    r"β\s*stabiliz|β稳定": "β-Stabilized Structure",
    r"dynamic\s*recrystal|DRX|动态recrystallization": "Dynamic Recrystallization",
    r"static\s*recrystal|SRX|静态recrystallization": "Static Recrystallization",
    r"adiabatic\s*shear|绝热剪切": "Adiabatic Shear Band",
    r"porosity|pore|孔隙": "Porosity",
}

#
PROPERTY_KEYWORDS = {
    r"UTS|ultimate\s*tensile|ultimate tensile strength": "Ultimate Tensile Strength",
    r"YS|yield\s*strength|yield strength": "Yield Strength",
    r"elongation|elongation|ductility": "Elongation/Ductility",
    r"hardness|维氏hardness|HV": "Hardness",
    r"fatigue|fatigue": "Fatigue Performance",
    r"fracture\s*toughness|断裂toughness": "Fracture Toughness",
    r"wear\s*resistance|耐磨": "Wear Resistance",
    r"corrosion|腐蚀": "Corrosion Resistance",
    r"biocompatib|生物相容": "Biocompatibility",
    r"elastic\s*modulus|弹性modulus|Young": "Elastic Modulus",
    r"ductility|toughness": "Ductility",
    r"strength.?ductility|强塑积|强ductility": "Strength-Ductility Balance",
    r"high.?temperature\s*strength|高温strength": "High-Temp Strength",
    r"creep|creep": "Creep Resistance",
    r"work.?harden|加工硬化": "Work Hardening",
    r"TWIP|twinning\s*induced\s*plastic": "TWIP Effect",
    r"TRIP|transformation\s*induced": "TRIP Effect",
}

#
CAUSAL_VERBS = [
    r"leads?\s*to", r"results?\s*in", r"causes?", r"promotes?",
    r"enhances?", r"improves?", r"increases?", r"decreases?",
    r"reduces?", r"refines?", r"suppresses?", r"inhibits?",
    r"facilitates?", r"enables?", r"contributes?\s*to",
    r"导致", r"使得", r"促进", r"抑制", r"提高", r"降低", r"改善",
    r"影响", r"决定", r"有助于",
]
CAUSAL_PATTERN = re.compile(
    r"(?P<cause>[A-Z][^.]{5,60}?)\s+(?:" + "|".join(CAUSAL_VERBS) + r")\s+(?P<effect>[a-z][^.]{5,60})",
    re.IGNORECASE
)

#

def extract_json_from_content(content: str) -> dict:
    """Module functionality."""
    m = re.search(r"```json\s*([\s\S]+?)\s*```", content)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    #
    try:
        return json.loads(content)
    except Exception:
        return {}


def text_list(obj) -> list:
    """Module functionality."""
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(text_list(item))
        return out
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(text_list(v))
        return out
    return [str(obj)]


def extract_elements_from_text(text: str) -> list:
    """Module functionality."""
    found = []
    #
    for m in re.finditer(r"(\d+\.?\d*)\s*([A-Z][a-z]?)\b", text):
        sym = m.group(2)
        if sym in ELEMENT_SET:
            found.append(sym)
    #
    for elem in ALLOY_ELEMENTS:
        if re.search(r"\b" + elem + r"\b", text):
            if elem not in found:
                found.append(elem)
    return found


def match_keywords(text: str, kw_dict: dict) -> list:
    """Module functionality."""
    found = []
    text_lower = text.lower()
    for pattern, label in kw_dict.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            if label not in found:
                found.append(label)
    return found


def extract_paper_title(folder_name: str) -> str:
    """Module functionality."""
    return folder_name[:60].strip()


#

def parse_all_papers() -> list:
    """
    """
    papers = []
    all_json_files = list(SEMANTIC_DIR.glob("*/semantic_results.json"))
    print(f"[INFO] Found {len(all_json_files)} papers with semantic extraction results")

    for jf in all_json_files:
        folder = jf.parent.name
        title = extract_paper_title(folder)
        try:
            with open(jf, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"  [SKIP] {folder[:40]}: read failed {e}")
            continue

        results = raw.get("results", [])

        #
        all_texts = []
        for chunk in results:
            content = chunk.get("content", "")
            chunk_obj = extract_json_from_content(content)
            if chunk_obj:
                texts = []
                for key in ["研究背景和动机", "研究方法", "实验result", "结论",
                            "关键概念和定义", "研究创新点", "未来工作方向"]:
                    texts.extend(text_list(chunk_obj.get(key, "")))
                all_texts.extend(texts)
            else:
                #
                all_texts.append(content)

        full_text = " ".join(all_texts)

        #
        elements = list(set(extract_elements_from_text(full_text)))
        #
        elements = [e for e in elements if e != "Ti"]

        processes = match_keywords(full_text, PROCESS_KEYWORDS)
        microstructures = match_keywords(full_text, MICRO_KEYWORDS)
        properties = match_keywords(full_text, PROPERTY_KEYWORDS)

        #
        #
        alloy_names = []
        for m in re.finditer(
            r"Ti[-–]\s*(?:\d+\.?\d*[A-Za-z]{1,3}[-–]?)+",
            full_text
        ):
            name = m.group()
            #
            name = re.sub(r"[^\x20-\x7E]", "", name).strip().strip("-–")
            name = re.sub(r"\s+", "", name).replace("–", "-")
            #
            if 4 < len(name) < 35 and re.match(r"^Ti-[\w\-\.]+$", name):
                if name not in alloy_names:
                    alloy_names.append(name)
        #
        alloy_names = alloy_names[:5]

        #
        beta_trans = None
        m_bt = re.search(r"[βb](?:eta)?\s*[Tt]rans(?:us)?\s*[≈~=]?\s*(\d{3,4})\s*°?\s*C", full_text)
        if m_bt:
            try:
                beta_trans = float(m_bt.group(1))
            except:
                pass

        #
        proc_temps = []
        for m in re.finditer(r"(\d{2,4})\s*°C", full_text):
            try:
                t = int(m.group(1))
                if 200 <= t <= 1400:
                    proc_temps.append(t)
            except:
                pass
        proc_temps = sorted(set(proc_temps))

        #
        mech_props = {}
        for m in re.finditer(r"(\d+\.?\d*)\s*(MPa|GPa|%|HV|HRC|J/m²|MJ/m³)", full_text):
            val, unit = float(m.group(1)), m.group(2)
            if unit == "MPa" and 100 < val < 3000:
                mech_props.setdefault("strength_MPa", []).append(val)
            elif unit == "GPa" and 0.1 < val < 500:
                mech_props.setdefault("modulus_GPa", []).append(val)
            elif unit == "%" and 0.1 < val < 60:
                mech_props.setdefault("elongation_%", []).append(val)
            elif unit == "HV" and 50 < val < 700:
                mech_props.setdefault("hardness_HV", []).append(val)

        #
        causal_pairs = []
        for m in CAUSAL_PATTERN.finditer(full_text):
            cause = m.group("cause").strip()
            effect = m.group("effect").strip()
            if 3 < len(cause) < 80 and 3 < len(effect) < 80:
                causal_pairs.append({"cause": cause, "effect": effect})

        paper = {
            "title": title,
            "folder": folder,
            "elements": elements,
            "alloy_names": alloy_names,
            "processes": processes,
            "microstructures": microstructures,
            "properties": properties,
            "beta_transus": beta_trans,
            "proc_temps": proc_temps,
            "mech_props": mech_props,
            "causal_pairs": causal_pairs[:20],  # Limit to 20 per paper
            "chunk_count": len(results),
        }
        papers.append(paper)

    print(f"[INFO] Successfully parsed {len(papers)} papers")
    return papers


# ─── Graph 1: Alloy Elements Network ──────────────────────

def build_elements_network(papers: list) -> dict:
    """
    """
    elem_freq = Counter()
    cooccur = Counter()

    for p in papers:
        elems = p["elements"]
        for e in elems:
            elem_freq[e] += 1
        #
        elems_sorted = sorted(set(elems))
        for i in range(len(elems_sorted)):
            for j in range(i + 1, len(elems_sorted)):
                cooccur[(elems_sorted[i], elems_sorted[j])] += 1

    return {"elem_freq": dict(elem_freq), "cooccur": dict(cooccur)}


def render_elements_network(data: dict, outpath: str, paper_count: int = 0):
    """Module functionality."""
    if not PYVIS_OK:
        return
    net = Network(
        height="720px", width="100%",
        bgcolor="#0d1117", font_color="white",
        directed=False,
    )
    net.set_options("""
    {
      "nodes": {
        "font": {"size": 16, "strokeWidth": 2, "strokeColor": "#0d1117"},
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "smooth": {"type": "continuous"},
        "shadow": true
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 120,
          "springConstant": 0.1
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 200}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    elem_freq = data["elem_freq"]
    cooccur = data["cooccur"]

    if not elem_freq:
    """
    Compute co-occurrence frequencies of process -> microstructure (co-occurring in same paper).
    """
    pm_matrix = Counter()  # (process, micro) -> count
    proc_freq = Counter()
    micro_freq = Counter()
    for p in papers:
        for proc in p["processes"]:
            proc_freq[proc] += 1
            for micro in p["microstructures"]:
                pm_matrix[(proc, micro)] += 1
        for micro in p["microstructures"]:
            micro_freq[micro] += 1
    return {
        "pm_matrix": {f"{k[0]}||{k[1]}": v for k, v in pm_matrix.items()},
        "proc_freq": dict(proc_freq),
        "micro_freq": dict(micro_freq),
    }


def render_proc_micro(data: dict, outpath: str):
    """
    if not PYVIS_OK:
        return

    net = Network(
        height="760px", width="100%",
        bgcolor="#111827", font_color="#f9fafb",
        directed=True,
    )
    net.set_options("""
    {
      "nodes": {"font": {"size": 14}, "shadow": true, "borderWidth": 2},
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
        "smooth": {"type": "curvedCW", "roundness": 0.2},
        "shadow": true
      },
      "physics": {
        "hierarchicalRepulsion": {
          "nodeDistance": 160,
          "centralGravity": 0.0,
          "springLength": 120,
          "springConstant": 0.01
        },
        "solver": "hierarchicalRepulsion",
        "stabilization": {"iterations": 300}
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 280,
          "nodeSpacing": 90
        }
      },
      "interaction": {"hover": true, "tooltipDelay": 100, "navigationButtons": true}
    }
    """)

    pm_matrix = {tuple(k.split("||")): v for k, v in data["pm_matrix"].items()}
    proc_freq = data["proc_freq"]
    micro_freq = data["micro_freq"]

    if not proc_freq:
    """
    mp_matrix = Counter()
    micro_freq = Counter()
    prop_freq = Counter()
    for p in papers:
        for micro in p["microstructures"]:
            micro_freq[micro] += 1
            for prop in p["properties"]:
                mp_matrix[(micro, prop)] += 1
        for prop in p["properties"]:
            prop_freq[prop] += 1
    return {
        "mp_matrix": {f"{k[0]}||{k[1]}": v for k, v in mp_matrix.items()},
        "micro_freq": dict(micro_freq),
        "prop_freq": dict(prop_freq),
    }


def render_micro_property(data: dict, outpath: str):
    """Module functionality."""
    if not PYVIS_OK:
        return

    net = Network(
        height="760px", width="100%",
        bgcolor="#0f172a", font_color="#f1f5f9",
        directed=True,
    )
    net.set_options("""
    {
      "nodes": {"font": {"size": 13}, "shadow": true},
      "edges": {
        "arrows": {"to": {"enabled": true}},
        "smooth": {"type": "dynamic"}
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.005,
          "springLength": 150,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 200}
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "levelSeparation": 300,
          "nodeSpacing": 80
        }
      },
      "interaction": {"hover": true, "navigationButtons": true}
    }
    """)

    mp_matrix = {tuple(k.split("||")): v for k, v in data["mp_matrix"].items()}
    micro_freq = data["micro_freq"]
    prop_freq = data["prop_freq"]

    if not micro_freq:
        return

    for micro, freq in micro_freq.items():
        net.add_node(
            f"MICRO:{micro}", label=micro,
            color="#10b981", shape="diamond",
            size=18 + freq * 2,
    """
    Build high-level causal knowledge graph (process -> microstructure -> property chain + alloy composition).
    Each edge has weight (co-occurrence frequency) and polarity (positive/negative).
    """
    """
    if not PYVIS_OK:
        return

    net = Network(
        height="800px", width="100%",
        bgcolor="#030712", font_color="#f9fafb",
        directed=True,
    )
    net.set_options("""
    {
      "nodes": {"font": {"size": 13, "strokeWidth": 2, "strokeColor": "#030712"}, "shadow": true},
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
        "smooth": {"type": "cubicBezier", "forceDirection": "horizontal"},
        "shadow": true
      },
      "physics": {
        "hierarchicalRepulsion": {
          "nodeDistance": 130,
          "centralGravity": 0.0,
          "springLength": 100,
          "springConstant": 0.01,
          "avoidOverlap": 1
        },
        "solver": "hierarchicalRepulsion",
        "stabilization": {"iterations": 400}
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "hubsize",
          "levelSeparation": 260,
          "nodeSpacing": 75,
          "treeSpacing": 200
        }
      },
      "interaction": {"hover": true, "tooltipDelay": 100, "navigationButtons": true, "keyboard": true}
    }
    """)

    LAYER_COLOR = {
    # Build global KG: papers as center nodes, connected to alloys/processes/microstructures/properties.
    if not PYVIS_OK:
        return

    net = Network(
        height="820px", width="100%",
        bgcolor="#0a0a0a", font_color="#e2e8f0",
        directed=True,
    )
    net.set_options("""
    {
      "nodes": {"font": {"size": 11}, "shadow": true},
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "smooth": {"type": "continuous"},
        "color": {"opacity": 0.4}
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.005,
          "springLength": 180,
          "springConstant": 0.05
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 300}
      },
      "interaction": {"hover": true, "navigationButtons": true}
    }
    """)

    """
    header = f"""
<div style="position:fixed;top:0;left:0;right:0;z-index:9999;
            background:linear-gradient(135deg,#1e293b,#0f172a);
            padding:12px 24px;border-bottom:1px solid #334155;
            font-family:'Segoe UI',sans-serif;">
  <div style="font-size:20px;font-weight:700;color:#f1f5f9;">{title}</div>
  <div style="font-size:12px;color:#94a3b8;margin-top:2px;">{subtitle}</div>
</div>
<div style="height:60px;"></div>
"""
    return html.replace("<body>", "<body>" + header, 1)


    """
    total = len(papers)
    # # statistics
    proc_cnt = Counter()
    micro_cnt = Counter()
    prop_cnt = Counter()
    elem_cnt = Counter()
    alloy_cnt = Counter()
    for p in papers:
        for x in p["processes"]:    proc_cnt[x] += 1
        for x in p["microstructures"]: micro_cnt[x] += 1
        for x in p["properties"]:   prop_cnt[x] += 1
        for x in p["elements"]:     elem_cnt[x] += 1
        for x in p["alloy_names"]:  alloy_cnt[x] += 1

    def table_rows(cnt: Counter, top_n=15) -> str:
        rows = ""
        for k, v in cnt.most_common(top_n):
            pct = v / total * 100
            bar = f'<div style="width:{min(pct*3,300):.0f}px;height:14px;background:#3b82f6;border-radius:3px;display:inline-block;"></div>'
            rows += f"<tr><td>{k}</td><td>{v}</td><td>{pct:.1f}%</td><td>{bar}</td></tr>"
        return rows

    #
    graph_links = ""
    for fname, title in graph_files.items():
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            graph_links += f'<a href="{fname}" target="_blank" style="display:inline-block;margin:8px;padding:10px 18px;background:#1d4ed8;color:white;text-decoration:none;border-radius:8px;font-size:14px;">📊 {title}</a>'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Ti-Alloy Knowledge Graph Report</title>
<style>
  body{{margin:0;font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;}}
  .hero{{background:linear-gradient(135deg,#1e3a5f,#0c1a2e);padding:40px 48px;border-bottom:1px solid #1e3a5f;}}
  .hero h1{{margin:0;font-size:28px;color:#f1f5f9;}}
  .hero p{{margin:8px 0 0;color:#94a3b8;font-size:14px;}}
  .stats-grid{{display:flex;gap:16px;flex-wrap:wrap;padding:24px 48px;}}
  .stat-card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px 28px;min-width:140px;}}
  .stat-card .num{{font-size:32px;font-weight:700;color:#60a5fa;}}
  .stat-card .lbl{{font-size:12px;color:#94a3b8;margin-top:4px;}}
  .section{{padding:24px 48px;}}
  .section h2{{font-size:18px;color:#f1f5f9;border-left:3px solid #3b82f6;padding-left:12px;margin-bottom:16px;}}
  table{{width:100%;border-collapse:collapse;font-size:13px;}}
  th{{background:#1e293b;color:#94a3b8;padding:8px 12px;text-align:left;font-weight:500;}}
  td{{padding:7px 12px;border-bottom:1px solid #1e293b;}}
  tr:hover td{{background:#1e293b;}}
  .graph-links{{padding:24px 48px;}}
  .graph-links h2{{font-size:18px;color:#f1f5f9;border-left:3px solid #3b82f6;padding-left:12px;margin-bottom:16px;}}
  footer{{padding:24px 48px;color:#475569;font-size:12px;border-top:1px solid #1e293b;margin-top:24px;}}
</style>
</head>
<body>
<div class="hero">
  <h1>Ti-Alloy Literature Knowledge Graph Report</h1>
  <p>Based on semantic extraction results from {total} titanium alloy papers, auto-built multi-level knowledge graph</p>
</div>

<div class="stats-grid">
  <div class="stat-card"><div class="num">{total}</div><div class="lbl">Total Papers</div></div>
  <div class="stat-card"><div class="num">{len(proc_cnt)}</div><div class="lbl">Process Types</div></div>
  <div class="stat-card"><div class="num">{len(micro_cnt)}</div><div class="lbl">Microstructure Types</div></div>
  <div class="stat-card"><div class="num">{len(prop_cnt)}</div><div class="lbl">Property Types</div></div>
  <div class="stat-card"><div class="num">{len(elem_cnt)}</div><div class="lbl">Element Types</div></div>
  <div class="stat-card"><div class="num">{len(alloy_cnt)}</div><div class="lbl">Alloy Systems</div></div>
</div>

<div class="graph-links">
  <h2>Interactive Knowledge Graphs (click to open in new window)</h2>
  {graph_links}
</div>

<div class="section">
  <h2>Top Alloy Elements (by occurrence)</h2>
  <table><tr><th>Element</th><th>Papers</th><th>Share</th><th>Frequency Bar</th></tr>
  {table_rows(elem_cnt, 15)}</table>
</div>

<div class="section">
  <h2>Top Processes</h2>
  <table><tr><th>Process</th><th>Papers</th><th>Share</th><th>Frequency Bar</th></tr>
  {table_rows(proc_cnt, 15)}</table>
</div>

<div class="section">
  <h2>Top Microstructures</h2>
  <table><tr><th>Microstructure</th><th>Papers</th><th>Share</th><th>Frequency Bar</th></tr>
  {table_rows(micro_cnt, 15)}</table>
</div>

<div class="section">
  <h2>Top Properties</h2>
  <table><tr><th>Property</th><th>Papers</th><th>Share</th><th>Frequency Bar</th></tr>
  {table_rows(prop_cnt, 15)}</table>
</div>

<div class="section">
  <h2>Frequent Alloy Systems (Top 20)</h2>
  <table><tr><th>Alloy Name</th><th>Papers</th><th>Share</th><th>Frequency Bar</th></tr>
  {table_rows(alloy_cnt, 20)}</table>
</div>

<footer>
  Generated by LitExtract Agent v4.0 | Ti-alloy Knowledge Graph Builder<br>
  Data source: semantic extraction results from {total} papers under semantic_outputs/
</footer>
</body>
</html>"""

    with open(outpath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [OK] {outpath}")


#

def main():
    print("=" * 60)
    print("  Ti-Alloy Knowledge Graph Builder")
    print("=" * 60)

    #
    papers = parse_all_papers()

    if not papers:
        print("[ERROR] Could not parse any papers, check semantic_outputs/ path")
        return

    #
    data_path = str(OUTPUT_DIR / "ti_knowledge_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Structured data exported: {data_path}")

    if not PYVIS_OK:
        print("[ERROR] pyvis not installed, cannot generate HTML graphs")
        print("Please run: pip install pyvis")
        return

    print("\n[BUILD] Building graph data...")

    #
    elem_data    = build_elements_network(papers)
    pm_data      = build_proc_micro_graph(papers)
    mp_data      = build_micro_property_graph(papers)
    causal_data  = build_causal_kg(papers)

    #
    print("\n[RENDER] Rendering interactive HTML graphs...")
    out = lambda name: str(OUTPUT_DIR / name)

    render_elements_network(elem_data, out("01_alloy_elements_network.html"), paper_count=len(papers))
    render_proc_micro(pm_data, out("02_processing_microstructure.html"))
    render_micro_property(mp_data, out("03_microstructure_property.html"))
    render_causal_kg(causal_data, out("04_causal_knowledge_graph.html"))
    render_global_kg(papers, out("05_global_kg.html"))

    #
    print("\n[REPORT] Generating statistics report...")
    graph_files = {
        "01_alloy_elements_network.html": "Alloy Elements Network",
        "02_processing_microstructure.html": "Processing–Microstructure",
        "03_microstructure_property.html": "Microstructure–Property",
        "04_causal_knowledge_graph.html": "Causal Knowledge Graph",
        "05_global_kg.html": "Global KG",
    }
    build_summary_report(papers, graph_files, out("00_summary_report.html"))

    print("\n" + "=" * 60)
    print(f"  All done! Output directory: {OUTPUT_DIR}")
    print("  File list:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size / 1024
        print(f"    {f.name:50s}  {size:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
