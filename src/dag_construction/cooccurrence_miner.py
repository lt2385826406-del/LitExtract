"""
Step1: Candidate Edge Generation via Co-occurrence Mining
"""

import json
import itertools
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional, Set


#
ENTITY_GROUPS = {
    "Element": "composition",
    "Alloy": "composition",
    "Processing": "processing",
    "Microstructure": "microstructure",
    "Property": "property",
}

#
ALLOWED_CROSS_GROUP_PAIRS = [
    ("Processing", "Microstructure"),
    ("Processing", "Property"),
    ("Microstructure", "Property"),
    ("Element", "Property"),
    ("Element", "Microstructure"),   
    ("Alloy", "Property"),
    ("Processing", "Alloy"),
    ("Microstructure", "Processing"),  
]

#
ALLOWED_SAME_GROUP_PAIRS = True

# ============================================================
#
# ============================================================
#
#
#
#
#
#
#
_LOW_FREQ_ELEMENT_SYMBOLS: Set[str] = {
    #
    "B", "Nb", "Zr", "Ta", "W", "Si", "Sn", "Y", "La", "Ce", "Nd", "Er",
    "O", "N", "H",
    #
}

_LOW_FREQ_MICROSTRUCTURE_NAMES: Set[str] = {
    #
    "tib", "tib2", "tic", "tin", "ti5si3",
    "silicides", "silicide",
    "borides", "boride",
    "carbides", "carbide",
    "rare earth", "rare-earth",
    "ω phase", "omega phase", "ω-phase", "omega-phase",
    "α₂ phase", "alpha-2 phase",
    "o phase", "orthorhombic phase",
    "y₂o₃", "y2o3", "la₂o₃", "la2o3",
}

def _is_low_freq_entity(name: str, entity_type: str) -> bool:
    """
    """
    import os
    if os.environ.get("WB_NO_DYNAMIC_FREQ") == "1":
        return False
    
    name_lower = name.lower().strip()
    
    if entity_type == "Element":
        #
        #
        #
        tokens = name_lower.replace("=", " ").replace(":", " ").replace(",", " ").split()
        for token in tokens:
            if token in {e.lower() for e in _LOW_FREQ_ELEMENT_SYMBOLS}:
                return True
        return False
    
    elif entity_type == "Microstructure":
        #
        return name_lower in _LOW_FREQ_MICROSTRUCTURE_NAMES
    
    else:
        #
        return False

# ============================================================
#
# ============================================================
#
#
_SYNONYM_CANONICAL_MAP: Dict[str, str] = {
    #
    "precipitation": "precipitation",
    "precipitates": "precipitation",
    "precipitate": "precipitation",
    "α precipitate": "precipitation",
    "α precipitates": "precipitation",
    "alpha precipitate": "precipitation",
    "alpha precipitates": "precipitation",
    "ω precipitates": "precipitation",
    "ω precipitate": "precipitation",
    "omega precipitates": "precipitation",
    "omega precipitate": "precipitation",
    "ω_iso precipitates": "precipitation",
    "secondary phase": "precipitation",
    "secondary phases": "precipitation",
    "second phase": "precipitation",
    "second phases": "precipitation",
    #
    "martensite": "martensite",
    "α'": "martensite",
    "α' martensite": "martensite",
    "α'-martensite": "martensite",
    "α′ martensite": "martensite",
    "α″": "martensite",
    "α″ martensite": "martensite",
    "acicular martensite": "martensite",
    "acicular α'": "martensite",
    "acicular alpha": "martensite",
    "acicular α phase": "martensite",
    "acicular alpha phase": "martensite",
    #
    "lamellar α": "lamellar α phase",
    "lamellar alpha": "lamellar α phase",
    "lamellar": "lamellar α phase",
    "lamellae": "lamellar α phase",
    "lamella": "lamellar α phase",
    "lamellar microstructure": "lamellar α phase",
    "lamellar structure": "lamellar α phase",
    #
    "equiaxed α": "equiaxed α phase",
    "equiaxed alpha": "equiaxed α phase",
    "equiaxed": "equiaxed α phase",
    "equiaxed grains": "equiaxed α phase",
    "equiaxed grain": "equiaxed α phase",
    "equiaxed microstructure": "equiaxed α phase",
    "equiaxed crystals": "equiaxed α phase",
    #
    "grain refinement": "grain refinement",
    "grain refining": "grain refinement",
    "refined grains": "grain refinement",
    "refined grain": "grain refinement",
    "fine grain": "grain refinement",
    "fine grains": "grain refinement",
    "ultrafine grains": "grain refinement",
    "ultrafine grain": "grain refinement",
    #
    "recrystallization": "recrystallization",
    "recrystallized": "recrystallization",
    "recrystallised": "recrystallization",
    "drx": "recrystallization",
    "dynamic recrystallization": "recrystallization",
    #
    "yield strength": "yield strength",
    "ys": "yield strength",
    "yield stress": "yield strength",
    "ultimate tensile strength": "ultimate tensile strength",
    "uts": "ultimate tensile strength",
    "tensile strength": "ultimate tensile strength",
    "elongation": "elongation",
    "ductility": "elongation",
    "elongation/ductility": "elongation",
    "fracture toughness": "fracture toughness",
    "kic": "fracture toughness",
    "k1c": "fracture toughness",
    "fatigue strength": "fatigue strength",
    "fatigue": "fatigue strength",
    "fatigue life": "fatigue strength",
    "hardness": "hardness",
    "microhardness": "hardness",
    "vickers hardness": "hardness",
}

# ============================================================
#
# ============================================================
#
#
_NOISE_PREFIXES = (
    "fig:", "fig.", "figure:", "figure.",
    "test:", "test-", "test_",
    "table:", "table.",
    "sample:", "sample-",
    "specimen:", "specimen-",
)

_NOISE_EXACT = {
    "", "null", "none", "n/a", "na", "-", "--",
}


def load_semantic_data(json_path: str) -> List[Dict[str, Any]]:
    """Module functionality."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _canonicalize_entity_name(name: str, entity_type: str) -> Optional[str]:
    """
    """
    if not name or not isinstance(name, str):
        return None
    
    name_stripped = name.strip()
    
    #
    if name_stripped.lower() in _NOISE_EXACT:
        return None
    
    #
    name_lower = name_stripped.lower()
    for prefix in _NOISE_PREFIXES:
        if name_lower.startswith(prefix):
            return None
    
    #
    if name_lower.replace(".", "").replace("-", "").isdigit():
        return None
    
    #
    if name_lower in _SYNONYM_CANONICAL_MAP:
        return _SYNONYM_CANONICAL_MAP[name_lower]
    
    #
    if entity_type == "Element" and "=" in name_stripped:
        base = name_stripped.split("=")[0].strip()
        if base:
            return base
    
    return name_stripped


def extract_entities_from_paper(paper: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    """
    entities: Dict[str, List[str]] = {
        "Element": [],
        "Alloy": [],
        "Processing": [],
        "Microstructure": [],
        "Property": [],
    }

    #
    if "elements" in paper:
        entities["Element"].extend(paper.get("elements", []))
    if "alloy_names" in paper:
        entities["Alloy"].extend(paper.get("alloy_names", []))
    if "processes" in paper:
        entities["Processing"].extend(paper.get("processes", []))
    if "microstructures" in paper:
        entities["Microstructure"].extend(paper.get("microstructures", []))
    if "properties" in paper:
        entities["Property"].extend(paper.get("properties", []))

    #
    for sample in paper.get("samples", []):
        comp = sample.get("composition", {})
        if comp and comp != "null":
            elems = comp.get("elements", "")
            if isinstance(elems, str) and elems:
                entities["Element"].extend([e.strip() for e in elems.split(",") if e.strip()])
            elif isinstance(elems, list):
                entities["Element"].extend([e.get("element", "").strip() for e in elems if isinstance(e, dict) and e.get("element")])
        proc = sample.get("process", {})
        if proc and proc != "null":
            method = proc.get("method", "")
            if method:
                entities["Processing"].append(method)
        micro = sample.get("microstructure", {})
        if micro and micro != "null":
            phase = micro.get("phase_type", "")
            if phase:
                entities["Microstructure"].append(phase)
        prop = sample.get("property", {})
        if prop and prop != "null":
            ptype = prop.get("type", "")
            if ptype:
                entities["Property"].append(ptype)

    #
    for k in entities:
        cleaned = []
        seen = set()
        for x in entities[k]:
            canonical = _canonicalize_entity_name(x, k)
            if canonical is not None and canonical.lower() not in seen:
                cleaned.append(canonical)
                seen.add(canonical.lower())
        entities[k] = cleaned

    return entities


def compute_cooccurrence(
    papers: List[Dict[str, Any]],
    window_size: int = 1,
    min_freq: int = 2,
) -> Tuple[Counter, Dict[str, List[Dict[str, Any]]]]:
    """
    """
    cooc_matrix: Counter = Counter()
    cooc_details: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for paper in papers:
        title = paper.get("title", "")
        folder = paper.get("folder", "")
        entities = extract_entities_from_paper(paper)

        #
        present_entities: List[Tuple[str, str]] = []
        for etype, enames in entities.items():
            for ename in enames:
                present_entities.append((etype, ename))

        #
        for (t1, n1), (t2, n2) in itertools.combinations(present_entities, 2):
            if t1 == t2 and n1 == n2:
                continue
            #
            key_parts = sorted([(t1, n1), (t2, n2)])
            ck = f"{key_parts[0][0]}|{key_parts[0][1]}||{key_parts[1][0]}|{key_parts[1][1]}"
            cooc_matrix[ck] += 1
            cooc_details[ck].append({
                "title": title,
                "folder": folder,
            })

    return cooc_matrix, cooc_details


def generate_candidate_edges(
    cooc_matrix: Counter,
    cooc_details: Dict[str, List[Dict[str, Any]]],
    papers: List[Dict[str, Any]],
    min_freq: int = 2,
    max_papers: int = 10000,
    direction_hints: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    """
    if direction_hints is None:
        direction_hints = {}

    candidates = []
    all_entity_counts = _count_individual_entities(papers)

    for cooc_key, freq in cooc_matrix.items():
        #
        parts = cooc_key.split("||")
        left = parts[0].split("|")
        right = parts[1].split("|")
        t1, n1 = left[0], left[1]
        t2, n2 = right[0], right[1]

        #
        effective_min = min_freq
        if freq >= 1 and freq < min_freq:
            if _is_low_freq_entity(n1, t1) or _is_low_freq_entity(n2, t2):
                effective_min = 1

        if freq < effective_min:
            continue

        #
        if not _is_allowed_pair(t1, t2):
            continue

        detail_list = cooc_details.get(cooc_key, [])

        #
        direction = _infer_direction(
            t1, n1, t2, n2,
            freq,
            all_entity_counts,
            direction_hints,
        )

        if direction == "forward" or direction == "both":
            candidates.append({
                "source_type": t1,
                "source_name": n1,
                "target_type": t2,
                "target_name": n2,
                "cooccurrence_freq": freq,
                "evidence_papers": detail_list[:max_papers],
                "direction": "forward",
                "confidence_raw": min(0.5 + freq * 0.05, 0.85),  # Frequency-based raw confidence
            })

        if direction == "reverse" or direction == "both":
            candidates.append({
                "source_type": t2,
                "source_name": n2,
                "target_type": t1,
                "target_name": n1,
                "cooccurrence_freq": freq,
                "evidence_papers": detail_list[:max_papers],
                "direction": "reverse",
                "confidence_raw": min(0.5 + freq * 0.05, 0.85),
            })

    return candidates


def _count_individual_entities(papers: List[Dict[str, Any]]) -> Counter:
    """Module functionality."""
    counter: Counter = Counter()
    for paper in papers:
        entities = extract_entities_from_paper(paper)
        seen_in_paper = set()
        for etype, enames in entities.items():
            for ename in enames:
                key = f"{etype}|{ename}"
                if key not in seen_in_paper:
                    counter[key] += 1
                    seen_in_paper.add(key)
    return counter


def _is_allowed_pair(type1: str, type2: str) -> bool:
    """Module functionality."""
    if type1 == type2 and ALLOWED_SAME_GROUP_PAIRS:
        return True
    if (type1, type2) in ALLOWED_CROSS_GROUP_PAIRS:
        return True
    if (type2, type1) in ALLOWED_CROSS_GROUP_PAIRS:
        return True
    return False


def _infer_direction(
    t1: str, n1: str, t2: str, n2: str,
    cooc_freq: int,
    entity_counts: Counter,
    direction_hints: Dict[str, str],
) -> str:
    """
    """
    #
    try:
        from .causal_builder import NODE_ORDER
    except Exception:
        #
        NODE_ORDER = ["Element", "Alloy", "Processing", "Microstructure", "Property"]
    order = {t: i for i, t in enumerate(NODE_ORDER)}
    order = {t: i for i, t in enumerate(NODE_ORDER)}
    if t1 in order and t2 in order:
        if order[t1] < order[t2]:
            return "forward"
        elif order[t1] > order[t2]:
            return "reverse"

    #
    count1 = entity_counts.get(f"{t1}|{n1}", 0)
    count2 = entity_counts.get(f"{t2}|{n2}", 0)
    if count1 > count2 * 2 and count2 > 0:
        return "forward"  
    if count2 > count1 * 2 and count1 > 0:
        return "reverse"

    #
    return "both"


def save_candidates(candidates: List[Dict[str, Any]], output_path: str):
    """Module functionality."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)


def load_candidates(input_path: str) -> List[Dict[str, Any]]:
    """Module functionality."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_step1(
    input_json: str,
    output_json: str,
    min_freq: int = 2,
) -> List[Dict[str, Any]]:
    """
    """
    print(f"[Step1] Loading semantic data: {input_json}")
    papers = load_semantic_data(input_json)
    print(f"         {len(papers)} papers total")

    print("[Step1] Computing co-occurrence matrix...")
    cooc_matrix, cooc_details = compute_cooccurrence(papers, min_freq=1)

    print(f"[Step1] Generating candidate edges (min_freq>={min_freq})...")
    candidates = generate_candidate_edges(
        cooc_matrix, cooc_details, papers,
        min_freq=min_freq,
    )

    print(f"[Step1] Generated {len(candidates)} candidate edges")
    save_candidates(candidates, output_json)
    print(f"[Step1] Candidate edges saved to: {output_json}")

    return candidates
