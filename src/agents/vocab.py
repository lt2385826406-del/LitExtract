# vocab.py
# ============================================================
# Controlled Vocabulary & Canonical Mapping
# For Material Knowledge Graph (KG) and Causal Hypothesis Graph (CHG)
# ============================================================

from typing import Dict, List

# ------------------------------------------------------------
# 1. Property (Mechanical / Physical Properties)
# ------------------------------------------------------------

PROPERTY_CANONICAL_MAP: Dict[str, str] = {
    # Ultimate tensile strength
    "ultimate tensile strength": "UTS",
    "tensile strength": "UTS",
    "uts": "UTS",
    "rm": "UTS",
    "σb": "UTS",

    # Yield strength
    "yield strength": "YS",
    "yield stress": "YS",
    "ys": "YS",
    "rp0.2": "YS",
    "σy": "YS",

    # Elongation / ductility
    "elongation": "EL",
    "elongation to failure": "EL",
    "ductility": "EL",
    "elongation (%)": "EL",

    # Hardness
    "hardness": "HV",
    "vickers hardness": "HV",
    "microhardness": "HV",
    "hv": "HV",
    "hrc": "HRC",

    # Fracture / toughness
    "fracture toughness": "K_IC",
    "kic": "K_IC",
    "impact toughness": "ImpactEnergy",
    "charpy energy": "ImpactEnergy",

    # Fatigue
    "fatigue life": "FatigueLife",
    "fatigue limit": "FatigueLimit",
    "endurance limit": "FatigueLimit",

    # Creep
    "creep rate": "CreepRate",
    "steady state creep rate": "CreepRate",
    "creep life": "CreepLife",
    "rupture life": "CreepLife",

    # Others
    "wear rate": "WearRate",
    "corrosion rate": "CorrosionRate",
}

PROPERTY_TEST_CONDITIONS = {
    "room temperature": "RT",
    "ambient": "RT",
    "rt": "RT",
    "high temperature": "HT",
    "elevated temperature": "HT",
    "low temperature": "LT",
    "creep": "Creep",
    "fatigue": "Fatigue",
    "quasi-static": "QuasiStatic",
}

# ------------------------------------------------------------
# 2. Microstructure
# ------------------------------------------------------------

PHASE_TYPE_MAP: Dict[str, str] = {
    "alpha": "α",
    "α": "α",
    "alpha phase": "α",

    "beta": "β",
    "β": "β",
    "beta phase": "β",

    "alpha+beta": "α+β",
    "α+β": "α+β",
    "duplex": "α+β",
    "bimodal": "α+β",

    "gamma": "γ",
    "γ": "γ",
    "gamma prime": "γ′",
    "γ′": "γ′",
    "delta": "δ",
    "δ": "δ",

    "laves": "Laves",
    "laves phase": "Laves",

    "martensite": "Martensite",
    "martensitic": "Martensite",

    "bcc": "BCC",
    "fcc": "FCC",
    "hcp": "HCP",
}

MORPHOLOGY_MAP: Dict[str, str] = {
    "equiaxed": "equiaxed",
    "equiaxed grains": "equiaxed",
    "lamellar": "lamellar",
    "lamellae": "lamellar",
    "acicular": "acicular",
    "needle-like": "acicular",
    "globular": "globular",
    "globularized": "globular",
    "columnar": "columnar",
    "columnar grains": "columnar",
    "dendritic": "dendritic",
}

GRAIN_SIZE_DESCRIPTOR_MAP: Dict[str, str] = {
    "ultrafine": "ultrafine",
    "ufg": "ultrafine",
    "fine": "fine",
    "fine-grained": "fine",
    "medium": "medium",
    "coarse": "coarse",
    "coarse-grained": "coarse",
    "refined": "refined",
    "grain refinement": "refined",
    "coarsened": "coarsened",
}

# ------------------------------------------------------------
# 3. Processing
# ------------------------------------------------------------

PROCESS_METHOD_MAP: Dict[str, str] = {
    "solution treated": "solution_treatment",
    "solution treatment": "solution_treatment",
    "solution heat treatment": "solution_treatment",

    "aging": "aging",
    "aged": "aging",

    "annealing": "annealing",
    "annealed": "annealing",

    "homogenization": "homogenization",
    "homogenized": "homogenization",

    "hot rolling": "hot_rolling",
    "hot rolled": "hot_rolling",

    "cold rolling": "cold_rolling",
    "cold rolled": "cold_rolling",

    "forging": "forging",
    "forged": "forging",

    "additive manufacturing": "additive_manufacturing",
    "am": "additive_manufacturing",
    "slm": "additive_manufacturing",
    "ebm": "additive_manufacturing",

    "hot isostatic pressing": "HIP",
    "hip": "HIP",
}

PROCESS_PARAMETER_KEYS = {
    "temperature",
    "time",
    "duration",
    "cooling_rate",
    "strain_rate",
    "pressure",
}

# ------------------------------------------------------------
# 4. Composition
# ------------------------------------------------------------

VALID_ELEMENTS: List[str] = [
    "Ti", "Al", "V", "Nb", "Mo", "Zr", "Sn", "Fe", "Cr", "Ni", "Co",
    "Ta", "W", "B", "C", "O", "N", "H", "Si", "Mn", "Cu"
]

COMPOSITION_UNITS = {
    "wt%": "wt%",
    "weight percent": "wt%",
    "at%": "at%",
    "atomic percent": "at%",
    "ppm": "ppm",
    "balance": "balance",
}

# ------------------------------------------------------------
# 5. Causal Graph (CHG)
# ------------------------------------------------------------

CAUSAL_CLAIM_TYPES = {
    "explicit_causal",
    "contrast_based",
    "mechanism_supported",
    "cooccurrence",
}

CAUSAL_POLARITY_MAP = {
    "increase": "increase",
    "increases": "increase",
    "decrease": "decrease",
    "decreases": "decrease",
    "promote": "promote",
    "promotes": "promote",
    "suppress": "suppress",
    "suppresses": "suppress",
    "inhibit": "suppress",
    "unknown": "unknown",
}

ALLOWED_CAUSAL_DIRECTIONS = {
    "composition->microstructure",
    "composition->property",
    "process->microstructure",
    "process->property",
    "microstructure->property",
}

# ------------------------------------------------------------
# 6. Figure / Evidence
# ------------------------------------------------------------

FIGURE_TYPE_MAP = {
    "sem": "SEM",
    "tem": "TEM",
    "om": "OM",
    "ebsd": "EBSD",
    "xrd": "XRD",
    "dsc": "DSC",
    "phase diagram": "PhaseDiagram",
}

EVIDENCE_TYPES = {
    "paragraph",
    "figure",
    "table",
    "caption",
}

# ------------------------------------------------------------
# 7. Utility Functions (recommended usage)
# ------------------------------------------------------------

def normalize_text(text: any) -> str:
    """Normalize text, handling any type of input gracefully"""
    if not text:
        return ""
    #
    text_str = str(text)
    return text_str.strip().lower()


def map_with_fallback(raw: any, mapping: Dict[str, str], default: str = None) -> str:
    if not raw:
        return default
    key = normalize_text(raw)
    return mapping.get(key, default or raw)


def map_property_type(raw: any) -> str:
    return map_with_fallback(raw, PROPERTY_CANONICAL_MAP, default="Unknown")


def map_process_method(raw: any) -> str:
    return map_with_fallback(raw, PROCESS_METHOD_MAP, default="unknown_process")


def map_phase_type(raw: any) -> str:
    return map_with_fallback(raw, PHASE_TYPE_MAP, default=str(raw))


def map_morphology(raw: any) -> str:
    return map_with_fallback(raw, MORPHOLOGY_MAP, default=str(raw))


def map_polarity(raw: any) -> str:
    return map_with_fallback(raw, CAUSAL_POLARITY_MAP, default="unknown")
