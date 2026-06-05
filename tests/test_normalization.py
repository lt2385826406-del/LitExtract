"""Tests for controlled vocabulary and normalization functions in normalization.py."""

import pytest
from semantic_extraction.normalization import (
    normalize_text,
    map_with_fallback,
    map_property_type,
    map_process_method,
    map_phase_type,
    map_morphology,
    map_polarity,
    VALID_ELEMENTS,
    PROPERTY_CANONICAL_MAP,
    PHASE_TYPE_MAP,
    MORPHOLOGY_MAP,
    PROCESS_METHOD_MAP,
    CAUSAL_POLARITY_MAP,
    ALLOWED_CAUSAL_DIRECTIONS,
    COMPOSITION_UNITS,
)


# ============================================================================
# normalize_text
# ============================================================================

@pytest.mark.parametrize("input_val,expected", [
    ("  Ti-6Al-4V  ", "ti-6al-4v"),
    ("Ti-6Al-4V", "ti-6al-4v"),
    ("  ", ""),
    ("", ""),
    (None, ""),
    (0, ""),     # 0 is falsy in Python, treated as empty input
    (42, "42"),
])
def test_normalize_text(input_val, expected):
    assert normalize_text(input_val) == expected


# ============================================================================
# map_with_fallback
# ============================================================================

def test_map_with_fallback_match():
    assert map_with_fallback("tensile strength", PROPERTY_CANONICAL_MAP) == "UTS"


def test_map_with_fallback_no_match_returns_original():
    assert map_with_fallback("weird_property", PROPERTY_CANONICAL_MAP) == "weird_property"


def test_map_with_fallback_default():
    assert map_with_fallback("weird_property", PROPERTY_CANONICAL_MAP, default="Unknown") == "Unknown"


def test_map_with_fallback_none_input():
    assert map_with_fallback(None, PROPERTY_CANONICAL_MAP, default="Unknown") == "Unknown"


# ============================================================================
# Property mapping
# ============================================================================

@pytest.mark.parametrize("raw,expected", [
    ("tensile strength", "UTS"),
    ("UTS", "UTS"),
    ("yield strength", "YS"),
    ("elongation", "EL"),
    ("hardness", "HV"),
    ("vickers hardness", "HV"),
    ("fracture toughness", "K_IC"),
    ("fatigue life", "FatigueLife"),
    ("creep rate", "CreepRate"),
])
def test_map_property_type(raw, expected):
    assert map_property_type(raw) == expected


def test_map_property_type_unknown():
    assert map_property_type("mystery value") == "Unknown"


# ============================================================================
# Process mapping
# ============================================================================

@pytest.mark.parametrize("raw,expected", [
    ("solution treatment", "solution_treatment"),
    ("aging", "aging"),
    ("annealing", "annealing"),
    ("hot rolling", "hot_rolling"),
    ("forging", "forging"),
    ("HIP", "HIP"),
    ("hot isostatic pressing", "HIP"),
])
def test_map_process_method(raw, expected):
    assert map_process_method(raw) == expected


def test_map_process_method_unknown():
    assert map_process_method("unknown_process") == "unknown_process"


# ============================================================================
# Phase type mapping
# ============================================================================

@pytest.mark.parametrize("raw,expected", [
    ("alpha", "α"),
    ("beta", "β"),
    ("α+β", "α+β"),
    ("bimodal", "α+β"),
    ("gamma prime", "γ′"),
    ("laves", "Laves"),
    ("martensite", "Martensite"),
])
def test_map_phase_type(raw, expected):
    assert map_phase_type(raw) == expected


# ============================================================================
# Morphology mapping
# ============================================================================

@pytest.mark.parametrize("raw,expected", [
    ("equiaxed", "equiaxed"),
    ("lamellar", "lamellar"),
    ("acicular", "acicular"),
    ("globular", "globular"),
])
def test_map_morphology(raw, expected):
    assert map_morphology(raw) == expected


# ============================================================================
# Polarity mapping
# ============================================================================

@pytest.mark.parametrize("raw,expected", [
    ("increase", "increase"),
    ("decrease", "decrease"),
    ("promote", "promote"),
    ("suppress", "suppress"),
    ("inhibit", "suppress"),
    ("unknown", "unknown"),
    ("random", "unknown"),
])
def test_map_polarity(raw, expected):
    assert map_polarity(raw) == expected


# ============================================================================
# Vocabulary integrity checks
# ============================================================================

def test_valid_elements_format():
    """All elements should be valid element symbols (1-2 letters, first uppercase)."""
    import re
    symbol_pattern = re.compile(r'^[A-Z][a-z]?$')
    for el in VALID_ELEMENTS:
        assert symbol_pattern.match(el), f"Element {el} is not a valid element symbol"


def test_allowed_causal_directions_format():
    """Ensure all causal directions follow the 'source->target' format."""
    for direction in ALLOWED_CAUSAL_DIRECTIONS:
        parts = direction.split("->")
        assert len(parts) == 2, f"Direction {direction} has wrong format"


def test_composition_units_keys():
    expected_keys = {"wt%", "weight percent", "at%", "atomic percent", "ppm", "balance"}
    assert set(COMPOSITION_UNITS.keys()) >= expected_keys
