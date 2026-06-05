"""Tests for co-occurrence miner constants and basic logic."""

import pytest
from dag_construction.cooccurrence_miner import (
    ENTITY_GROUPS,
    ALLOWED_CROSS_GROUP_PAIRS,
    ALLOWED_SAME_GROUP_PAIRS,
    _LOW_FREQ_ELEMENT_SYMBOLS,
)


# ============================================================================
# Entity group mapping
# ============================================================================

def test_entity_groups_cover_all_types():
    """Ensure entity group mappings cover expected entity types."""
    expected_types = {"Element", "Alloy", "Processing", "Microstructure", "Property"}
    assert set(ENTITY_GROUPS.keys()) == expected_types


def test_entity_groups_map_to_valid_groups():
    """Groups should map to known domains."""
    valid_groups = {"composition", "processing", "microstructure", "property"}
    for group_name in ENTITY_GROUPS.values():
        assert group_name in valid_groups


# ============================================================================
# Cross-group pair constraints
# ============================================================================

def test_cross_group_pairs_use_known_types():
    known_types = set(ENTITY_GROUPS.keys())
    for src, tgt in ALLOWED_CROSS_GROUP_PAIRS:
        assert src in known_types, f"Unknown source type: {src}"
        assert tgt in known_types, f"Unknown target type: {tgt}"


def test_cross_group_pairs_are_distinct():
    """Each pair should be a distinct direction."""
    assert len(ALLOWED_CROSS_GROUP_PAIRS) == len(set(ALLOWED_CROSS_GROUP_PAIRS))


def test_same_group_pairs_allowed():
    """Same-group pairs should be allowed for aggregation."""
    assert ALLOWED_SAME_GROUP_PAIRS is True


# ============================================================================
# Low-frequency element symbols
# ============================================================================

def test_low_freq_elements_set_type():
    assert isinstance(_LOW_FREQ_ELEMENT_SYMBOLS, set)


def test_low_freq_elements_have_known_candidates():
    """B, Nb, Zr and other micro-alloying elements should be present."""
    expected = {"B", "Nb", "Zr", "Ta", "W", "O", "N", "H"}
    assert expected <= _LOW_FREQ_ELEMENT_SYMBOLS
