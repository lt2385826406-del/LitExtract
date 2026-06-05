"""Tests for causal DAG builder module."""

import pytest
import networkx as nx
from dag_construction.dag_causal_builder import (
    CausalHypothesisGraph,
    ClaimType,
    Polarity,
    Strength,
    NODE_ORDER,
)


# ============================================================================
# NODE_ORDER
# ============================================================================

def test_node_order_has_expected_types():
    expected = ["Element", "Alloy", "Processing", "Microstructure", "Property"]
    assert NODE_ORDER == expected


def test_node_order_is_causal_chain():
    """The order should represent a causal chain: composition → processing → microstructure → property."""
    micro_idx = NODE_ORDER.index("Microstructure")
    element_idx = NODE_ORDER.index("Element")
    processing_idx = NODE_ORDER.index("Processing")
    property_idx = NODE_ORDER.index("Property")
    
    # Element and Processing come before Microstructure
    assert element_idx < micro_idx
    assert processing_idx < micro_idx
    # Microstructure comes before Property
    assert micro_idx < property_idx


# ============================================================================
# ClaimType, Polarity, Strength enums
# ============================================================================

def test_claim_type_values():
    values = {c.value for c in ClaimType}
    assert "explicit_causal" in values
    assert "contrast_based" in values
    assert "mechanism_supported" in values
    assert "cooccurrence" in values


def test_polarity_values():
    values = {p.value for p in Polarity}
    assert "promote" in values
    assert "suppress" in values
    assert "increase" in values
    assert "decrease" in values


def test_strength_values():
    values = {s.value for s in Strength}
    assert "weak" in values
    assert "medium" in values
    assert "strong" in values


# ============================================================================
# CausalHypothesisGraph
# ============================================================================

@pytest.fixture
def chg():
    return CausalHypothesisGraph()


def test_chg_initialization(chg):
    assert isinstance(chg.cg, nx.MultiDiGraph)
    assert isinstance(chg.evidence_map, dict)
    assert len(chg.cg.nodes) == 0
    assert len(chg.cg.edges) == 0


def test_generate_edge_id(chg):
    eid = chg._generate_edge_id("Ti", "UTS", "increase", "explicit_causal")
    assert eid == "Ti->UTS:increase:explicit_causal"


def test_calculate_strength_strong(chg):
    result = chg._calculate_strength("mechanism_supported", 0.9, 3)
    assert result == "strong"


def test_calculate_strength_medium(chg):
    result = chg._calculate_strength("explicit_causal", 0.6, 1)
    assert result == "medium"


def test_calculate_strength_weak(chg):
    result = chg._calculate_strength("contrast_based", 0.3, 0)
    assert result == "weak"


def test_add_causal_edge(chg):
    evidence_ids = ["ev1", "ev2"]
    chg.add_causal_edge(
        src="Ti",
        dst="UTS",
        polarity="increase",
        claim_type="explicit_causal",
        confidence=0.8,
        evidence_ids=evidence_ids,
        evidence_text="Ti increases UTS"
    )
    assert chg.cg.has_edge("Ti", "UTS")
    # Evidence is stored in edge attributes, not in evidence_map directly
    edge_data = chg.cg["Ti"]["UTS"]
    assert len(edge_data) >= 1


def test_add_causal_edge_multi(chg):
    """Adding two edges between same nodes with different polarities creates two edges."""
    chg.add_causal_edge("Ti", "UTS", "increase", "explicit_causal", 0.8, ["ev1"])
    chg.add_causal_edge("Ti", "UTS", "decrease", "contrast_based", 0.4, ["ev2"])
    # MultiDiGraph should store both edges
    assert chg.cg.number_of_edges() == 2


def test_add_causal_edge_duplicate_merges(chg):
    """Adding same edge twice should merge evidence and average confidence."""
    chg.add_causal_edge("Ti", "UTS", "increase", "explicit_causal", 0.6, ["ev1"])
    chg.add_causal_edge("Ti", "UTS", "increase", "explicit_causal", 0.8, ["ev2"])
    # Same edge key → should still be 1 edge (merged)
    assert chg.cg.number_of_edges() == 1
