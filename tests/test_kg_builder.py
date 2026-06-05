"""Tests for knowledge graph builder, schema, and visualization."""

import pytest
from agents.kg_schema import Node, Edge, KnowledgeGraph
from kg_construction.build_graph import KGBuilder


# ============================================================================
# Node
# ============================================================================

def test_node_creation():
    node = Node(
        node_id="n1",
        node_type="Element",
        key="Ti",
        name="Titanium",
        properties={"atomic_number": 22}
    )
    assert node.node_id == "n1"
    assert node.node_type == "Element"
    assert node.key == "Ti"
    assert node.name == "Titanium"
    assert node.properties["atomic_number"] == 22


def test_node_defaults():
    node = Node(node_id="n2", node_type="Property", key="UTS", name="Tensile Strength")
    assert node.properties == {}
    assert isinstance(node.created_at, str)


# ============================================================================
# Edge
# ============================================================================

def test_edge_creation():
    edge = Edge(
        edge_id="e1",
        relationship_type="composition_to_microstructure",
        source_node_id="n1",
        target_node_id="n2",
        confidence=0.85,
        evidence_ids=["ev1", "ev2"],
    )
    assert edge.edge_id == "e1"
    assert edge.relationship_type == "composition_to_microstructure"
    assert edge.confidence == 0.85
    assert edge.evidence_ids == ["ev1", "ev2"]


def test_edge_defaults():
    edge = Edge(
        edge_id="e2",
        relationship_type="microstructure_to_property",
        source_node_id="n1",
        target_node_id="n2",
    )
    assert edge.confidence == 0.0  # default in the Edge dataclass
    assert edge.evidence_ids == []
    assert edge.value is None
    assert edge.unit is None


# ============================================================================
# KnowledgeGraph
# ============================================================================

def test_kg_add_node():
    kg = KnowledgeGraph()
    node = Node(node_id="n1", node_type="Element", key="Ti", name="Titanium")
    kg.add_node(node)
    assert len(kg.nodes) == 1
    # nodes is a dict keyed by node_id
    assert "n1" in kg.nodes
    assert kg.nodes["n1"].key == "Ti"


def test_kg_add_edge():
    kg = KnowledgeGraph()
    src = Node(node_id="n1", node_type="Element", key="Ti", name="Titanium")
    tgt = Node(node_id="n2", node_type="Microstructure", key="equiaxed", name="Equiaxed")
    kg.add_node(src)
    kg.add_node(tgt)
    edge = Edge(
        edge_id="e1",
        relationship_type="composition_to_microstructure",
        source_node_id="n1",
        target_node_id="n2",
    )
    kg.add_edge(edge)
    assert len(kg.edges) == 1


def test_kg_get_node_by_key():
    kg = KnowledgeGraph()
    node = Node(node_id="n1", node_type="Element", key="Ti", name="Titanium")
    kg.add_node(node)
    found = kg.get_node_by_key("Element", "Ti")
    assert found is not None
    assert found.name == "Titanium"


def test_kg_get_node_by_key_not_found():
    kg = KnowledgeGraph()
    assert kg.get_node_by_key("Element", "Nonexistent") is None


# ============================================================================
# KGBuilder
# ============================================================================

def test_kgbuilder_upsert_node_new():
    builder = KGBuilder()
    node = builder.upsert_node("Element", "Ti", "Titanium", atomic_number=22)
    assert node.node_type == "Element"
    assert node.key == "Ti"
    assert node.name == "Titanium"
    assert node.properties.get("atomic_number") == 22
    assert len(builder.kg.nodes) == 1


def test_kgbuilder_upsert_node_update():
    builder = KGBuilder()
    node1 = builder.upsert_node("Element", "Ti", "Titanium", atomic_number=22)
    node2 = builder.upsert_node("Element", "Ti", "Titanium", atomic_number=22, density=4.5)
    # Should return same node (upsert) with updated properties
    assert node2.node_id == node1.node_id
    assert node2.properties.get("density") == 4.5
    assert len(builder.kg.nodes) == 1  # No duplicate


def test_kgbuilder_add_edge():
    builder = KGBuilder()
    src = builder.upsert_node("Element", "Ti", "Titanium")
    tgt = builder.upsert_node("Microstructure", "equiaxed", "Equiaxed")
    edge = builder.add_edge("composition_to_microstructure", src, tgt, confidence=0.9)
    assert edge.relationship_type == "composition_to_microstructure"
    assert edge.confidence == 0.9
    assert len(builder.kg.edges) == 1


def test_kgbuilder_add_edge_with_value():
    builder = KGBuilder()
    src = builder.upsert_node("Composition", "Al", "Aluminum")
    tgt = builder.upsert_node("Property", "UTS", "Tensile Strength")
    edge = builder.add_edge(
        "composition_to_property", src, tgt,
        confidence=0.8, value=950, unit="MPa"
    )
    assert edge.value == 950
    assert edge.unit == "MPa"
