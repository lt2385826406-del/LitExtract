"""Pytest configuration and shared fixtures for LitExtract tests."""

import os
import sys
import pytest
from pathlib import Path


# Add project source directory to Python path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture
def sample_semantic_data():
    """Minimal semantic extraction JSON fixture for testing."""
    return [
        {
            "paper_id": "P001",
            "sample_id": "S1",
            "composition": {"Ti": "balance", "Al": "6.0", "V": "4.0"},
            "process": [
                {"method": "forging", "temperature": "950C"},
                {"method": "annealing", "temperature": "800C", "time": "2h"}
            ],
            "microstructure": {
                "phase": "alpha+beta",
                "morphology": "equiaxed",
                "grain_size": "10um"
            },
            "property": [
                {"type": "UTS", "value": 950, "unit": "MPa"},
                {"type": "EL", "value": 12, "unit": "%"}
            ],
            "causal_relations": [
                {"cause": "annealing", "effect": "equiaxed", "polarity": "promote"},
                {"cause": "equiaxed", "effect": "EL", "polarity": "increase"}
            ]
        },
        {
            "paper_id": "P002",
            "sample_id": "S1",
            "composition": {"Ti": "balance", "Al": "6.0", "V": "4.0", "B": "0.1"},
            "process": [
                {"method": "forging", "temperature": "1050C"},
                {"method": "aging", "temperature": "500C", "time": "8h"}
            ],
            "microstructure": {
                "phase": "alpha+beta",
                "morphology": "lamellar",
                "grain_size": "5um"
            },
            "property": [
                {"type": "UTS", "value": 1050, "unit": "MPa"},
                {"type": "EL", "value": 8, "unit": "%"}
            ],
            "causal_relations": [
                {"cause": "forging_temperature", "effect": "lamellar", "polarity": "promote"},
                {"cause": "lamellar", "effect": "UTS", "polarity": "increase"},
                {"cause": "B_addition", "effect": "grain_refinement", "polarity": "promote"}
            ]
        }
    ]


@pytest.fixture
def sample_composition():
    """Sample composition dict for testing."""
    return {"Ti": "balance", "Al": "6.0", "V": "4.0", "Fe": "0.05", "O": "0.15"}


@pytest.fixture
def sample_knowledge_graph_data():
    """Minimal KG data for DAG tests."""
    return {
        "nodes": [
            {"id": "Ti", "type": "Element", "name": "Titanium"},
            {"id": "Al", "type": "Element", "name": "Aluminum"},
            {"id": "V", "type": "Element", "name": "Vanadium"},
            {"id": "alpha_beta", "type": "Microstructure", "name": "alpha+beta"},
            {"id": "equiaxed", "type": "Microstructure", "name": "equiaxed"},
            {"id": "UTS", "type": "Property", "name": "Ultimate Tensile Strength"},
        ],
        "edges": [
            {"source": "Al", "target": "alpha_beta", "weight": 0.85, "relationship_type": "composition_to_microstructure"},
            {"source": "V", "target": "alpha_beta", "weight": 0.75, "relationship_type": "composition_to_microstructure"},
            {"source": "alpha_beta", "target": "equiaxed", "weight": 0.60, "relationship_type": "microstructure_to_microstructure"},
            {"source": "equiaxed", "target": "UTS", "weight": 0.70, "relationship_type": "microstructure_to_property"},
        ]
    }
