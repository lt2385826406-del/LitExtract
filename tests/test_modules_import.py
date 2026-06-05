"""Smoke tests for PDF parser and vision modules."""

import pytest


# ============================================================================
# Import checks — pdf_parser
# ============================================================================

def test_import_pdf_to_images():
    """Verify pdf_to_images module can be imported."""
    from pdf_parser import pdf_to_images
    assert pdf_to_images is not None


# ============================================================================
# Import checks — vision
# ============================================================================

def test_import_image_preprocessing():
    """Verify image_preprocessing module can be imported."""
    from vision import image_preprocessing
    assert image_preprocessing is not None


def test_import_figure_matcher():
    """Verify figure_matcher module can be imported."""
    from vision import figure_matcher
    assert figure_matcher is not None


# ============================================================================
# Import checks — validation
# ============================================================================

def test_import_statistics():
    """Verify statistics module can be imported."""
    from validation import statistics
    assert statistics is not None


# ============================================================================
# Import checks — kg_construction
# ============================================================================

def test_import_export_neo4j():
    """Verify export_neo4j module can be imported."""
    from kg_construction import export_neo4j
    assert export_neo4j is not None


# ============================================================================
# Import checks — ocr_alignment
# ============================================================================

def test_import_subgraph_alignment():
    """Verify subgraph_alignment module can be imported."""
    from ocr_alignment import subgraph_alignment
    assert subgraph_alignment is not None
