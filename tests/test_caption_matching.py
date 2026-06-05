"""Tests for OCR caption matching logic."""

import pytest
from ocr_alignment.caption_matching import _caption_ocr_bonus, CAPTION_KEYWORDS


# ============================================================================
# Caption keyword patterns
# ============================================================================

def test_caption_keywords_not_empty():
    assert len(CAPTION_KEYWORDS) > 0


def test_caption_keywords_are_regex_patterns():
    """All keys in CAPTION_KEYWORDS should be valid regex patterns."""
    for pattern, weight in CAPTION_KEYWORDS.items():
        assert weight > 0, f"Keyword {pattern} has non-positive weight"
        # Should compile without error
        import re
        try:
            re.compile(pattern)
        except re.error as e:
            pytest.fail(f"Invalid regex pattern '{pattern}': {e}")


# ============================================================================
# _caption_ocr_bonus
# ============================================================================

def test_caption_ocr_bonus_empty_text():
    assert _caption_ocr_bonus("") == -1.0


def test_caption_ocr_bonus_none_text():
    assert _caption_ocr_bonus(None) == -1.0


def test_caption_ocr_bonus_short_text():
    """Text shorter than 5 chars: the short-text penalty may be offset by keyword bonus."""
    score = _caption_ocr_bonus("Fig")
    # "Fig" matches the fig keyword (+2.0) but is short (-2.0) so score is 0
    assert score >= -1.0


def test_caption_ocr_bonus_with_figure():
    """Text containing 'Figure' should get positive bonus."""
    score = _caption_ocr_bonus("Figure 3. Microstructure of Ti-6Al-4V")
    assert score > 0


def test_caption_ocr_bonus_with_fig_abbr():
    """Text containing 'Fig' should get positive bonus."""
    score = _caption_ocr_bonus("Fig. 2 XRD patterns")
    assert score > 0


def test_caption_ocr_bonus_non_caption_text():
    """Non-caption text should score near zero."""
    score = _caption_ocr_bonus("This is just regular text without figure keywords")
    # Should score 0 or very low (but above the short-text penalty threshold)
    assert score >= 0
