"""Tests for ff_ai_assistant.utils."""

import pytest

from ff_ai_assistant.utils import normalize_player_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Kenneth Walker III", "kenneth walker"),
        ("Odell Beckham Jr.", "odell beckham"),
        ("A.J. Green", "aj green"),
        ("Patrick Mahomes II", "patrick mahomes"),
        ("Le'Veon Bell", "leveon bell"),
        ("  spaced  name  ", "spaced name"),
    ],
)
def test_normalize_player_name_examples(raw: str, expected: str) -> None:
    """Known inputs from docstring / notebook usage should normalize consistently."""
    assert normalize_player_name(raw) == expected
