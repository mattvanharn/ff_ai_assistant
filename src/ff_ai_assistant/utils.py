# Utility functions for the project
import re


def normalize_player_name(name: str) -> str:
    """
    Normalize player names so ADP and nflreadpy spellings match:
    - Remove apostrophes (Le'Veon -> LeVeon) so variants match
    - Remove periods (A.J. Green -> AJ Green)
    - Merge spaced initials (A. J. Green -> A J Green -> AJ Green) so both spellings match
    - Standardize whitespace to single spaces
    - Remove suffixes (Jr., Sr., II, III, IV, V) only at the end of the name
    - Lowercase the name
    """
    name = name.strip()
    # Remove apostrophes so "Le'Veon" and "LeVeon" normalize the same
    name = re.sub(r"'", "", name)
    # Remove periods
    name = re.sub(r"\.", "", name)
    # Standardize whitespace
    name = re.sub(r"\s+", " ", name)
    # Merge spaced single-letter initials (A J Green -> AJ Green) so "A. J. Green" matches "A.J. Green"
    name = re.sub(r"\b([a-zA-Z]) ([a-zA-Z])\b", r"\1\2", name)
    # Remove suffixes only if they're at the end, possibly after a space
    name = re.sub(r"(?:\s+)(Jr|Sr|III|II|IV|V)$", "", name, flags=re.I)
    # Remove any trailing whitespace left after removing the suffix
    name = name.strip()
    return name.lower()
