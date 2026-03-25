"""Tests for ff_ai_assistant.config."""

from pathlib import Path

from ff_ai_assistant import config


def test_project_root_is_absolute_and_looks_like_repo() -> None:
    """Config resolves repo root from __file__; it should exist and hold pyproject.toml."""
    root = config.PROJECT_ROOT
    assert isinstance(root, Path)
    assert root.is_absolute()
    assert (root / "pyproject.toml").is_file()


def test_data_paths_are_under_project_root() -> None:
    """Processed data and chroma paths should live inside the repo, not arbitrary dirs."""
    assert config.PROCESSED_DATA_DIR.is_relative_to(config.PROJECT_ROOT)
    assert config.CHROMA_DIR.is_relative_to(config.PROJECT_ROOT)
    assert config.COMBINED_PARQUET.name == "combined_stats_adp.parquet"


def test_league_constants() -> None:
    """Smoke check for documented defaults (half-PPR, 12-team)."""
    assert config.SCORING_FORMAT == "half_ppr"
    assert config.LEAGUE_SIZE == 12
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
