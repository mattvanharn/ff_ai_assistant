"""SQLite database layer for text-to-SQL queries.

Loads processed parquet files into an in-memory SQLite database so the LLM
can generate and execute SQL queries against structured fantasy football data.

Tables created:
    player_seasons — one row per player per season (points, ranks, ADP, etc.)
    weekly_stats   — one row per player per week (detailed game-level stats)
"""

import sqlite3

import polars as pl

from ff_ai_assistant.config import (
    COMBINED_PARQUET,
    WEEKLY_PARQUET,
    POSITIONS,
)


def _create_connection() -> sqlite3.Connection:
    """Create an in-memory SQLite database and load parquet data into it."""
    conn = sqlite3.connect(":memory:")

    combined = pl.read_parquet(COMBINED_PARQUET)
    combined = combined.filter(pl.col("position").is_in(POSITIONS))
    combined.to_pandas().to_sql("player_seasons", conn, index=False, if_exists="replace")

    weekly = pl.read_parquet(WEEKLY_PARQUET)
    weekly = weekly.filter(pl.col("position").is_in(POSITIONS))
    weekly.to_pandas().to_sql("weekly_stats", conn, index=False, if_exists="replace")

    return conn


_conn: sqlite3.Connection | None = None


def get_connection() -> sqlite3.Connection:
    """Return a cached database connection (created once per process)."""
    global _conn  # noqa: PLW0603
    if _conn is None:
        _conn = _create_connection()
    return _conn


def get_schema() -> str:
    """Return a human-readable schema description for the LLM prompt.

    The LLM needs table names, column names, and types to write valid SQL.
    This introspects the database and formats the schema as text.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        col_lines = [f"  {col[1]} ({col[2]})" for col in columns]
        parts.append(f"Table: {table}\n" + "\n".join(col_lines))

    return "\n\n".join(parts)


def get_sample_rows(table: str, n: int = 3) -> str:
    """Return a few sample rows from a table, formatted for the LLM prompt.

    Seeing real data helps the LLM understand column contents (e.g. that
    'position' contains 'QB', 'RB', not full words like 'Quarterback').
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table} LIMIT {n}")  # noqa: S608
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]

    lines = [", ".join(col_names)]
    for row in rows:
        lines.append(", ".join(str(v) for v in row))
    return "\n".join(lines)


def execute_query(sql: str) -> list[dict]:
    """Execute a SELECT query and return results as a list of dicts.

    Only SELECT statements are allowed — prevents the LLM from modifying data.

    Raises:
        ValueError: If the query is not a SELECT statement.
    """
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(sql)

    col_names = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return [dict(zip(col_names, row)) for row in rows]


def format_results(results: list[dict], max_rows: int = 20) -> str:
    """Format query results as readable text for the LLM answer prompt."""
    if not results:
        return "(No results)"

    display = results[:max_rows]
    lines = []
    for row in display:
        parts = [f"{k}: {v}" for k, v in row.items()]
        lines.append(" | ".join(parts))

    text = "\n".join(lines)
    if len(results) > max_rows:
        text += f"\n... ({len(results) - max_rows} more rows)"
    return text


if __name__ == "__main__":
    print("=== Schema ===")
    print(get_schema())
    print("\n=== Sample rows (player_seasons) ===")
    print(get_sample_rows("player_seasons"))
    print("\n=== Test query ===")
    results = execute_query(
        "SELECT player_display_name, position, season, seasonal_fantasy_points "
        "FROM player_seasons WHERE position = 'RB' AND season = 2024 "
        "ORDER BY seasonal_fantasy_points DESC LIMIT 5"
    )
    for row in results:
        print(row)
