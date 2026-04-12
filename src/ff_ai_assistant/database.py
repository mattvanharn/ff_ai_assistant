"""DuckDB database layer for text-to-SQL queries.

Queries parquet files directly via DuckDB's native parquet support —
no pandas conversion needed.

Tables created:
    player_seasons — one row per player per season (points, ranks, ADP, etc.)
    weekly_stats   — one row per player per week (detailed game-level stats)
"""

import duckdb

from ff_ai_assistant.config import (
    COMBINED_PARQUET,
    WEEKLY_PARQUET,
    POSITIONS,
)

_WEEKLY_STATS_SCHEMA = """Table: weekly_stats
    -- Identity (all players)
    player_id (VARCHAR), player_name (VARCHAR), player_display_name (VARCHAR)
    position (VARCHAR), season (BIGINT), week (BIGINT)
    team (VARCHAR), opponent_team (VARCHAR), game_id (VARCHAR)

    -- Fantasy scoring
    fantasy_points_half_ppr (DOUBLE), fantasy_points (DOUBLE),
  fantasy_points_ppr (DOUBLE)

    -- Passing (QBs; occasionally WR/RB on trick plays)
    completions (BIGINT), attempts (BIGINT), passing_yards (BIGINT),
  passing_tds (BIGINT)
    passing_interceptions (BIGINT), passing_air_yards (BIGINT),
  passing_yards_after_catch (BIGINT)
    passing_first_downs (BIGINT), passing_epa (DOUBLE), passing_cpoe (DOUBLE)
    passing_2pt_conversions (BIGINT), sacks_suffered (BIGINT),
  sack_yards_lost (BIGINT)
    sack_fumbles (BIGINT), sack_fumbles_lost (BIGINT), pacr (DOUBLE)

    -- Rushing (QBs and RBs; occasionally WR/TE on designed plays)
    carries (BIGINT), rushing_yards (BIGINT), rushing_tds (BIGINT)
    rushing_fumbles (BIGINT), rushing_fumbles_lost (BIGINT)
    rushing_first_downs (BIGINT), rushing_epa (DOUBLE),
  rushing_2pt_conversions (BIGINT)
    special_teams_tds (BIGINT)

    -- Receiving (WRs, TEs, RBs)
    receptions (BIGINT), targets (BIGINT), receiving_yards (BIGINT),
  receiving_tds (BIGINT)
    receiving_air_yards (BIGINT), receiving_yards_after_catch (BIGINT)
    receiving_fumbles (BIGINT), receiving_fumbles_lost (BIGINT)
    receiving_first_downs (BIGINT), receiving_epa (DOUBLE),
  receiving_2pt_conversions (BIGINT)
    racr (DOUBLE), target_share (DOUBLE), air_yards_share (DOUBLE), wopr
  (DOUBLE)

    -- Kicking (Ks only)
    fg_made (BIGINT), fg_att (BIGINT), fg_missed (BIGINT), fg_blocked
  (BIGINT)
    fg_long (BIGINT), fg_pct (DOUBLE)
    fg_made_0_19 (BIGINT), fg_made_20_29 (BIGINT), fg_made_30_39 (BIGINT)
    fg_made_40_49 (BIGINT), fg_made_50_59 (BIGINT), fg_made_60_ (BIGINT)
    fg_missed_0_19 (BIGINT), fg_missed_20_29 (BIGINT), fg_missed_30_39
  (BIGINT)
    fg_missed_40_49 (BIGINT), fg_missed_50_59 (BIGINT), fg_missed_60_
  (BIGINT)
    fg_made_distance (BIGINT), fg_missed_distance (BIGINT),
  fg_blocked_distance (BIGINT)
    fg_made_list (VARCHAR), fg_missed_list (VARCHAR), fg_blocked_list
  (VARCHAR)
    pat_made (BIGINT), pat_att (BIGINT), pat_missed (BIGINT)
    pat_blocked (BIGINT), pat_pct (DOUBLE)

    -- Note: individual defensive player stats available via def_* columns
  (not standard fantasy)"""


def _create_connection() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB database and load parquet data into it."""
    conn = duckdb.connect()

    positions_sql = ", ".join(f"'{p}'" for p in POSITIONS)

    conn.execute(f"""
        CREATE TABLE player_seasons AS
        SELECT * FROM read_parquet('{COMBINED_PARQUET}')
        WHERE position IN ({positions_sql})
    """)

    conn.execute(f"""
        CREATE TABLE weekly_stats AS
        SELECT * FROM read_parquet('{WEEKLY_PARQUET}')
        WHERE position IN ({positions_sql})
    """)

    return conn


_conn: duckdb.DuckDBPyConnection | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
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
    # player_seasons: show all columns via DESCRIBE (25 columns, all relevant)
    columns = conn.execute("DESCRIBE player_seasons").fetchall()
    col_lines = [f" {col[0]} ({col[1]})" for col in columns]
    player_seasons_schema = "Table: player_seasons\n" + "\n".join(col_lines)

    return player_seasons_schema + "\n\n" + _WEEKLY_STATS_SCHEMA


def get_sample_rows(table: str, n: int = 3) -> str:
    """Return a few sample rows from a table, formatted for the LLM prompt.

    Seeing real data helps the LLM understand column contents (e.g. that
    'position' contains 'QB', 'RB', not full words like 'Quarterback').
    """
    conn = get_connection()
    result = conn.execute(f"SELECT * FROM {table} LIMIT {n}")  # noqa: S608
    col_names = [desc[0] for desc in result.description]
    rows = result.fetchall()

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
    result = conn.execute(sql)
    col_names = [desc[0] for desc in result.description]
    rows = result.fetchall()
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
