"""Text-to-SQL chain: natural language → SQL → results → LLM answer.

Two-step LLM process:
    1. User question + schema/samples → LLM generates a SQL query
    2. SQL results + original question → LLM writes a natural-language answer

You'll build this file in stages:
    Step 1: Write SQL_GENERATION_PROMPT (tell the LLM about the schema and rules)
    Step 2: Write ANSWER_PROMPT (tell the LLM to summarize results)
    Step 3: Implement get_sql_chain() (wire prompts → LLM → database → LLM)
    Step 4: Test with the __main__ block
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq
from ff_ai_assistant.config import GROQ_MODEL, GROQ_TEMPERATURE

from ff_ai_assistant.database import (
    get_schema,
    execute_query,
    format_results,
)

load_dotenv()

SQL_GENERATION_PROMPT = """You are an expert DuckDB SQL analyst for a fantasy football database.
Given a question, write a syntactically correct DuckDB SQL query that answers it.

Database schema:
{schema}

Rules:
- Return ONLY the SQL query. No explanation, no markdown fences, no prose.
- Use exact column names from the schema above. Do not invent column names.
- Do not hallucinate table names. Only use the tables listed in the schema.
- Use player_display_name for player name lookups (not player_name).
- Use player_seasons for season totals and rankings; use weekly_stats for weekly or game-level questions.
- Always include LIMIT for open-ended queries. Default to 12 unless the question specifies a number.
- Do not use LIMIT when the question asks for all of a player's stats across a full season or all weeks.
- For player names with apostrophes, use two single quotes: 'Ja''Marr Chase'.
- Use DuckDB/PostgreSQL syntax. Do not use SQLite-specific functions.

Domain notes:
- ADP = Average Draft Position. Lower ADP = earlier draft pick = higher expected value.
- Scoring is half-PPR (0.5 points per reception).
- Fantasy points columns: fantasy_points_half_ppr (weekly), seasonal_fantasy_points (season total).
- Positions: QB, RB, WR, TE, K, DST. League size: 12 teams, seasons 2018-2025.

Examples:
Q: How many games did Christian McCaffrey play in 2024?
SQL: SELECT COUNT(*) AS games_played FROM weekly_stats WHERE player_display_name = 'Christian McCaffrey' AND season = 2024;

Q: Who were the top 10 QBs by half-PPR fantasy points in 2024?
SQL: SELECT player_display_name, seasonal_fantasy_points FROM player_seasons WHERE position = 'QB' AND season = 2024 ORDER BY seasonal_fantasy_points DESC LIMIT 10;

Q: Give me Ja'Marr Chase's weekly scoring for all of 2023.
SQL: SELECT week, fantasy_points_half_ppr FROM weekly_stats WHERE player_display_name = 'Ja''Marr Chase' AND season = 2023 ORDER BY week;

Q: Give me all of the players with 35+ point weeks in 2025.
SQL: SELECT player_display_name, position, week, fantasy_points_half_ppr FROM weekly_stats WHERE fantasy_points_half_ppr >= 35 AND season = 2025 ORDER BY fantasy_points_half_ppr DESC;

Question: {question}
SQL:"""


ANSWER_PROMPT = """You are a fantasy football assistant answering questions about player stats.

A SQL query was run to answer the following question: {question}

This is the SQL query that was run: {sql}

This is the result of the SQL query: {results}

If the results of the query are empty then simply say that no players fit that criteria
(i.e., if the query is asking for all players who scored 100+ fantasy points in a week in 2025,
then say "No players scored 100+ fantasy points in a single week in 2025.")

Summarize the query results as a natural language answer.
"""


def extract_select_sql(raw: str) -> str:
    """Strip common LLM noise (think tags, markdown fences, prose) from SQL output."""

    return re.sub(r"```(?:sql)?\n?", "", raw).strip("`").strip()


def get_sql_chain():
    """Build the text-to-SQL chain with retry logic"""

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    schema = get_schema()

    def chain(question: str) -> str:
        # print("===DEBUG===")
        # print(f"Question: {question}")
        sql_prompt = SQL_GENERATION_PROMPT.format(schema=schema, question=question)
        # print(f"SQL Prompt: {sql_prompt}")

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.0,
            messages=[{"role": "user", "content": sql_prompt}],
        )
        sql = response.choices[0].message.content.strip()
        sql = extract_select_sql(sql)
        # print(f"SQL: {sql}")

        try:
            results = execute_query(sql)
            # print(f"Results: {results}")
        except Exception as e:
            retry_prompt = (
                f"The following SQL query failed with this error:\n{e}\n\n"
                f"Broken query:\n{sql}\n\n"
                f"Rewrite the query to fix the error. Return only the corrected SQL."
            )
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.0,
                messages=[{"role": "user", "content": retry_prompt}],
            )
            sql = response.choices[0].message.content.strip()
            sql = extract_select_sql(sql)

            results = execute_query(sql)  # let this one raise if it fails again

        formatted = format_results(results=results)

        answer_prompt = ANSWER_PROMPT.format(
            question=question, sql=sql, results=formatted
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=GROQ_TEMPERATURE,
            messages=[{"role": "user", "content": answer_prompt}],
        )
        return response.choices[0].message.content.strip()

    return chain


def ask_sql(question: str) -> str:
    """Ask a structured question via text-to-SQL"""

    chain = get_sql_chain()
    return chain(question)


if __name__ == "__main__":
    print("FF AI Assistant — Text-to-SQL Mode")
    print("Ask precise questions about player stats, rankings, comparisons.")
    print("Type 'quit' to exit.\n")

    chain = get_sql_chain()
    while True:
        question = input("Question: ")
        if question.lower() in ("quit", "exit", "q"):
            break
        try:
            answer = chain(question)
            print(f"\n{answer}\n")
        except Exception as e:
            print(f"\nError: {e}\n")
