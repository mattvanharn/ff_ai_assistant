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

# TODO Step 1: SQL generation prompt
# TODO Step 2: Answer prompt
# TODO Step 3: get_sql_chain()
# TODO Step 4: ask_sql() convenience wrapper
