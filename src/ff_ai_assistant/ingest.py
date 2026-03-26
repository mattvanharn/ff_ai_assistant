"""Document ingestion for RAG pipeline (DEFERRED).

This module is reserved for a future phase when truly unstructured text
exists (in-season articles, Reddit/Twitter, beat writer notes).

The previous implementation converted structured player-season stats into
templated text documents for embedding. This was abandoned because all
documents followed the same template, making them semantically similar to
the embedding model despite containing different stats. Ranking and
comparison queries produced poor retrieval results.

Text-to-SQL (sql_chain.py) is the current approach for structured data.
RAG will be revisited when an unstructured corpus exists.
"""
