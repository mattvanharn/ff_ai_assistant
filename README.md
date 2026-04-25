# Fantasy Football AI Assistant

[![CI](https://github.com/mattvanharn/ff_ai_assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/mattvanharn/ff_ai_assistant/actions)

An ML-powered fantasy football assistant that uses a natural language interface over structured historical data, deep learning projections, and analytics to provide draft recommendations, trade analysis, and waiver wire targets.

## What It Does

Ask questions like:

- "Who were the top 5 RBs in 2024?"
- "Which players had an ADP under 50 but finished outside the top 100?"
- "Compare Saquon Barkley's last 3 seasons side by side"
- "How many WRs scored over 200 half-PPR points in the last 5 years?"

The system converts natural language into SQL, executes it against the historical database, and summarizes the results in plain English. Future phases add ML projections and analytics (VOR, draft optimizer, trade evaluator), followed by a lightweight web app demo layer.

## Architecture

```
Question
    ↓
LLM router (future: SQL vs RAG)
    ↓
┌─────────────────────────────────────┐
│  Text-to-SQL (Phase 1 — current)    │
│  NL → SQL → SQLite → LLM summary   │
│                                     │
│  ML Projections (Phase 2)           │
│  Features → PyTorch → projected pts │
│                                     │
│  Analytics (Phase 2)                │
│  VOR, draft optimizer, trade eval   │
│                                     │
│  Web App Demo (Phase 3)             │
│  Streamlit/FastAPI UI showcase      │
│                                     │
│  RAG (Phase 4 — deferred)           │
│  In-season articles & reports       │
└─────────────────────────────────────┘
    ↓
Answer
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Data processing | Polars |
| Database | SQLite (in-memory from parquets) |
| LLM | Groq API |
| NL interface | LangChain |
| ML (primary) | XGBoost / LightGBM (Phase 2) |
| ML (comparison) | PyTorch feedforward NN (Phase 2) |
| Player stats | nflreadpy |
| ADP data | FantasyPros |
| Scoring | Half-PPR |
| Testing | pytest |
| Package manager | uv |
| Python version | 3.14+ |

## Setup

```bash
git clone https://github.com/mattvanharn/ff_ai_assistant.git
cd ff_ai_assistant

# Install uv (Arch: sudo pacman -S uv), then install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)
```

Run scripts with `uv run python scripts/fetch_stats.py`.

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Data pipeline | Done | Fetch scripts, exploration notebook, processed parquets |
| 2. Text-to-SQL | In progress | NL → SQL → answer over historical stats |
| 3. Feature engineering | Planned | ML features from historical + weekly data |
| 4. Projection model | Planned | PyTorch model for next-season fantasy points |
| 5. Analytics layer | Planned | VOR, draft optimizer, trade evaluator |
| 6. Web app demo layer | Planned | Lightweight UI to demo core capabilities |
| 7. RAG (deferred) | Future | In-season unstructured text when corpus exists |

## License

MIT
