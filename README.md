# Fantasy Football AI Assistant

An ML-powered fantasy football assistant that uses deep learning projections, analytics, and a RAG-backed natural language interface to provide draft recommendations, trade analysis, and waiver wire targets — grounded in real stats, ADP data, and historical patterns.

## What It Does

Ask questions like:

- "Who are the best value RBs to target in rounds 3-5?"
- "Is Ja'Marr Chase worth his ADP this year?"
- "How often do first-round RBs finish as RB1?"
- "Who are some late-round breakout candidates based on historical patterns?"

The system generates player projections using a deep learning model trained on historical data, computes value-based draft recommendations, and explains its reasoning using retrieved historical comparables.

## Architecture

```
Question
    ↓
┌───┴────────────────────────────────┐
│                                    │
│  ML Projections    Analytics       │
│  (PyTorch model)   (VOR, scarcity) │
│       ↓                ↓           │
│  Projected pts    Draft optimizer  │
│                                    │
│  RAG Layer                         │
│  (historical comparables)          │
│                                    │
└───┬────────────────────────────────┘
    ↓
LLM synthesis (Groq)
    ↓
Answer + Reasoning
```

## Tech Stack

| Component | Tool |
|-----------|------|
| ML framework | PyTorch |
| Data processing | Polars |
| RAG framework | LangChain |
| Vector database | ChromaDB |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM | Groq API |
| Player stats | nflreadpy |
| ADP data | FantasyPros |
| Scoring | Half-PPR |
| Testing | pytest |
| Python version | 3.14+ (see `pyproject.toml` and `.python-version`) |

## Setup

```bash
# Clone the repo
git clone https://github.com/mattvanharn/ff_ai_assistant.git
cd ff_ai_assistant

# Install uv (Arch: sudo pacman -S uv), then install dependencies
# uv picks a compatible Python using .python-version and pyproject.toml
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
| 2. Feature engineering | In progress | ML features from historical data |
| 3. Projection model | Planned | PyTorch model for next-season fantasy points |
| 4. Analytics layer | Planned | VOR, draft optimizer, trade evaluator |
| 5. RAG + LLM interface | Planned | Natural language Q&A over projections + history |

## License

MIT
