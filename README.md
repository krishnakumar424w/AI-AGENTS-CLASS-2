# Placement Assistant: Day 2 programming kit

SoDak EduTech, Agentic AI Track, Day 2. Open `HANDOUT.html` in a browser for the full exercise sheet.

You need Python 3.10+ and either a Groq API key or a Gemini API key. No database server: the placement data lives in memory
and the agent's memory uses SQLite, which comes with Python.

## What you build

| Part | You write | Where |
|---|---|---|
| 1 Tools | `get_student`, `list_open_drives`, `book_interview_slot` (two sample tools are given) | `app/tools/placement_tools.py` |
| 2 A small agent | `run_tool` and the loop in `ask` | `app/agent.py` |
| 3 Memory | the schema, the SQL, and wiring memory into the agent | `schema/agent.sql`, `app/memory.py`, `app/agent.py` |
| Lab | `notify_student`, append-only history, paging through history | see the handout |

## Start

```bash
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Preferred: Groq
export GROQ_API_KEY=your-groq-key        # Windows: set GROQ_API_KEY=your-groq-key
export GROQ_MODEL=llama-3.3-70b-versatile

# Fallback: Gemini
export GEMINI_API_KEY=your-gemini-key    # Windows: set GEMINI_API_KEY=your-gemini-key

pytest tests/test_part1_tools.py         # the sample tools pass already
python -m scripts.chat --mock            # after Part 2: talk to your agent, no quota
python -m scripts.chat                   # uses Groq if GROQ_API_KEY is set, otherwise Gemini
python -m scripts.chat --db agent.db     # after Part 3: it remembers
```

Tests never call the model provider, so they never use your quota.
