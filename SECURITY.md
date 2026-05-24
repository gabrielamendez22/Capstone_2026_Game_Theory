# SECURITY.md — API Key and Data Discipline

Read this before touching anything related to API keys, credentials, or raw data.

---

## API Keys

### Where They Live

All API keys live in `.env` in the project root. **This file is in `.gitignore`
and must never be committed.**

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

### How to Load Them

```python
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("ANTHROPIC_API_KEY")
```

### What Never Happens

- Keys never appear in source code
- Keys never appear in comments
- Keys never appear in print statements or logs
- Keys never appear in CSV, JSON, or database output
- Keys never appear in git commits, diffs, or PR descriptions
- If a key is accidentally committed: rotate it immediately at the provider console

---

## Raw Data Files

Files in `data/raw/` are **read-only**. Do not:
- Modify any `.json`, `.csv`, or `.db` file in `data/raw/`
- Delete raw output files
- Overwrite existing runs with new runs

If you need to reprocess data, copy it to `data/processed/` first.

---

## Database Files

SQLite `.db` files are written by the experiment scripts.
- Do not run `DROP TABLE` or `DELETE FROM` on live experiment databases
- Do not write to a database that is currently being written by a running experiment
- For analysis, open databases in **read-only mode**:

```python
conn = sqlite3.connect("file:experiment.db?mode=ro", uri=True)
```

---

## Git Hygiene

The `.gitignore` must always include:

```
.env
*.db
*.log
__pycache__/
.DS_Store
```

Data CSVs from `data/raw/` should be committed only intentionally, not by default.
Large result files (>10MB) should not be committed at all — use external storage.

---

## LLM API Costs

These experiments make real API calls that incur real costs.
Before running any experiment:

1. Verify `TOTAL_ROUNDS` and the number of `MATCHUPS`
2. Estimate: `cost ≈ MATCHUPS × TOTAL_ROUNDS × 2 calls × avg_tokens × price_per_token`
3. Start with a **single matchup, 5 rounds** to validate before scaling
4. Monitor token usage in the `token_usage_a/b` columns of the output

Never run the full matchup matrix against the most expensive models (Opus, GPT-4o)
without reviewing estimated costs first.