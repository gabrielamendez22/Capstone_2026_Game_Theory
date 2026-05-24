---
name: experiment-runner
description: Use when running, modifying, debugging, or extending an experiment script in /experiments/ — including PD, Commons, or Cheap-Talk. Covers pre-flight cost checks, condition switching, prompt versioning, model registry updates, and common LLM parsing failures.
---

# Experiment Runner Skill

## Pre-Flight Checklist (run BEFORE any full experiment)

Before executing any experiment script, verify:

1. **`OPPONENT_CONDITION`** is set correctly at the top of the script
   (`"undisclosed"`, `"ai"`, or `"human"`)
2. **`TOTAL_ROUNDS`** matches intended design (currently 20)
3. **`PROMPT_VERSION`** has been incremented if the system prompt changed
4. **`MATCHUPS`** list reflects only the matchups you want to run
5. **API keys** are loaded (run a quick `os.getenv("ANTHROPIC_API_KEY")[:8]` check)
6. **Cost estimate**: `len(MATCHUPS) × TOTAL_ROUNDS × 2 × ~150 tokens × price`

## Always Pilot First

Never run the full matchup list cold. Always:

```python
# Reduce to one matchup and 5 rounds first
MATCHUPS = [("claude_opus", "gpt4o")]
TOTAL_ROUNDS = 5
```

Run, inspect the resulting CSV, verify:
- Actions are clean `C` / `D`
- Beliefs are floats 0–1
- No truncated JSON in `raw_output_a/b`
- Compliance rate at or near 100%

Only then restore full settings and run.

## Adding a New Model to the Registry

In `build_model_registry()`:

```python
"new_model_key": (
    ChatProvider(
        model="exact-model-id",
        api_key=PROVIDER_API_KEY,
        temperature=0.6,
        max_tokens=150,
    ),
    "Display Label",
    0.6,  # temperature stored in CSV
),
```

Then add the key to `MATCHUPS` tuples and update `MODEL_COLORS` in
`dashboard/data_loader.py` to keep the dashboard consistent.

## Modifying the System Prompt

If you change `SYSTEM_PROMPT`:

1. **Increment `PROMPT_VERSION`** (e.g. `"v4.0"` → `"v4.1"`)
2. Run a 5-round pilot before any full experiment
3. Document the change in the commit message ("v4.1: removed cooperate-only example")

Never run analysis that mixes data from different `prompt_version` values without
explicitly comparing them as a variable.

## Common Parsing Failures

These are the failure modes seen in real data:

| Failure | Example | Cause | Fix |
|---|---|---|---|
| Truncated JSON | `{"belief": 0.5, "action":` | `max_tokens` too low | Raise to 150+ |
| Markdown fences | ` ```json\n{...}\n``` ` | Gemini default behavior | Already handled in `parse_response` |
| Verbose prose | `"I think we should COOPERATE"` | Prompt not strict enough | Reinforce JSON-only instruction |
| Unicode quotes | `"action": "COOPERATE"` (curly quotes) | Auto-formatting | Pre-process raw text with `.replace()` |

Always check `raw_output_a/b` columns when investigating anomalies.

## Switching Conditions

Run each condition as a **separate full experiment**, not within one script:

```bash
# Edit OPPONENT_CONDITION = "undisclosed", then:
python experiments/prisoners_dilemma_langchain.py

# Edit OPPONENT_CONDITION = "ai", then:
python experiments/prisoners_dilemma_langchain.py

# Edit OPPONENT_CONDITION = "human", then:
python experiments/prisoners_dilemma_langchain.py
```

The output filename includes the condition, so files don't overwrite.

## When an Experiment Crashes Mid-Run

The SQLite `.db` file is always safe — every round is committed.
The CSV is **only written at the end** of a full run.

To recover partial data:
```python
import sqlite3, pandas as pd
conn = sqlite3.connect("file:pd_experiment_ai_xxx.db?mode=ro", uri=True)
df = pd.read_sql_query("SELECT * FROM rounds", conn)
df.to_csv("partial_recovery.csv", index=False)
```

## Anti-Patterns (do NOT do)

- Running all three conditions inside a single `for` loop — gets confused, produces messy output filenames
- Hardcoding model names instead of using the registry keys
- Running with `MAX_RETRIES = 0` — single API hiccups will skew data
- Editing the CSV manually after a run — always reprocess from `.db`
