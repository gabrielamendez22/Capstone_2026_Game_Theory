# CLAUDE.md — Strategic Coherence in LLMs
## ESADE MiBA Capstone 2026

Read this file first. It is the single source of truth for this project.

---

## What This Project Is

We are investigating whether large language models exhibit **stable, transferable
strategic dispositions** across structurally distinct game-theoretic environments,
or whether their behavior is context-fragmented and prompt-sensitive.

This is an **AI systems research project**, not a human-AI behavioral comparison.
The games (PD, Commons, Cheap-Talk) are instruments to probe AI strategic cognition.

**Core construct:** Cross-Environment Consistency Score (CECS) — do behavioral
parameters estimated in one game predict behavior in a structurally distinct game
for the same model?

---

## Repository Structure

```
/
├── CLAUDE.md               ← you are here
├── SECURITY.md             ← read before touching API keys or data
├── METHODOLOGY.md          ← research definitions, parameters, games
├── .env                    ← NEVER READ OR MODIFY (API keys)
├── .gitignore
├── requirements.txt
├── environment.yml
│
├── experiments/
│   ├── prisoners_dilemma_langchain.py   ← primary PD experiment script
│   └── commons_dilemma.py               ← CD experiment (in progress)
│
├── data/
│   ├── raw/                ← JSON and CSV output files (do not modify)
│   └── processed/          ← cleaned CSVs for analysis
│
├── analysis/
│   ├── parameter_estimation.py
│   └── coherence_metrics.py
│
└── dashboard/
    ├── app.py              ← Dash dashboard entry point
    ├── data_loader.py
    ├── assets/style.css
    └── tabs/               ← overview, simulation, metrics, fingerprints, explorer
```

---

## Current State (April 2026)

- Prisoner's Dilemma: **complete** — pilot data collected across 3 conditions
  (undisclosed / ai / human), temperatures 0.3 and 0.6
- Commons Dilemma: **in progress** — design phase
- Cheap-Talk/Signaling: **not started**
- Dashboard: **live** — reads from CSV, runs locally on port 8050

---

## Experimental Conditions

Every experiment has one `OPPONENT_CONDITION` variable at the top of the script:

| Value | Meaning |
|---|---|
| `"undisclosed"` | Opponent identity not mentioned |
| `"ai"` | Told opponent is another AI model |
| `"human"` | Told opponent is a human (deception condition) |

Output files are named with the condition: `pd_results_{condition}_{timestamp}.csv`

---

## Model Registry

Models are defined in `build_model_registry()` in each experiment script.
Current keys and their labels:

| Key | Label | Provider |
|---|---|---|
| `claude_opus` | Claude Opus | Anthropic |
| `claude_sonnet` | Claude Sonnet | Anthropic |
| `gpt4o` | GPT-4o | OpenAI |
| `gpt4o_mini` | GPT-4o-mini | OpenAI |
| `gemini_pro` | Gemini 2.5 Flash | Google |
| `gemini_flash` | Gemini 2.5 Flash Lite | Google |

---

## Key Configuration Variables

In each experiment script (top of file):

```python
TOTAL_ROUNDS      = 20        # rounds per game session
PROMPT_VERSION    = "v4.0"    # increment when prompt changes
TEMPERATURE       = 0.6       # primary; 0.3 for low-variance baseline
HISTORY_WINDOW    = None      # None = full history injected each round
OPPONENT_CONDITION = "ai"     # see table above
MAX_RETRIES       = 1         # retry on parse failure before defaulting D
```

---

## Output Files

Each run produces two files in the project root (or configured path):

| File | When written | Use |
|---|---|---|
| `pd_experiment_{condition}_{timestamp}.db` | After every round (crash-safe) | Permanent raw store |
| `pd_results_{condition}_{timestamp}.csv` | End of full run | Analysis and dashboard |

**The SQLite .db file is always safe.** The CSV is only complete if the run finishes.
For partial runs, query the .db directly.

### CSV Schema (one row per round per game)

`game_id, condition, matchup, round, model_a, action_a, belief_a, payoff_a,
cumulative_a, raw_output_a, token_usage_a, response_time_a, temperature_a,
prompt_version, model_b, action_b, belief_b, payoff_b, cumulative_b,
raw_output_b, token_usage_b, response_time_b, temperature_b, timestamp`

Actions are stored as `C` or `D`. Beliefs are floats 0–1.

---

## Coding Rules

1. **Never hardcode API keys** — always `os.getenv()` from `.env`
2. **Never modify raw data files** in `data/raw/` — treat as read-only
3. **One OPPONENT_CONDITION per run** — change it at the top, do not branch
4. **Payoff matrix must satisfy T > R > P > S** — currently T=5, R=3, P=1, S=0
5. **Response format is JSON only** — `{"belief": float, "action": "COOPERATE or DEFECT"}`
6. **Parse action to C/D immediately** — never store raw "COOPERATE" strings in analysis
7. **Commit data files separately** from code changes — keep git history clean

---

## Dashboard

Run from the `dashboard/` directory:
```bash
cd dashboard
python app.py
```
Opens at `http://127.0.0.1:8050`

The dashboard reads from a hardcoded CSV path in `data_loader.py`.
Update `_find_csv()` there when switching to a new results file.

---

## See Also

- `SECURITY.md` — API key handling, .env discipline, what never gets committed
- `METHODOLOGY.md` — Strategic profile vector, parameter definitions, game designs