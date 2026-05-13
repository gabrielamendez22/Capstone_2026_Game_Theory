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
│   ├── prisoners_dilemma_langchain.py   ← primary PD experiment script (v4.4)
│   ├── cheap_talk.py                    ← Cheap-Talk experiment script (v2, canonical)
│   └── commons_dilemma_langchain.py     ← CD experiment (pending merge from feature branch)
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

## Current State (May 2026)

- Prisoner's Dilemma: **pilot complete** — data across 3 conditions (undisclosed / ai / human),
  T=0.3 and T=0.6; prompt at v4.4 with mandatory chain-of-thought; 20 rounds per session
- Commons Dilemma: **partial pilot** — implementation in `feature/commons-dilemma` branch;
  3 conditions run; needs branch cleanup and merge before analysis
- Cheap-Talk / Signaling: **v2 implementation complete** — `experiments/cheap_talk.py` with
  6 identity conditions, belief tracking, 6 models; pilot data collected (undisclosed condition);
  full-condition runs pending; v1 legacy script at root (`cheap-talk-v1.py`) — do not use for new runs
- Dashboard: **live** — reads from `data/raw/`, runs locally on port 8050

---

## Experimental Conditions

### Prisoner's Dilemma & Commons Dilemma — `OPPONENT_CONDITION`

| Value | Meaning |
|---|---|
| `"undisclosed"` | Opponent identity not mentioned |
| `"ai"` | Told opponent is another AI model |
| `"human"` | Told opponent is a human (deception condition — model still plays as AI) |

Output files: `pd_results_{condition}_{timestamp}.csv`

### Cheap-Talk — `IDENTITY_CONDITION`

The Cheap-Talk game has asymmetric roles (Sender / Receiver), so identity framing must be
applied per-role. Six conditions:

| Value | Sender is told | Receiver is told | Sender persona | Receiver persona |
|---|---|---|---|---|
| `"undisclosed"` | nothing | nothing | none | none |
| `"ai_vs_ai"` | partner is AI | partner is AI | none | none |
| `"ai_vs_human_informed"` | partner plays as human | nothing | none | play as human |
| `"ai_vs_human_blind"` | nothing | nothing | none | play as human |
| `"human_vs_human_declared"` | nothing | nothing | play as human, state it | play as human |
| `"human_vs_human_silent"` | nothing | nothing | play as human, silent | play as human |

Output files: `cheap_talk_results_{identity_condition}_{timestamp}.csv`

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
TOTAL_ROUNDS      = 20        # rounds per game session (PD); 10 for Cheap-Talk
PROMPT_VERSION    = "v4.4"    # PD current; increment when prompt changes
TEMPERATURE       = 0.8       # PD current; global variable — flows to all models
HISTORY_WINDOW    = None      # None = full history injected each round
OPPONENT_CONDITION = "ai"     # PD/CD — see table above
IDENTITY_CONDITION = "undisclosed"  # Cheap-Talk — see table above
MAX_RETRIES       = 2         # retry on parse failure before defaulting D
NUM_REPLICATIONS  = 1         # repeat each matchup; raise to ≥3 for statistics
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