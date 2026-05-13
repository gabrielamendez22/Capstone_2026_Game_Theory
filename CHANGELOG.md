# CHANGELOG

One entry per significant work session. Most recent at the top.
Each entry states **what** changed, **why**, and **what is still open**.

---

## 2026-05-13 — Repository structure cleanup

### Files moved / structure fixed
The project root was used as a dumping ground for experiment outputs, scripts, and artifacts.
All files are now in the locations defined in CLAUDE.md:

| Before | After |
|---|---|
| `commons_dilemma_langchain.py` (root) | `experiments/commons_dilemma_langchain.py` |
| `prisoners_dilemma_langchain.py` (root) | `experiments/prisoners_dilemma_langchain.py` |
| `import json.py` (root) | `analysis/json_to_csv.py` |
| `commons_dilemma_dev_log.py` (root) | `docs/commons_dilemma_dev_log.py` |
| 11 × `cd_experiment_*.db` (root) | `data/raw/` |
| 11 × `cd_results_*.csv` (root) | `data/raw/` |
| 4 × `pd_experiment_*.db` (root) | `data/raw/` |
| 4 × `pd_results_*.csv` (root) | `data/raw/` |

Created directories: `experiments/`, `data/raw/`, `analysis/`, `docs/`.

### Output paths fixed in both experiment scripts
Both `commons_dilemma_langchain.py` and `prisoners_dilemma_langchain.py` previously wrote
outputs to the current working directory. Now use `pathlib.Path(__file__).parent.parent / "data" / "raw"`
— identical to the canonical pattern established in `feature/cheap-talk`.

### Artifacts removed
- `requirements.txt~` (editor backup)
- `__pycache__/` removed from git tracking

### `.gitignore` updated
Added: `__pycache__/`, `*.pyc`, `*.py~`, `requirements.txt~`

### Open issues (not fixed here — code changes, separate session)
- OpenAI models hard-coded at `temperature=1.0` in `commons_dilemma_langchain.py`;
  all other models use `0.6`. Global `TEMPERATURE` variable not propagated to OpenAI.
  **Impact:** cross-model temperature comparison invalid in current pilot data.
- `TOTAL_ROUNDS = 5` is too short for reliable θ/β estimation. Raise to 20 before full run.

---

## 2026-04-29 — Commons Dilemma pilot runs (3 conditions, 5 rounds)

### What changed (`d328b56`)
- Pilot runs completed for all three opponent conditions: undisclosed / ai / human.
- 5 rounds per session, 6 models, standard matchup list.
- Results written to root (now moved to `data/raw/` — see 2026-05-13 entry).

### What was found
- Pool depletion occurred in several undisclosed runs, suggesting models extract aggressively
  without identity framing.
- Behaviour differed across conditions — qualitative signal that OPPONENT_CONDITION matters
  in the CD setting, consistent with PD findings.

---

## 2026-04-29 — Script stabilisation: v2.1 and v2.2

### v2.2 (`c979fe9`) — Gemini `max_output_tokens` raised to 1024
**Why:** Gemini generates thinking tokens before visible output. At 150 tokens, JSON was
frequently truncated mid-response. 1024 gives sufficient headroom.
Also added a truncated-JSON fallback parser: extracts the last valid `}` if output is cut off.

### v2.1 (`2cd1bf1`) — Robust parser for Gemini prose responses
**Why:** Gemini wrapped JSON in prose sentences rather than outputting it directly.
Parser now strips all text before the first `{` and after the last `}`.

---

## 2026-04-29 — v2.0: competitive prompt, OPPONENT_CONDITION, model fixes (`29456ea`)

### What changed
- `OPPONENT_CONDITION` variable added (undisclosed / ai / human) — mirrors PD architecture.
- System prompt rewritten with explicitly competitive framing.
- Gemini model IDs updated to current API names (were stale from initial implementation).
- `.env` key loading fixed — was hardcoded in first attempt.

---

## 2026-04-26 — First Commons Dilemma implementation (`036de1f`, `1fc61ee`)

### What was built
- `commons_dilemma_langchain.py`: LangChain-based multi-model experiment for the Commons
  Dilemma (shared resource pool).
- Game parameters: pool=100, regen=20/round (fixed), 2 players, max extraction=20/round.
- Sustainable share (10 units/player) intentionally hidden — models must discover it.
- Parameters tracked: extraction, belief, payoff, cumulative (θ computed post-hoc).
- SQLite + CSV output, same schema pattern as PD script.
- First pilot run completed; identified Gemini parse failures (→ fixed in v2.1/v2.2).
