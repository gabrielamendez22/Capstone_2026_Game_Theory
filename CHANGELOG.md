# CHANGELOG

One entry per significant work session. Most recent at the top.
Each entry states **what** changed, **why**, and **what is still open**.

---

## 2026-05-11 — Cheap-Talk v2: full 6-model schema with belief tracking

### What changed
- `experiments/cheap_talk.py` (v2) created as the canonical Cheap-Talk script.
  Replaces the root-level `cheap-talk-v1.py` for all new runs.
- Expanded from 3 to **6 models** (added Claude Opus, GPT-4o-mini, Gemini Flash Lite).
- Added **`IDENTITY_CONDITION`** variable (6 values) mirroring `OPPONENT_CONDITION` in PD/CD.
  The 6 conditions allow asymmetric persona injection per role (Sender / Receiver).
- Added **belief tracking**: Sender reports P(Receiver follows message); Receiver reports
  P(Sender is truthful). These are the β-equivalent fields for Cheap-Talk.
- Schema rename: `condition` → `game_condition` (aligned/misaligned); `identity_condition`
  added as separate column.
- Full run: 900 rows (6 models × 90 matchups × 10 rounds) at T=0.6, undisclosed condition,
  0 parse failures after increasing token limits.
- Root `cheap-talk-v1.py` also patched with belief tracking in last commit (`f068bfd`) but
  this script is now **legacy** — do not start new runs from it.

### Why
The v1 script (3 models, no belief, no identity condition) could not support Δm computation
or the β cross-game comparison required by METHODOLOGY.md. v2 restores structural parity
with the PD script.

### Open
- `TOTAL_ROUNDS = 5` in `experiments/cheap_talk.py` is set to PILOT — restore to 10 before
  full-condition runs.
- `MAX_RETRIES = 1` — raise to 2 (consistent with PD) before full runs.
- Identity conditions `ai_vs_ai`, `ai_vs_human_informed`, `ai_vs_human_blind`,
  `human_vs_human_declared`, `human_vs_human_silent` not yet run.
- `cheap-talk-v1.py` at project root is a legacy artifact — should be removed or
  clearly marked deprecated once all data re-runs are done.

---

## 2026-05-03 to 2026-05-04 — Cheap-Talk v1 pilot runs

### What changed
- `cheap-talk-v1.py` created (3 models, aligned/misaligned conditions, fixed/rotated roles).
- Two pilot runs: T=0.7 (180 rounds) and T=0.3 (180 rounds).
- Results written to `data/raw/`.
- `cheap_talk_summary.md` added summarising sender truthfulness, receiver scepticism,
  deception success rates.

### Key findings (v1 pilot)
- All models were fully truthful in aligned condition.
- Misaligned: deception rates 40-53%; Gemini most trusting receiver (67-80% follow rate).
- Temperature had small effect — strategic dispositions stable at T=0.3 vs T=0.7.

---

## 2026-04-29 — Commons Dilemma pilot runs (feature/commons-dilemma branch)

### What changed
- `commons_dilemma_langchain.py` (v2.2): OPPONENT_CONDITION added, Gemini max_tokens=1024.
- Pilot runs for undisclosed/ai/human conditions (5 rounds each).
- Results at branch root (non-standard location — needs `data/raw/` move on merge).

### Open (on that branch)
- OpenAI models hard-coded to temperature=1.0 while others use 0.6 — global TEMPERATURE
  variable not propagated. Corrupts cross-model temperature comparison.
- TOTAL_ROUNDS=5 is too short for reliable ρ/β estimation — raise to 20 before full run.
- Files at branch root, not `data/raw/` — clean up before merge.

---

## 2026-04-27 — Repository restructure + PD experiment audit

### Files moved / structure fixed
The project root was used as a dumping ground for experiment outputs and scripts.
All files are now in the locations defined in `CLAUDE.md`:

| Before | After |
|---|---|
| `prisoners_dilemma_langchain.py` (root) | `experiments/prisoners_dilemma_langchain.py` |
| `pd_experiment_*.db` (root) | `data/raw/` |
| `pd_results_*.csv` (root) | `data/raw/` |
| `import json.py` (root) | `analysis/json_to_csv.py` |
| `requirements.txt~` (root) | deleted (editor backup artifact) |

Created: `experiments/`, `data/raw/`, `data/processed/`, `analysis/`.

`dashboard/data_loader.py` updated: `_find_csv()` replaced by `_find_csvs()` which
auto-scans `data/raw/` and concatenates all CSVs — no manual path update needed when
new result files are added.

---

### Prompt bumped v4.0 → v4.1
**What:** Added one line to the RESPONSE FORMAT block of the system prompt:
> *"Any text outside the JSON will cause your response to be rejected."*

**Why:** Pilot data showed 6.5% parse-failure rate (63/960 rounds), 61 of which came
from Gemini Flash. The line is a format enforcement instruction — not a strategic
framing change — so it is classified as **legitimate** per the change protocol.

**Canonical updated in:** `.claude/skills/prompt-engineering/SKILL.md` and
`~/.claude/skills/prompt-engineering/SKILL.md` (global copy).
**Logged in:** `METHODOLOGY.md` → Prompt Version History table.

**Still open:** sensitivity check (v4.0 vs v4.1, Claude Opus × GPT-4o, 5 sessions
of 10 rounds) must be run before the next full experiment.

---

### Experiment script improvements (`experiments/prisoners_dilemma_langchain.py`)

| Change | Reason |
|---|---|
| `TEMPERATURE = 0.6` global variable added | Was hardcoded per-model; filenames were renamed manually post-run. Now flows to all models and auto-generates `_temp6_` in the filename. |
| `NUM_REPLICATIONS = 1` added | Makes replication count explicit. Raise to ≥3 before drawing statistical conclusions. Currently 1 = no behavior change. |
| `MAX_RETRIES` 1 → 2 | One retry was insufficient for Gemini; parse defaults to D inflate defection counts and corrupt ρ and β. |
| `max_output_tokens` 300 → 500 for Gemini models | Gemini preamble before JSON caused mid-response truncation. 500 tokens still minimal; valid response is ~40 chars. |
| Output paths → `data/raw/` | Outputs were written to root and moved manually. Now resolved automatically via `pathlib` relative to the script location. |

---

### Known open issues (as of this session)

| Issue | Impact | Required action |
|---|---|---|
| T=0.3 only exists for `undisclosed` condition | H3 (temperature → variance) untestable across conditions | Run T=0.3 for `ai` and `human` |
| 1 replication per matchup | No confidence intervals; single-game conclusions | Raise `NUM_REPLICATIONS` to 3 |
| Claude Sonnet + GPT-4o-mini never defect | ρ = NaN; reciprocity unanalyzable for these models | Investigate ceiling effect vs genuine finding |
| Gemini Flash pilot data partially contaminated | 61 defaults-to-D from old 300-token limit | Re-run Gemini matchups under v4.1 |
| v4.1 sensitivity check pending | Protocol requires check before scaling | Run sensitivity check (template in prompt-engineering skill) |

---

### Claude Code skills — setup for teammates
Project skills live in `.claude/skills/`. To make them invocable via the `Skill`
tool, copy them once to the global skills directory:

```bash
cp -r .claude/skills/prompt-engineering ~/.claude/skills/
cp -r .claude/skills/data-analysis-pd   ~/.claude/skills/
cp -r .claude/skills/experiment-runner  ~/.claude/skills/
cp -r .claude/skills/dashboard-dev      ~/.claude/skills/
```

Re-run after any `git pull` that modifies a skill file.
