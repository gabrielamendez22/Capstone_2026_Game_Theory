# CHANGELOG

One entry per significant work session. Most recent at the top.
Each entry states **what** changed, **why**, and **what is still open**.

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
