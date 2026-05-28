# RESEARCH JOURNAL
## Strategic Coherence in Large Language Models: Evidence from Game-Theoretic Environments
### ESADE MiBA Capstone 2026

**Research team:** Giorgio Fiorentino, Samreen Siddique, Gabriela Mendez
**Journal target:** TBD (candidates: JASSS / Computational Social Science / Nature Machine Intelligence)
**Last updated:** 2026-05-14

---

## How to Use This Document

This is the authoritative lab notebook for the project. It lives on `main` and is updated
after every significant session. Its three purposes:

1. **Record decisions with rationale** — so nothing is lost in reconstruction
2. **Track empirical findings as they emerge** — paper drafts from this, not from memory
3. **Map findings to paper sections** — writing phase is fast when this is complete

`CHANGELOG.md` (on each branch) tracks *code* changes.
This journal tracks *research* decisions, findings, and paper implications.

---

## Repository Map (as of 2026-05-14)

Three active branches. Each has its own state — this section is the single source of truth
for what exists where.

### `main` — Integration branch, PD complete, human priors added

| Location | Contents | Status |
|---|---|---|
| `experiments/prisoners_dilemma_langchain.py` | PD script v4.4 | Production-ready |
| `data/raw/` | 7 PD result files (3 conditions × T=0.3/0.6/0.8) | Complete pilot |
| `data/human_benchmarks/` | Data.csv, alldata.dta, Gneezy 2005 PDF, Anwar & Georgalos 2026 PDF | Added 2026-05-14 |
| `analysis/human_prior_builder.py` | Computes human behavioral priors from benchmark data | Added 2026-05-14 |
| `analysis/human_priors.json` | Generated priors dict + 3 game prompt strings | Added 2026-05-14 |
| `CLAUDE.md` | **Stale** — says CT "not started", CD "design phase" | Needs update from CT branch |
| `METHODOLOGY.md` | **Stale** — prompt history ends at v4.1 | Needs update from CT branch |

PD data inventory on main:

| File | Condition | Temperature | Date |
|---|---|---|---|
| pd_results_undisclosed_temp3_20260423 | undisclosed | 0.3 | Apr 23 |
| pd_results_undisclosed_temp6_20260423 | undisclosed | 0.6 | Apr 23 |
| pd_results_undisclosed_temp8_20260504 | undisclosed | 0.8 | May 4 |
| pd_results_ai_temp6_20260423 | ai | 0.6 | Apr 23 |
| pd_results_ai_temp8_20260504 | ai | 0.8 | May 4 |
| pd_results_human_temp6_20260423 | human | 0.6 | Apr 23 |
| pd_results_human_temp8_20260504 | human | 0.8 | May 4 |

---

### `feature/cheap-talk` — CT v2 complete, pilot data collected

| Location | Contents | Status |
|---|---|---|
| `experiments/cheap_talk.py` | CT v2: 6 identity conditions, belief tracking, 6 models | **Canonical** — use this |
| `cheap-talk-v1.py` (root) | CT v1: 3 models, old schema | **Legacy** — do not use for new runs |
| `data/raw/` | CT pilot files + PD files (carried from main) | Mixed schemas (see below) |
| `CLAUDE.md`, `METHODOLOGY.md`, `CHANGELOG.md` | Updated to May 2026 state | Up to date |

CT data inventory on `feature/cheap-talk`:

| File | Script used | Schema | Identity condition | Notes |
|---|---|---|---|---|
| cheap_talk_results_20260503 | v1 | old (no belief, no identity) | aligned/misaligned | T=0.7 pilot |
| cheap_talk_results_20260504 | v1 | old | aligned/misaligned | T=0.3 pilot |
| cheap_talk_results_undisclosed_20260506 | v1 or transition | old | undisclosed | smoke test |
| cheap_talk_results_undisclosed_20260507 | v1 or transition | old | undisclosed | smoke test |
| cheap_talk_results_20260510_135946 | v2 | new (belief + identity) | undisclosed | smoke test |
| cheap_talk_results_20260510_140437 | v2 | new | undisclosed | 900 rows, 0 parse fails |
| cheap_talk_results_20260511_101447 | v2 | new | undisclosed | 900 rows, 0 parse fails |

**Critical open issues on this branch:**
- `TOTAL_ROUNDS = 5` in `experiments/cheap_talk.py` — marked PILOT, must restore to 10
- `MAX_RETRIES = 1` — must raise to 2 (consistent with PD) before full runs
- Identity conditions `ai_vs_ai`, `ai_vs_human_informed`, `ai_vs_human_blind`,
  `human_vs_human_declared`, `human_vs_human_silent` — not yet run
- v1/v2 schema break: old CSVs cannot contribute β to CECS computation

---

### `feature/commons-dilemma` — CD pilot complete, structural cleanup done

| Location | Contents | Status |
|---|---|---|
| `experiments/commons_dilemma_langchain.py` | CD script v2.2 | Pilot-ready |
| `experiments/prisoners_dilemma_langchain.py` | PD script **v4.1** (old) | Will be superseded on merge |
| `data/raw/` | 11 CD result files + 4 PD files | Pilot data |
| `docs/commons_dilemma_dev_log.py` | Design decisions log | Reference only |
| `CHANGELOG.md` | Full branch history | Up to date |

CD data inventory on `feature/commons-dilemma`:

| File | Condition | Rounds | Notes |
|---|---|---|---|
| cd_results_20260426 | undisclosed | 5 | First implementation |
| cd_results_20260429 | undisclosed | 5 | v2.0 rerun |
| cd_results_undisclosed_20260429_* (7 files) | undisclosed | 5 | Multiple short runs |
| cd_results_ai_20260429 | ai | 5 | Pilot |
| cd_results_human_20260429 | human | 5 | Pilot |

**Critical open issues on this branch:**
- OpenAI models hard-coded at `temperature=1.0`; all others at `0.6` — temperature NOT
  controlled across models. All existing CD data is contaminated for cross-model temp comparison.
- `TOTAL_ROUNDS = 5` — far too short for reliable θ/β estimation; raise to 20
- Branch has no `CLAUDE.md` or `METHODOLOGY.md` (diverged before they existed)

---

## Paper Structure Tracker

| Paper Section | Status | Blocking on |
|---|---|---|
| Abstract | Not started | All results |
| 1. Introduction | Draft notes below | — |
| 2. Literature Review | Anchors listed below | — |
| 3.1 Methods: PD design | Ready to write | — |
| 3.2 Methods: CD design | Draft ready | Full CD runs |
| 3.3 Methods: CT design | Draft ready | Full CT runs |
| 3.4 Methods: Models | Ready to write | — |
| 3.5 Methods: Conditions + Perturbation test | Design complete | Implementation |
| 4.1 Results: PD (ρ, β) | Can draft from pilot | — |
| 4.2 Results: CD (θ, β) | Blocked | Fix temp bug; full CD runs (20 rounds) |
| 4.3 Results: CT (η, γ, β) | Blocked | Full CT runs (all identity conditions) |
| 4.4 Results: CECS cross-game | Blocked | All three games complete |
| 4.5 Results: Opponent sensitivity (Δm) | Blocked | All conditions per game |
| 4.6 Results: Perturbation test | Blocked | Human prior implementation |
| 5. Discussion | Not started | Results |
| 6. Limitations | Draft notes below | — |
| 7. Conclusion | Not started | Results |

---

## Chronological Session Log

---

### 2026-05-14 — Human behavioral priors built; journal moved to main

**Context:** Perturbation test design requires human behavioral priors to construct
the `human_sim` prompt — one AI acting with calibrated human-like behavior as opponent.
Data files (Dvorak & Fehrler 2024, Abatayo & Lynham 2022) were at project root, untracked.

**D1: Human priors computed from empirical data (not invented)**
All three game priors are anchored to real experimental data, not assumed.
This is essential for the paper — the "human simulation" condition must be defensible
as empirically calibrated, not a made-up caricature of human behavior.

**D2: `pd_bcr_for_prior` uses T1 (no-comm, 39.6%), not T13+T14 (55.6%)**
T13+T14 cooperation rates are inflated by: (a) communication allowed, (b) indefinitely
repeated game (δ=0.80). Our LLM setup has neither. T1 (no-comm, imperfect monitoring)
is the closest structural match — acknowledged as a lower bound.
For the conditional reciprocity estimates (rho_pos=96%, rho_neg=35%), T13+T14 is used
because the larger sample is cleaner — labelled as upper-bound estimates in the script.

**D3: Research journal moved to `main`**
Journal was created on `feature/cheap-talk`. Moving to `main` so it is the stable,
always-accessible reference across all branches, not dependent on which branch is checked out.

**Verified computed values (all 11 passed ±0.02 tolerance):**

| Metric | Value | Source |
|---|---|---|
| PD BCR T13+T14 (all supergames, round 1) | 0.907 | Dvorak & Fehrler 2024 |
| PD BCR T13+T14 (supergame 1, round 1) | 0.556 | Dvorak & Fehrler 2024 |
| PD rho T13+T14 | 0.610 | Dvorak & Fehrler 2024 |
| PD overall coop T13+T14 | 0.897 | Dvorak & Fehrler 2024 |
| PD BCR T1 no-comm (→ prior anchor) | **0.396** | Dvorak & Fehrler 2024 |
| PD rho T1 no-comm | 0.489 | Dvorak & Fehrler 2024 |
| CD strategyB overall | 0.583 | Abatayo & Lynham 2022 |
| CD strategyB round 1 | 0.661 | Abatayo & Lynham 2022 |
| CD strategyB late rounds | 0.589 | Abatayo & Lynham 2022 |
| CD low-inflow strategyB | 0.531 | Abatayo & Lynham 2022 |
| CD high-inflow strategyB | 0.649 | Abatayo & Lynham 2022 |

**CT priors (hardcoded from Gneezy 2005):**
- Sender honesty rate (η): 64%
- Receiver compliance (γ): 78%
- Deception low-stakes: 36%, high-stakes: 52%

**Paper implications:**
- Methods 3.5: Human prior construction is documentable as an empirical calibration step.
  Table of sources and design differences belongs in the paper as a footnote or appendix.
- Discussion: The gap between T13+T14 (90.7%) and T1 (39.6%) cooperation illustrates
  how sensitive human baselines are to game structure — supports the argument that our
  LLM behavioral parameters also depend on structural context (the CECS question).

---

### 2026-05-13 — Full repository audit; structure standardised; documentation updated

**Context:** Both feature branches had files in wrong locations, stale documentation,
and no CHANGELOG on `feature/commons-dilemma`.

**D4: `feature/commons-dilemma` structural cleanup**
Moved all scripts, data, and artifacts to canonical locations (see repository map above).
Both experiment scripts on that branch now use `pathlib` to write to `data/raw/`, matching
the pattern established on `feature/cheap-talk`. `requirements.txt~` and `__pycache__/`
removed; `.gitignore` updated.

**D5: CHANGELOG added to `feature/commons-dilemma`**
Branch had diverged before CHANGELOG was created on main. Added with full retroactive history
from initial CD implementation (Apr 26) through May 13 cleanup.

**D6: CLAUDE.md and METHODOLOGY.md updated on `feature/cheap-talk`**
- CLAUDE.md: current state corrected; IDENTITY_CONDITION table for CT added; file structure updated
- METHODOLOGY.md: prompt version history extended to v4.4 (v4.2/v4.3/v4.4 were undocumented)
- Note: these updates have NOT been merged to `main` yet — main's docs are stale

**Outstanding after this session:**
- `main` CLAUDE.md and METHODOLOGY.md still stale — need to bring updates from CT branch
- CT script: TOTAL_ROUNDS and MAX_RETRIES still need fixing before full runs
- CD temperature bug: OpenAI models at T=1.0 not fixed yet (code change, separate session)

---

### 2026-05-11 — Cheap-Talk v2: 6 models, 6 identity conditions, belief tracking

**Context:** v1 pilot (3 models, aligned/misaligned only) proved LLMs respond to incentive
structure. But it could not support CECS computation: no β field, no Δm axis,
only 3 of 6 models represented.

**D7: Six identity conditions for CT (vs 3 for PD/CD)**
PD and CD use a simple 3-value `OPPONENT_CONDITION` — but CT roles are asymmetric
(Sender vs Receiver), requiring identity framing to be applied per-role independently.
Six conditions form a 2×2 structure (who is told what × who plays as what):

| Condition | Effect on Sender | Effect on Receiver |
|---|---|---|
| `undisclosed` | No framing | No framing |
| `ai_vs_ai` | Told partner is AI | Told partner is AI |
| `ai_vs_human_informed` | Told partner plays as human | Plays as human (silent) |
| `ai_vs_human_blind` | No framing | Plays as human (silent) |
| `human_vs_human_declared` | Plays as human, declares it | Plays as human (silent) |
| `human_vs_human_silent` | Plays as human (silent) | Plays as human (silent) |

*Paper implication:* CT Δm is richer than PD/CD Δm — it decomposes into Sender-side and
Receiver-side effects. Worth highlighting in Results 4.5.

**D8: Belief fields added to CT**
- Sender: P(Receiver will follow my message) — Sender's model of the Receiver
- Receiver: P(Sender's message is truthful) — Receiver skepticism
These are the β-equivalent fields for CT, enabling cross-game β comparison for CECS.

**D9: v1/v2 schema break — old CT data cannot contribute β**
CSVs from May 3–4 (v1) have no belief field and use `condition` (not `game_condition +
identity_condition`). They can describe sender truthfulness and deception rates, but β
cannot be computed from them. All full-condition CECS runs must use the v2 schema.

**Findings from v1 pilot (May 3–4, 180 rounds each run):**

| Finding | T=0.7 | T=0.3 | Interpretation |
|---|---|---|---|
| Aligned sender truthfulness | 100% all models | 100% all models | Equilibrium honesty when incentives align |
| Misaligned sender truthfulness | Claude 53%, GPT 40%, Gemini 47% | Claude 40%, GPT 43%, Gemini 47% | Strategic deception under misaligned payoffs |
| Misaligned receiver follow rate | Claude 37%, GPT 50%, Gemini 80% | Claude 30%, GPT 47%, Gemini 67% | Wide inter-model variance in credulity |
| Deception success | Claude 37%, GPT 30%, Gemini 23% | Claude 33%, GPT 33%, Gemini 30% | Moderate; receivers partially discount senders |
| Temperature effect | — | Small | Strategic dispositions stable at T=0.3 vs 0.7 |

**Anomaly to flag in paper:** Gemini receiver follow rate of 67–80% under misaligned
conditions (sender lying 47% of the time) — super-credulous behavior that persists
across rounds. This is a striking between-model strategic personality difference.

**Paper implications (Results 4.3 and Discussion):**
- Table: sender truthfulness rates (aligned/misaligned) × model × temperature
- Table: receiver skepticism and deception success
- Discussion paragraph: Gemini credulity as a strategic personality finding

---

### 2026-05-03 to 2026-05-04 — CT v1 pilot; structure and summary

First implementation. 3 models, aligned/misaligned conditions, fixed/rotated roles,
10 rounds per matchup, 18 matchups per condition. See findings table above (D9).
`cheap_talk_summary.md` added to repo with full result tables.

---

### 2026-04-29 — Commons Dilemma pilot (3 conditions, 5 rounds)

**D10: Sustainable extraction threshold (θ) hidden from prompt**
Pool: 100 units, regen: 20/round, 2 players → sustainable share = 10 units/player/round.
This is NOT shown to models. θ must emerge from behavior, not instruction-following.
Academically important: we probe strategic discovery, not compliance.

**D11: OPPONENT_CONDITION added to CD (mirrors PD)**
Same 3-value variable: undisclosed / ai / human. Enables Δm computation for CD.

**Preliminary observations from pilot (5 rounds — indicative only):**
- Pool depletion occurred in several undisclosed runs → aggressive extraction without framing
- Qualitative signal: condition affects CD behavior, consistent with PD findings
- Results not publishable at 5 rounds — too short for reliable θ/β estimation

**Open (must fix before full CD runs):**
- OpenAI T=1.0 vs others T=0.6 — temperature bug corrupts cross-model comparison
- Need 20 rounds for meaningful θ estimation (pool depletion dynamics visible)

---

### 2026-04-27 — PD pilot complete; repository restructured; prompt v4.1

**D12: Prompt v4.1 — format enforcement line**
Gemini Flash: 6.5% parse failure rate (61/960 rounds defaulted to DEFECT).
Defaults inflate defection counts and corrupt ρ and β.
Added: "Any text outside the JSON will cause your response to be rejected."
Classified as format change (not framing) — justified and documented in METHODOLOGY.md.

**D13: Global TEMPERATURE variable (replaces per-model hardcode)**
Before: temperature set individually per model → filenames required manual renaming.
After: one global variable flows to all models and auto-generates filename suffix.

**Known data quality issues from PD pilot (still open):**
- T=0.3 only for undisclosed condition → H3 (temperature → variance) untestable across conditions
- 1 replication per matchup → no confidence intervals
- Claude Sonnet + GPT-4o-mini never defect → ρ = NaN for these models (ceiling effect?)
- Gemini Flash pilot data partially contaminated (300-token limit pre-v4.1)
- v4.1 sensitivity check still pending (required before treating v4.4 data as comparable)

---

## Running Notes for Paper Sections

### 1. Introduction — Draft Framing

**Research question:**
Do large language models possess coherent strategic architectures — behavioral parameters
that respond predictably to structural game features — or does their behavior reflect
prompt-level pattern matching that collapses under structural variation?

**Contribution claim:**
We introduce the Cross-Environment Consistency Score (CECS), a behavioral fingerprinting
metric that measures strategic coherence across structurally distinct game-theoretic
environments, and apply it to six state-of-the-art LLMs across three games.

**Opening hook:**
LLMs are increasingly deployed as agents in economic and social environments.
Understanding whether their strategic behavior is coherent or context-fragmented has
direct implications for their reliability as social actors. The existing literature tests
LLMs in individual games; we test whether behavioral signatures transfer across games.

### 2. Literature Review — Key Anchors

| Topic | Source | Relevance |
|---|---|---|
| LLMs in PD | Dvorak & Fehrler (2024) AEJ:Micro 16(3) | Human baseline + data source |
| LLMs in bargaining | Ferraz et al. (2025) | Scale of evidence (400k decisions) |
| LLM morality | Capraro (2025) | Prosocial LLM behavior baseline |
| LLMs in cooperation | Sreedhar & Chilton (2025) | LLMs replicate lab PGG results |
| Human-AI PD | Anwar & Georgalos (2026) arXiv:2603.15852 | TfT vs GRIM dominant strategies |
| Cheap-talk baseline | Gneezy (2005) AER 95(1) | Human CT honesty/compliance |
| CPR baseline | Abatayo & Lynham (2022) | Human CD extraction rates |
| Crawford-Sobel | Crawford & Sobel (1982) | Foundational cheap-talk theory |

*To add: LLM strategic reasoning / ToM literature; AI agent game theory literature.*

### 3. Methods — Key Decisions to Document

- **Why three games?** PD (dyadic reciprocity), CD (collective action, no direct reciprocity),
  CT (information asymmetry) — three structurally distinct environments that activate
  different strategic mechanisms. If behavior is coherent, parameters should correlate.
- **Why these six models?** Two per family (large/small), three families (Anthropic, OpenAI,
  Google) — enables family effects and scale effects within H1.
- **Prompt neutrality controls:** neutral objective framing, both C and D examples shown,
  round total hidden, belief timing explicit, identical prompt across all models.
- **Temperature rationale:** T=0.8 (primary), T=0.3 (low-variance robustness check).
  Global variable ensures same temperature across all models within a run.
- **Human simulation prior construction:** empirically calibrated from Dvorak & Fehrler
  (2024) and Abatayo & Lynham (2022); design differences documented in Methods appendix.

### 6. Limitations — Running List

1. Single replication per matchup in pilot — no confidence intervals; plan ≥3 for final
2. Prompt sensitivity not fully ruled out — v4.1 sensitivity check still pending
3. Temperature not standardized across all existing data (CD: OpenAI at T=1.0)
4. 5-round CD pilot too short for θ estimation — full 20-round runs needed
5. v1 CT data (pre-schema change) cannot contribute β to CECS
6. No human participant baseline run — sourced from literature (document explicitly)
7. Model versions fixed at experiment time — findings may not generalize to future versions
8. T13+T14 human PD baseline is inflated vs our setup (communication + indefinite horizon)

---

## Key Definitions (Quick Reference)

| Symbol | Name | Game | Formula / Operationalisation |
|---|---|---|---|
| ρ | Conditional reciprocity | PD | P(C\|opp C at t−1) − P(C\|opp D at t−1) |
| θ | Exploitation threshold | CD | Pool size at which extraction first exceeds regen/N |
| η | Signal honesty | CT | Proportion of rounds where message = true state |
| γ | Receiver gullibility | CT | P(follow\|msg=H) − P(follow\|msg=L) in misaligned |
| β | Belief calibration | All | MAE: (1/T) Σ \|stated_belief_t − actual_opp_action_t\| |
| CECS | Cross-env consistency | All | Spearman rank corr of parameters across games per model |
| Δm | Opponent sensitivity | All | ‖Sm_AI − Sm_Human‖ Euclidean distance |

## Immediate Next Steps (Prioritised)

1. **Fix CT script before next run:** `TOTAL_ROUNDS = 10`, `MAX_RETRIES = 2`
2. **Run CT under remaining 5 identity conditions** to collect full Δm data
3. **Fix CD temperature bug** (`commons_dilemma_langchain.py`: propagate global TEMPERATURE to OpenAI)
4. **Run CD full experiment** (20 rounds, all 3 conditions)
5. **Implement perturbation test** in PD and CD scripts (`human_sim` condition using priors from `human_priors.json`)
6. **Bring CLAUDE.md and METHODOLOGY.md updates from `feature/cheap-talk` to `main`**
7. **Run PD sensitivity check** (v4.0 vs v4.1 — still pending from April)
8. **Merge feature branches to main** once full runs complete
9. **Parameter estimation pass** (ρ, θ, η, γ, β per model per game)
10. **CECS computation and Δm** — headline results for the paper
