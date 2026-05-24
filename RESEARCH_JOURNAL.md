# RESEARCH JOURNAL
## Strategic Coherence in Large Language Models: Evidence from Game-Theoretic Environments
### ESADE MiBA Capstone 2026

**Authors:** Giorgio Fiorentino, Samreen Siddique, Gabriela Mendez *(add co-authors)*
**Supervisor:** *(add)*
**Journal target:** *(e.g., JASSS / Computational Social Science / Nature Machine Intelligence)*

---

## How to Use This Document

This is a living lab notebook, written as the research happens.
Its purpose is threefold:
1. Record every significant decision with its rationale — so nothing is lost in reconstruction
2. Track empirical findings as they emerge — so the paper can be drafted from this, not from memory
3. Map each finding to a paper section — so the writing phase is fast

**Entry format:** date, one-line title, structured content below.
Write here *before* pushing code changes when a decision is made. Write here *after* running experiments when results are in.

---

## Paper Structure Tracker

Use this to know what each experiment contributes to the paper.

| Paper Section | Status | Source |
|---|---|---|
| Abstract | Not started | — |
| 1. Introduction | Draft notes below | RQ, H1-H4 |
| 2. Literature Review | Not started | See reference anchors below |
| 3. Methodology: Experimental Design | Partial | METHODOLOGY.md |
| 3.1 Game 1: Prisoner's Dilemma | Data complete (pilot) | PD data files |
| 3.2 Game 2: Commons Dilemma | Data partial | CD feature branch |
| 3.3 Game 3: Cheap-Talk | Data partial (undisclosed only) | CT data files |
| 3.4 Models | Complete | Build_model_registry() |
| 3.5 Conditions | Complete design | METHODOLOGY.md |
| 4. Results: CECS | Blocked — need all 3 games | Pending CD/CT full runs |
| 4.1 PD parameters (ρ, β) | Can draft | PD pilot data |
| 4.2 CD parameters (θ, β) | Can draft (partial) | CD pilot data |
| 4.3 CT parameters (η, γ, β) | Blocked — full runs needed | CT identity conditions pending |
| 4.4 Cross-game CECS | Blocked | All games needed |
| 4.5 Opponent Sensitivity (Δm) | Blocked — need all 3 conditions per game | Pending |
| 5. Discussion | Not started | — |
| 6. Limitations | Draft notes below | — |
| 7. Conclusion | Not started | — |

---

## 2026-05-13 — Full repository audit; research journal created

### Context
Project has three active branches: `main` (PD complete), `feature/cheap-talk`
(currently checked out, v2 complete + pilot data), `feature/commons-dilemma` (pilot data,
structural issues). Academic paper objective confirmed: prosociality in AI via game theory.

### Decisions Made

**D1: Research journal format**
Created this file as the authoritative lab notebook. CHANGELOG.md continues to track code
changes; this journal tracks research decisions, findings, and paper implications.
*Rationale:* As experiments accumulate, the gap between "what we did" and "what we can write"
widens fast. A journal written in real time is the only reliable way to reconstruct the paper's
Methods and Discussion sections.

**D2: CLAUDE.md and METHODOLOGY.md updated to reflect current state (May 2026)**
- CLAUDE.md: Current state corrected (Cheap-Talk now complete; CD partial; PD v4.4).
- CLAUDE.md: IDENTITY_CONDITION table for Cheap-Talk added.
- METHODOLOGY.md: Prompt version history extended to v4.4.
- CHANGELOG.md: Entries added for May 2026 cheap-talk v2 work and CD pilot.

**D3: Canonical script locations confirmed**
- `experiments/prisoners_dilemma_langchain.py` — PD (v4.4)
- `experiments/cheap_talk.py` — Cheap-Talk (v2, canonical)
- `cheap-talk-v1.py` at root — **legacy**, do not use for new runs
- `experiments/commons_dilemma_langchain.py` — CD (pending merge)

### Open Structural Issues (prioritized)
1. `experiments/cheap_talk.py` has `TOTAL_ROUNDS = 5` (PILOT) — restore to 10 before full run
2. `experiments/cheap_talk.py` has `MAX_RETRIES = 1` — raise to 2
3. CD branch: OpenAI models hard-coded at temperature=1.0 — must fix before full CD runs
4. CD branch: files at root, not `data/raw/` — fix on merge
5. `cheap-talk-v1.py` root artifact — quarantine or delete after v2 full runs confirmed

### Paper Implications
None from this session — housekeeping only.

---

## 2026-05-11 — Cheap-Talk v2: full schema, 6 models, belief tracking

### Context
v1 pilot (3 models, aligned/misaligned only) confirmed LLMs respond to incentive structure.
But it could not support the CECS computation (no β, no Δm) and used only 3 of 6 models.
v2 was designed to restore parity with PD architecture.

### Decisions Made

**D4: Six identity conditions for Cheap-Talk (replacing 3-value OPPONENT_CONDITION)**
The PD/CD "human" condition is a simple deception framing — the observing model is told its
opponent is human, but the opponent plays normally as an AI.
In Cheap-Talk, roles are asymmetric (Sender vs Receiver), so identity framing must be
applied per-role independently. This requires 6 conditions.

The six conditions form a 2×2 factorial (who is told what / who plays as what) plus two
"both play as human" variants for the perturbation test design.

*Why this matters for the paper:* Δm in Cheap-Talk is richer than in PD. In PD, Δm
is a scalar (AI vs human framing). In Cheap-Talk, Δm has multiple dimensions: does
being told your partner is human change your honesty as Sender? Your skepticism as Receiver?

**D5: Belief fields added to Cheap-Talk**
- Sender: P(Receiver will follow my message) — measures Sender's model of the Receiver
- Receiver: P(Sender's message is truthful) — measures Receiver's skepticism
These are the β-equivalent fields for Cheap-Talk, enabling cross-game β comparison.

**D6: Data schema break — old CSVs are not directly comparable with v2**
CSVs from 2026-05-03 and 2026-05-04 (v1 pilot) have the old schema (no belief, old
`condition` column instead of `game_condition` + `identity_condition`). They can be used
to describe sender truthfulness and deception rates, but NOT to compute β for cross-game
CECS. All full-condition runs going forward use the v2 schema from `experiments/cheap_talk.py`.

### Findings (from 900-row undisclosed run, 2026-05-11)

*To be filled after analysis — schema confirmed correct, 0 parse failures reported.*

- Run parameters: T=0.6, undisclosed condition, 6 models, 90 matchups, 10 rounds each
- No parse failures confirms v2 architecture is production-ready

### Paper Implications
- **Methods 3.3:** Cheap-Talk design is now fully specified and implemented. The 6-condition
  identity design can be described as a contribution — it goes beyond simple deception framing
  to isolate role-specific effects.
- **Results 4.3:** Will populate from full-condition runs.
- **Results 4.5 (Δm):** Cheap-Talk Δm will be richer than PD/CD — worth highlighting.

---

## 2026-05-03 to 2026-05-04 — Cheap-Talk v1 pilot

### Context
First implementation of the signaling game. 3 models (Claude Sonnet, GPT-4o, Gemini Flash),
aligned vs misaligned conditions, fixed and rotated roles.

### Key Findings

| Observation | Finding | Interpretation |
|---|---|---|
| Aligned condition | 100% truthfulness across all models | LLMs default to honesty when incentives are shared — consistent with equilibrium |
| Misaligned — sender truthfulness | 40-53% lie rates across models | LLMs do engage in strategic deception when payoffs reward it |
| Misaligned — receiver skepticism | Varies: Claude 30-37% follow; Gemini 67-80% | Significant between-model variance in credulity |
| Temperature effect | Small (T=0.3 vs T=0.7 results similar) | Strategic dispositions appear robust to temperature in this range |

### Hypotheses Status After This Data

| Hypothesis | Direction |
|---|---|
| H1 (scale → CECS) | Cannot test yet — need full cross-game data |
| H2 (instruction-tuning → norm-conforming) | Suggestive: Claude more sceptical receiver, less exploitable |
| H3 (temperature → variance) | Weak: small effect in pilot; needs systematic H3 test |
| H4 (Δm neg. correlated with CECS) | Cannot test yet |

### Anomaly: Gemini remains highly trusting even after repeated deceptions
- Gemini receiver follow rate: 67-80% in misaligned condition despite sender deception rate of 47%
- This is super-credulous behavior that persists through the game
- **Paper note:** This could be a striking finding about model-specific strategic personalities.
  Worth a dedicated paragraph in Results and Discussion.

### Paper Implications
- **Results 4.3:** Table of sender truthfulness rates (aligned / misaligned) and receiver
  follow rates across models. Deception success rates.
- **Discussion:** The Gemini credulity anomaly. Between-model strategic personality differences.
  Whether "deception" in LLMs is strategic or stochastic.

---

## 2026-04-29 — Commons Dilemma pilot (3 conditions, 5 rounds)

### Context
Commons Dilemma tests whether cooperative behavior survives when direct dyadic reciprocity
is removed. Pool: 100 units, regen: 20/round fixed, 2 players, max extraction: 20.
Sustainable share: 10 units/player/round. Sustainable share NOT shown in prompt — models
must discover it themselves.

### Key Design Decision: Sustainable Share Hidden
We do not tell models the sustainable extraction threshold. This forces θ to emerge from
behavior rather than instruction-following. Important for academic validity — we are probing
strategic discovery, not instruction compliance.

### Open Issues from Pilot
1. Only 5 rounds per session — too short to observe learning or pool depletion dynamics
2. OpenAI models ran at T=1.0 while others at T=0.6 — temperature not controlled
3. Files at branch root (not `data/raw/`) — affects reproducibility

### Paper Implications
- **Methods 3.2:** Pool collapse (θ) as a behavioral proxy for resource exploitation threshold.
  Note that hidden sustainable share design is a strength for ecological validity.
- **Limitations:** 5-round pilot is insufficient for θ estimation — acknowledge, plan full runs.

---

## 2026-04-27 — Repository restructure + PD audit

### Decisions Made

**D7: Prompt v4.1 — format enforcement line added**
Gemini Flash had 6.5% parse failure rate (61 of 960 rounds defaulting to DEFECT).
These defaults artificially inflate defection counts and corrupt ρ and β estimates.
Added: "Any text outside the JSON will cause your response to be rejected."
Classified as format change (not framing) — justified.

**D8: TEMPERATURE global variable (was per-model hardcode)**
Before v4.1 restructure, temperature was set individually per model in the registry.
This meant filenames had to be renamed manually to reflect the temperature run.
Now: one global `TEMPERATURE` variable flows to all models and auto-generates filename suffix.

**D9: Output paths standardized to data/raw/**
Before: scripts wrote to root directory. Files moved manually post-run.
After: `pathlib` resolves `data/raw/` relative to script location — automatic.

### Known Data Quality Issues (as of this session)
- T=0.3 only for undisclosed condition — H3 untestable across conditions until corrected
- 1 replication per matchup — no confidence intervals
- Claude Sonnet + GPT-4o-mini never defect → ρ = NaN for these models; investigate
- Gemini Flash pilot data partially contaminated by 300-token limit (pre-v4.1)

### Paper Implications
- **Methods 3.5 (Controls):** Global temperature variable ensures temperature is held constant
  across models within each condition run. This should be documented as a control.
- **Limitations:** Single replication per matchup → no confidence intervals. Plan ≥3 replications
  for the final analysis.

---

## Running Notes for Paper Sections

### 1. Introduction — Draft Notes

**Research question (one sentence):**
Do large language models possess coherent strategic architectures — behavioral parameters that
respond predictably to structural game features — or does their behavior reflect prompt-level
pattern matching that collapses under structural variation?

**Contribution claim:**
We introduce the Cross-Environment Consistency Score (CECS), a behavioral fingerprinting
metric that measures strategic coherence across structurally distinct game-theoretic
environments, and apply it to six state-of-the-art LLMs across three games.

**Why this matters (hook for intro):**
LLMs are increasingly deployed as agents in economic and social environments. Understanding
whether their strategic behavior is coherent or context-fragmented has direct implications
for their reliability as social actors.

### 2. Literature Review — Key Anchors

| Topic | Source | Relevance |
|---|---|---|
| LLMs in PD | Dvorak et al. (2025) | Baseline: 10-15% lower cooperation vs humans; spillover |
| LLMs in bargaining | Ferraz et al. (2025) | 400k decisions, 17 models vs 4.1k humans |
| LLM morality | Capraro (2025) | 106 instructions; prosocial LLM behavior |
| LLMs in cooperation | Sreedhar & Chilton (2025) | LLMs replicate lab PGG results |

*Note: Add cheap-talk / signaling literature. Crawford-Sobel (1982) is the foundational model
for cheap-talk signaling. Add LLM deception / honesty literature.*

### 6. Limitations — Running List

1. Single replication per matchup in pilot — no confidence intervals
2. Prompt sensitivity not fully ruled out — v4.1 sensitivity check pending
3. Temperature not standardized across all conditions (CD branch, OpenAI at T=1.0)
4. 5-round CD sessions too short for θ estimation
5. Cheap-Talk v1 data not comparable with v2 schema (no belief, different condition names)
6. No human participant baseline — sourced from literature (document this choice explicitly)
7. Model versions fixed at time of experiment — findings may not generalize to future model versions

---

## Key Definitions (Quick Reference)

| Symbol | Name | Game | Formula |
|---|---|---|---|
| ρ | Conditional reciprocity | PD | P(C\|opp C t-1) − P(C\|opp D t-1) |
| θ | Exploitation threshold | CD | Pool size at first extraction > regen/N |
| η | Signal honesty | CT | Prop of rounds where message = true state |
| γ | Receiver gullibility | CT | P(follow\|msg=H) − P(follow\|msg=L) in misaligned |
| β | Belief calibration | All | Mean absolute error: \|stated_belief − actual_opp_action\| |
| CECS | Cross-env consistency score | All | Spearman rank corr of parameters across games |
| Δm | Opponent sensitivity | All | \|\|Sm_AI − Sm_Human\|\| (Euclidean distance) |
