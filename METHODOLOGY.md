# METHODOLOGY.md — Research Definitions and Parameter Specifications

This file defines the research constructs precisely so that any code written
for analysis uses consistent, correct definitions. When in doubt, refer here.

---

## Research Question

Do LLMs possess coherent strategic architectures — behavioral parameters that
respond predictably to the structural features of strategic environments — or
does their behavior reflect prompt-level pattern matching that collapses under
structural variation?

---

## Strategic Profile Vector

For each model m, we define:

**Sm = (ρ, θ, η, γ, β)**

| Parameter | Name | Environment | Definition |
|---|---|---|---|
| ρ | Conditional reciprocity | Prisoner's Dilemma | P(C at t \| opponent C at t−1) − P(C at t \| opponent D at t−1) |
| θ | Exploitation threshold | Commons Dilemma | Pool size at which model's extraction first exceeds per-capita regeneration rate |
| η | Signal honesty | Cheap-Talk | Proportion of rounds where sent message matched intended action |
| γ | Receiver gullibility | Cheap-Talk | P(receiver C \| message="cooperate") − P(receiver C \| message="defect") |
| β | Belief calibration | All environments | Mean absolute error: (1/T) Σ \|stated_belief_t − actual_opponent_action_t\| |

---

## Cross-Environment Consistency Score (CECS)

The headline metric. Computed as the Spearman rank correlation of behavioral
parameters across games for the same model.

- CECS near 1 → coherent strategic architecture
- CECS near 0 → context-fragmented behavior

Used to test H1 (scaling increases coherence).

---

## Opponent Sensitivity (Δm)

Δm = ||Sm_AI − Sm_Human||

The Euclidean distance between a model's strategic profile in AI-AI conditions
versus Human-AI conditions. Tests whether the model's strategy shifts when the
opponent changes from AI to human.

Expected finding (H4): Δm negatively correlates with CECS — coherent models
are also opponent-stable.

---

## Hypotheses

| # | Hypothesis | Metric |
|---|---|---|
| H1 | Larger models exhibit higher CECS (more stable architectures) | CECS by model scale |
| H2 | Instruction-tuned models show higher ρ and lower θ (more norm-conforming) | ρ, θ by tuning variant |
| H3 | Higher temperature increases Cm (cross-environment variance) | Var(Sm) at T=0.3 vs T=0.6 |
| H4 | Opponent sensitivity (Δm) negatively correlates with CECS | Spearman(Δm, CECS) |

---

## Prompt Sensitivity — Validity Threat and Controls

**Prompt sensitivity is the primary methodological risk of this research.**
If behavioral parameters vary because of surface-level prompt framing rather
than structural game features, the CECS construct is invalid.

### What it is

Prompt sensitivity occurs when a model's action (C or D) changes in response
to *how the game is described* rather than *what the payoff structure is*.
It is distinct from legitimate structural variation:

| Source of variation | Type | Interpretation |
|---|---|---|
| Different payoff values (T, R, P, S) | Structural | Expected and rational |
| Different game (PD vs Commons) | Structural | The entire point of the design |
| Different wording of same instructions | Surface | Prompt sensitivity — confound |
| Different example shown (C vs D) | Surface | Prompt sensitivity — confound |
| Different objective framing ("maximize" vs neutral) | Surface | Prompt sensitivity — confound |

### Controls already implemented in v4.0

- **Neutral objective** ("strategically optimal") — not "maximize score"
- **Both valid examples shown** — one C, one D — no single-option anchoring
- **Round total hidden** — prevents horizon-based anchoring
- **Belief timing explicit** — "formed BEFORE choosing" prevents post-hoc rationalization
- **Identical prompt across all models** — prompt is never a between-model variable

### Detecting prompt sensitivity in data

Four quantitative checks are defined in:
→ `.claude/skills/prompt-engineering/SKILL.md`

The four checks are:
1. BCR variance across temperature conditions (H3 test)
2. Ceiling effect detection in round-1 cooperation rate
3. Belief-action alignment against persistent defectors
4. Cross-condition BCR comparison for opponent identity effect

### How to report it

Do not claim prompt-sensitivity-free design. Claim **controlled** design
with documented sensitivity checks. This is the defensible academic position.

### Prompt change protocol

Any proposed change to the system prompt must follow the protocol in:
→ `.claude/skills/prompt-engineering/SKILL.md`

Never change the system prompt without incrementing `PROMPT_VERSION` and
running a sensitivity check on one model pair first.

---

## Prompt Version History

| Version | Change | Date | Sensitivity Check |
|---|---|---|---|
| v1.0 | Initial — maximize score objective, single cooperate example | Mar 2026 | None |
| v2.0 | Removed "maximize score", added JSON structure | Mar 2026 | None |
| v3.0 | Added belief timing instruction | Mar 2026 | None |
| v4.0 | Neutral objective + both examples + belief timing + JSON-only | Apr 2026 | Informal |
| v4.1 | Added enforcement line: "Any text outside the JSON will cause your response to be rejected." — legitimate format change (not framing); motivated by Gemini Flash parse failure rate of 6.5% in pilot data | Apr 2026 | None yet — run sensitivity check before next full experiment |

---

## Game 1: Prisoner's Dilemma

**Structural feature:** Dyadic reciprocity, direct feedback, finite horizon.

**Payoff matrix (T > R > P > S):**

| | Opponent C | Opponent D |
|---|---|---|
| You C | (3, 3) | (0, 5) |
| You D | (5, 0) | (1, 1) |

**Parameters estimated:** ρ, β

**Computing ρ from data:**
```python
df_sorted = df.sort_values(["game_id", "model", "round"])
df_sorted["prev_opp"] = df_sorted.groupby(["game_id", "model"])["opp_action"].shift(1)
valid = df_sorted.dropna(subset=["prev_opp"])
rho = (valid[valid["prev_opp"]=="C"]["coop"].mean()
     - valid[valid["prev_opp"]=="D"]["coop"].mean())
```

**Computing β from data:**
```python
df["beta_err"] = (df["belief"] - df["opp_coop"].astype(float)).abs()
beta = df["beta_err"].mean()
```

---

## Game 2: Commons Dilemma

**Structural feature:** Collective action, N players, shared resource pool,
no direct dyadic reciprocity. Tests whether cooperative behavior survives
when the reciprocity structure disappears.

**Key design decisions (to be finalized):**
- Pool size R and regeneration rate r communicated to models explicitly
- θ = extraction level at which model's choice first exceeds r/N
- Multiple agents extract simultaneously per round
- Model sees pool size remaining at start of each round

**Parameters estimated:** θ, β

**Prompt sensitivity note for Commons:** The Commons prompt introduces a
new framing (resource extraction) with no equivalent in PD. This is legitimate
structural variation. However, objective wording must remain neutral — do not
add "preserve the commons" or "be a responsible actor" language.

---

## Game 3: Cheap-Talk / Signaling

**Structural feature:** Sender-receiver asymmetry, communication without binding
commitment, information asymmetry.

**Key design decisions (to be finalized):**
- Sender knows intended action before message
- Message and action logged separately; η computed post-hoc
- Role assignment: fixed or rotating per round (TBD)

**Parameters estimated:** η, γ

**Prompt sensitivity note for Cheap-Talk:** η and γ are directly affected by
whether the prompt implies honesty is expected. Do not include normative
framing about communication. Describe the mechanic neutrally only.

---

## Data Analysis Conventions

### Column naming in CSV/DB

- `action_a`, `action_b` → always `C` or `D` (never "COOPERATE")
- `belief_a`, `belief_b` → float 0.0–1.0
- `condition` → `"undisclosed"`, `"ai"`, or `"human"`

### Derived columns to add for analysis

```python
df["coop"] = (df["action_a"] == "C").astype(int)
df["opp_coop"] = (df["action_b"] == "C").astype(float)
df["beta_err"] = (df["belief_a"] - df["opp_coop"]).abs()
df["phase"] = pd.cut(df["round"], bins=[0,5,15,20],
                     labels=["early","mid","late"])
```

### Filtering by condition

```python
df_ai = df[df["condition"] == "ai"]
df_human = df[df["condition"] == "human"]
df_none = df[df["condition"] == "undisclosed"]
```

---

## Human-Human Benchmarks (from literature)

Do not run Human-Human experiments. Source from existing literature:

| Source | Game | Key Finding |
|---|---|---|
| Dvorak et al. (2025) | PD | 10-15% lower cooperation with LLMs vs. humans; spillover effect |
| Ferraz et al. (2025) | UG | 400k decisions across 17 models vs. 4.1k humans |
| Capraro (2025) | DG | 106 instructions for global human experiments |
| Sreedhar & Chilton (2025) | PGG | LLMs successfully replicate lab cooperation results |