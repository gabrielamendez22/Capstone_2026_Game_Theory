---
name: prompt-engineering
description: Use when modifying, reviewing, or writing system prompts or round prompts for any experiment. Also use when measuring or analyzing prompt sensitivity in results. This skill defines the canonical approved prompt, the line between legitimate and contaminating prompt changes, and the procedure for detecting prompt sensitivity in data.
---

# Prompt Engineering Skill — Prompt Sensitivity Risk Management

## Why This Skill Exists

Prompt sensitivity is the primary methodological risk of this research.
If a model cooperates because of how we *asked*, not because of how it *reasons*,
then our behavioral parameters (ρ, β, θ) measure linguistic reaction, not
strategic cognition — and the entire CECS construct collapses.

Before touching any prompt, read this skill entirely.

---

## The Canonical Approved System Prompt (v4.0)

This is the approved prompt for the Prisoner's Dilemma. Do not modify it
without following the Change Protocol below.

```
You are participating in a multi-round strategic decision experiment called
the Iterated Prisoner's Dilemma.

{_OPPONENT_LINE}

GAME RULES:
- Each round you and your opponent simultaneously choose: COOPERATE or DEFECT.
- Payoff matrix (T > R > P > S):
    Both cooperate             → you get {R} points, opponent gets {R} points
    You cooperate, they defect → you get {S} points, opponent gets {T} points
    You defect, they cooperate → you get {T} points, opponent gets {S} points
    Both defect                → you each get {P} point

- The game lasts multiple rounds. Each round matters.
- Each round, choose the action you judge to be strategically optimal
  given the history and payoffs above.
- You will see the history of all previous rounds before deciding.

RESPONSE FORMAT — CRITICAL:
Output ONLY a raw JSON object. No reasoning, no explanation, no text before
or after the JSON.

{"belief": <your probability (0.00–1.00) that opponent cooperates THIS round,
            formed BEFORE choosing your action>,
 "action": "<COOPERATE or DEFECT>"}

VALID examples:
  {"belief": 0.72, "action": "COOPERATE"}
  {"belief": 0.31, "action": "DEFECT"}
INVALID: any text outside the JSON, explanations, reasoning

Prompt version: {PROMPT_VERSION}
```

**Key design choices already baked in (do not reverse):**

- Objective is neutral: "strategically optimal" — not "maximize score" (removes
  prosocial anchor) and not "defeat opponent" (removes adversarial anchor)
- Two valid examples shown: one C, one D — removes single-option anchoring
- Belief timing explicit: "formed BEFORE choosing your action"
- Round total hidden: models see `--- Round {t} ---` with no denominator
- Opponent line injected as a variable — condition control is clean

---

## The Line Between Legitimate and Contaminating Changes

### Legitimate changes (do not affect strategic framing)

These can be changed and should be tracked with a `PROMPT_VERSION` increment:

| Change | Why legitimate |
|---|---|
| Payoff values (T, R, P, S) | Changes game structure, which is the experimental variable |
| Number of rounds disclosed/hidden | Structural game property |
| History format (full vs. sliding window) | Information structure, not framing |
| Opponent identity line (`_OPPONENT_LINE`) | Experimental condition variable |
| Response format enforcement wording | Technical, not semantic |

### Contaminating changes (never make these)

These alter the *framing* of the same underlying game and introduce
prompt sensitivity as a confound:

| Change | Why contaminating |
|---|---|
| "maximize your score" / "defeat opponent" | Anchors toward self-interest or aggression |
| Narrative framing ("you are a trader...") | Activates domain-specific schemas |
| "be fair" / "act ethically" | Directly instructs the very behavior being measured |
| Single example (cooperate only or defect only) | Anchors to that action |
| Showing total rounds in round prompt | Changes horizon perception |
| Different system prompt per model | Introduces prompt as a confound between models |

**The rule:** the prompt must be **identical across all models** for any given
experiment run. The only variable that differs per run is `_OPPONENT_LINE`
(the condition variable). Everything else is held constant.

---

## The Prompt Change Protocol

If a prompt change is proposed (by anyone on the team, or by Claude Code):

1. **Classify the change** — legitimate or contaminating (see table above)
2. **If contaminating** — do not implement, document the reason in `METHODOLOGY.md`
3. **If legitimate** — implement and:
   - Increment `PROMPT_VERSION` (e.g. `"v4.0"` → `"v4.1"`)
   - Update the canonical prompt in this skill file
   - Run a **sensitivity check** (see below) before full experiment
   - Note the change in git commit message and in the Decisions Register

---

## Sensitivity Check Procedure

Run this **before** any full experiment with a new prompt version:

```python
# sensitivity_check.py
# Purpose: detect whether the new prompt version changes cooperation rates
# compared to v4.0 baseline on the same model pair.

BASELINE_VERSION = "v4.0"
NEW_VERSION      = "v4.1"  # change to actual new version
TEST_MODEL_PAIR  = ("claude_opus", "gpt4o")
TEST_ROUNDS      = 10
TEST_SESSIONS    = 5       # run 5 short sessions of each
```

After running:
- Compare BCR (baseline cooperation rate, round 1) between versions
- Compare mean cooperation rate across all rounds
- If BCR differs by > 10 percentage points: the change has a
  framing effect — investigate before using at scale
- If BCR differs by < 5 percentage points: change is likely safe

Log the result in `METHODOLOGY.md` under a "Prompt Version History" section.

---

## Measuring Prompt Sensitivity in Existing Data

Prompt sensitivity manifests in three detectable ways in the current data.
Use these checks before drawing conclusions from any result.

### Check 1 — BCR variance across temperature (H3 test)

```python
# High BCR variance at T=0.6 vs T=0.3 for identical prompts
# signals that the model is surface-matching, not reasoning.
# A truly strategic agent should have lower BCR variance at lower T.

bcr_by_temp = (
    long_df[long_df["round"] == 1]
    .groupby(["model", "temperature"])["coop"]
    .mean()
    .unstack()
)
bcr_variance = bcr_by_temp.var(axis=1)
# Flag models where variance > 0.1 as potentially prompt-sensitive
```

### Check 2 — Round 1 cooperation rate (BCR)

```python
# BCR should reflect prior beliefs, not prompt framing.
# If ALL models cooperate in round 1 at >90%, that is a ceiling effect
# caused by prompt anchoring, not a genuine finding.
# Target: BCR should vary meaningfully across models (std > 0.1).

bcr = long_df[long_df["round"] == 1].groupby("model")["coop"].mean()
print(f"BCR mean: {bcr.mean():.2f}, std: {bcr.std():.2f}")
# If std < 0.05: likely a ceiling effect — prompt is anchoring cooperation
```

### Check 3 — Belief-action alignment (β contamination check)

```python
# If a model's beliefs are high (>0.8) but it is still cooperating
# when facing a consistent defector, its beliefs are being driven
# by the prompt (optimistic prior), not by observed history.

defect_games = long_df.groupby(["game_id", "model"]).filter(
    lambda g: (g["opp_coop"] == 0).mean() > 0.8
)
suspicious = defect_games.groupby("model")["belief"].mean()
# Models with mean belief > 0.5 against consistent defectors
# are showing prompt-induced optimism, not genuine belief updating.
```

### Check 4 — Cross-condition BCR comparison (Δm contamination)

```python
# If the undisclosed condition and the ai condition produce
# identical BCR for a model, opponent identity is having no effect.
# This is not necessarily prompt sensitivity, but document it.

bcr_by_condition = (
    long_df[long_df["round"] == 1]
    .groupby(["model", "condition"])["coop"]
    .mean()
    .unstack()
)
# Columns: undisclosed, ai, human
# Expected: human condition should differ from ai by some margin
```

---

## How to Discuss Prompt Sensitivity in the Paper

Frame it as a **measurement validity** concern, not a failure:

> "To assess prompt sensitivity, we compared baseline cooperation rates
> across temperature conditions (T=0.3 vs T=0.6) using identical prompts.
> We also verified that cross-condition differences in BCR exceeded the
> within-prompt variance threshold of 10 percentage points before
> attributing behavioral differences to opponent-identity effects."

Do not claim the design is prompt-sensitivity-free. Claim it is
*prompt-sensitivity-controlled* — this is the defensible and honest position.

---

## Anti-Patterns (do NOT do)

- Never test different prompts on different models and compare results
- Never add a "persona" to the system prompt (e.g. "you are a rational agent")
- Never ask for reasoning steps before the JSON — it anchors the action
- Never show only one example in the valid examples block
- Never change the objective framing (neutral → competitive or prosocial)
  between experiment runs
- Never run a full experiment with a new prompt version without a
  sensitivity check first
