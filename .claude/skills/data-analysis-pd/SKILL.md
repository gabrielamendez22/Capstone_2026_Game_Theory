---
name: data-analysis-pd
description: Use when computing strategic profile parameters (ρ, β, θ, η, γ), CECS, or Δm from experiment CSV data, or when writing analysis scripts and notebooks. Covers the canonical pandas patterns, edge cases in the current dataset, and the long-format reshape that makes parameter computation correct.
---

# Data Analysis Skill — Prisoner's Dilemma and Strategic Profile

## Canonical Loading Pattern

Always use this loader pattern. It handles all current CSV variants.

```python
import pandas as pd
import numpy as np

def load_pd(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df
```

## The Long-Format Reshape (CRITICAL)

The CSV stores **one row per round per game**, with two players' data side-by-side
(`_a` and `_b` suffixes). For per-model analysis, reshape to **one row per
player per round**:

```python
def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide CSV to long format: one row per player per round."""
    def _side(player: str) -> pd.DataFrame:
        other = "b" if player == "a" else "a"
        cols = {
            f"model_{player}":      "model",
            f"action_{player}":     "action",
            f"belief_{player}":     "belief",
            f"payoff_{player}":     "payoff",
            f"cumulative_{player}": "cumulative",
            f"action_{other}":      "opp_action",
            f"belief_{other}":      "opp_belief",
        }
        base = df[list(cols.keys()) +
                  ["game_id", "matchup", "round", "condition"]].rename(columns=cols)
        base["side"] = player
        return base

    long = pd.concat([_side("a"), _side("b")], ignore_index=True)
    long["coop"]     = (long["action"]     == "C").astype(int)
    long["opp_coop"] = (long["opp_action"] == "C").astype(float)
    long["beta_err"] = (long["belief"] - long["opp_coop"]).abs()
    return long
```

This pattern matches `dashboard/data_loader.py::compute_long_df()`. Use it.

## Computing ρ (Conditional Reciprocity)

```python
def compute_rho(long_df: pd.DataFrame) -> pd.Series:
    """ρ = P(C | opp_C at t-1) − P(C | opp_D at t-1) per model."""
    df = long_df.sort_values(["game_id", "model", "round"]).copy()
    df["prev_opp"] = df.groupby(["game_id", "model"])["opp_action"].shift(1)
    valid = df.dropna(subset=["prev_opp"])

    def _rho(grp):
        c_mask = grp["prev_opp"] == "C"
        d_mask = grp["prev_opp"] == "D"
        if not c_mask.any() or not d_mask.any():
            return np.nan
        return float(grp.loc[c_mask, "coop"].mean() -
                     grp.loc[d_mask, "coop"].mean())

    return valid.groupby("model").apply(_rho)
```

**Edge case:** if a model never defected in the dataset, ρ is `NaN` (no D in
prev_opp). Document this in the result. Don't impute zero.

## Computing β (Belief Calibration)

```python
def compute_beta(long_df: pd.DataFrame) -> pd.Series:
    """β = mean|belief_t − actual_opp_action_t| per model."""
    return long_df.groupby("model")["beta_err"].mean()
```

Range: 0 (perfect) to 1 (always wrong). 0.5 = no better than random.

## Computing CECS (Cross-Environment Consistency Score)

CECS requires data from **at least two games**. With only PD data available,
CECS cannot yet be computed — write code that gracefully reports this:

```python
def compute_cecs(profile_vectors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    profile_vectors: {game_name: DataFrame with index=model, columns=parameters}
    Returns Spearman rank correlation per pair of games per model.
    """
    if len(profile_vectors) < 2:
        raise ValueError("CECS requires data from at least 2 games")
    # ... pairwise Spearman across games ...
```

Until Commons Dilemma data exists, CECS scaffolding stays as a stub.

## Filtering by Condition

```python
df_ai     = df[df["condition"] == "ai"]
df_human  = df[df["condition"] == "human"]
df_undsc  = df[df["condition"] == "undisclosed"]
```

When comparing across conditions, use `groupby(["model", "condition"])`.

## Computing Δm (Opponent Sensitivity)

```python
def compute_delta_m(profile_ai: pd.DataFrame,
                    profile_human: pd.DataFrame) -> pd.Series:
    """Δm = Euclidean distance between AI-condition and human-condition profiles."""
    common = profile_ai.index.intersection(profile_human.index)
    diff = profile_ai.loc[common] - profile_human.loc[common]
    return np.sqrt((diff ** 2).sum(axis=1))
```

## Edge Cases in the Current Pilot Data

These are real issues seen in the existing CSV files — handle them:

1. **Gemini Flash truncated JSON**
   `raw_output_b` ends with `"action":` and parser defaults to D.
   Visible in early rounds before `max_tokens` was raised.
   Filter with: `df[~df["raw_output_b"].str.contains("```json", na=False)]`
   if studying clean compliance, or accept as-is for behavior analysis.

2. **Beliefs of exactly 0.5 in defection rounds**
   Often signals a parsing fallback, not a genuine 50% belief.
   Cross-check by inspecting `raw_output` for that round.

3. **Zero token usage in older runs**
   `token_usage_a/b` is `{"prompt": 0, "completion": 0}` in early data.
   The newer LangChain version logs real counts. Filter or accept depending
   on whether token usage is part of the analysis.

## Phase-Based Analysis (Backward Induction Test)

```python
df["phase"] = pd.cut(df["round"],
                     bins=[0, 5, 15, 20],
                     labels=["early", "mid", "late"])
phase_coop = df.groupby(["model", "phase"])["coop"].mean().unstack()
```

Late-phase cooperation drops are the signature of backward induction. This is
how the dashboard's "Backward Induction Index" is computed.

## Plotting Conventions

- Use the model colors from `dashboard/data_loader.py::model_color()`
- Anthropic = purple, OpenAI = green, Google = blue
- For belief trajectories, defect actions render as red `x` markers
- Always include zero-line and 0.5-line as reference

## Anti-Patterns (do NOT do)

- Don't compute ρ from the wide-format CSV directly — always reshape to long first
- Don't impute missing ρ as 0 — it means "untestable", not "no reciprocity"
- Don't average parameters across conditions when condition is a study variable
- Don't drop rounds with non-compliant outputs silently — log them explicitly
