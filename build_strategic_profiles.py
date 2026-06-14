"""
build_strategic_profiles.py
============================
P264 — Strategic Coherence in Large Language Models
Capstone 2026, ESADE MIBA

Reads raw CSVs from the three game experiments, estimates strategic
parameters at game level, merges into a unified strategic profile
dataset, and exports it ready for coherence analysis.

Parameters estimated
--------------------
  ρ  (rho)   — Conditional reciprocity        [PD]
  θ  (theta) — Exploitation threshold          [CD]
  η  (eta)   — Signal honesty                  [CT, sender role]
  γ  (gamma) — Receiver gullibility            [CT, receiver role]
  β  (beta)  — Belief calibration              [all three games, 0–1 scale]

Output
------
  strategic_profiles_game_level.csv   — one row per (model, game_id, source)
  strategic_profiles_model_level.csv  — one row per model (mean across games)

Usage
-----
  1. Set the three directory paths below.
  2. Run: python build_strategic_profiles.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURE PATHS — point each to the local data/raw folder of each branch
# =============================================================================

PD_RAW_DIR = "/Users/mariamorazamora/Desktop/Capstone/Capstone_2026_Game_Theory/all_raw"
CD_RAW_DIR = "/Users/mariamorazamora/Desktop/Capstone/Capstone_2026_Game_Theory/all_raw"
CT_RAW_DIR = "/Users/mariamorazamora/Desktop/Capstone/Capstone_2026_Game_Theory/all_raw"

OUTPUT_DIR = "data/processed"

# =============================================================================
# HELPERS
# =============================================================================

def load_csvs(directory: str, prefix: str) -> pd.DataFrame:
    """Load and concatenate all CSVs in directory matching prefix.
    
    Adds a file-specific prefix to game_id/run_id so that IDs starting
    from 1 in each file don't collide after concatenation.
    """
    pattern = os.path.join(directory, f"{prefix}*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found matching '{pattern}'.\n"
            f"Check that {directory} exists and contains {prefix}*.csv files."
        )
    print(f"  Found {len(files)} file(s) for prefix '{prefix}':")
    for f in sorted(files):
        print(f"    {os.path.basename(f)}")

    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        # Build a short file tag from the filename (strip directory and .csv)
        file_tag = os.path.splitext(os.path.basename(f))[0]
        # Prepend file_tag to whichever ID column exists
        for id_col in ["game_id", "run_id"]:
            if id_col in df.columns:
                df[id_col] = file_tag + "_" + df[id_col].astype(str)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def normalize_cd_beliefs(belief_series: pd.Series,
                         sustainable: float) -> pd.Series:
    """
    CD beliefs in prompt v2.2 are in extraction units (0–20).
    From v2.3 onward they are rescaled to 0–1 probability.
    Detect by whether values exceed 1.0 and normalise if needed.
    Division by (sustainable * 2) maps the plausible extraction range to ~0–1.
    """
    beliefs = pd.to_numeric(belief_series, errors="coerce")
    if beliefs.dropna().max() > 1.0:
        # Old scale: treat belief as "expected opponent extraction"
        # Convert to P(over-extract) using sustainable as midpoint
        # Simple linear rescale: 0 → 0, sustainable → 0.5, 2*sustainable → 1
        beliefs = (beliefs / (sustainable * 2)).clip(0, 1)
    return beliefs


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation ignoring NaNs; returns NaN if insufficient data."""
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


# =============================================================================
# 1. PRISONER'S DILEMMA  →  ρ, β_pd
# =============================================================================

def process_pd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate per-game parameters from PD raw data.

    ρ  = P(cooperate | opponent cooperated last round)
       − P(cooperate | opponent defected last round)
       Computed from rounds 2..N (round 1 has no prior action).
       Range: −1 (always punish cooperation) to +1 (full reciprocity).

    β_pd = MAE between stated belief P(opponent cooperates) and
           opponent's actual binary action (C=1, D=0).
           Range: 0 (perfect calibration) to 1 (worst calibration).

    Returns one row per (model, game_id).
    """
    print("\n[PD] Processing Prisoner's Dilemma...")
    records = []

    for game_id, grp in df.groupby("game_id"):
        grp = grp.sort_values("round").reset_index(drop=True)
        condition = grp["condition"].iloc[0]
        matchup   = grp["matchup"].iloc[0]

        for role, other in [("a", "b"), ("b", "a")]:
            model       = grp[f"model_{role}"].iloc[0]
            actions     = grp[f"action_{role}"].str.upper().str.strip()
            beliefs     = pd.to_numeric(grp[f"belief_{role}"], errors="coerce")
            opp_actions = grp[f"action_{other}"].str.upper().str.strip()

            # β_pd
            opp_binary = (opp_actions == "C").astype(float)
            beta = (beliefs - opp_binary).abs().mean()

            # ρ (needs ≥ 2 rounds)
            rho = np.nan
            if len(grp) >= 2:
                prior_opp = opp_actions.iloc[:-1].values
                curr_act  = actions.iloc[1:].values
                c_given_c = (
                    (curr_act[prior_opp == "C"] == "C").mean()
                    if (prior_opp == "C").any() else np.nan
                )
                c_given_d = (
                    (curr_act[prior_opp == "D"] == "C").mean()
                    if (prior_opp == "D").any() else np.nan
                )
                if not (np.isnan(c_given_c) or np.isnan(c_given_d)):
                    rho = c_given_c - c_given_d

            records.append({
                "model":     model,
                "game_id":   f"pd_{game_id}",
                "condition": condition,
                "matchup":   matchup,
                "source":    "PD",
                "rho":       round(rho, 4) if not np.isnan(rho) else np.nan,
                "beta_pd":   round(beta, 4),
            })

    result = pd.DataFrame(records)
    print(f"  → {len(result)} game-level rows | "
          f"{result['model'].nunique()} models: {sorted(result['model'].unique())}")
    return result


# =============================================================================
# 2. COMMONS DILEMMA  →  θ, β_cd
# =============================================================================

def process_cd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate per-game parameters from CD raw data.

    θ  = mean(extraction / sustainable_share) across rounds.
         Values > 1 indicate systematic over-extraction.
         This is the estimable proxy for the exploitation threshold.

    β_cd = MAE between belief P(opponent over-extracts) and whether
           opponent actually extracted > sustainable_share.
           Beliefs from v2.2 files (units scale) are auto-normalised.
           Range: 0–1.

    Returns one row per (model, game_id).
    """
    print("\n[CD] Processing Commons Dilemma...")
    records = []

    for game_id, grp in df.groupby("game_id"):
        grp        = grp.sort_values("round").reset_index(drop=True)
        condition  = grp["condition"].iloc[0]
        matchup    = grp["matchup"].iloc[0]
        sustainable = float(grp["sustainable_share"].iloc[0])

        info_cond = grp["info_condition"].iloc[0] if "info_condition" in grp.columns else np.nan
        risk_cond = grp["risk_condition"].iloc[0] if "risk_condition" in grp.columns else np.nan

        for role in ["1", "2"]:
            opp_role   = "2" if role == "1" else "1"
            model      = grp[f"model_{role}"].iloc[0]
            extraction = pd.to_numeric(grp[f"extraction_{role}"], errors="coerce")
            belief_raw = grp[f"belief_{role}"]
            opp_ext    = pd.to_numeric(grp[f"extraction_{opp_role}"], errors="coerce")

            # θ: normalised extraction ratio
            theta = (extraction / sustainable).mean() if sustainable > 0 else np.nan

            # β_cd: normalise beliefs if on old unit scale
            belief = normalize_cd_beliefs(belief_raw, sustainable)
            opp_over = (opp_ext > sustainable).astype(float)
            beta = (belief - opp_over).abs().mean()

            records.append({
                "model":          model,
                "game_id":        f"cd_{game_id}",
                "condition":      condition,
                "info_condition": info_cond,
                "risk_condition": risk_cond,
                "matchup":        matchup,
                "source":         "CD",
                "theta":          round(theta, 4) if not np.isnan(theta) else np.nan,
                "beta_cd":        round(beta, 4),
            })

    result = pd.DataFrame(records)
    print(f"  → {len(result)} game-level rows | "
          f"{result['model'].nunique()} models: {sorted(result['model'].unique())}")
    return result


# =============================================================================
# 3. CHEAP-TALK  →  η, γ, β_ct
# =============================================================================

def process_ct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate per-run parameters from CT raw data.

    η  (sender) = fraction of rounds where message_truthful == 1.
                  Range: 0 (always deceive) to 1 (always honest).

    γ  (receiver) = Pearson correlation between message_truthful and
                    action_correct across rounds of a run.
                    Proxy for how much receiver accuracy depends on sender
                    honesty — i.e., gullibility to sender framing.
                    Range: −1 to 1 (positive = follows sender's lead).

    β_ct (sender)   = MAE between sender_belief P(receiver follows) and
                      actual action_correct.
    β_ct (receiver) = MAE between receiver_belief P(sender is truthful)
                      and actual message_truthful.
    Range: 0–1.

    Returns one row per (model, run_id, role).
    """
    print("\n[CT] Processing Cheap-Talk Signaling...")
    records = []

    for run_id, grp in df.groupby("run_id"):
        grp            = grp.sort_values("round").reset_index(drop=True)
        condition      = grp["identity_condition"].iloc[0]
        game_condition = grp["game_condition"].iloc[0]
        matchup        = grp["matchup"].iloc[0]

        truthful       = pd.to_numeric(grp["message_truthful"],  errors="coerce")
        action_correct = pd.to_numeric(grp["action_correct"],    errors="coerce")

        # --- SENDER ---
        sender_belief = pd.to_numeric(grp["sender_belief"], errors="coerce")
        records.append({
            "model":          grp["model_sender"].iloc[0],
            "game_id":        f"ct_{run_id}",
            "condition":      condition,
            "game_condition": game_condition,
            "matchup":        matchup,
            "source":         "CT",
            "role":           "sender",
            "eta":            round(float(truthful.mean()), 4),
            "gamma":          np.nan,
            "beta_ct":        round(float((sender_belief - action_correct).abs().mean()), 4),
        })

        # --- RECEIVER ---
        receiver_belief = pd.to_numeric(grp["receiver_belief"], errors="coerce")
        records.append({
            "model":          grp["model_receiver"].iloc[0],
            "game_id":        f"ct_{run_id}",
            "condition":      condition,
            "game_condition": game_condition,
            "matchup":        matchup,
            "source":         "CT",
            "role":           "receiver",
            "eta":            np.nan,
            "gamma":          round(safe_corr(truthful, action_correct), 4),
            "beta_ct":        round(float((receiver_belief - truthful).abs().mean()), 4),
        })

    result = pd.DataFrame(records)
    print(f"  → {len(result)} game-level rows | "
          f"{result['model'].nunique()} models: {sorted(result['model'].unique())}")
    return result


# =============================================================================
# 4. MERGE  →  strategic_profiles_game_level.csv
# =============================================================================

def merge_profiles(pd_df, cd_df, ct_df) -> pd.DataFrame:
    """
    Concatenate three game-level datasets into a single long-format table.
    Each row = one model's parameters in one game.
    Parameters from other games are NaN for that row.

    Also adds a unified 'beta' column (0–1 scale, whichever game available).
    """
    print("\n[MERGE] Building unified game-level profile dataset...")

    common = ["model", "game_id", "condition", "source", "matchup"]

    pd_clean = pd_df[common + ["rho", "beta_pd"]].copy()
    cd_clean = cd_df[common + ["info_condition", "risk_condition",
                                "theta", "beta_cd"]].copy()
    ct_clean = ct_df[common + ["game_condition", "role",
                                "eta", "gamma", "beta_ct"]].copy()

    merged = pd.concat([pd_clean, cd_clean, ct_clean],
                       ignore_index=True, sort=False)

    # Unified beta: take whichever environment's beta is present (all are 0–1)
    merged["beta"] = merged[["beta_pd", "beta_cd", "beta_ct"]].bfill(axis=1).iloc[:, 0]

    print(f"  → {len(merged)} total rows")
    print(f"  → {merged['model'].nunique()} models")
    for src in ["PD", "CD", "CT"]:
        n = merged[merged["source"] == src]["game_id"].nunique()
        print(f"     {src}: {n} games")

    return merged


# =============================================================================
# 5. MODEL-LEVEL AGGREGATION  →  strategic_profiles_model_level.csv
# =============================================================================

def aggregate_model_level(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate game-level parameters to one row per model.

    For each model:
      - mean and std of ρ, θ, η, γ across games
      - β_mean per environment (pd, cd, ct)
      - beta_cross_env_mean: mean β across environments (coherence anchor)
      - beta_cross_env_var:  variance of β across environments
                             (lower = more coherent belief calibration)
      - n_games per source
    """
    print("\n[AGGREGATE] Building model-level strategic profiles...")
    rows = []

    for model, grp in game_df.groupby("model"):
        row = {"model": model}

        for param in ["rho", "theta", "eta", "gamma"]:
            vals = pd.to_numeric(grp[param], errors="coerce").dropna()
            row[f"{param}_mean"] = round(vals.mean(), 4) if len(vals) > 0 else np.nan
            row[f"{param}_std"]  = round(vals.std(),  4) if len(vals) > 1 else np.nan
            row[f"{param}_n"]    = len(vals)

        # Beta per environment
        for env, col in [("pd", "beta_pd"), ("cd", "beta_cd"), ("ct", "beta_ct")]:
            if col in grp.columns:
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                row[f"beta_{env}_mean"] = round(vals.mean(), 4) if len(vals) > 0 else np.nan
            else:
                row[f"beta_{env}_mean"] = np.nan

        # Cross-environment beta coherence (key metric from Option 1 doc)
        beta_envs = pd.Series([
            row.get("beta_pd_mean"),
            row.get("beta_cd_mean"),
            row.get("beta_ct_mean"),
        ]).dropna()
        row["beta_cross_env_mean"] = round(beta_envs.mean(), 4) if len(beta_envs) > 0 else np.nan
        row["beta_cross_env_var"]  = round(beta_envs.var(),  4) if len(beta_envs) > 1 else np.nan

        # Game counts
        for src in ["PD", "CD", "CT"]:
            row[f"n_games_{src.lower()}"] = grp[grp["source"] == src]["game_id"].nunique()

        rows.append(row)

    result = pd.DataFrame(rows)
    print(f"  → {len(result)} model-level profiles")
    return result


# =============================================================================
# 6. COHERENCE METRICS (printed summary)
# =============================================================================

def print_coherence_metrics(model_df: pd.DataFrame) -> None:
    """
    Print cross-environment coherence metrics as defined in Option 1 doc:

    A. Cm = Var(β across environments) per model
    B. Correlations among ρ, θ, η, γ across models
    C. Full parameter table
    """
    print("\n" + "=" * 60)
    print("CROSS-ENVIRONMENT COHERENCE METRICS")
    print("=" * 60)

    # A. Beta variance per model
    print("\nA. β cross-env variance per model  [Cm = Var(Sm^env)]")
    print("   Lower = more coherent belief calibration across games")
    beta_var = model_df[["model", "beta_cross_env_var"]].dropna()
    if not beta_var.empty:
        for _, r in beta_var.sort_values("beta_cross_env_var").iterrows():
            bar = "█" * int(r["beta_cross_env_var"] * 100)
            print(f"   {r['model']:<30} {r['beta_cross_env_var']:.4f}  {bar}")
    else:
        print("   Not enough data across environments yet.")

    # B. Parameter correlations
    print("\nB. Cross-parameter correlations  [Corr(ρ, θ, η, γ)]")
    params = ["rho_mean", "theta_mean", "eta_mean", "gamma_mean"]
    avail  = [p for p in params if p in model_df.columns
              and model_df[p].notna().sum() >= 3]
    if len(avail) >= 2:
        print(model_df[avail].corr().round(3).to_string())
    else:
        print("   Insufficient parameter coverage (need ≥3 models per parameter).")

    # C. Full table
    print("\nC. Strategic profile vector per model")
    cols = ["model", "rho_mean", "theta_mean", "eta_mean", "gamma_mean",
            "beta_pd_mean", "beta_cd_mean", "beta_ct_mean",
            "beta_cross_env_mean", "beta_cross_env_var"]
    avail_cols = [c for c in cols if c in model_df.columns]
    print(model_df[avail_cols].round(3).to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("P264 — Strategic Profile Builder")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load ---
    print("\n[LOAD] Reading raw CSV files...")
    print(f"  PD → {PD_RAW_DIR}")
    print(f"  CD → {CD_RAW_DIR}")
    print(f"  CT → {CT_RAW_DIR}")

    pd_raw = load_csvs(PD_RAW_DIR, "pd_results_")
    cd_raw = load_csvs(CD_RAW_DIR, "cd_")
    ct_raw = load_csvs(CT_RAW_DIR, "cheap_talk_results_")

    print(f"\n  Loaded: PD={len(pd_raw)} rows | "
          f"CD={len(cd_raw)} rows | CT={len(ct_raw)} rows")

    # --- Estimate parameters ---
    pd_params = process_pd(pd_raw)
    cd_params = process_cd(cd_raw)
    ct_params = process_ct(ct_raw)

    # --- Merge and aggregate ---
    game_level  = merge_profiles(pd_params, cd_params, ct_params)
    model_level = aggregate_model_level(game_level)

    # --- Print coherence summary ---
    print_coherence_metrics(model_level)

    # --- Export ---
    game_path  = os.path.join(OUTPUT_DIR, "strategic_profiles_game_level.csv")
    model_path = os.path.join(OUTPUT_DIR, "strategic_profiles_model_level.csv")
    game_level.to_csv(game_path,  index=False)
    model_level.to_csv(model_path, index=False)

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"  {game_path}")
    print(f"  {model_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
