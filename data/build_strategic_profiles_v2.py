"""
build_strategic_profiles_v2.py
================================
P264 — Strategic Coherence in Large Language Models
Capstone 2026, ESADE MIBA

Revised per Carlos Carrasco-Farré feedback (June 2026):

Changes from v1
---------------
  θ  — Redefined as exploitation intensity: mean(extraction_i /
        (pool_before_regen × regen_rate / num_players)) per game.
        Sustainable play = 1.0; over-extraction > 1.0.
        regen_rate is inferred from sustainable_share column
        (stored as percentage, e.g. 2 → 0.20).

  η  — Split into η_aligned and η_misaligned; Δη = η_aligned − η_misaligned
        is the primary estimate. Pooled η is reported for reference only.

  γ  — Computed on misaligned rounds only, where the sender has
        an incentive to deceive. Pooled γ is reported for reference only.

  β  — No longer pooled across environments. Reported separately as
        β_pd, β_cd, β_ct. Cross-environment coherence is measured as
        the Pearson correlation among per-model β values, not their mean.

Parameters estimated
--------------------
  ρ    — Conditional reciprocity                [PD]
  θ    — Exploitation intensity (pool-relative) [CD]
  η_a  — Signal honesty, aligned rounds         [CT sender]
  η_m  — Signal honesty, misaligned rounds      [CT sender]
  Δη   — Honesty drop under misalignment        [CT sender, primary]
  γ_m  — Receiver gullibility, misaligned only  [CT receiver, primary]
  β_pd — Belief calibration                     [PD]
  β_cd — Belief calibration                     [CD]
  β_ct — Belief calibration                     [CT]

Output
------
  strategic_profiles_game_level_v2.csv   — one row per (model, game_id, source)
  strategic_profiles_model_level_v2.csv  — one row per model
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURE PATHS
# =============================================================================

RAW_DIR    = "/Users/mariamorazamora/Desktop/Capstone/Capstone_2026_Game_Theory/all_raw"   # single folder with all CSVs from all three games
OUTPUT_DIR = "data/processed"

# =============================================================================
# HELPERS
# =============================================================================

def load_csvs(directory: str, prefix: str) -> pd.DataFrame:
    """Load and concatenate all CSVs matching prefix, with unique game IDs."""
    pattern = os.path.join(directory, f"{prefix}*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found: '{pattern}'\n"
            f"Check that {directory} exists and contains {prefix}*.csv files."
        )
    print(f"  [{prefix}] {len(files)} files")
    dfs = []
    for f in files:
        df       = pd.read_csv(f)
        file_tag = os.path.splitext(os.path.basename(f))[0]
        for id_col in ["game_id", "run_id"]:
            if id_col in df.columns:
                df[id_col] = file_tag + "_" + df[id_col].astype(str)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Pearson r ignoring NaNs; NaN if fewer than 3 valid pairs."""
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


# =============================================================================
# 1. PRISONER'S DILEMMA  →  ρ, β_pd
# =============================================================================

def process_pd(df: pd.DataFrame) -> pd.DataFrame:
    """
    ρ = P(C | opp cooperated last round) − P(C | opp defected last round)
        Computed from rounds 2..N. Range: −1 to +1.

    β_pd = MAE(stated P(opp cooperates), actual opp action ∈ {0,1})
    """
    print("\n[PD] Processing...")
    records = []

    for game_id, grp in df.groupby("game_id"):
        grp       = grp.sort_values("round").reset_index(drop=True)
        condition = grp["condition"].iloc[0]
        matchup   = grp["matchup"].iloc[0]

        for role, other in [("a", "b"), ("b", "a")]:
            model       = grp[f"model_{role}"].iloc[0]
            actions     = grp[f"action_{role}"].str.upper().str.strip()
            beliefs     = pd.to_numeric(grp[f"belief_{role}"], errors="coerce")
            opp_actions = grp[f"action_{other}"].str.upper().str.strip()

            # β_pd
            opp_binary = (opp_actions == "C").astype(float)
            beta_pd    = (beliefs - opp_binary).abs().mean()

            # ρ
            rho = np.nan
            if len(grp) >= 2:
                prior_opp = opp_actions.iloc[:-1].values
                curr_act  = actions.iloc[1:].values
                cgc = (curr_act[prior_opp == "C"] == "C").mean() \
                      if (prior_opp == "C").any() else np.nan
                cgd = (curr_act[prior_opp == "D"] == "C").mean() \
                      if (prior_opp == "D").any() else np.nan
                if not (np.isnan(cgc) or np.isnan(cgd)):
                    rho = cgc - cgd

            records.append({
                "model":     model,
                "game_id":   f"pd_{game_id}",
                "condition": condition,
                "matchup":   matchup,
                "source":    "PD",
                "rho":       round(rho,    4) if pd.notna(rho)    else np.nan,
                "beta_pd":   round(beta_pd, 4),
            })

    result = pd.DataFrame(records)
    print(f"  → {len(result)} rows | {result['model'].nunique()} models")
    return result


# =============================================================================
# 2. COMMONS DILEMMA  →  θ (pool-relative), β_cd
# =============================================================================

def process_cd(df: pd.DataFrame) -> pd.DataFrame:
    """
    θ = mean(extraction_i / sustainable_per_player_t) across rounds.

    sustainable_per_player_t = pool_before_regen_t × regen_rate / num_players

    regen_rate is inferred from the sustainable_share column:
      - If sustainable_share > 1  →  it is stored as a percentage integer
        (e.g. 2 means 2%, so regen_rate = 0.02).  This matches files where
        regeneration = pool × 0.20 and the column stores 2 (i.e. 20% / 10).
      - Wait — from the data: pool=100, regen=20, sustainable_share=2.
        regen/pool = 0.20, but col=2. So col = regen_rate × 10.
        Therefore regen_rate = sustainable_share / 10 / 10... let me be precise:
        regen = pool × (sustainable_share / 100) × 10?  No.
        Observed: regen=20 always (until collapse), pool starts at 100.
        sustainable_share=2. So regen_rate = 20/100 = 0.20, and col=2 = 0.20×10.
        → regen_rate = sustainable_share_col × 0.10

    θ > 1 means over-extraction relative to the sustainable share at that
    pool size. θ = 1.0 is the sustainable benchmark.

    β_cd = MAE(belief P(opp over-extracts), actual opp over-extracts ∈ {0,1})
           Beliefs > 1.0 (old unit scale, v2.2) are auto-normalised to 0–1
           by dividing by max_extraction (sustainable_per_player × 5 as upper bound),
           clipped to [0,1].
    """
    print("\n[CD] Processing...")
    records = []

    for game_id, grp in df.groupby("game_id"):
        grp         = grp.sort_values("round").reset_index(drop=True)
        condition   = grp["condition"].iloc[0]
        matchup     = grp["matchup"].iloc[0]
        info_cond   = grp["info_condition"].iloc[0] \
                      if "info_condition" in grp.columns else np.nan
        risk_cond   = grp["risk_condition"].iloc[0] \
                      if "risk_condition" in grp.columns else np.nan
        num_players = float(grp["num_players"].iloc[0]) \
                      if "num_players" in grp.columns else 2.0

        # Infer regen_rate from sustainable_share column
        # sustainable_share=2 → regen_rate = 0.20 (verified from data)
        sus_col     = float(grp["sustainable_share"].iloc[0])
        regen_rate  = sus_col * 0.10   # e.g. 2 → 0.20

        pool_before = pd.to_numeric(grp["pool_before_regen"], errors="coerce")

        # sustainable units per player per round (varies with pool)
        sus_per_player = pool_before * regen_rate / num_players

        for role in ["1", "2"]:
            opp_role   = "2" if role == "1" else "1"
            model      = grp[f"model_{role}"].iloc[0]
            extraction = pd.to_numeric(grp[f"extraction_{role}"], errors="coerce")
            opp_ext    = pd.to_numeric(grp[f"extraction_{opp_role}"], errors="coerce")
            belief_raw = pd.to_numeric(grp[f"belief_{role}"], errors="coerce")

            # θ: pool-relative exploitation intensity
            theta_vals = extraction / sus_per_player
            theta      = theta_vals.replace([np.inf, -np.inf], np.nan).mean()

            # β_cd: auto-normalise old-scale beliefs (> 1.0)
            belief = belief_raw.copy()
            if belief.dropna().max() > 1.0:
                # old unit scale: normalise by (sus_per_player × 5) as upper bound
                upper = sus_per_player * 5
                belief = (belief / upper).clip(0, 1)

            opp_over = (opp_ext > sus_per_player).astype(float)
            beta_cd  = (belief - opp_over).abs().mean()

            records.append({
                "model":          model,
                "game_id":        f"cd_{game_id}",
                "condition":      condition,
                "info_condition": info_cond,
                "risk_condition": risk_cond,
                "matchup":        matchup,
                "source":         "CD",
                "theta":          round(float(theta), 4) if pd.notna(theta) else np.nan,
                "beta_cd":        round(float(beta_cd), 4),
            })

    result = pd.DataFrame(records)
    print(f"  → {len(result)} rows | {result['model'].nunique()} models")
    return result


# =============================================================================
# 3. CHEAP-TALK  →  η_aligned, η_misaligned, Δη, γ_misaligned, β_ct
# =============================================================================

def process_ct(df: pd.DataFrame) -> pd.DataFrame:
    """
    η_aligned    = fraction truthful in aligned rounds (sender incentive = honest)
    η_misaligned = fraction truthful in misaligned rounds (sender incentive = deceive)
    Δη           = η_aligned − η_misaligned  [PRIMARY: honesty drop under misalignment]

    γ_misaligned = Pearson r(message_truthful, action_correct) in misaligned rounds
                   [PRIMARY: gullibility when sender has incentive to deceive]

    β_ct (sender)   = MAE(sender_belief P(receiver follows), action_correct)
    β_ct (receiver) = MAE(receiver_belief P(sender truthful), message_truthful)

    Returns two rows per run_id: one for sender role, one for receiver role.

    IMPORTANT — game_condition structure:
    Each run_id is entirely aligned OR entirely misaligned (not mixed).
    Δη and γ_mis must therefore be computed at file level by comparing
    the mean η across aligned runs vs misaligned runs for the same
    (file_tag, matchup, model_sender) group.
    """
    print("\n[CT] Processing...")

    if "game_condition" not in df.columns:
        df = df.copy()
        df["game_condition"] = "unknown"

    # ── Step 1: run-level records (β, η per run, γ per run) ──────────────────
    records = []
    for run_id, grp in df.groupby("run_id"):
        grp            = grp.sort_values("round").reset_index(drop=True)
        condition      = grp["identity_condition"].iloc[0] \
                         if "identity_condition" in grp.columns \
                         else grp.get("condition", pd.Series(["unknown"])).iloc[0]
        matchup        = grp["matchup"].iloc[0]
        game_cond      = grp["game_condition"].iloc[0]   # aligned or misaligned
        file_tag       = run_id.rsplit("_", 1)[0]        # strip trailing _<run_id_num>

        truthful       = pd.to_numeric(grp["message_truthful"],  errors="coerce")
        action_correct = pd.to_numeric(grp["action_correct"],    errors="coerce")
        sender_belief  = pd.to_numeric(grp["sender_belief"],     errors="coerce")
        receiver_belief= pd.to_numeric(grp["receiver_belief"],   errors="coerce")

        eta_run        = truthful.mean()
        gamma_run      = safe_corr(truthful, action_correct)
        beta_ct_sender = (sender_belief   - action_correct).abs().mean()
        beta_ct_recv   = (receiver_belief - truthful).abs().mean()

        base = {
            "game_id":      f"ct_{run_id}",
            "file_tag":     file_tag,
            "matchup":      matchup,
            "condition":    condition,
            "game_cond":    game_cond,
            "source":       "CT",
            "eta_run":      round(float(eta_run), 4),
        }
        records.append({**base,
            "model":       grp["model_sender"].iloc[0],
            "role":        "sender",
            "gamma_run":   np.nan,
            "beta_ct":     round(float(beta_ct_sender), 4),
        })
        records.append({**base,
            "model":       grp["model_receiver"].iloc[0],
            "role":        "receiver",
            "gamma_run":   round(float(gamma_run), 4) if pd.notna(gamma_run) else np.nan,
            "beta_ct":     round(float(beta_ct_recv), 4),
        })

    run_df = pd.DataFrame(records)

    # ── Step 2: compute Δη and γ_mis at (file_tag, matchup, model, role) level ─
    # For each file, average η across aligned runs → eta_aligned
    # average η across misaligned runs → eta_misaligned
    # Δη = eta_aligned − eta_misaligned

    def compute_delta(grp):
        aligned    = grp[grp["game_cond"] == "aligned"]["eta_run"]
        misaligned = grp[grp["game_cond"] == "misaligned"]["eta_run"]
        eta_a = aligned.mean()    if len(aligned)    > 0 else np.nan
        eta_m = misaligned.mean() if len(misaligned) > 0 else np.nan
        delta  = (eta_a - eta_m)  if pd.notna(eta_a) and pd.notna(eta_m) else np.nan
        return pd.Series({
            "eta_aligned":    round(float(eta_a), 4) if pd.notna(eta_a) else np.nan,
            "eta_misaligned": round(float(eta_m), 4) if pd.notna(eta_m) else np.nan,
            "delta_eta":      round(float(delta),  4) if pd.notna(delta) else np.nan,
        })

    def compute_gamma_mis(grp):
        mis = grp[grp["game_cond"] == "misaligned"]["gamma_run"].dropna()
        return pd.Series({
            "gamma_mis": round(float(mis.mean()), 4) if len(mis) > 0 else np.nan
        })

    sender_grp   = run_df[run_df["role"] == "sender"]
    receiver_grp = run_df[run_df["role"] == "receiver"]

    delta_df = sender_grp.groupby(
        ["file_tag", "matchup", "model"]
    ).apply(compute_delta).reset_index()

    gamma_df = receiver_grp.groupby(
        ["file_tag", "matchup", "model"]
    ).apply(compute_gamma_mis).reset_index()

    # Merge back onto run_df
    run_df = run_df.merge(
        delta_df[["file_tag", "matchup", "model",
                  "eta_aligned", "eta_misaligned", "delta_eta"]],
        on=["file_tag", "matchup", "model"], how="left"
    )
    run_df = run_df.merge(
        gamma_df[["file_tag", "matchup", "model", "gamma_mis"]],
        on=["file_tag", "matchup", "model"], how="left"
    )

    # Clean up: keep only sender rows for η cols, receiver for γ
    run_df.loc[run_df["role"] == "receiver",
               ["eta_aligned", "eta_misaligned", "delta_eta"]] = np.nan
    run_df.loc[run_df["role"] == "sender", "gamma_mis"] = np.nan

    # Rename eta_run to eta_pooled for reference
    run_df = run_df.rename(columns={"eta_run": "eta_pooled"})
    run_df = run_df.drop(columns=["file_tag", "game_cond", "gamma_run"])

    result = run_df
    print(f"  → {len(result)} rows | {result['model'].nunique()} models")
    delta_ok = result["delta_eta"].notna().sum()
    gamma_ok = result["gamma_mis"].notna().sum()
    print(f"     delta_eta non-null: {delta_ok} | gamma_mis non-null: {gamma_ok}")
    return result


# =============================================================================
# 4. MERGE
# =============================================================================

def merge_profiles(pd_df, cd_df, ct_df) -> pd.DataFrame:
    """Long-format merge: one row per (model, game_id). Parameters from
    other environments are NaN for each row."""
    print("\n[MERGE] Building unified game-level dataset...")

    common   = ["model", "game_id", "condition", "source", "matchup"]
    pd_clean = pd_df[common + ["rho", "beta_pd"]].copy()
    cd_clean = cd_df[common + ["info_condition", "risk_condition",
                                "theta", "beta_cd"]].copy()
    ct_clean = ct_df[common + ["role", "eta_pooled", "eta_aligned", "eta_misaligned",
                                "delta_eta", "gamma_mis", "beta_ct"]].copy()

    merged = pd.concat([pd_clean, cd_clean, ct_clean],
                       ignore_index=True, sort=False)

    print(f"  → {len(merged)} total rows | {merged['model'].nunique()} models")
    for src in ["PD", "CD", "CT"]:
        n = merged[merged["source"] == src]["game_id"].nunique()
        print(f"     {src}: {n} games")

    return merged


# =============================================================================
# 5. MODEL-LEVEL AGGREGATION
# =============================================================================

def aggregate_model_level(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per model. Key columns:
      rho_mean / rho_std
      theta_mean / theta_std
      eta_aligned_mean, eta_misaligned_mean, delta_eta_mean (primary)
      gamma_mis_mean (primary)
      beta_pd_mean, beta_cd_mean, beta_ct_mean  (reported separately)
      beta_corr_pd_cd, beta_corr_pd_ct, beta_corr_cd_ct  (cross-env coherence)
    """
    print("\n[AGGREGATE] Building model-level profiles...")
    rows = []

    # collect per-env beta means for correlation later
    beta_by_model = {}

    for model, grp in game_df.groupby("model"):
        row = {"model": model}

        # ρ
        rho = pd.to_numeric(grp["rho"], errors="coerce").dropna()
        row["rho_mean"] = round(rho.mean(), 4) if len(rho) > 0 else np.nan
        row["rho_std"]  = round(rho.std(),  4) if len(rho) > 1 else np.nan
        row["rho_n"]    = len(rho)

        # θ
        theta = pd.to_numeric(grp["theta"], errors="coerce").dropna()
        row["theta_mean"] = round(theta.mean(), 4) if len(theta) > 0 else np.nan
        row["theta_std"]  = round(theta.std(),  4) if len(theta) > 1 else np.nan
        row["theta_n"]    = len(theta)

        # η (aligned, misaligned, Δη) — sender rows only
        sender = grp[grp["role"] == "sender"] if "role" in grp.columns else grp
        for col, key in [("eta_aligned", "eta_a"), ("eta_misaligned", "eta_m"),
                         ("delta_eta", "delta_eta")]:
            if col in sender.columns:
                v = pd.to_numeric(sender[col], errors="coerce").dropna()
                row[f"{key}_mean"] = round(v.mean(), 4) if len(v) > 0 else np.nan
                row[f"{key}_std"]  = round(v.std(),  4) if len(v) > 1 else np.nan
                row[f"{key}_n"]    = len(v)

        # γ misaligned — receiver rows only
        receiver = grp[grp["role"] == "receiver"] if "role" in grp.columns else grp
        if "gamma_mis" in receiver.columns:
            gm = pd.to_numeric(receiver["gamma_mis"], errors="coerce").dropna()
            row["gamma_mis_mean"] = round(gm.mean(), 4) if len(gm) > 0 else np.nan
            row["gamma_mis_std"]  = round(gm.std(),  4) if len(gm) > 1 else np.nan
            row["gamma_mis_n"]    = len(gm)

        # β per environment (separate, not pooled)
        for env, col in [("pd", "beta_pd"), ("cd", "beta_cd"), ("ct", "beta_ct")]:
            if col in grp.columns:
                v = pd.to_numeric(grp[col], errors="coerce").dropna()
                row[f"beta_{env}_mean"] = round(v.mean(), 4) if len(v) > 0 else np.nan
                row[f"beta_{env}_n"]    = len(v)
            else:
                row[f"beta_{env}_mean"] = np.nan
                row[f"beta_{env}_n"]    = 0

        # game counts
        for src in ["PD", "CD", "CT"]:
            row[f"n_games_{src.lower()}"] = \
                grp[grp["source"] == src]["game_id"].nunique()

        beta_by_model[model] = {
            "pd": row.get("beta_pd_mean"),
            "cd": row.get("beta_cd_mean"),
            "ct": row.get("beta_ct_mean"),
        }
        rows.append(row)

    result = pd.DataFrame(rows)

    # Cross-environment β correlations across models (coherence anchor)
    beta_df = pd.DataFrame(beta_by_model).T.apply(pd.to_numeric, errors="coerce")
    result["beta_corr_pd_cd"] = safe_corr(beta_df["pd"], beta_df["cd"])
    result["beta_corr_pd_ct"] = safe_corr(beta_df["pd"], beta_df["ct"])
    result["beta_corr_cd_ct"] = safe_corr(beta_df["cd"], beta_df["ct"])

    print(f"  → {len(result)} model-level profiles")
    return result


# =============================================================================
# 6. COHERENCE SUMMARY
# =============================================================================

def print_coherence_metrics(model_df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("CROSS-ENVIRONMENT COHERENCE METRICS  (v2)")
    print("=" * 65)

    llm_mask = ~model_df["model"].str.startswith("Human Prior")
    ml = model_df[llm_mask].copy()

    # A. Strategic profile table
    print("\nA. Strategic profile vector (LLM models)")
    cols = ["model", "rho_mean", "theta_mean", "delta_eta_mean",
            "gamma_mis_mean", "beta_pd_mean", "beta_cd_mean", "beta_ct_mean"]
    avail = [c for c in cols if c in ml.columns]
    print(ml[avail].round(3).to_string(index=False))

    # B. Cross-env β correlations
    print("\nB. β cross-environment correlations (across models)")
    for pair, col in [("PD–CD", "beta_corr_pd_cd"),
                      ("PD–CT", "beta_corr_pd_ct"),
                      ("CD–CT", "beta_corr_cd_ct")]:
        val = ml[col].iloc[0] if col in ml.columns else np.nan
        print(f"   r(β_{pair}) = {val:.3f}" if pd.notna(val) else f"   r(β_{pair}) = n/a")

    # C. Cross-parameter correlations
    print("\nC. Cross-parameter correlations  [Corr(ρ, θ, Δη, γ_mis)]")
    params = ["rho_mean", "theta_mean", "delta_eta_mean", "gamma_mis_mean"]
    avail_p = [p for p in params if p in ml.columns and ml[p].notna().sum() >= 3]
    if len(avail_p) >= 2:
        print(ml[avail_p].corr().round(3).to_string())
    else:
        print("   Insufficient coverage.")

    # D. Δη summary (primary η estimate)
    print("\nD. Δη per model  (η_aligned − η_misaligned; higher = more strategic deception)")
    if "delta_eta_mean" in ml.columns:
        for _, r in ml[["model", "eta_a_mean", "eta_m_mean", "delta_eta_mean"]].dropna().iterrows():
            bar = "█" * int(abs(r.get("delta_eta_mean", 0) or 0) * 20)
            print(f"   {r['model']:<30} Δη={r['delta_eta_mean']:+.3f}  "
                  f"(aligned={r.get('eta_a_mean', float('nan')):.3f}  "
                  f"mis={r.get('eta_m_mean', float('nan')):.3f})  {bar}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("P264 — Strategic Profile Builder  v2")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[LOAD] Reading raw CSVs from {RAW_DIR}")
    pd_raw = load_csvs(RAW_DIR, "pd_results_")
    cd_raw = load_csvs(RAW_DIR, "cd_")
    ct_raw = load_csvs(RAW_DIR, "cheap_talk_results_")
    print(f"\n  PD={len(pd_raw)} rows | CD={len(cd_raw)} rows | CT={len(ct_raw)} rows")

    pd_params = process_pd(pd_raw)
    cd_params = process_cd(cd_raw)
    ct_params = process_ct(ct_raw)

    game_level  = merge_profiles(pd_params, cd_params, ct_params)
    model_level = aggregate_model_level(game_level)

    print_coherence_metrics(model_level)

    game_path  = os.path.join(OUTPUT_DIR, "strategic_profiles_game_level_v2.csv")
    model_path = os.path.join(OUTPUT_DIR, "strategic_profiles_model_level_v2.csv")
    game_level.to_csv(game_path,  index=False)
    model_level.to_csv(model_path, index=False)

    print("\n" + "=" * 65)
    print("OUTPUT FILES")
    print("=" * 65)
    print(f"  {game_path}")
    print(f"  {model_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
