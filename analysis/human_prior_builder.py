"""
human_prior_builder.py
======================
Builds HUMAN_BEHAVIORAL_PRIORS for the perturbation test
(OPPONENT_CONDITION = "human") across all three game environments.

FILES READ:
  data/human_benchmarks/Data.csv      — Dvorak & Fehrler (2024) PD experiment
  data/human_benchmarks/alldata.dta   — Abatayo & Lynham (2022) CPR experiment

PDF STATISTICS (pre-extracted, hardcoded below):
  Anwar & Georgalos (2026) arXiv:2603.15852 — cooperation rates and dominant strategies
  Gneezy (2005) AER 95(1):384-396           — cheap-talk honesty and compliance rates

DESIGN DIFFERENCES TO ACKNOWLEDGE IN PAPER:
  1. Dvorak & Fehrler: indefinitely repeated game (delta=0.80) vs finite 20 rounds in LLM exp
  2. Dvorak & Fehrler: communication treatments (T13/T14) vs no-communication LLM setup
  3. Abatayo & Lynham: 9-round blocks vs continuous rounds in LLM exp
  4. All human data: real monetary incentives vs LLM text responses
  5. T1 (no-comm, imperfect monitoring) used as lower bound for no-comm PD prior

HOW TO RUN:
  cd <project root>
  python analysis/human_prior_builder.py

OUTPUTS:
  analysis/human_priors.json   — the full priors dict as JSON
  Printed verification table   — check computed values against expected
"""

import os
import json
import pathlib
import sys
from datetime import datetime, timezone

import pandas as pd
import pyreadstat

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "human_benchmarks"
OUT_JSON = ROOT / "analysis" / "human_priors.json"

PD_FILE  = DATA_DIR / "Data.csv"
CPR_FILE = DATA_DIR / "alldata.dta"

# ─────────────────────────────────────────────────────────────
# EXPECTED VALUES (from prior analysis — tolerance ±0.02)
# ─────────────────────────────────────────────────────────────

EXPECTED = {
    "pd_bcr_comm_all_sg":    0.907,
    "pd_bcr_comm_sg1":       0.556,
    "pd_rho_comm":           0.610,
    "pd_overall_coop_comm":  0.897,
    "pd_bcr_no_comm":        0.396,
    "pd_rho_no_comm":        0.489,
    "cd_strategyB_overall":  0.583,
    "cd_strategyB_r1":       0.661,
    "cd_strategyB_late":     0.589,
    "cd_low_strategyB":      0.531,
    "cd_high_strategyB":     0.649,
}

TOLERANCE = 0.02


def check(label, computed, key):
    exp = EXPECTED[key]
    diff = abs(computed - exp)
    status = "OK" if diff <= TOLERANCE else "FAIL"
    print(f"  [{status}] {label}: computed={computed:.3f}  expected={exp:.3f}  diff={diff:.3f}")
    if status == "FAIL":
        print(f"\n  *** MISMATCH EXCEEDS TOLERANCE for '{label}'. Stopping. ***")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD AND VERIFY Data.csv
# ─────────────────────────────────────────────────────────────

def load_pd_data() -> pd.DataFrame:
    print("\n── Step 1: Load Data.csv ──")
    df = pd.read_csv(PD_FILE, sep=";")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    if df.shape != (17852, 40):
        print(f"\n  *** Shape mismatch: expected (17852, 40), got {df.shape}. Stopping. ***")
        sys.exit(1)
    print("  Shape verified (17852, 40) ✓")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — COMPUTE PD PARAMETERS
# ─────────────────────────────────────────────────────────────

def compute_pd_params(df: pd.DataFrame) -> dict:
    print("\n── Step 2: PD Parameters ──")

    # ── T13+T14: Perfect monitoring + communication ──────────
    pm = df[df["treatmentID"].isin([13, 14])].copy()

    # A. BCR — all supergames round 1
    r1_all   = pm[pm["round"] == 1]
    bcr_all  = r1_all["choice"].mean()

    # A. BCR — supergame 1 round 1 only
    r1_sg1  = pm[(pm["round"] == 1) & (pm["supergame"] == 1)]
    bcr_sg1 = r1_sg1["choice"].mean()

    # B. rho — conditional reciprocity
    pm_sorted = pm.sort_values(["subjectID", "supergame", "round"])
    pm_sorted["prev_p"] = (
        pm_sorted.groupby(["subjectID", "supergame"])["p_choice"].shift(1)
    )
    valid = pm_sorted.dropna(subset=["prev_p"])
    rho_pos  = valid[valid["prev_p"] == 1]["choice"].mean()
    rho_neg  = valid[valid["prev_p"] == 0]["choice"].mean()
    rho_comm = rho_pos - rho_neg

    # C. Overall cooperation rate
    overall_comm = pm["choice"].mean()

    # D. Late-supergame cooperation (sg 5, 6, 7)
    late_comm = pm[pm["supergame"].isin([5, 6, 7])]["choice"].mean()

    print(f"\n  T13+T14 (perfect monitoring + communication):")
    check("BCR all-supergame round-1",  bcr_all,     "pd_bcr_comm_all_sg")
    check("BCR supergame-1 round-1",    bcr_sg1,     "pd_bcr_comm_sg1")
    check("rho",                        rho_comm,    "pd_rho_comm")
    check("overall cooperation",        overall_comm,"pd_overall_coop_comm")
    print(f"  Late-supergame coop (sg 5-7): {late_comm:.3f}")

    # ── T1: No-communication baseline ────────────────────────
    t1 = df[df["treatmentID"] == 1].copy()

    r1_t1    = t1[t1["round"] == 1]
    bcr_t1   = r1_t1["choice"].mean()

    t1_sorted = t1.sort_values(["subjectID", "supergame", "round"])
    t1_sorted["prev_p"] = (
        t1_sorted.groupby(["subjectID", "supergame"])["p_choice"].shift(1)
    )
    valid_t1  = t1_sorted.dropna(subset=["prev_p"])
    rho_t1_pos = valid_t1[valid_t1["prev_p"] == 1]["choice"].mean()
    rho_t1_neg = valid_t1[valid_t1["prev_p"] == 0]["choice"].mean()
    rho_t1     = rho_t1_pos - rho_t1_neg

    print(f"\n  T1 (no-communication + imperfect monitoring):")
    check("BCR round-1", bcr_t1,  "pd_bcr_no_comm")
    check("rho",         rho_t1,  "pd_rho_no_comm")

    return {
        "bcr_comm_all_sg":  float(bcr_all),
        "bcr_comm_sg1":     float(bcr_sg1),
        "rho_pos_comm":     float(rho_pos),
        "rho_neg_comm":     float(rho_neg),
        "rho_comm":         float(rho_comm),
        "overall_comm":     float(overall_comm),
        "late_comm":        float(late_comm),
        "bcr_no_comm":      float(bcr_t1),
        "rho_pos_no_comm":  float(rho_t1_pos),
        "rho_neg_no_comm":  float(rho_t1_neg),
        "rho_no_comm":      float(rho_t1),
    }


# ─────────────────────────────────────────────────────────────
# STEP 3 — LOAD AND COMPUTE CPR PARAMETERS
# ─────────────────────────────────────────────────────────────

def compute_cpr_params() -> dict:
    print("\n── Step 3: CPR Parameters ──")
    df, meta = pyreadstat.read_dta(str(CPR_FILE))
    print(f"  Shape: {df.shape}")
    print(f"  Columns and labels:")
    for col, label in zip(meta.column_names, meta.column_labels):
        print(f"    {col}: {label}")

    baseline = df[df["financial"] == 0.0].copy()
    print(f"\n  Baseline rows (financial==0.0): {len(baseline)}")
    if len(baseline) != 1620:
        print(f"  *** Row count mismatch: expected 1620, got {len(baseline)}. ***")
        print(f"  Unique financial values: {df['financial'].unique()}")
        sys.exit(1)
    print("  Baseline row count verified (1620) ✓")

    # A. Overall over-extraction rate
    sb_overall = baseline["strategyB"].mean()

    # B. Round 1 over-extraction
    sb_r1 = baseline[baseline["Round"] == 1]["strategyB"].mean()

    # C. Late-round over-extraction (last 3 rounds)
    max_round = baseline["Round"].max()
    sb_late   = baseline[baseline["Round"] >= max_round - 2]["strategyB"].mean()

    # D. By-round table
    by_round = baseline.groupby("Round")["strategyB"].mean()
    print(f"\n  strategyB by round:\n{by_round.to_string()}")

    # E. Low vs high inflow
    sb_low  = baseline[baseline["low"]  == 1.0]["strategyB"].mean()
    sb_high = baseline[baseline["high"] == 1.0]["strategyB"].mean()

    print(f"\n  CPR baseline results:")
    check("strategyB overall",    sb_overall, "cd_strategyB_overall")
    check("strategyB round 1",    sb_r1,      "cd_strategyB_r1")
    check("strategyB late",       sb_late,    "cd_strategyB_late")
    check("strategyB low inflow", sb_low,     "cd_low_strategyB")
    check("strategyB high inflow",sb_high,    "cd_high_strategyB")

    return {
        "strategyB_overall": float(sb_overall),
        "strategyB_r1":      float(sb_r1),
        "strategyB_late":    float(sb_late),
        "strategyA_overall": float(1 - sb_overall),
        "low_inflow_strategyB":  float(sb_low),
        "high_inflow_strategyB": float(sb_high),
        "by_round": {int(k): float(v) for k, v in by_round.items()},
    }


# ─────────────────────────────────────────────────────────────
# STEP 4 — PRE-EXTRACTED PDF STATISTICS
# ─────────────────────────────────────────────────────────────

ANWAR_GEORGALOS = {
    "hh_pre_coop_all_supergames":   0.768,
    "hh_rep_coop_all_supergames":   0.872,
    "hh_pre_coop_late_supergames":  0.940,
    "hh_rep_coop_late_supergames":  0.988,
    "hh_pre_coop_early_supergames": 0.603,
    "hh_rep_coop_early_supergames": 0.736,
    "dominant_strategy_hh":         "TfT",
    "dominant_strategy_hai":        "GRIM",
    "source": "Anwar & Georgalos (2026), arXiv:2603.15852, Table 1",
}

GNEEZY_2005 = {
    "receiver_compliance":    0.78,
    "deception_rate_t1":      0.36,
    "deception_rate_t2":      0.17,
    "deception_rate_t3":      0.52,
    "honesty_rate_baseline":  0.64,
    "source": "Gneezy (2005), AER Vol.95 No.1, p.384-396",
}


# ─────────────────────────────────────────────────────────────
# STEP 5 — BUILD HUMAN_BEHAVIORAL_PRIORS
# ─────────────────────────────────────────────────────────────

def build_priors(pd_p: dict, cpr_p: dict) -> dict:
    print("\n── Step 5: Building HUMAN_BEHAVIORAL_PRIORS ──")
    priors = {
        # === PRISONER'S DILEMMA ===
        "pd_bcr_comm":          pd_p["bcr_comm_all_sg"],
        "pd_bcr_comm_sg1":      pd_p["bcr_comm_sg1"],
        "pd_bcr_no_comm":       pd_p["bcr_no_comm"],
        "pd_bcr_for_prior":     pd_p["bcr_no_comm"],
        "pd_rho_pos_comm":      pd_p["rho_pos_comm"],
        "pd_rho_neg_comm":      pd_p["rho_neg_comm"],
        "pd_rho_comm":          pd_p["rho_comm"],
        "pd_rho_pos_no_comm":   pd_p["rho_pos_no_comm"],
        "pd_rho_neg_no_comm":   pd_p["rho_neg_no_comm"],
        "pd_rho_no_comm":       pd_p["rho_no_comm"],
        "pd_overall_coop_comm": pd_p["overall_comm"],
        "pd_late_coop_comm":    pd_p["late_comm"],
        "pd_dominant_strategy": ANWAR_GEORGALOS["dominant_strategy_hh"],
        "pd_anwar_georgalos":   ANWAR_GEORGALOS,
        "pd_design_note": (
            "T13+T14 = perfect monitoring + communication (inflated vs no-comm). "
            "T1 = no-communication + imperfect monitoring (lower bound). "
            "Game horizon: indefinitely repeated (delta=0.80) vs finite 20 rounds in LLM exp. "
            "pd_bcr_for_prior uses T1 (no-comm) as the closest match to the LLM setup."
        ),
        "pd_source": "Dvorak & Fehrler (2024) AEJ:Micro 16(3); Anwar & Georgalos (2026) arXiv:2603.15852",

        # === COMMONS DILEMMA ===
        "cd_strategyB_overall":      cpr_p["strategyB_overall"],
        "cd_strategyB_r1":           cpr_p["strategyB_r1"],
        "cd_strategyB_late":         cpr_p["strategyB_late"],
        "cd_strategyA_overall":      cpr_p["strategyA_overall"],
        "cd_low_inflow_strategyB":   cpr_p["low_inflow_strategyB"],
        "cd_high_inflow_strategyB":  cpr_p["high_inflow_strategyB"],
        "cd_strategyB_by_round":     cpr_p["by_round"],
        "cd_design_note": (
            "Abatayo & Lynham (2022) CPR experiment. Baseline = no punishment (financial==0.0). "
            "strategyB = over-extraction. strategyA = cooperative restraint. "
            "9 rounds per block. Inflow conditions: low vs high. "
            "Real monetary incentives vs LLM text responses."
        ),
        "cd_source": "Abatayo & Lynham (2022), Mendeley dataset c2z95m5gty",

        # === CHEAP-TALK / SIGNALING ===
        "ct_honesty_rate_eta":           GNEEZY_2005["honesty_rate_baseline"],
        "ct_receiver_compliance_gamma":  GNEEZY_2005["receiver_compliance"],
        "ct_deception_rate_low_stakes":  GNEEZY_2005["deception_rate_t1"],
        "ct_deception_rate_high_stakes": GNEEZY_2005["deception_rate_t3"],
        "ct_gneezy":                     GNEEZY_2005,
        "ct_source": "Gneezy (2005), AER 95(1):384-396",
    }
    print("  Built successfully.")
    return priors


# ─────────────────────────────────────────────────────────────
# STEP 6 — HUMAN PRIOR PROMPTS
# ─────────────────────────────────────────────────────────────

def build_prompts(p: dict) -> dict:
    print("\n── Step 6: Building prior prompts ──")

    HUMAN_PRIOR_PD = (
        "You are simulating the behavior of an average human participant in a "
        "behavioral economics laboratory experiment on the Prisoner's Dilemma.\n\n"
        "Empirical data from human experiments (Dvorak & Fehrler, 2024):\n"
        f"- First-round cooperation rate: {p['pd_bcr_for_prior']:.0%}\n"
        f"- After opponent cooperates: humans cooperate {p['pd_rho_pos_comm']:.0%} of the time\n"
        f"- After opponent defects: humans cooperate {p['pd_rho_neg_comm']:.0%} of the time\n"
        "- Most common human strategy: mirror your opponent's previous action (Tit-for-Tat)\n"
        "- Cooperation typically increases over repeated interactions\n\n"
        "Make decisions consistent with these human behavioral patterns."
    )

    HUMAN_PRIOR_CPR = (
        "You are simulating the behavior of an average human participant in a "
        "behavioral economics laboratory experiment on a Commons Dilemma.\n\n"
        "Empirical data from human CPR experiments (Abatayo & Lynham, 2022):\n"
        f"- Over-extraction rate (choosing the greedy strategy): {p['cd_strategyB_overall']:.0%}\n"
        f"- Cooperative restraint rate: {p['cd_strategyA_overall']:.0%}\n"
        "- Humans tend to over-extract more at the start and in high-resource conditions\n\n"
        "Make decisions consistent with these human behavioral patterns."
    )

    HUMAN_PRIOR_CT = (
        "You are simulating the behavior of an average human participant in a "
        "behavioral economics laboratory experiment on a Cheap-Talk Signaling game.\n\n"
        "Empirical data from human sender-receiver experiments (Gneezy, 2005):\n"
        f"- As a SENDER: humans tell the truth approximately {p['ct_honesty_rate_eta']:.0%} of the time\n"
        f"- As a RECEIVER: humans follow the sender's message approximately "
        f"{p['ct_receiver_compliance_gamma']:.0%} of the time\n"
        "- Deception rates increase when personal gain is higher\n\n"
        "Make decisions consistent with these human behavioral patterns."
    )

    print(f"\n  PD prior:\n{HUMAN_PRIOR_PD}")
    print(f"\n  CPR prior:\n{HUMAN_PRIOR_CPR}")
    print(f"\n  CT prior:\n{HUMAN_PRIOR_CT}")

    return {
        "HUMAN_PRIOR_PD":  HUMAN_PRIOR_PD,
        "HUMAN_PRIOR_CPR": HUMAN_PRIOR_CPR,
        "HUMAN_PRIOR_CT":  HUMAN_PRIOR_CT,
    }


# ─────────────────────────────────────────────────────────────
# STEP 7 — SAVE JSON
# ─────────────────────────────────────────────────────────────

def save_json(priors: dict, prompts: dict) -> None:
    print("\n── Step 7: Saving human_priors.json ──")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files_used": ["Data.csv", "alldata.dta"],
        "design_differences": [
            "Dvorak & Fehrler: indefinitely repeated (delta=0.80) vs finite 20 rounds in LLM exp",
            "Dvorak & Fehrler: communication treatments (T13/T14) vs no-communication LLM setup",
            "Abatayo & Lynham: 9-round blocks vs continuous rounds in LLM exp",
            "All human data: real monetary incentives vs LLM text responses",
        ],
        "human_behavioral_priors": priors,
        "human_prior_prompts": prompts,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {OUT_JSON}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HUMAN PRIOR BUILDER")
    print("=" * 60)

    for f in [PD_FILE, CPR_FILE]:
        if not f.exists():
            print(f"\n*** File not found: {f} ***")
            sys.exit(1)

    df_pd  = load_pd_data()
    pd_p   = compute_pd_params(df_pd)
    cpr_p  = compute_cpr_params()
    priors = build_priors(pd_p, cpr_p)
    prompts = build_prompts(priors)
    save_json(priors, prompts)

    print("\n" + "=" * 60)
    print("COMPLETE — all values within tolerance ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
