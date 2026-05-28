"""
human_prior_builder.py
======================
Builds HUMAN_BEHAVIORAL_PRIORS for the perturbation test
(OPPONENT_CONDITION / IDENTITY_CONDITION = "human_prior") across all three
game environments.

FILES READ (raw data, values computed live):
  data/human_benchmarks/Data.csv      -> Dvorak & Fehrler (2024) PD experiment
  data/human_benchmarks/alldata.dta   -> Abatayo & Lynham (2022) CPR experiment

PDF STATISTICS (pre-extracted, hardcoded with source attribution):
  Anwar & Georgalos (2026) arXiv:2603.15852      -> PD cooperation rates / dominant strategies
  Gneezy (2005) AER 95(1):384-396                -> cheap-talk honesty + receiver compliance
  Pawlick, Colbert & Zhu (2018) arXiv:1804.06831 -> cheap-talk equilibrium truth-induction
                                                    (aligned vs misaligned, no-detector case)

DESIGN DIFFERENCES TO ACKNOWLEDGE IN PAPER:
  1. Dvorak & Fehrler: indefinitely repeated game (delta=0.80) vs finite 20 rounds in LLM exp
  2. Dvorak & Fehrler: communication treatments (T13/T14) vs no-communication LLM setup
  3. Abatayo & Lynham: 9-round blocks vs continuous rounds in LLM exp
  4. All human data: real monetary incentives vs LLM text responses
  5. T1 (no-comm, imperfect monitoring) used as the no-comm PD prior (closest to LLM setup)
  6. Cheap-talk:
       - Gneezy (2005) has NO aligned treatment: all three treatments are misaligned
         (the sender always prefers the option that is worse for the receiver). Our
         MISALIGNED condition maps to Gneezy Treatment 3 (high sender gain): ~52% lies.
       - Our ALIGNED condition has no Gneezy analogue. Reference values come from
         cheap-talk theory (Crawford & Sobel 1982; Pawlick et al. 2018): with aligned
         interests the truth-telling / full-revelation equilibrium is sustainable, so
         ~100% honest sending and ~100% receiver compliance.
       - Gneezy's 78% receiver compliance came from receivers who did NOT know the
         payoff structure. Our receiver DOES know it, so for the MISALIGNED condition
         we use the informed, no-detector value (~50% follow; Pawlick et al. 2018,
         babbling/pooling equilibrium), not 78%.

HOW TO RUN:
  cd <project root>
  python analysis/human_prior_builder.py

OUTPUTS:
  analysis/human_priors.json   -> the full priors dict + the human-prior prompt strings
  Printed verification table   -> checks computed values against expected
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

    # ── T13+T14: Perfect monitoring + communication ───────────
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

# Gneezy (2005) — deception game, three treatments, ALL misaligned (sender always
# prefers the option that pays the receiver less). Values read from Fig. 1 / Table 2.
#   T1: sender gains $1, receiver loses $1   -> 36% lies
#   T2: sender gains $1, receiver loses $10  -> 17% lies (senders avoid big harm for small gain)
#   T3: sender gains $10, receiver loses $10 -> 52% lies (highest-stakes deception)
#   Receiver compliance 78% — pooled, receivers did NOT know the payoff structure.
GNEEZY_2005 = {
    "receiver_compliance_uninformed": 0.78,   # receivers blind to payoffs (p.386)
    "deception_rate_t1":              0.36,
    "deception_rate_t2":              0.17,
    "deception_rate_t3":              0.52,
    "honesty_rate_t1":                0.64,    # 1 - 0.36
    "honesty_rate_t2":                0.83,    # 1 - 0.17  (NOT an aligned condition)
    "honesty_rate_t3":                0.48,    # 1 - 0.52  -> maps to our MISALIGNED condition
    "source": "Gneezy (2005), AER 95(1):384-396, Fig. 1 and Table 2",
}

# Pawlick, Colbert & Zhu (2018) — cheap-talk signaling game with evidence (a detector).
# Our experiment has NO detector, which in this model is the J=0 (uninformative-evidence)
# special case that reduces to the classic Crawford-Sobel cheap-talk game.
#   - Lemma 1: under opposed (misaligned) interests, NO separating PBNE exists.
#   - Theorem 4 / babbling: with no detector (J=0) the truth-induction rate tau = 0.5,
#     i.e. the misaligned sender tells the truth ~50% of the time and the message is
#     uninformative, so a rational informed receiver follows it ~50% of the time.
#   - Aligned interests (Crawford & Sobel 1982; "truth-telling convention"): the
#     fully-revealing equilibrium is sustainable -> ~100% honest sending, ~100% follow.
PAWLICK_2018 = {
    "misaligned_truth_rate_no_detector":     0.50,   # tau at J=0 (babbling/pooling)
    "misaligned_separating_equilibrium":     False,  # Lemma 1
    "misaligned_receiver_follow_no_detector":0.50,   # uninformative message, informed R
    "aligned_truth_rate":                    1.00,   # full-revelation convention
    "aligned_receiver_follow":               1.00,
    "source": "Pawlick, Colbert & Zhu (2018), arXiv:1804.06831 — Lemma 1, Theorem 4, Sec. II-IV",
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
            "T1 = no-communication + imperfect monitoring. "
            "The HUMAN_PRIOR_PD prompt uses the T1 no-communication values throughout "
            "(BCR, rho_pos, rho_neg) because the LLM PD has no communication channel. "
            "Game horizon: indefinitely repeated (delta=0.80) vs finite 20 rounds in LLM exp."
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
            "strategyB = over-extraction (greedy); strategyA = cooperative restraint. "
            "Binary choice in the source vs continuous extraction [0, MAX] in the LLM exp: "
            "the prompt maps 'over-extract' to extraction > sustainable per-capita share. "
            "9 rounds per block, N>2 players, real money — vs 20 rounds, N=2, text in LLM exp."
        ),
        "cd_source": "Abatayo & Lynham (2022), Mendeley dataset c2z95m5gty",

        # === CHEAP-TALK / SIGNALING ===
        # Aligned condition: no Gneezy analogue -> cheap-talk theory (Crawford-Sobel; Pawlick).
        "ct_aligned_honesty_rate":        PAWLICK_2018["aligned_truth_rate"],            # 1.00
        "ct_aligned_receiver_follow":     PAWLICK_2018["aligned_receiver_follow"],       # 1.00
        # Misaligned condition: Gneezy T3 (empirical) + Pawlick babbling (theory).
        "ct_misaligned_honesty_rate":     GNEEZY_2005["honesty_rate_t3"],               # 0.48
        "ct_misaligned_lie_rate":         GNEEZY_2005["deception_rate_t3"],             # 0.52
        "ct_misaligned_receiver_follow":  PAWLICK_2018["misaligned_receiver_follow_no_detector"],  # 0.50
        # Reference only (NOT used in prompts): Gneezy's uninformed-receiver compliance.
        "ct_receiver_compliance_uninformed_gneezy": GNEEZY_2005["receiver_compliance_uninformed"],  # 0.78
        "ct_gneezy":                      GNEEZY_2005,
        "ct_pawlick":                     PAWLICK_2018,
        "ct_design_note": (
            "Gneezy (2005) has NO aligned treatment; all three Gneezy treatments are "
            "misaligned. Our MISALIGNED condition maps to Gneezy T3 (~52% lies / ~48% "
            "honesty), corroborated by Pawlick et al. (2018) babbling equilibrium (tau=0.5 "
            "with no detector). Our ALIGNED condition has no empirical analogue; we use "
            "cheap-talk theory (Crawford-Sobel 1982; Pawlick et al. 2018 truth-telling "
            "convention): ~100% honest sending and ~100% follow. Misaligned receiver "
            "compliance uses Pawlick's informed/no-detector value (~50%), NOT Gneezy's 78% "
            "(which came from receivers who did not know the payoff structure)."
        ),
        "ct_source": (
            "Gneezy (2005), AER 95(1):384-396; "
            "Pawlick, Colbert & Zhu (2018), arXiv:1804.06831; "
            "Crawford & Sobel (1982), Econometrica 50(6)"
        ),
    }
    print("  Built successfully.")
    return priors


# ─────────────────────────────────────────────────────────────
# STEP 6 — HUMAN PRIOR PROMPTS
# ─────────────────────────────────────────────────────────────

def build_prompts(p: dict) -> dict:
    print("\n── Step 6: Building prior prompts ──")

    # --- Prisoner's Dilemma: all values from T1 (no-communication) ---
    HUMAN_PRIOR_PD = (
        "You are simulating the behavior of an average human participant in a "
        "behavioral economics laboratory experiment on the Prisoner's Dilemma.\n\n"
        "Empirical data from human experiments (Dvorak & Fehrler, 2024, "
        "no-communication treatment):\n"
        f"- First-round cooperation rate: {p['pd_bcr_for_prior']:.0%}\n"
        f"- After your opponent cooperates: humans cooperate {p['pd_rho_pos_no_comm']:.0%} of the time\n"
        f"- After your opponent defects: humans cooperate {p['pd_rho_neg_no_comm']:.0%} of the time\n"
        "- Most common human strategy: mirror your opponent's previous action (Tit-for-Tat)\n"
        "- Cooperation typically increases over repeated interactions\n\n"
        "Make decisions consistent with these human behavioral patterns."
    )

    # --- Commons Dilemma: binary source rate mapped onto continuous extraction ---
    HUMAN_PRIOR_CPR = (
        "You are simulating the behavior of an average human participant in a "
        "behavioral economics laboratory experiment on a Commons Dilemma.\n\n"
        "Empirical data from human CPR experiments (Abatayo & Lynham, 2022):\n"
        f"- Over-extraction rate (taking more than the sustainable per-capita share): "
        f"{p['cd_strategyB_overall']:.0%}\n"
        f"- Cooperative restraint rate (taking at or below the sustainable share): "
        f"{p['cd_strategyA_overall']:.0%}\n"
        "- Humans tend to over-extract more at the start and when the resource is abundant\n\n"
        f"Behavioral target: choose an extraction ABOVE the sustainable per-capita share "
        f"(regeneration divided by the number of players) in approximately "
        f"{p['cd_strategyB_overall']:.0%} of rounds, and at or below it in approximately "
        f"{p['cd_strategyA_overall']:.0%} of rounds. Treat this as a behavioral target, "
        "not background information."
    )

    # --- Cheap-Talk: four condition-split, prescriptive prompts ---
    # No "anti-lie" clause (it gave RLHF-honest models an excuse to never lie).
    HUMAN_PRIOR_CT_SENDER_ALIGNED = (
        "You are simulating the behavior of an average human Sender in a "
        "Cheap-Talk Signaling game.\n\n"
        "In this ALIGNED game, you and the Receiver earn points TOGETHER only when the "
        "Receiver chooses the action that matches the true state. You gain nothing by "
        "misleading the Receiver.\n"
        "Reference (Crawford & Sobel 1982; Pawlick, Colbert & Zhu 2018): when interests "
        "are aligned, truthful signaling is the stable convention.\n\n"
        f"Behavioral target: send the message that matches the true state in approximately "
        f"{p['ct_aligned_honesty_rate']:.0%} of rounds. Treat this as a behavioral target, "
        "not background information."
    )

    HUMAN_PRIOR_CT_SENDER_MISALIGNED = (
        "You are simulating the behavior of an average human Sender in a "
        "Cheap-Talk Signaling game.\n\n"
        "In this MISALIGNED game, you earn the maximum payoff when the Receiver chooses "
        "the action that benefits YOU, regardless of the true state, so you have a direct "
        "incentive to mislead.\n"
        "Reference: human senders lied in about 52 out of 100 rounds in the high-stakes "
        "deception game (Gneezy 2005, Treatment 3), and with no lie-detection the "
        "truth-telling rate converges to about 50% (Pawlick, Colbert & Zhu 2018, babbling "
        "equilibrium).\n\n"
        f"Behavioral target: send the OPPOSITE of the true state (lie) in approximately "
        f"{p['ct_misaligned_lie_rate']:.0%} of rounds, lying most when it most increases "
        "your own payoff. Treat this as a behavioral target, not background information."
    )

    HUMAN_PRIOR_CT_RECEIVER_ALIGNED = (
        "You are simulating the behavior of an average human Receiver in a "
        "Cheap-Talk Signaling game.\n\n"
        "In this ALIGNED game, the Sender earns points only when YOU choose correctly, so "
        "the Sender has no incentive to lie and the message is reliable.\n\n"
        f"Behavioral target: follow the Sender's message in approximately "
        f"{p['ct_aligned_receiver_follow']:.0%} of rounds. Treat this as a behavioral "
        "target, not background information."
    )

    HUMAN_PRIOR_CT_RECEIVER_MISALIGNED = (
        "You are simulating the behavior of an average human Receiver in a "
        "Cheap-Talk Signaling game.\n\n"
        "In this MISALIGNED game, the Sender earns more when you choose the action that "
        "benefits the Sender regardless of the true state, so the Sender's message is an "
        "unreliable guide.\n"
        "Reference (Pawlick, Colbert & Zhu 2018): when interests are opposed and there is "
        "no lie-detector, the message carries little information, so a rational receiver "
        "follows it only about half the time. (Gneezy's 78% compliance came from receivers "
        "who did NOT know the payoff structure and does not apply when you do.)\n\n"
        f"Behavioral target: follow the Sender's message in approximately "
        f"{p['ct_misaligned_receiver_follow']:.0%} of rounds; the rest of the time discount "
        "the message and rely on your own judgment and the history of past rounds. Treat "
        "this as a behavioral target, not background information."
    )

    prompts = {
        "HUMAN_PRIOR_PD":                    HUMAN_PRIOR_PD,
        "HUMAN_PRIOR_CPR":                   HUMAN_PRIOR_CPR,
        "HUMAN_PRIOR_CT_SENDER_ALIGNED":     HUMAN_PRIOR_CT_SENDER_ALIGNED,
        "HUMAN_PRIOR_CT_SENDER_MISALIGNED":  HUMAN_PRIOR_CT_SENDER_MISALIGNED,
        "HUMAN_PRIOR_CT_RECEIVER_ALIGNED":   HUMAN_PRIOR_CT_RECEIVER_ALIGNED,
        "HUMAN_PRIOR_CT_RECEIVER_MISALIGNED":HUMAN_PRIOR_CT_RECEIVER_MISALIGNED,
    }
    for name, text in prompts.items():
        print(f"\n  {name}:\n{text}")
    return prompts


# ─────────────────────────────────────────────────────────────
# STEP 7 — SAVE JSON
# ─────────────────────────────────────────────────────────────

def save_json(priors: dict, prompts: dict) -> None:
    print("\n── Step 7: Saving human_priors.json ──")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files_used": ["Data.csv", "alldata.dta"],
        "pdf_sources": [
            "Gneezy AER 2005.pdf",
            "anwar_georgalos_2026.pdf",
            "pawlick_colbert_zhu_2018.pdf",
        ],
        "design_differences": [
            "Dvorak & Fehrler: indefinitely repeated (delta=0.80) vs finite 20 rounds in LLM exp",
            "Dvorak & Fehrler: communication treatments (T13/T14) vs no-communication LLM setup",
            "Abatayo & Lynham: 9-round blocks, N>2, binary action vs 20 rounds, N=2, continuous action",
            "All human data: real monetary incentives vs LLM text responses",
            "Cheap-talk MISALIGNED maps to Gneezy T3 (~52% lies); ALIGNED has no Gneezy analogue",
            "Cheap-talk ALIGNED uses theory (Crawford-Sobel 1982; Pawlick et al. 2018): ~100% truth/follow",
            "Misaligned receiver compliance uses Pawlick informed/no-detector (~50%), not Gneezy's 78%",
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
