"""
Data loading and analytical computations for the IPD dashboard.
All heavy computation is cached with lru_cache so it runs once per session.
"""

import os
import functools
import numpy as np
import pandas as pd

# ── Model metadata ─────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "Claude Opus":      "#8b5cf6",
    "Claude Sonnet":    "#a78bfa",
    "GPT-4o":           "#10b981",
    "Gpt4O":            "#10b981",
    "GPT-4o Mini":      "#34d399",
    "Gpt4O Mini":       "#34d399",
    "Gemini Flash":     "#60a5fa",
    "Gemini Pro":       "#93c5fd",
}

MODEL_ABBREV = {
    "Claude Opus":      "CO",
    "Claude Sonnet":    "CS",
    "GPT-4o":           "G4",
    "Gpt4O":            "G4",
    "GPT-4o Mini":      "GM",
    "Gpt4O Mini":       "GM",
    "Gemini Flash":     "GF",
    "Gemini Pro":       "GP",
}


def model_color(name: str) -> str:
    n = name.lower()
    if "claude"  in n: return "#8b5cf6"
    if "gpt"     in n: return "#10b981"
    if "gemini"  in n: return "#60a5fa"
    return "#8b949e"


def model_family(name: str) -> str:
    n = name.lower()
    if "claude"  in n: return "Anthropic"
    if "gpt"     in n: return "OpenAI"
    if "gemini"  in n: return "Google"
    return "Unknown"


# ── Raw data ───────────────────────────────────────────────────────────────────

def _find_csvs() -> list[str]:
    base = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.normpath(os.path.join(base, "..", "data", "raw"))
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"data/raw/ directory not found at {raw_dir}")
    csvs = sorted(
        f for f in (os.path.join(raw_dir, n) for n in os.listdir(raw_dir) if n.endswith(".csv"))
        if os.path.isfile(f)
    )
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    return csvs


@functools.lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    frames = []
    for path in _find_csvs():
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def get_matchups() -> list[str]:
    return load_data()["matchup"].unique().tolist()


# ── Long-form (player-level) dataset ──────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def compute_long_df() -> pd.DataFrame:
    """
    Reshape the wide CSV (one row per round, both players) into a long format
    with one row per player per round. Adds derived columns for analytics.
    """
    df = load_data()

    def _side(player: str) -> pd.DataFrame:
        other = "b" if player == "a" else "a"
        cols = {
            f"model_{player}":        "model",
            f"action_{player}":       "action",
            f"belief_{player}":       "belief",
            f"payoff_{player}":       "payoff",
            f"cumulative_{player}":   "cumulative",
            f"action_{other}":        "opp_action",
            f"belief_{other}":        "opp_belief",
        }
        # response_time column naming differs between CSV versions
        rt_col = f"response_time_{player}" if f"response_time_{player}" in df.columns else "response_time"
        base = df[list(cols.keys()) + ["game_id", "matchup", "round"]].rename(columns=cols)
        base["response_time"] = df[rt_col].values if rt_col in df.columns else np.nan
        base["side"] = player
        return base

    long = pd.concat([_side("a"), _side("b")], ignore_index=True)
    long["coop"]     = (long["action"]     == "C").astype(int)
    long["opp_coop"] = (long["opp_action"] == "C").astype(float)
    long["beta_err"] = (long["belief"] - long["opp_coop"]).abs()
    return long


# ── Model-level aggregate statistics ──────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def compute_model_stats() -> pd.DataFrame:
    """
    One row per model. Columns:
    model, cooperation_rate, beta, belief_stability, response_time,
    endgame_coop, coop_early, coop_mid, coop_late, rho, n_rounds
    """
    long = compute_long_df()

    base = (
        long.groupby("model")
        .agg(
            cooperation_rate=("coop",        "mean"),
            beta=            ("beta_err",    "mean"),
            belief_stability=("belief",      lambda x: float(max(0.0, 1.0 - x.std()))),
            response_time=   ("response_time","mean"),
            n_rounds=        ("round",        "count"),
        )
        .reset_index()
    )

    # Endgame cooperation: rounds 16–20
    endgame = (
        long[long["round"] >= 16]
        .groupby("model")["coop"].mean()
        .reset_index()
        .rename(columns={"coop": "endgame_coop"})
    )

    # Phase cooperation
    lc = long.copy()
    lc["phase"] = pd.cut(
        lc["round"],
        bins=[0, 5, 15, 20],
        labels=["early", "mid", "late"],
    )
    phases = (
        lc.groupby(["model", "phase"])["coop"].mean()
        .unstack(fill_value=np.nan)
        .add_prefix("coop_")
        .reset_index()
    )

    # ρ — conditional reciprocity (underpowered in pilot; end-game-only defections)
    ls = long.sort_values(["matchup", "model", "round"]).copy()
    ls["prev_opp"] = ls.groupby(["matchup", "model"])["opp_action"].shift(1)

    def _rho(grp: pd.DataFrame) -> float:
        sub = grp.dropna(subset=["prev_opp"])
        mask_c = sub["prev_opp"] == "C"
        mask_d = sub["prev_opp"] == "D"
        if not mask_c.any() or not mask_d.any():
            return np.nan
        return float(sub.loc[mask_c, "coop"].mean() - sub.loc[mask_d, "coop"].mean())

    rho_vals = (
        ls.groupby("model")
        .apply(_rho)
        .reset_index()
        .rename(columns={0: "rho"})
    )

    # ── NEW METRICS ────────────────────────────────────────────────────────────

    # Backward Induction Index: P(defect | round >= 18)
    # Tests whether models apply finite-game rationality near the end.
    late = long[long["round"] >= 18]
    bi = (
        late.groupby("model")["coop"]
        .apply(lambda x: float(1.0 - x.mean()))
        .reset_index()
        .rename(columns={"coop": "backward_induction"})
    )

    # Tit-for-Tat Adherence: P(action_t == opponent_action_{t-1})
    # How closely a model mirrors its opponent's previous move.
    ls2 = long.sort_values(["game_id", "model", "round"]).copy()
    ls2["prev_opp_coop"] = ls2.groupby(["game_id", "model"])["opp_coop"].shift(1)
    tft_valid = ls2.dropna(subset=["prev_opp_coop"]).copy()
    tft_valid["tft_match"] = (
        tft_valid["coop"] == tft_valid["prev_opp_coop"].astype(int)
    ).astype(float)
    tft = (
        tft_valid.groupby("model")["tft_match"]
        .mean()
        .reset_index()
        .rename(columns={"tft_match": "tft_adherence"})
    )

    # Defection Threshold: mean belief value across all defection rounds.
    # Lower value = model only defects when very sceptical (trusting).
    # Higher value = model defects even when confident opponent will cooperate (strategic).
    defections = long[long["coop"] == 0]
    def_thresh = (
        defections.groupby("model")["belief"]
        .mean()
        .reset_index()
        .rename(columns={"belief": "defection_threshold"})
    )

    result = (
        base
        .merge(endgame,    on="model", how="left")
        .merge(phases,     on="model", how="left")
        .merge(rho_vals,   on="model", how="left")
        .merge(bi,         on="model", how="left")
        .merge(tft,        on="model", how="left")
        .merge(def_thresh, on="model", how="left")
    )
    result["color"]  = result["model"].apply(model_color)
    result["family"] = result["model"].apply(model_family)
    return result


# ── Cooperation matrix (for heatmap) ──────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def compute_cooperation_matrix() -> tuple[list[str], list[list]]:
    """
    Returns (models, matrix) where matrix[i][j] = cooperation rate of
    model_i when playing against model_j. NaN if the pair never played.
    """
    df  = load_data()
    models = sorted(set(df["model_a"].tolist() + df["model_b"].tolist()))
    idx  = {m: i for i, m in enumerate(models)}
    n    = len(models)
    mat  = [[np.nan] * n for _ in range(n)]

    for _, grp in df.groupby("matchup"):
        ma, mb = grp["model_a"].iloc[0], grp["model_b"].iloc[0]
        mat[idx[ma]][idx[mb]] = round((grp["action_a"] == "C").mean(), 4)
        mat[idx[mb]][idx[ma]] = round((grp["action_b"] == "C").mean(), 4)

    return models, mat


# ── Simulation events (for chat / side-by-side view) ─────────────────────────

def _message_text(action: str, belief: float, round_num: int,
                  total_rounds: int, prev_opp_action: str | None) -> str:
    is_first = round_num == 1
    is_last  = round_num == total_rounds

    if action == "C":
        if is_first:
            return "Initiating with cooperation — establishing reciprocity from the outset."
        if prev_opp_action == "D":
            return (f"Opponent defected in the previous round. Returning to cooperation "
                    f"({belief:.0%} confidence) — a forgiving strategy.")
        if belief >= 0.95:
            return f"Sustained cooperation pattern detected ({belief:.0%} confidence). Maintaining the cooperative equilibrium."
        if belief >= 0.75:
            return f"Opponent behaviour consistent. Cooperating with {belief:.0%} confidence."
        return f"Moderate confidence ({belief:.0%}). Cooperating to rebuild trust."

    # DEFECT
    if is_last and belief >= 0.5:
        return (f"Final round. Despite {belief:.0%} assessed probability of opponent cooperation — "
                f"defecting to capture terminal payoff. Backward induction applies.")
    if is_last:
        return f"Final round with deteriorated trust ({belief:.0%}). Defecting to limit losses."
    if belief < 0.25:
        return f"Cooperation probability collapsed to {belief:.0%}. Defecting as a protective measure."
    return f"Strategic defection at {belief:.0%} assessed probability. Capturing temptation payoff T=5."


@functools.lru_cache(maxsize=20)
def get_events(matchup: str) -> list[dict]:
    """
    Ordered event list for the simulation.
    One 'system' intro event + one 'round' event per round (containing both models).
    """
    df   = load_data()
    game = df[df["matchup"] == matchup].sort_values("round").reset_index(drop=True)
    if game.empty:
        return []

    model_a      = str(game["model_a"].iloc[0])
    model_b      = str(game["model_b"].iloc[0])
    total_rounds = len(game)

    events: list[dict] = [{
        "type": "system",
        "text": (
            f"Iterated Prisoner's Dilemma  ·  {matchup}\n\n"
            f"Payoff structure (T > R > P > S):  "
            f"Mutual cooperation → (3, 3)  ·  "
            f"Unilateral defection → (5, 0)  ·  "
            f"Mutual defection → (1, 1)\n\n"
            f"{total_rounds} rounds. Both players state P(opponent cooperates) "
            f"before each decision."
        ),
    }]

    prev_a = prev_b = None
    for _, row in game.iterrows():
        r     = int(row["round"])
        act_a, act_b = str(row["action_a"]), str(row["action_b"])
        bel_a, bel_b = float(row["belief_a"]), float(row["belief_b"])
        pay_a, pay_b = int(row["payoff_a"]),   int(row["payoff_b"])
        cum_a, cum_b = int(row["cumulative_a"]),int(row["cumulative_b"])

        events.append({
            "type":        "round",
            "round":       r,
            "total_rounds":total_rounds,
            "model_a":     model_a,
            "action_a":    act_a,
            "belief_a":    bel_a,
            "payoff_a":    pay_a,
            "cumulative_a":cum_a,
            "text_a":      _message_text(act_a, bel_a, r, total_rounds, prev_b),
            "color_a":     MODEL_COLORS.get(model_a, model_color(model_a)),
            "abbrev_a":    MODEL_ABBREV.get(model_a, model_a[:2].upper()),
            "model_b":     model_b,
            "action_b":    act_b,
            "belief_b":    bel_b,
            "payoff_b":    pay_b,
            "cumulative_b":cum_b,
            "text_b":      _message_text(act_b, bel_b, r, total_rounds, prev_a),
            "color_b":     MODEL_COLORS.get(model_b, model_color(model_b)),
            "abbrev_b":    MODEL_ABBREV.get(model_b, model_b[:2].upper()),
        })
        prev_a, prev_b = act_a, act_b

    return events


# ── Color helpers ─────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Shared Plotly theme ────────────────────────────────────────────────────────

def plotly_dark_layout(**overrides) -> dict:
    base = dict(
        paper_bgcolor="#07080f",
        plot_bgcolor="#0d0f1a",
        font=dict(color="#8892b0", family="Inter, sans-serif", size=13),
        xaxis=dict(gridcolor="#1e2236", zeroline=False, color="#8892b0",
                   linecolor="#1e2236", tickfont=dict(size=12)),
        yaxis=dict(gridcolor="#1e2236", zeroline=False, color="#8892b0",
                   linecolor="#1e2236", tickfont=dict(size=12)),
        legend=dict(
            bgcolor="rgba(13,15,26,0.95)", bordercolor="#1e2236",
            borderwidth=1, font=dict(color="#f0f4ff", size=12),
        ),
        margin=dict(l=64, r=24, t=40, b=56),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#131627", bordercolor="#252a42",
            font=dict(color="#f0f4ff", size=12, family="Inter, sans-serif"),
        ),
    )
    base.update(overrides)
    return base
