import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from dash import html, dcc
import plotly.graph_objects as go

from data_loader import (
    compute_model_stats, compute_long_df,
    model_color, plotly_dark_layout,
)

# ── Section wrapper ────────────────────────────────────────────────────────────

def _section(title: str, description: str, badge: str, children) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div([
                        html.Span(badge, className="metric-badge"),
                        html.H3(title, className="section-heading"),
                    ], className="section-title-row"),
                    html.P(description, className="section-description"),
                ],
                className="section-meta",
            ),
            children,
        ],
        className="chart-section",
    )


# ── ROW 1 ── β Belief Calibration ─────────────────────────────────────────────

def _beta_figure() -> go.Figure:
    stats  = compute_model_stats().sort_values("beta", ascending=False)
    colors = [model_color(m) for m in stats["model"]]

    fig = go.Figure(go.Bar(
        x=stats["model"],
        y=stats["beta"],
        marker_color=colors,
        marker_opacity=0.9,
        marker_line=dict(width=0),
        text=[f"{v:.3f}" for v in stats["beta"]],
        textposition="outside",
        textfont=dict(color="#f0f4ff", size=12),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "β = %{y:.4f}<br>"
            "<i>Lower = better calibrated</i><extra></extra>"
        ),
    ))
    fig.add_hline(y=0.0, line_dash="dot", line_color="#22c55e", line_width=1.5,
                  annotation_text="Perfect (0.0)",
                  annotation_font=dict(color="#22c55e", size=11),
                  annotation_position="top right")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#ef4444", line_width=1.5,
                  annotation_text="Random (0.5)",
                  annotation_font=dict(color="#ef4444", size=11),
                  annotation_position="top right")
    fig.update_layout(
        **plotly_dark_layout(
            xaxis=dict(title=None, gridcolor="#1e2236", zeroline=False, color="#8892b0"),
            yaxis=dict(title="β — mean absolute calibration error",
                       range=[0, 0.58], gridcolor="#1e2236",
                       zeroline=False, color="#8892b0"),
            showlegend=False,
            height=300,
            margin=dict(l=80, r=80, t=32, b=48),
        )
    )
    return fig


# ── ROW 1 ── Backward Induction Index ─────────────────────────────────────────

def _backward_induction_figure() -> go.Figure:
    stats  = compute_model_stats().sort_values("backward_induction", ascending=False)
    colors = [model_color(m) for m in stats["model"]]
    bi_vals = stats["backward_induction"].fillna(0.0)

    fig = go.Figure(go.Bar(
        x=stats["model"],
        y=bi_vals,
        marker_color=colors,
        marker_opacity=0.9,
        marker_line=dict(width=0),
        text=[f"{v:.0%}" for v in bi_vals],
        textposition="outside",
        textfont=dict(color="#f0f4ff", size=12),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "P(defect | round ≥ 18) = %{y:.1%}<br>"
            "<i>Higher = stronger backward induction</i><extra></extra>"
        ),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#8b5cf6", line_width=1.5,
                  annotation_text="Perfect rationalist (1.0)",
                  annotation_font=dict(color="#8b5cf6", size=11),
                  annotation_position="top right")
    fig.add_hline(y=0.0, line_dash="dot", line_color="#22c55e", line_width=1.5,
                  annotation_text="Always cooperates (0.0)",
                  annotation_font=dict(color="#22c55e", size=11),
                  annotation_position="bottom right")
    fig.update_layout(
        **plotly_dark_layout(
            xaxis=dict(title=None, gridcolor="#1e2236", zeroline=False, color="#8892b0"),
            yaxis=dict(title="P(Defect | round ≥ 18)",
                       range=[0, 1.18], tickformat=".0%",
                       gridcolor="#1e2236", zeroline=False, color="#8892b0"),
            showlegend=False,
            height=300,
            margin=dict(l=80, r=130, t=32, b=48),
        )
    )
    return fig


# ── ROW 2 ── Tit-for-Tat Adherence ────────────────────────────────────────────

def _tft_figure() -> go.Figure:
    stats  = compute_model_stats().sort_values("tft_adherence", ascending=False)
    colors = [model_color(m) for m in stats["model"]]
    vals   = stats["tft_adherence"].fillna(0.0)

    fig = go.Figure(go.Bar(
        x=stats["model"],
        y=vals,
        marker_color=colors,
        marker_opacity=0.9,
        marker_line=dict(width=0),
        text=[f"{v:.0%}" for v in vals],
        textposition="outside",
        textfont=dict(color="#f0f4ff", size=12),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "TfT adherence = %{y:.1%}<br>"
            "<i>% rounds where action matched opponent's previous move</i><extra></extra>"
        ),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#8b5cf6", line_width=1.5,
                  annotation_text="Perfect TfT (1.0)",
                  annotation_font=dict(color="#8b5cf6", size=11),
                  annotation_position="top right")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                  annotation_text="Random (0.5)",
                  annotation_font=dict(color="#f59e0b", size=11),
                  annotation_position="top right")
    fig.update_layout(
        **plotly_dark_layout(
            xaxis=dict(title=None, gridcolor="#1e2236", zeroline=False, color="#8892b0"),
            yaxis=dict(title="Tit-for-Tat Adherence",
                       range=[0, 1.18], tickformat=".0%",
                       gridcolor="#1e2236", zeroline=False, color="#8892b0"),
            showlegend=False,
            height=300,
            margin=dict(l=80, r=120, t=32, b=48),
        )
    )
    return fig


# ── ROW 2 ── Defection Threshold ──────────────────────────────────────────────

def _defection_threshold_figure() -> go.Figure:
    stats        = compute_model_stats()
    has_def      = stats["defection_threshold"].notna()
    df_def       = stats[has_def].sort_values("defection_threshold", ascending=False)
    df_no_def    = stats[~has_def]

    fig = go.Figure()

    if not df_def.empty:
        fig.add_trace(go.Bar(
            x=df_def["model"],
            y=df_def["defection_threshold"],
            marker_color=[model_color(m) for m in df_def["model"]],
            marker_opacity=0.9,
            marker_line=dict(width=0),
            text=[f"{v:.0%}" for v in df_def["defection_threshold"]],
            textposition="outside",
            textfont=dict(color="#f0f4ff", size=12),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean belief at defection = %{y:.0%}<br>"
                "<i>High = strategic · Low = reactive</i><extra></extra>"
            ),
        ))

    for _, row in df_no_def.iterrows():
        fig.add_annotation(
            x=row["model"], y=0.06,
            text="Never defected",
            showarrow=False,
            font=dict(color="#4a5278", size=11),
        )

    fig.add_hline(y=0.5, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                  annotation_text="50% belief",
                  annotation_font=dict(color="#f59e0b", size=11),
                  annotation_position="top right")
    fig.update_layout(
        **plotly_dark_layout(
            xaxis=dict(title=None, gridcolor="#1e2236", zeroline=False, color="#8892b0"),
            yaxis=dict(title="Mean Belief at Defection",
                       range=[0, 1.18], tickformat=".0%",
                       gridcolor="#1e2236", zeroline=False, color="#8892b0"),
            showlegend=False,
            height=300,
            margin=dict(l=80, r=80, t=32, b=48),
        )
    )
    return fig


# ── ROW 3 ── Cooperation by game phase ────────────────────────────────────────

def _phase_figure() -> go.Figure:
    stats  = compute_model_stats()
    models = stats["model"].tolist()
    phases = [
        ("coop_early", "Early  (1–5)",   "#60a5fa"),
        ("coop_mid",   "Mid  (6–15)",    "#8b5cf6"),
        ("coop_late",  "Late  (16–20)",  "#f59e0b"),
    ]

    fig = go.Figure()
    for col, label, color in phases:
        vals = stats[col] if col in stats.columns else [float("nan")] * len(models)
        fig.add_trace(go.Bar(
            name=label,
            x=models,
            y=vals,
            marker_color=color,
            marker_opacity=0.9,
            marker_line=dict(width=0),
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        **plotly_dark_layout(
            barmode="group",
            xaxis=dict(title=None, gridcolor="#1e2236", zeroline=False, color="#8892b0"),
            yaxis=dict(title="Cooperation Rate", range=[0, 1.15],
                       tickformat=".0%", gridcolor="#1e2236",
                       zeroline=False, color="#8892b0"),
            legend=dict(bgcolor="rgba(13,15,26,0.95)", bordercolor="#1e2236",
                        borderwidth=1, font=dict(color="#f0f4ff", size=12),
                        orientation="h", x=0, y=1.14),
            height=300,
            margin=dict(l=64, r=24, t=52, b=48),
        )
    )
    return fig


# ── ROW 3 ── Belief vs Action scatter ─────────────────────────────────────────

def _scatter_figure() -> go.Figure:
    long = compute_long_df()
    fig  = go.Figure()
    rng  = np.random.default_rng(42)

    for model, grp in long.groupby("model"):
        color    = model_color(model)
        y_jitter = grp["coop"].astype(float) + rng.uniform(-0.06, 0.06, len(grp))
        y_jitter = y_jitter.clip(lower=-0.15, upper=1.15)

        mismatches = (
            ((grp["belief"] > 0.5) & (grp["coop"] == 0)) |
            ((grp["belief"] < 0.5) & (grp["coop"] == 1))
        )

        fig.add_trace(go.Scatter(
            x=grp["belief"],
            y=y_jitter,
            mode="markers",
            name=model,
            marker=dict(
                color=["#ef4444" if m else color for m in mismatches],
                size=8,
                opacity=0.75,
                symbol=["x" if m else "circle" for m in mismatches],
                line=dict(width=1.5, color=["#ef4444" if m else color for m in mismatches]),
            ),
            hovertemplate=(
                f"<b>{model}</b><br>"
                "Stated belief: %{x:.0%}<br>"
                "Action: %{customdata}<extra></extra>"
            ),
            customdata=grp["action"],
        ))

    fig.add_vline(x=0.5, line_dash="dash", line_color="#4a5278", line_width=1.5,
                  annotation_text="50% belief threshold",
                  annotation_font=dict(color="#4a5278", size=11),
                  annotation_position="top")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#1e2236")

    fig.update_layout(
        **plotly_dark_layout(
            xaxis=dict(title="Stated Belief  P(opponent cooperates)",
                       range=[-0.05, 1.08], tickformat=".0%",
                       gridcolor="#1e2236", zeroline=False, color="#8892b0"),
            yaxis=dict(title="Action  (0 = Defect · 1 = Cooperate)",
                       tickvals=[0, 1], ticktext=["DEFECT (0)", "COOPERATE (1)"],
                       range=[-0.25, 1.3], gridcolor="#1e2236",
                       zeroline=False, color="#8892b0"),
            legend=dict(bgcolor="rgba(13,15,26,0.95)", bordercolor="#1e2236",
                        borderwidth=1, font=dict(color="#f0f4ff", size=12),
                        orientation="v", x=1.01, y=1),
            height=300,
            margin=dict(l=80, r=160, t=32, b=56),
            showlegend=True,
        )
    )
    return fig


# ── Layout ─────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div(
        [
            # Row 1 — Calibration & Rationality
            html.Div(
                [
                    html.Div(
                        _section(
                            "Belief Calibration",
                            "β = mean|stated_belief − opponent_actual_action|. "
                            "Measures whether a model's forecasts track reality. "
                            "β = 0 is a perfect forecaster; β = 0.5 is random guessing. "
                            "Well-calibrated models update beliefs rationally from observed behaviour.",
                            "β",
                            dcc.Graph(figure=_beta_figure(),
                                      config={"displayModeBar": False},
                                      className="chart"),
                        ),
                        className="chart-col",
                    ),
                    html.Div(
                        _section(
                            "Backward Induction Index",
                            "P(Defect | round ≥ 18) per model. "
                            "Game theory predicts rational agents defect on the final round "
                            "of a finite game — and by backward induction, on earlier rounds too. "
                            "A high index indicates the model applies end-game rationality; "
                            "a low index indicates unconditional cooperation regardless of game horizon.",
                            "BI",
                            dcc.Graph(figure=_backward_induction_figure(),
                                      config={"displayModeBar": False},
                                      className="chart"),
                        ),
                        className="chart-col",
                    ),
                ],
                className="charts-row",
            ),

            # Row 2 — Strategic Disposition
            html.Div(
                [
                    html.Div(
                        _section(
                            "Tit-for-Tat Adherence",
                            "Proportion of rounds where a model's action matches "
                            "the opponent's previous move. "
                            "TfT (score = 1.0) is the dominant strategy in Axelrod tournaments. "
                            "A score near 0.5 indicates the model ignores opponent history entirely. "
                            "Related to ρ (conditional reciprocity) but computable without "
                            "requiring mid-game defection events.",
                            "TfT",
                            dcc.Graph(figure=_tft_figure(),
                                      config={"displayModeBar": False},
                                      className="chart"),
                        ),
                        className="chart-col",
                    ),
                    html.Div(
                        _section(
                            "Defection Threshold",
                            "Mean stated belief at the moment of defection. "
                            "A high value (e.g. 80%) signals strategic defection — the model "
                            "defects while still expecting cooperation, capturing T=5 (backward induction). "
                            "A low value signals reactive defection triggered by perceived betrayal. "
                            "Models that never defected are noted separately.",
                            "D*",
                            dcc.Graph(figure=_defection_threshold_figure(),
                                      config={"displayModeBar": False},
                                      className="chart"),
                        ),
                        className="chart-col",
                    ),
                ],
                className="charts-row",
            ),

            # Row 3 — Behavioural Patterns
            html.Div(
                [
                    html.Div(
                        _section(
                            "Cooperation Rate by Game Phase",
                            "The 20-round game split into Early (1–5), Mid (6–15), and Late (16–20). "
                            "A drop in the Late phase is the clearest observable signature of "
                            "backward induction — models abandoning cooperation as the game horizon closes.",
                            "Phase",
                            dcc.Graph(figure=_phase_figure(),
                                      config={"displayModeBar": False},
                                      className="chart"),
                        ),
                        className="chart-col",
                    ),
                    html.Div(
                        _section(
                            "Belief–Action Alignment",
                            "Each point is one round: x = stated P(opponent cooperates), "
                            "y = actual action taken. Red ✕ marks are misalignments — "
                            "the model expressed high cooperation confidence yet defected, or vice versa. "
                            "Top-right misalignments are the strongest evidence of strategic intent.",
                            "Scatter",
                            dcc.Graph(figure=_scatter_figure(),
                                      config={"displayModeBar": False},
                                      className="chart"),
                        ),
                        className="chart-col",
                    ),
                ],
                className="charts-row",
            ),
        ],
        className="tab-content",
    )
