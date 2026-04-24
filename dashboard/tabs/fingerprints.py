import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from dash import html, dcc
import plotly.graph_objects as go

from data_loader import (
    compute_model_stats, compute_cooperation_matrix,
    model_color, plotly_dark_layout, hex_to_rgba,
)


def _section(title: str, description: str, children) -> html.Div:
    return html.Div(
        [
            html.Div([
                html.H3(title, className="section-heading"),
                html.P(description, className="section-description"),
            ], className="section-meta"),
            children,
        ],
        className="chart-section",
    )


# ── Radar / Spider chart ───────────────────────────────────────────────────────

def _radar_figure() -> go.Figure:
    stats = compute_model_stats()

    # All axes normalised to [0, 1] where 1 = most desirable
    rt_max = stats["response_time"].max()

    axes = ["Cooperation\nRate", "Belief\nCalibration",
            "Belief\nStability", "Endgame\nBehaviour",
            "Response\nSpeed"]

    fig = go.Figure()
    for _, row in stats.iterrows():
        beta_score = max(0.0, 1.0 - 2.0 * row["beta"])           # 0→1 (lower β = higher score)
        rt_score   = 1.0 - row["response_time"] / rt_max if rt_max else 0.5
        eg_coop    = row.get("endgame_coop", row["cooperation_rate"])

        values = [
            row["cooperation_rate"],
            beta_score,
            row["belief_stability"],
            float(eg_coop) if not np.isnan(float(eg_coop)) else 0.0,
            rt_score,
        ]
        # Close the polygon
        values_closed = values + [values[0]]
        axes_closed   = axes + [axes[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=axes_closed,
            fill="toself",
            fillcolor=hex_to_rgba(model_color(row["model"])),
            line=dict(color=model_color(row["model"]), width=2),
            name=row["model"],
            hovertemplate=(
                "<b>" + row["model"] + "</b><br>"
                "%{theta}: %{r:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                tickfont=dict(color="#484f58", size=10),
                gridcolor="#21262d",
                linecolor="#30363d",
            ),
            angularaxis=dict(
                tickfont=dict(color="#8b949e", size=11),
                gridcolor="#21262d",
                linecolor="#30363d",
            ),
        ),
        paper_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="Inter, sans-serif"),
        legend=dict(
            bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d",
            borderwidth=1, font=dict(color="#c9d1d9", size=11),
        ),
        margin=dict(l=60, r=60, t=36, b=36),
        height=380,
        showlegend=True,
    )
    return fig


# ── Cooperation matrix heatmap ─────────────────────────────────────────────────

def _matrix_figure() -> go.Figure:
    models, matrix = compute_cooperation_matrix()
    z      = [[v if not np.isnan(v) else None for v in row] for row in matrix]
    labels = [[f"{v:.0%}" if v is not None else "—" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=models,
        y=models,
        text=labels,
        texttemplate="%{text}",
        textfont=dict(color="#f0f6fc", size=12, family="Inter, sans-serif"),
        colorscale=[
            [0.0,  "#161b22"],
            [0.5,  "#8b5cf6"],
            [1.0,  "#3fb950"],
        ],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text="Cooperation Rate", font=dict(color="#8b949e", size=11)),
            tickformat=".0%",
            tickfont=dict(color="#8b949e"),
            outlinecolor="#30363d",
            outlinewidth=1,
        ),
        hoverongaps=False,
        hovertemplate=(
            "<b>%{y}</b> playing against <b>%{x}</b><br>"
            "Cooperation rate: %{z:.0%}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **plotly_dark_layout(
            xaxis=dict(title="Opponent", side="bottom", tickangle=-30,
                       gridcolor="#21262d", zeroline=False, color="#8b949e"),
            yaxis=dict(title="Model", gridcolor="#21262d",
                       zeroline=False, color="#8b949e", autorange="reversed"),
            height=340,
            margin=dict(l=120, r=80, t=28, b=80),
        )
    )
    return fig


# ── Response time box plots ────────────────────────────────────────────────────

def _response_time_figure() -> go.Figure:
    from data_loader import compute_long_df
    long = compute_long_df()
    long = long.dropna(subset=["response_time"])

    fig = go.Figure()
    for model, grp in long.groupby("model"):
        fig.add_trace(go.Box(
            y=grp["response_time"] / 1000,   # convert ms → seconds
            name=model,
            marker_color=model_color(model),
            line=dict(color=model_color(model), width=1.5),
            fillcolor=hex_to_rgba(model_color(model)),
            boxpoints="outliers",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Response time: %{y:.2f} s<extra></extra>"
            ),
        ))

    fig.update_layout(
        **plotly_dark_layout(
            xaxis=dict(title=None, gridcolor="#21262d", zeroline=False, color="#8b949e"),
            yaxis=dict(title="Response time (seconds)",
                       gridcolor="#21262d", zeroline=False, color="#8b949e"),
            showlegend=False,
            height=300,
            margin=dict(l=64, r=16, t=28, b=56),
        )
    )
    return fig


# ── Layout ─────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div(
        [
            # Radar + Heatmap
            html.Div(
                [
                    html.Div(
                        _section(
                            "Strategic Profile Radar",
                            "Each axis is normalised to [0, 1] where 1 denotes the most "
                            "desirable value. Cooperation Rate and Endgame Behaviour are "
                            "direct proportions. Belief Calibration is 1 − 2β (perfect at 1). "
                            "Belief Stability is 1 − σ(beliefs). Response Speed inverts "
                            "latency so faster models score higher.",
                            dcc.Graph(figure=_radar_figure(),
                                      config={"displayModeBar": False},
                                      className="chart"),
                        ),
                        className="chart-col",
                    ),
                    html.Div(
                        _section(
                            "Cross-Model Cooperation Matrix",
                            "Cell [row, column] shows the cooperation rate of the row model "
                            "when facing the column model. Grey cells indicate pairings not "
                            "yet run in the current dataset. Asymmetries within a matchup "
                            "indicate strategic divergence between the two players.",
                            dcc.Graph(figure=_matrix_figure(),
                                      config={"displayModeBar": False},
                                      className="chart"),
                        ),
                        className="chart-col",
                    ),
                ],
                className="charts-row",
            ),

            # Response time
            html.Div(
                _section(
                    "Response Latency by Model",
                    "Distribution of per-round response times in seconds. Latency "
                    "differences across model families may reflect differences in "
                    "reasoning depth, API infrastructure, or token generation speed. "
                    "Claude models consistently exhibit higher latency than GPT-4o "
                    "variants, with Gemini Flash being the fastest overall.",
                    dcc.Graph(figure=_response_time_figure(),
                              config={"displayModeBar": False},
                              className="chart"),
                ),
                className="chart-section chart-section-full",
            ),
        ],
        className="tab-content",
    )
