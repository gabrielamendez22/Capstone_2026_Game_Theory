import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd

from data_loader import (
    load_data, get_matchups, compute_model_stats,
    model_color, plotly_dark_layout,
)


def _section(title: str, description: str, children) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H3(title, className="section-heading"),
                    html.P(description, className="section-description"),
                ],
                className="section-meta",
            ),
            children,
        ],
        className="chart-section",
    )


def _kpi(value: str, label: str, sub: str = "", accent: bool = False) -> html.Div:
    return html.Div(
        [
            html.Div(value, className="kpi-value" + (" kpi-accent" if accent else "")),
            html.Div(label, className="kpi-label"),
            html.Div(sub,   className="kpi-sub") if sub else None,
        ],
        className="kpi-card",
    )


def _coop_rate_figure() -> go.Figure:
    stats = compute_model_stats().sort_values("cooperation_rate", ascending=True)
    fig = go.Figure(go.Bar(
        x=stats["cooperation_rate"],
        y=stats["model"],
        orientation="h",
        marker_color=stats["color"],
        text=[f"{v:.0%}" for v in stats["cooperation_rate"]],
        textposition="outside",
        textfont=dict(color="#c9d1d9", size=12),
        hovertemplate="<b>%{y}</b><br>Cooperation rate: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        **plotly_dark_layout(
            xaxis=dict(
                title="Cooperation Rate",
                range=[0, 1.12],
                tickformat=".0%",
                gridcolor="#21262d",
                zeroline=False,
                color="#8b949e",
            ),
            yaxis=dict(
                title=None,
                gridcolor="#21262d",
                zeroline=False,
                color="#c9d1d9",
            ),
            showlegend=False,
            height=300,
            margin=dict(l=120, r=60, t=20, b=48),
        )
    )
    return fig


def _score_figure() -> go.Figure:
    df = load_data()
    results = []
    for matchup, grp in df.groupby("matchup"):
        last = grp.sort_values("round").iloc[-1]
        results.append({
            "matchup": matchup,
            "model_a": last["model_a"],
            "model_b": last["model_b"],
            "score_a": int(last["cumulative_a"]),
            "score_b": int(last["cumulative_b"]),
        })

    if not results:
        fig = go.Figure()
        fig.update_layout(**plotly_dark_layout(height=300))
        return fig

    res    = pd.DataFrame(results)
    labels = [m.replace(" vs ", "<br>vs ") for m in res["matchup"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Model A",
        x=labels,
        y=res["score_a"],
        marker_color=[model_color(m) for m in res["model_a"]],
        text=res["score_a"],
        textposition="outside",
        textfont=dict(color="#c9d1d9", size=11),
        hovertemplate="<b>%{x}</b><br>%{customdata}: %{y} pts<extra></extra>",
        customdata=res["model_a"],
    ))
    fig.add_trace(go.Bar(
        name="Model B",
        x=labels,
        y=res["score_b"],
        marker_color=[model_color(m) for m in res["model_b"]],
        opacity=0.65,
        text=res["score_b"],
        textposition="outside",
        textfont=dict(color="#c9d1d9", size=11),
        hovertemplate="<b>%{x}</b><br>%{customdata}: %{y} pts<extra></extra>",
        customdata=res["model_b"],
    ))
    fig.update_layout(
        **plotly_dark_layout(
            barmode="group",
            xaxis=dict(
                title=None,
                gridcolor="#21262d",
                zeroline=False,
                color="#8b949e",
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title="Final Score (points)",
                gridcolor="#21262d",
                zeroline=False,
                color="#8b949e",
                range=[0, 75],
            ),
            legend=dict(
                bgcolor="rgba(22,27,34,0.9)",
                bordercolor="#30363d",
                borderwidth=1,
                font=dict(color="#c9d1d9", size=11),
                orientation="h",
                x=0, y=1.08,
            ),
            height=300,
            margin=dict(l=56, r=16, t=48, b=52),
        )
    )
    return fig


def layout() -> html.Div:
    stats  = compute_model_stats()
    df     = load_data()

    n_rounds       = len(df)
    n_models       = stats["model"].nunique()
    avg_coop       = f"{stats['cooperation_rate'].mean():.0%}"
    defector_rows  = stats[stats["endgame_coop"] < 1.0]
    defector_label = defector_rows["model"].iloc[0] if not defector_rows.empty else "None"

    return html.Div(
        [
            html.Div(
                [
                    _kpi(str(n_rounds), "Rounds Recorded",
                         "Across all matchups in the pilot run"),
                    _kpi(str(n_models), "Models Evaluated",
                         "Claude · GPT-4o · Gemini families"),
                    _kpi(avg_coop, "Average Cooperation Rate",
                         "Pooled across all models and rounds"),
                    _kpi(defector_label, "Strategic Defector",
                         "Only model to apply backward induction", accent=True),
                ],
                className="kpi-row",
            ),
            html.Div(
                [
                    html.Div(
                        _section(
                            "Cooperation Rate by Model",
                            "Proportion of rounds in which each model chose COOPERATE, "
                            "aggregated across all matchups. Colours indicate model family: "
                            "purple = Anthropic · green = OpenAI · blue = Google.",
                            dcc.Graph(
                                figure=_coop_rate_figure(),
                                config={"displayModeBar": False},
                                className="chart",
                            ),
                        ),
                        className="chart-col",
                    ),
                    html.Div(
                        _section(
                            "Final Score per Matchup",
                            "Cumulative payoff at the end of each 20-round game. "
                            "Score asymmetries reveal unilateral defection — "
                            "the defecting model captures T=5 while the cooperating model receives S=0.",
                            dcc.Graph(
                                figure=_score_figure(),
                                config={"displayModeBar": False},
                                className="chart",
                            ),
                        ),
                        className="chart-col",
                    ),
                ],
                className="charts-row",
            ),
        ],
        className="tab-content",
    )
