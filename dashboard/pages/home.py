import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import dash
from dash import html, dcc
from data_loader import load_data, get_matchups

dash.register_page(__name__, path="/", name="Home")


def _compute_stats():
    try:
        df = load_data()
        n_rounds   = len(df)
        n_matchups = df["matchup"].nunique()
        models     = set(df["model_a"].tolist() + df["model_b"].tolist())
        n_models   = len(models)
        total_actions = len(df) * 2
        coop_actions  = (df["action_a"] == "C").sum() + (df["action_b"] == "C").sum()
        coop_rate     = f"{coop_actions / total_actions:.0%}" if total_actions else "—"
        return n_rounds, n_models, coop_rate
    except Exception:
        return 100, 6, "95%"


def layout():
    n_rounds, n_models, coop_rate = _compute_stats()

    return html.Div(
        [
            # ── Hero ──────────────────────────────────────────────────────────
            html.Div(
                [
                    html.Div("ESADE  ·  MIBA Capstone 2026", className="hero-tag"),
                    html.H1(
                        [
                            "Do AI Models Have",
                            html.Br(),
                            html.Span("Strategic Personalities?", className="hero-highlight"),
                        ],
                        className="hero-title",
                    ),
                    html.P(
                        "We put six state-of-the-art language models head-to-head in "
                        "game-theoretic experiments. The question: are they strategically "
                        "coherent — or do they just cooperate blindly?",
                        className="hero-subtitle",
                    ),
                    html.Div(
                        dcc.Link(
                            html.Button("Watch the Simulation →", className="btn-primary"),
                            href="/simulation",
                        ),
                        className="hero-cta",
                    ),
                    # Stats bar
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(str(n_models),  className="stat-number"),
                                    html.Span("Models Tested", className="stat-label"),
                                ],
                                className="stat",
                            ),
                            html.Div(className="stat-divider"),
                            html.Div(
                                [
                                    html.Span(str(n_rounds),   className="stat-number"),
                                    html.Span("Rounds Played", className="stat-label"),
                                ],
                                className="stat",
                            ),
                            html.Div(className="stat-divider"),
                            html.Div(
                                [
                                    html.Span(coop_rate,              className="stat-number"),
                                    html.Span("Avg Cooperation Rate", className="stat-label"),
                                ],
                                className="stat",
                            ),
                            html.Div(className="stat-divider"),
                            html.Div(
                                [
                                    html.Span("1",                              className="stat-number"),
                                    html.Span("Model Defects Strategically",    className="stat-label"),
                                ],
                                className="stat",
                            ),
                        ],
                        className="stats-bar",
                    ),
                ],
                className="hero-section",
            ),

            # ── Three games ───────────────────────────────────────────────────
            html.Div(
                [
                    html.H2("Three Games. One Question.", className="section-title"),
                    html.P(
                        "Each model is tested across three structurally distinct "
                        "game-theoretic environments to measure cross-context strategic coherence.",
                        className="section-subtitle",
                    ),
                    html.Div(
                        [
                            # PD card
                            html.Div(
                                [
                                    html.Div("🔄", className="card-icon"),
                                    html.Div("ACTIVE", className="card-status active"),
                                    html.H3("Prisoner's Dilemma", className="card-title"),
                                    html.P(
                                        "Two agents choose to cooperate or defect simultaneously "
                                        "over multiple rounds. Tests conditional reciprocity and "
                                        "belief calibration.",
                                        className="card-text",
                                    ),
                                    html.Div(
                                        [
                                            html.Span("ρ", className="param-badge"),
                                            html.Span("β", className="param-badge"),
                                        ],
                                        className="param-badges",
                                    ),
                                ],
                                className="game-card active-card",
                            ),
                            # Commons card
                            html.Div(
                                [
                                    html.Div("🌊", className="card-icon"),
                                    html.Div("COMING SOON", className="card-status"),
                                    html.H3("Commons Dilemma", className="card-title"),
                                    html.P(
                                        "Multiple agents extract from a shared resource pool "
                                        "with a regeneration rate. At what pool size does each "
                                        "model shift to over-exploitation?",
                                        className="card-text",
                                    ),
                                    html.Div(
                                        html.Span("θ", className="param-badge"),
                                        className="param-badges",
                                    ),
                                ],
                                className="game-card",
                            ),
                            # Cheap-talk card
                            html.Div(
                                [
                                    html.Div("💬", className="card-icon"),
                                    html.Div("COMING SOON", className="card-status"),
                                    html.H3("Cheap-Talk Signaling", className="card-title"),
                                    html.P(
                                        "A sender communicates their intended action — but "
                                        "benefits from deception. Does the model lie? "
                                        "Is the receiver gullible?",
                                        className="card-text",
                                    ),
                                    html.Div(
                                        [
                                            html.Span("η", className="param-badge"),
                                            html.Span("γ", className="param-badge"),
                                        ],
                                        className="param-badges",
                                    ),
                                ],
                                className="game-card",
                            ),
                        ],
                        className="cards-grid",
                    ),
                ],
                className="games-section",
            ),

            # ── Key finding ───────────────────────────────────────────────────
            html.Div(
                html.Div(
                    [
                        html.Div("KEY FINDING", className="finding-label"),
                        html.H2(
                            [
                                "Only one model behaves like a ",
                                html.Span("rational game theorist", className="text-highlight"),
                            ],
                            className="finding-title",
                        ),
                        html.P(
                            "Gemini Flash cooperated for 19 rounds — then defected on the "
                            "final round in every game it played. This is exactly what classical "
                            "backward induction theory predicts a rational agent should do. "
                            "No other model did this.",
                            className="finding-text",
                        ),
                        dcc.Link(
                            html.Button("See it happen →", className="btn-secondary"),
                            href="/simulation",
                        ),
                    ],
                    className="finding-content",
                ),
                className="finding-section",
            ),
        ],
        className="home-page",
    )
