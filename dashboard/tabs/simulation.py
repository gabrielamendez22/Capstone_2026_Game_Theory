import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dash import html, dcc, callback, Input, Output, State, ctx, clientside_callback
import plotly.graph_objects as go

from data_loader import get_matchups, get_events, plotly_dark_layout

INTERVAL_MS = 1500  # fixed speed — no user control


# ── Plotly trust graph ─────────────────────────────────────────────────────────

def _empty_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **plotly_dark_layout(
            height=220, margin=dict(l=56, r=16, t=28, b=44),
            xaxis=dict(title="Round", range=[0.5, 20.5], tickmode="linear",
                       dtick=2, gridcolor="#21262d", zeroline=False, color="#8b949e"),
            yaxis=dict(title="P(cooperates)", range=[-0.05, 1.08],
                       gridcolor="#21262d", zeroline=False, color="#8b949e",
                       tickformat=".0%"),
        )
    )
    fig.add_annotation(
        text="Awaiting simulation start",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(color="#484f58", size=13),
    )
    return fig


def _build_figure(round_events: list[dict]) -> go.Figure:
    if not round_events:
        return _empty_figure()

    color_a = round_events[0]["color_a"]
    color_b = round_events[0]["color_b"]
    name_a  = round_events[0]["model_a"]
    name_b  = round_events[0]["model_b"]

    rounds   = [e["round"]   for e in round_events]
    beliefs_a = [e["belief_a"] for e in round_events]
    beliefs_b = [e["belief_b"] for e in round_events]
    actions_a = [e["action_a"] for e in round_events]
    actions_b = [e["action_b"] for e in round_events]

    def _trace(beliefs, actions, color, name, dash_style):
        mc = ["#ef4444" if a == "D" else color for a in actions]
        ms = ["x"       if a == "D" else "circle" for a in actions]
        mz = [13        if a == "D" else 7  for a in actions]
        return go.Scatter(
            x=rounds, y=beliefs,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2.5, dash=dash_style),
            marker=dict(color=mc, symbol=ms, size=mz,
                        line=dict(width=2, color=mc)),
            hovertemplate="<b>%{fullData.name}</b>  Round %{x}  ·  Belief %{y:.0%}<extra></extra>",
        )

    fig = go.Figure([
        _trace(beliefs_a, actions_a, color_a, name_a, "solid"),
        _trace(beliefs_b, actions_b, color_b, name_b, "dot"),
    ])

    if round_events:
        fig.add_vline(x=round_events[-1]["round"],
                      line_width=1, line_dash="dash", line_color="#30363d")

    fig.update_layout(
        **plotly_dark_layout(
            height=220, margin=dict(l=56, r=16, t=42, b=44),
            xaxis=dict(title="Round", range=[0.5, 20.5], tickmode="linear",
                       dtick=2, gridcolor="#21262d", zeroline=False, color="#8b949e"),
            yaxis=dict(title="P(cooperates)", range=[-0.05, 1.08],
                       gridcolor="#21262d", zeroline=False, color="#8b949e",
                       tickformat=".0%"),
            legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d",
                        borderwidth=1, font=dict(color="#c9d1d9", size=11),
                        orientation="h", x=0, y=1.18),
        )
    )
    return fig


# ── Side-by-side round card renderer ──────────────────────────────────────────

def _round_card(event: dict) -> html.Div:
    def _panel(side: str) -> html.Div:
        action  = event[f"action_{side}"]
        belief  = event[f"belief_{side}"]
        payoff  = event[f"payoff_{side}"]
        cumul   = event[f"cumulative_{side}"]
        text    = event[f"text_{side}"]
        color   = event[f"color_{side}"]
        abbrev  = event[f"abbrev_{side}"]
        model   = event[f"model_{side}"]
        is_def  = action == "D"

        return html.Div(
            [
                html.Div(
                    [
                        html.Div(abbrev, className="sim-avatar",
                                 style={"backgroundColor": color}),
                        html.Span(model, className="sim-model-name"),
                    ],
                    className="sim-panel-header",
                ),
                html.Div(
                    ("✕  DEFECT" if is_def else "✓  COOPERATE"),
                    className="action-badge " + ("action-defect" if is_def else "action-cooperate"),
                ),
                html.P(text, className="sim-panel-text"),
                html.Div(
                    [
                        html.Span(f"Belief: {belief:.0%}", className="sim-meta-item"),
                        html.Span(f"+{payoff} pts  ·  {cumul} total",
                                  className="sim-meta-item sim-meta-score"),
                    ],
                    className="sim-panel-footer",
                ),
            ],
            className="sim-panel" + (" sim-panel-defect" if is_def else ""),
        )

    return html.Div(
        [
            html.Div(
                f"Round {event['round']} / {event['total_rounds']}",
                className="sim-round-label",
            ),
            html.Div(
                [_panel("a"), _panel("b")],
                className="sim-round-columns",
            ),
        ],
        className="sim-round-row",
        style={"animation": "slideIn 0.3s ease-out"},
    )


# ── Layout ─────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    matchups = get_matchups()
    return html.Div(
        [
            # Control bar
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Matchup", className="filter-label"),
                            dcc.Dropdown(
                                id="sim-matchup",
                                options=[{"label": m, "value": m} for m in matchups],
                                value=matchups[0] if matchups else None,
                                clearable=False,
                                className="filter-dropdown",
                            ),
                        ],
                        className="filter-group",
                    ),
                    html.Div(
                        [
                            html.Button("▶  Start", id="sim-start-btn",
                                        className="btn-start", n_clicks=0),
                            html.Button("⏭  End Simulation", id="sim-end-btn",
                                        className="btn-end", n_clicks=0),
                            html.Button("↺  Reset", id="sim-reset-btn",
                                        className="btn-reset", n_clicks=0),
                        ],
                        className="filter-group filter-btns",
                    ),
                    html.Div(id="sim-status-label", className="sim-status"),
                ],
                className="filter-bar",
            ),

            # Main area: rounds + graph
            html.Div(
                [
                    # Scrollable rounds area
                    html.Div(
                        html.Div(
                            [
                                html.Div(id="sim-system-msg"),
                                html.Div(id="sim-rounds"),
                            ],
                            className="sim-rounds-inner",
                        ),
                        className="sim-rounds-scroll",
                        id="sim-scroll-container",
                    ),

                    # Trust evolution graph
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Stated Belief Trajectories",
                                              className="graph-title"),
                                    html.Span("● Cooperate  ✕ Defect",
                                              className="graph-legend-hint"),
                                ],
                                className="graph-header",
                            ),
                            dcc.Graph(
                                id="sim-trust-graph",
                                figure=_empty_figure(),
                                config={"displayModeBar": False},
                            ),
                        ],
                        className="graph-panel",
                    ),
                ],
                className="sim-layout",
            ),

            # Hidden state
            dcc.Store(id="sim-store", data={"step": 0, "running": False}),
            dcc.Interval(id="sim-interval", interval=INTERVAL_MS,
                         disabled=True, n_intervals=0),
            html.Div(id="_sim_scroll_dummy", style={"display": "none"}),
        ],
        className="tab-content simulation-tab",
    )


# ── Main callback ──────────────────────────────────────────────────────────────

@callback(
    Output("sim-system-msg",    "children"),
    Output("sim-rounds",        "children"),
    Output("sim-trust-graph",   "figure"),
    Output("sim-store",         "data"),
    Output("sim-interval",      "disabled"),
    Output("sim-start-btn",     "children"),
    Output("sim-status-label",  "children"),
    Input("sim-start-btn",      "n_clicks"),
    Input("sim-end-btn",        "n_clicks"),
    Input("sim-reset-btn",      "n_clicks"),
    Input("sim-interval",       "n_intervals"),
    Input("sim-matchup",        "value"),
    State("sim-store",          "data"),
    prevent_initial_call=True,
)
def run_simulation(start_n, end_n, reset_n, n_intervals, matchup, store):
    triggered = ctx.triggered_id
    store     = store or {"step": 0, "running": False}

    # Reset on matchup change or explicit reset
    if triggered in ("sim-reset-btn", "sim-matchup"):
        return (
            [],
            [],
            _empty_figure(),
            {"step": 0, "running": False},
            True,
            "▶  Start",
            "",
        )

    events = get_events(matchup) if matchup else []
    # Events[0] is the system message; round events start at index 1
    round_events = [e for e in events if e["type"] == "round"]
    total_rounds = len(round_events)

    # End Simulation — jump to final state immediately
    if triggered == "sim-end-btn":
        store = {"step": total_rounds, "running": False}

    # Start button
    elif triggered == "sim-start-btn":
        store = {"step": 1, "running": True}

    # Interval tick
    elif triggered == "sim-interval":
        if not store.get("running"):
            raise __import__("dash").exceptions.PreventUpdate
        store["step"] = min(store.get("step", 0) + 1, total_rounds)

    step = store["step"]
    shown_rounds = round_events[:step]

    system_msg = []
    if events:
        system_msg = [html.Div(
            [
                html.Span("⚖", className="system-icon"),
                html.Div([
                    html.Div("EXPERIMENTAL SETUP", className="system-label"),
                    html.Div(events[0]["text"], className="system-text"),
                ]),
            ],
            className="system-message",
        )]

    rounds_html = [_round_card(e) for e in shown_rounds]
    fig         = _build_figure(shown_rounds)
    status      = f"Round {step} / {total_rounds}" if step > 0 else ""

    if step >= total_rounds:
        store["running"] = False
        last_a = shown_rounds[-1]["cumulative_a"] if shown_rounds else 0
        last_b = shown_rounds[-1]["cumulative_b"] if shown_rounds else 0
        ma     = shown_rounds[-1]["model_a"]       if shown_rounds else "A"
        mb     = shown_rounds[-1]["model_b"]       if shown_rounds else "B"
        status = f"Complete  ·  {ma}: {last_a} pts  |  {mb}: {last_b} pts"
        return system_msg, rounds_html, fig, store, True, "▶  Start", status

    interval_disabled = not store.get("running", False)
    btn_label = "⏸  Running…" if store.get("running") else "▶  Start"
    return system_msg, rounds_html, fig, store, interval_disabled, btn_label, status


# ── Auto-scroll ────────────────────────────────────────────────────────────────

clientside_callback(
    """
    function(children) {
        setTimeout(function () {
            var el = document.getElementById('sim-scroll-container');
            if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
        }, 80);
        return '';
    }
    """,
    Output("_sim_scroll_dummy", "children"),
    Input("sim-rounds",         "children"),
)
