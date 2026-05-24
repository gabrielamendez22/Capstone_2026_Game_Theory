import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import dash
from dash import html, dcc, callback, Input, Output, State, ctx, clientside_callback
import plotly.graph_objects as go

from data_loader import get_matchups, get_events

dash.register_page(__name__, path="/simulation", name="Simulation")

SPEED_OPTIONS = [
    {"label": "Fast  (0.5 s)",  "value": 500},
    {"label": "Normal (1.5 s)", "value": 1500},
    {"label": "Slow  (3 s)",    "value": 3000},
]


# ── Plotly helpers ─────────────────────────────────────────────────────────────

def _base_layout(height=260):
    return dict(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#8b949e", family="Inter, sans-serif", size=12),
        xaxis=dict(
            title="Round",
            showgrid=True, gridcolor="#21262d",
            range=[0.5, 20.5], tickmode="linear", dtick=2,
            zeroline=False, color="#8b949e",
            title_font=dict(size=11),
        ),
        yaxis=dict(
            title="Belief  P(opponent cooperates)",
            showgrid=True, gridcolor="#21262d",
            range=[-0.05, 1.08],
            zeroline=False, color="#8b949e",
            tickformat=".0%",
            title_font=dict(size=11),
        ),
        legend=dict(
            bgcolor="rgba(22,27,34,0.85)",
            bordercolor="#30363d", borderwidth=1,
            font=dict(color="#c9d1d9", size=11),
            orientation="h", x=0, y=1.18,
        ),
        margin=dict(l=60, r=16, t=44, b=48),
        height=height,
        hovermode="x unified",
    )


def make_empty_figure():
    fig = go.Figure()
    fig.update_layout(**_base_layout())
    fig.add_annotation(
        text="Select a matchup and press Start Simulation",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(color="#484f58", size=13),
    )
    return fig


def make_figure(events_shown: list[dict]) -> go.Figure:
    data_a = [e for e in events_shown if e["type"] == "model_a"]
    data_b = [e for e in events_shown if e["type"] == "model_b"]
    fig    = go.Figure()

    def _add_trace(data, dash_style, side_label):
        if not data:
            return
        color  = data[0]["color"]
        name   = data[0]["model_name"]
        rounds  = [e["round"]  for e in data]
        beliefs = [e["belief"] for e in data]
        actions = [e["action"] for e in data]

        marker_colors  = ["#ef4444" if a == "D" else color for a in actions]
        marker_symbols = ["x"       if a == "D" else "circle" for a in actions]
        marker_sizes   = [13        if a == "D" else 7 for a in actions]

        fig.add_trace(go.Scatter(
            x=rounds, y=beliefs,
            mode="lines+markers",
            name=f"{name} ({side_label})",
            line=dict(color=color, width=2.5, dash=dash_style),
            marker=dict(
                color=marker_colors,
                symbol=marker_symbols,
                size=marker_sizes,
                line=dict(width=2, color=marker_colors),
            ),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Round %{x}  ·  Belief %{y:.0%}<extra></extra>"
            ),
        ))

    _add_trace(data_a, "solid", "right")
    _add_trace(data_b, "dot",   "left")

    # Vertical "now" marker at current round
    if data_a:
        fig.add_vline(
            x=data_a[-1]["round"],
            line_width=1, line_dash="dash", line_color="#30363d",
        )

    fig.update_layout(**_base_layout())
    return fig


# ── Chat bubble renderer ───────────────────────────────────────────────────────

def render_event(event: dict) -> html.Div:
    # System intro message
    if event["type"] == "system":
        return html.Div(
            html.Div(
                [
                    html.Span("⚖", className="system-icon"),
                    html.Div(
                        [
                            html.Div("GAME SETUP", className="system-label"),
                            html.Div(event["text"], className="system-text"),
                        ]
                    ),
                ],
                className="system-message",
            ),
            className="msg-wrapper center",
            style={"animation": "fadeIn 0.4s ease-out"},
        )

    is_a      = event["type"] == "model_a"
    is_defect = event["action"] == "D"

    action_text  = "✕  DEFECT"    if is_defect else "✓  COOPERATE"
    action_class = "action-defect" if is_defect else "action-cooperate"
    bubble_extra = " defect-bubble" if is_defect else ""
    bubble_side  = "right-bubble"   if is_a     else "left-bubble"

    avatar = html.Div(
        event["abbrev"],
        className="avatar",
        style={"backgroundColor": event["color"]},
    )

    bubble = html.Div(
        [
            html.Div(
                [
                    html.Span(event["model_name"], className="bubble-model-name"),
                    html.Span(f"Round {event['round']}", className="bubble-round"),
                ],
                className="bubble-header",
            ),
            html.Div(action_text, className=f"action-badge {action_class}"),
            html.Div(event["text"], className="bubble-text"),
            html.Div(
                [
                    html.Span(f"Belief: {event['belief']:.0%}", className="bubble-belief"),
                    html.Span(
                        f"+{event['payoff']} pts  ·  {event['cumulative']} total",
                        className="bubble-payoff",
                    ),
                ],
                className="bubble-footer",
            ),
        ],
        className=f"bubble {bubble_side}{bubble_extra}",
    )

    children = [bubble, avatar] if is_a else [avatar, bubble]
    return html.Div(
        children,
        className=f"msg-wrapper {'right' if is_a else 'left'}",
        style={"animation": "slideIn 0.32s ease-out"},
    )


# ── Page layout ────────────────────────────────────────────────────────────────

def layout():
    matchups = get_matchups()
    return html.Div(
        [
            # Filter bar
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Matchup", className="filter-label"),
                            dcc.Dropdown(
                                id="matchup-select",
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
                            html.Label("Speed", className="filter-label"),
                            dcc.Dropdown(
                                id="speed-select",
                                options=SPEED_OPTIONS,
                                value=1500,
                                clearable=False,
                                className="filter-dropdown speed-dropdown",
                            ),
                        ],
                        className="filter-group",
                    ),
                    html.Div(
                        [
                            html.Button(
                                "▶  Start Simulation",
                                id="start-btn",
                                className="btn-start",
                                n_clicks=0,
                            ),
                            html.Button(
                                "↺  Reset",
                                id="reset-btn",
                                className="btn-reset",
                                n_clicks=0,
                            ),
                        ],
                        className="filter-group filter-btns",
                    ),
                    html.Div(id="sim-status", className="sim-status"),
                ],
                className="filter-bar",
            ),

            # Chat + graph area
            html.Div(
                [
                    # Chat panel
                    html.Div(
                        html.Div(id="chat-messages", className="chat-messages"),
                        className="chat-panel",
                        id="chat-scroll-container",
                    ),

                    # Graph panel
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Trust Evolution", className="graph-title"),
                                    html.Span(
                                        "● Cooperate  ✕ Defect",
                                        className="graph-legend-hint",
                                    ),
                                ],
                                className="graph-header",
                            ),
                            dcc.Graph(
                                id="trust-graph",
                                figure=make_empty_figure(),
                                config={"displayModeBar": False},
                                className="trust-graph",
                            ),
                        ],
                        className="graph-panel",
                    ),
                ],
                className="sim-layout",
            ),

            # Hidden Dash internals
            dcc.Store(id="sim-store", data={"step": 0, "running": False}),
            dcc.Interval(id="sim-interval", interval=1500, disabled=True, n_intervals=0),
            html.Div(id="_scroll_dummy", style={"display": "none"}),
        ],
        className="simulation-page",
    )


# ── Main simulation callback ───────────────────────────────────────────────────

@callback(
    Output("chat-messages",    "children"),
    Output("trust-graph",      "figure"),
    Output("sim-store",        "data"),
    Output("sim-interval",     "disabled"),
    Output("sim-interval",     "interval"),
    Output("start-btn",        "children"),
    Output("sim-status",       "children"),
    Input("start-btn",         "n_clicks"),
    Input("reset-btn",         "n_clicks"),
    Input("sim-interval",      "n_intervals"),
    Input("matchup-select",    "value"),
    State("speed-select",      "value"),
    State("sim-store",         "data"),
    prevent_initial_call=True,
)
def run_simulation(start_n, reset_n, n_intervals, matchup, speed, store):
    triggered = ctx.triggered_id
    store     = store or {"step": 0, "running": False}
    speed     = speed or 1500

    # ── Reset / matchup change ────────────────────────────────────────────────
    if triggered in ("reset-btn", "matchup-select"):
        return (
            [],
            make_empty_figure(),
            {"step": 0, "running": False},
            True,
            speed,
            "▶  Start Simulation",
            "",
        )

    # ── Start button ──────────────────────────────────────────────────────────
    if triggered == "start-btn":
        store = {"step": 1, "running": True}

    # ── Interval tick ─────────────────────────────────────────────────────────
    elif triggered == "sim-interval":
        if not store.get("running"):
            raise dash.exceptions.PreventUpdate
        store["step"] = store.get("step", 0) + 1

    step = store["step"]

    if not matchup:
        raise dash.exceptions.PreventUpdate

    try:
        events = get_events(matchup)
    except Exception:
        raise dash.exceptions.PreventUpdate

    total  = len(events)
    shown  = events[:step]

    messages = [render_event(e) for e in shown]
    fig      = make_figure(shown)

    rounds_shown = len([e for e in shown if e["type"] == "model_a"])
    total_rounds = len([e for e in events if e["type"] == "model_a"])
    status       = f"Round {rounds_shown} / {total_rounds}" if rounds_shown else ""

    # ── Simulation complete ───────────────────────────────────────────────────
    if step >= total:
        store["running"] = False
        data_a   = [e for e in events if e["type"] == "model_a"]
        data_b   = [e for e in events if e["type"] == "model_b"]
        final_a  = data_a[-1]["cumulative"] if data_a else 0
        final_b  = data_b[-1]["cumulative"] if data_b else 0
        name_a   = data_a[0]["model_name"]  if data_a else "A"
        name_b   = data_b[0]["model_name"]  if data_b else "B"
        status   = f"Complete  ·  {name_a}: {final_a} pts  |  {name_b}: {final_b} pts"
        return messages, fig, store, True, speed, "↺  Replay", status

    return messages, fig, store, False, speed, "⏸  Running…", status


# ── Auto-scroll chat to bottom on new messages ─────────────────────────────────

clientside_callback(
    """
    function(children) {
        setTimeout(function () {
            var el = document.getElementById('chat-scroll-container');
            if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
        }, 80);
        return '';
    }
    """,
    Output("_scroll_dummy",  "children"),
    Input("chat-messages",   "children"),
)
