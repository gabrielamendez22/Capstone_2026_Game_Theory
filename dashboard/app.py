"""
Strategic Coherence in LLMs — Research Dashboard
ESADE MIBA Capstone 2026

Run:  python app.py
Open: http://127.0.0.1:8050
"""

import dash
from dash import Dash, html, dcc, callback, Input, Output, ctx

import tabs.overview     as overview
import tabs.simulation   as simulation
import tabs.metrics      as metrics
import tabs.fingerprints as fingerprints
import tabs.explorer     as explorer

TABS = [
    ("overview",     "Overview"),
    ("simulation",   "Simulation"),
    ("metrics",      "Strategic Metrics"),
    ("fingerprints", "Model Fingerprints"),
    ("explorer",     "Data Explorer"),
]

app = Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Strategic Coherence in LLMs \u00b7 ESADE 2026"
server = app.server  # required for gunicorn / Render deployment

app.layout = html.Div(
    [
        # ── Fixed site header with nav ────────────────────────────────────────
        html.Header(
            html.Div(
                [
                    # Brand
                    html.Div(
                        [
                            html.Span("\u25c8", className="nav-logo-icon"),
                            html.Span("Strategic AI", className="nav-brand-text"),
                        ],
                        className="nav-brand",
                    ),

                    # Nav links
                    html.Nav(
                        [
                            html.Button(
                                label,
                                id=f"nav-btn-{tab}",
                                className="nav-link" + (" active" if tab == "overview" else ""),
                                n_clicks=0,
                            )
                            for tab, label in TABS
                        ],
                        className="nav-links",
                    ),

                    # Badge
                    html.Div("ESADE MIBA \u00b7 2026", className="nav-badge"),
                ],
                className="nav-container",
            ),
            className="site-header",
        ),

        # ── Content panels ────────────────────────────────────────────────────
        html.Main(
            [
                html.Div(
                    overview.layout(),
                    id="panel-overview",
                    className="tab-panel",
                    style={"display": "block"},
                ),
                html.Div(
                    simulation.layout(),
                    id="panel-simulation",
                    className="tab-panel",
                    style={"display": "none"},
                ),
                html.Div(
                    metrics.layout(),
                    id="panel-metrics",
                    className="tab-panel",
                    style={"display": "none"},
                ),
                html.Div(
                    fingerprints.layout(),
                    id="panel-fingerprints",
                    className="tab-panel",
                    style={"display": "none"},
                ),
                html.Div(
                    explorer.layout(),
                    id="panel-explorer",
                    className="tab-panel",
                    style={"display": "none"},
                ),
            ],
            className="site-main",
        ),

        dcc.Store(id="active-tab-store", data="overview"),
    ],
    className="app-wrapper",
)


# ── Navigation callbacks ───────────────────────────────────────────────────────

@callback(
    Output("active-tab-store", "data"),
    [Input(f"nav-btn-{tab}", "n_clicks") for tab, _ in TABS],
    prevent_initial_call=True,
)
def set_active_tab(*_):
    triggered = ctx.triggered_id
    if triggered and triggered.startswith("nav-btn-"):
        return triggered[len("nav-btn-"):]
    return dash.no_update


@callback(
    [Output(f"panel-{tab}", "style") for tab, _ in TABS] +
    [Output(f"nav-btn-{tab}", "className") for tab, _ in TABS],
    Input("active-tab-store", "data"),
)
def show_active_panel(active):
    styles  = [{"display": "block"} if tab == active else {"display": "none"}
               for tab, _ in TABS]
    classes = ["nav-link active" if tab == active else "nav-link"
               for tab, _ in TABS]
    return styles + classes


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
