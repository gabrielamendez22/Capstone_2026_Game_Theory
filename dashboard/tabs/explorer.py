import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
from dash import html, dcc, callback, Input, Output, State, dash_table
import pandas as pd

from data_loader import load_data, get_matchups


# ── Column definitions for the DataTable ──────────────────────────────────────

DISPLAY_COLS = [
    {"name": "Game",       "id": "game_id",       "type": "numeric"},
    {"name": "Matchup",    "id": "matchup",        "type": "text"},
    {"name": "Round",      "id": "round",          "type": "numeric"},
    {"name": "Model A",    "id": "model_a",        "type": "text"},
    {"name": "Action A",   "id": "action_a",       "type": "text"},
    {"name": "Belief A",   "id": "belief_a",       "type": "numeric"},
    {"name": "Payoff A",   "id": "payoff_a",       "type": "numeric"},
    {"name": "Score A",    "id": "cumulative_a",   "type": "numeric"},
    {"name": "Model B",    "id": "model_b",        "type": "text"},
    {"name": "Action B",   "id": "action_b",       "type": "text"},
    {"name": "Belief B",   "id": "belief_b",       "type": "numeric"},
    {"name": "Payoff B",   "id": "payoff_b",       "type": "numeric"},
    {"name": "Score B",    "id": "cumulative_b",   "type": "numeric"},
]


def _table_style() -> dict:
    cell = {
        "backgroundColor": "#161b22",
        "color": "#c9d1d9",
        "border": "1px solid #21262d",
        "padding": "8px 12px",
        "fontFamily": "Inter, sans-serif",
        "fontSize": "12px",
    }
    header = {
        "backgroundColor": "#0d1117",
        "color": "#8b949e",
        "fontWeight": "600",
        "border": "1px solid #21262d",
        "fontSize": "11px",
        "letterSpacing": "0.5px",
        "textTransform": "uppercase",
    }
    return dict(
        style_cell=cell,
        style_header=header,
        style_data_conditional=[
            {"if": {"row_index": "odd"},
             "backgroundColor": "#0d1117"},
            {"if": {"filter_query": "{action_a} = 'D'", "column_id": "action_a"},
             "color": "#f85149", "fontWeight": "700"},
            {"if": {"filter_query": "{action_b} = 'D'", "column_id": "action_b"},
             "color": "#f85149", "fontWeight": "700"},
            {"if": {"filter_query": "{action_a} = 'C'", "column_id": "action_a"},
             "color": "#3fb950"},
            {"if": {"filter_query": "{action_b} = 'C'", "column_id": "action_b"},
             "color": "#3fb950"},
            {"if": {"state": "selected"},
             "backgroundColor": "rgba(139,92,246,0.12)",
             "border": "1px solid #8b5cf6"},
        ],
        style_filter={"backgroundColor": "#161b22", "color": "#f0f6fc",
                      "border": "1px solid #30363d"},
        style_table={"overflowX": "auto", "borderRadius": "8px",
                     "border": "1px solid #21262d"},
    )


# ── Layout ─────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    df      = load_data()
    records = df[list({c["id"] for c in DISPLAY_COLS} & set(df.columns))].to_dict("records")

    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.Div([
                        html.H3("Raw Data Explorer", className="section-heading"),
                        html.P(
                            "Full round-by-round dataset from all recorded matchups. "
                            "Use the column headers to filter or sort. Click any row to "
                            "inspect the complete raw model output for that round.",
                            className="section-description",
                        ),
                    ], className="section-meta"),
                    html.Div(
                        [
                            html.Button(
                                "↓  Export filtered CSV",
                                id="exp-export-btn",
                                className="btn-export",
                                n_clicks=0,
                            ),
                            dcc.Download(id="exp-download"),
                        ],
                        className="export-row",
                    ),
                ],
                className="explorer-header",
            ),

            # DataTable
            dash_table.DataTable(
                id="exp-table",
                columns=DISPLAY_COLS,
                data=records,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_action="native",
                page_size=20,
                row_selectable="single",
                selected_rows=[],
                **_table_style(),
            ),

            # Raw output detail panel
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("RAW MODEL OUTPUT", className="detail-label"),
                            html.Div(
                                "Select a row above to inspect the unprocessed API response for that round.",
                                id="exp-detail-content",
                                className="detail-content",
                            ),
                        ],
                        className="detail-panel",
                    ),
                ],
                className="detail-wrapper",
            ),
        ],
        className="tab-content",
    )


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output("exp-detail-content", "children"),
    Input("exp-table",           "selected_rows"),
    State("exp-table",           "derived_virtual_data"),
    prevent_initial_call=True,
)
def show_detail(selected_rows: list, virtual_data: list) -> list:
    if not selected_rows or not virtual_data:
        return ["Select a row to view raw model output."]

    row   = virtual_data[selected_rows[0]]
    df    = load_data()
    match = df[
        (df["game_id"] == row.get("game_id")) &
        (df["round"]   == row.get("round"))
    ]
    if match.empty:
        return ["Row data unavailable."]

    r = match.iloc[0]

    def _raw_block(side: str, model: str) -> html.Div:
        raw_col = f"raw_output_{side}"
        raw     = str(r[raw_col]) if raw_col in r.index else "{}"
        try:
            parsed  = json.dumps(json.loads(raw), indent=2)
        except Exception:
            parsed  = raw
        action  = str(r[f"action_{side}"])
        return html.Div(
            [
                html.Div(
                    [
                        html.Span(model, className="detail-model-name"),
                        html.Span(
                            "✕ DEFECT" if action == "D" else "✓ COOPERATE",
                            className="action-badge " +
                                      ("action-defect" if action == "D" else "action-cooperate"),
                        ),
                    ],
                    className="detail-model-header",
                ),
                html.Pre(parsed, className="detail-raw"),
            ],
            className="detail-model-block",
        )

    return html.Div(
        [
            html.Div(
                f"Matchup: {r['matchup']}  ·  Round {int(r['round'])}",
                className="detail-round-label",
            ),
            html.Div(
                [
                    _raw_block("a", str(r["model_a"])),
                    _raw_block("b", str(r["model_b"])),
                ],
                className="detail-columns",
            ),
        ]
    )


@callback(
    Output("exp-download", "data"),
    Input("exp-export-btn", "n_clicks"),
    State("exp-table",      "derived_virtual_data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks: int, virtual_data: list):
    if not n_clicks or not virtual_data:
        raise __import__("dash").exceptions.PreventUpdate
    df_out = pd.DataFrame(virtual_data)
    return dcc.send_data_frame(df_out.to_csv, "ipd_filtered_export.csv", index=False)
