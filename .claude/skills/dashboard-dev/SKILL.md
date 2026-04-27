---
name: dashboard-dev
description: Use when modifying, extending, or debugging the Dash dashboard in /dashboard/. Covers the tab/panel architecture, the dark-theme styling system, the lru_cache data loading pattern, how to add a new chart without breaking existing ones, and the Plotly layout helpers.
---

# Dashboard Development Skill

## Architecture

```
dashboard/
├── app.py                  ← entry point, tab routing
├── data_loader.py          ← shared data layer with lru_cache
├── assets/style.css        ← dark theme; Dash auto-loads
└── tabs/
    ├── overview.py         ← KPIs + cooperation rate + score
    ├── simulation.py       ← live replay of one matchup
    ├── metrics.py          ← β, BI, TfT, defection threshold, etc.
    ├── fingerprints.py     ← radar chart + cooperation matrix
    └── explorer.py         ← raw DataTable with row inspector
```

Each tab module exports a `layout()` function returning a `html.Div`.
`app.py` assembles them into hidden panels and shows one based on nav clicks.

## Adding a New Tab

1. Create `dashboard/tabs/new_tab.py` with a `layout()` function
2. In `app.py`, add to `TABS`:
   ```python
   TABS = [..., ("new_tab", "New Tab Name")]
   ```
3. Import at the top of `app.py`:
   ```python
   import tabs.new_tab as new_tab
   ```
4. Add to `app.layout` Main:
   ```python
   html.Div(new_tab.layout(), id="panel-new_tab",
            className="tab-panel", style={"display": "none"}),
   ```

The nav callbacks auto-handle the routing — no other changes needed.

## Adding a New Chart to an Existing Tab

Pattern used everywhere in `metrics.py` and `overview.py`:

```python
def _section(title: str, description: str, children) -> html.Div:
    return html.Div(
        [html.Div([
            html.H3(title, className="section-heading"),
            html.P(description, className="section-description"),
         ], className="section-meta"),
         children],
        className="chart-section",
    )

def _my_new_figure() -> go.Figure:
    stats = compute_model_stats()
    fig = go.Figure(go.Bar(...))
    fig.update_layout(**plotly_dark_layout(...))
    return fig

# Then inside layout():
_section(
    "Chart Title",
    "One-paragraph description of what this measures and how to read it.",
    dcc.Graph(figure=_my_new_figure(),
              config={"displayModeBar": False},
              className="chart"),
)
```

For two-column rows, wrap two `_section(...)` calls in a `charts-row` div.

## The Data Loader is Cached

`data_loader.py` uses `@functools.lru_cache(maxsize=1)` on the heavy functions:
- `load_data()`
- `compute_long_df()`
- `compute_model_stats()`
- `compute_cooperation_matrix()`

This means the CSV is read once per server startup. **If you change the CSV,
restart the server.** Don't try to invalidate the cache — just restart.

When pointing the dashboard at a new CSV, edit `_find_csv()` in
`data_loader.py` to add the new filename.

## Color System

| Family | Color | Hex |
|---|---|---|
| Anthropic (Claude) | Purple | `#8b5cf6` |
| OpenAI (GPT) | Green | `#10b981` |
| Google (Gemini) | Blue | `#60a5fa` |
| Defect highlight | Red | `#ef4444` |
| Cooperate highlight | Green | `#3fb950` |

Always use `model_color(name)` from `data_loader.py` — never hardcode.

For semi-transparent fills (radar polygons, hover regions), use
`hex_to_rgba(hex_color, alpha=0.13)`.

## Plotly Layout

Always use `plotly_dark_layout(**overrides)` from `data_loader.py`. It enforces:
- Background colors matching the dark theme
- Inter font, correct sizes
- Grid color `#21262d`
- Hover styling

```python
fig.update_layout(
    **plotly_dark_layout(
        xaxis=dict(title="Round", range=[0, 20], gridcolor="#21262d"),
        yaxis=dict(title="Cooperation rate", tickformat=".0%"),
        height=300,
    )
)
```

## CSS Conventions

The dark theme lives entirely in `assets/style.css`. Key utility classes:

- `.kpi-card` — hover-able stat card
- `.chart-section` — wraps a chart with title and description
- `.chart-section-full` — full-width version
- `.charts-row` — two-column grid of `.chart-col` items
- `.action-cooperate` / `.action-defect` — the green/red action pills
- `.metric-badge` — italic purple parameter badge (β, ρ, etc.)
- `.filter-dropdown` — dark-themed Dropdown override

When adding new UI, prefer reusing classes over adding new CSS.

## Performance Tips

- Charts render once on page load and on filter callbacks
- Heavy aggregations belong in `data_loader.py` with `lru_cache`, not in tab files
- `dash_table.DataTable` with `page_size=20` handles thousands of rows fine
- Don't put expensive operations inside `layout()` — they run on every page load
- Auto-scroll uses a clientside callback (already implemented in `simulation.py`)

## Anti-Patterns (do NOT do)

- Don't fetch data inside `layout()` — use cached functions from `data_loader.py`
- Don't hardcode colors — use `model_color()` and the constants in `data_loader.py`
- Don't create CSS files outside `assets/` — Dash won't auto-serve them
- Don't bypass `plotly_dark_layout()` — your chart will look out of place
- Don't try to invalidate `lru_cache` programmatically — restart the server
