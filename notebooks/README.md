# Analysis Notebooks

**ESADE MIBA Capstone 2026 — Strategic Coherence in LLMs**

Reproducible analysis of all three game-theoretic experiments.  
Every number is computed from raw data — no hardcoded conclusions.  
All estimates include **95% confidence intervals**.

---

## Notebooks

| Notebook | Game | Key Parameters | Data source |
|---|---|---|---|
| [`prisoners_dilemma_analysis.ipynb`](prisoners_dilemma_analysis.ipynb) | Prisoner's Dilemma | ρ, β, TfT, BI | `all_raw/pd_results_*.csv` |
| [`commons_dilemma_analysis.ipynb`](commons_dilemma_analysis.ipynb) | Commons Dilemma | θ, β, collapse rate | `all_raw/cd_*.csv` |
| [`cheap_talk_analysis.ipynb`](cheap_talk_analysis.ipynb) | Cheap-Talk Signaling | η, Δη, γ, β | `all_raw/cheap_talk_results_*.csv` |

---

## What Each Notebook Covers

### Prisoner's Dilemma
- Framing effect: COOPERATE/DEFECT vs EXPAND/HOLD (CD vs HE)
- Cooperation rates by condition (`undisclosed` · `ai` · `human` · `human_prior`)
- Per-model cooperation rates
- Round-by-round dynamics with Early / Mid / Late phase breakdown
- Strategic metrics: ρ (conditional reciprocity), β (belief calibration), TfT adherence, Backward Induction index
- Matchup heatmap
- Kruskal-Wallis + Holm-Bonferroni pairwise tests

### Commons Dilemma
- Pool collapse rates by condition
- Exploitation intensity θ by condition and 2×2 design (info × risk)
- Per-model θ
- Round-by-round pool dynamics
- Belief calibration β
- Kruskal-Wallis tests for conditions, models, info, and risk

### Cheap-Talk Signaling
- Sender honesty η by game condition (aligned vs misaligned)
- Honesty drop Δη per sender model — tests H2 (incentive-contingent deception)
- Receiver gullibility γ under misalignment
- Identity condition effect on η
- Round-by-round honesty and receiver accuracy
- Belief calibration β for sender and receiver roles
- Mann-Whitney U (H2) + Kruskal-Wallis + Holm-Bonferroni

---

## How to Run

```bash
# From the project root
pip install jupyter pandas numpy matplotlib seaborn scipy statsmodels
jupyter notebook notebooks/
```

Notebooks read data automatically from `../all_raw/` — no path changes needed.

---

## Statistical Methods

| Method | Used for |
|---|---|
| Wilson score interval | All proportion CIs (cooperation rate, honesty rate, collapse rate) |
| t-interval | Continuous variable CIs (θ, β, TfT) |
| Bootstrap (n=1000) | ρ, Δη, γ — difference/correlation CIs |
| Kruskal-Wallis | Omnibus test across conditions or models |
| Mann-Whitney U | Pairwise tests, H2 honesty test |
| Holm-Bonferroni | Multiple comparison correction |

All tests operate at the **game level** (one observation per complete game) to avoid round-level pseudo-replication.
