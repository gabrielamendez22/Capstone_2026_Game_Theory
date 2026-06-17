# Report Structure — Working Outline

Capstone report: *Strategic Coherence in Large Language Models: Evidence from
Game-Theoretic Environments* (ESADE MiBA 2026).

Spine follows the supervisor (Carlos) structure; writing format modelled on Sreedhar
et al. (2025). Full rationale: `docs/superpowers/specs/2026-06-11-capstone-report-structure-design.md`.
Source-to-section mapping: `docs/report/lit-review-source-map.md`.

**Status legend:** ✅ drafted · ✍️ in progress · ⬜ not started

| § | Section | Contents | Status |
|---|---|---|---|
| — | **Front matter** | Title, authors, abstract (write last), keywords | ⬜ (abstract last) |
| 1 | **Introduction** | Deployment hook → fingerprints evidence → gap → our study (6 models, 3 games, S-vector) → H1–H3 confirmatory + exploratory → contributions → roadmap | ✅ |
| 2 | **Literature Review** | Two streams below | ✅ |
| 2.1 | — Behavioural games | LLMs as strategic players; fingerprints; machine psychology; prompt-driven vs genuine disposition; mechanistic (persona vectors) | ✅ |
| 2.2 | — Agents | Prosociality & human-behaviour prediction; human–AI interaction & opponent identity (mixed/conditional → motivates H3); communication/negotiation; deployment stakes; the gap | ✅ |
| 3 | **Methods** | Three subsections below | ✅ |
| 3.1 | — How we collected data | 6-model registry; conditions; human-prior calibration; replication; SQLite+CSV pipeline & schema | ✅ |
| 3.2 | — How we prompted | Prompt design; neutrality controls; JSON format; versioning; prompt-sensitivity controls | ✅ |
| 3.3 | — How we played the games | PD (ρ, β); Commons (θ = exploitation intensity, β); Cheap-Talk (Δη, γ_mis, β); profile-similarity matrix; structure-vs-identity manipulation; β analyzed per-environment (not pooled) | ✅ |
| 4 | **Results** | Plot-heavy | ⬜ |
| 4.1 | — Prisoner's Dilemma | Cooperation curves, ρ, per-model fingerprints | ⬜ |
| 4.2 | — Commons Dilemma | θ exploitation intensity, extraction patterns | ⬜ |
| 4.3 | — Cheap-Talk | Δη honesty drop under misalignment, γ_mis, β | ⬜ |
| 4.4 | — Synthesis | **Confirmatory:** H1 differentiation (KW + logistic mixed models), H2 incentive deception (Δη), H3 structure > identity. **Exploratory:** β coherence, human comparison (descriptive), cross-role awareness | ⬜ |
| 4.5 | — Robustness checks | Battery of 10; detail in appendix | ⬜ |
| 5 | **What We Learn & Recommendations for Deploying Agentic AI** | Interpretation; model "personalities"; managerial recommendations; limitations & future work | ⬜ |
| — | **References** | Auto-generated (Pandoc + apa.csl from `references.bib`) | ✅ (auto) |
| — | **Appendices** | Full prompts; model versions; per-condition tables; 10 robustness-check details; figures; reproducibility | ⬜ |

## Hypotheses (post-June-2026 reframe, per Carlos)

The original H1–H4 (scaling / tuning / temperature / opponent-sensitivity) are retired:
scaling and opponent-sensitivity are not supported, tuning is untestable (no base
models), and temperature is untestable (single temperature in the consolidated dataset).
Current set:

**Confirmatory:**
- **H1 — Differentiation.** Models have statistically distinct strategic profiles (ρ, θ, Δη, γ). *Test:* KW + logistic mixed models. Robustness: HIGH.
- **H2 — Incentive-contingent honesty.** Misalignment lowers sender honesty (Δη > 0), model-specific. *Test:* logistic mixed model `truthful ~ misaligned × model + (1|run)`. Robustness: HIGH.
- **H3 — Structure over identity** (revised H5). Game structure (incentive alignment, information, risk, action framing) drives behavior more than disclosed opponent identity. *Test:* condition + structural features as fixed effects (round level). Robustness: MODERATE–HIGH.

**Exploratory (explicitly labelled, validate on held-out runs / extra models):**
- β coherence across environments (β_PD↔β_CT vs β_CD); n = 6 → exploratory.
- Human comparison — **descriptive only** (no participant-level data → no inferential test).
- Cross-role strategic awareness (Δη vs γ_mis), r = −0.61, p ≈ 0.20.

Guardrail: keep confirmatory vs post-hoc clearly separated; report all tested alternatives.

## Build
- Source: `report/metadata.yaml` (title block + settings) + `report/sections/*.md`
  (one file per section, concatenated in filename order). **Split per section to avoid
  merge conflicts — see root `CLAUDE.md` §9.** No YAML front matter in section files.
- Citations: citekeys → `references.bib` → APA via `apa.csl`. Append new bib entries only.
- Compile: `./report/build.sh` → `report/report.pdf` (needs a LaTeX engine; see script).

## Data status caveat
Per-model parameters and significance tests are **computed** (strategic-profile analysis
v2, June 2026 — `build_strategic_profiles_v2.py`; summarized in
`docs/report/strategic_profiles_report_v2.html`). §4 can be drafted from those results.
Carlos's *main* analysis (logistic mixed models for ρ and γ; structural-feature models
for H3) is still to be run — treat the current KW / Mann-Whitney numbers as preliminary,
and do not fabricate beyond the v2 output.
