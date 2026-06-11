# Capstone Report Structure — Design Spec

**Project:** Strategic Coherence in LLMs (ESADE MiBA Capstone 2026)
**Branch:** `feature/report`
**Date:** 2026-06-11
**Authors:** Giorgio Fiorentino, Samreen Siddique, Gabriela Mendez
**Status:** Approved structure — section drafting to follow

---

## Purpose

This document fixes the **structure** of the capstone report so it can be drafted
section by section without re-litigating organization. It is the single source of
truth for what goes where. Content is grounded in the actual repository (scripts,
data, `METHODOLOGY.md`, `RESEARCH_JOURNAL.md`) and in the source papers in
`fonti per capstone/` — not invented.

## Format decision

- **Deliverable type:** ESADE MiBA capstone report (academic backbone, no mandated
  ESADE template — designed from academic conventions + capstone needs).
- **Spine authority:** the supervisor (Carlos) structure, which is leaner and more
  results-forward than a standard empirical paper. Specifically: no standalone
  Theoretical Framework section, no formal Conclusion — the framework folds into
  Methods, and the paper closes on managerial recommendations. Robustness checks
  are a first-class results component.

## Structural templates (from `fonti per capstone/`)

| Paper | File | What we borrow |
|---|---|---|
| Payne & Allou-Cros (2025), *Strategic Intelligence in LLMs: Evidence from Evolutionary Game Theory* | `2507.02618v1.pdf` | Explicit threefold contribution; "strategic fingerprints" framing |
| Akata et al. (2025), *Playing Repeated Games with LLMs* | `2305.16867v2.pdf` | Game-by-game results organization; behavioral game theory framing; prompt-as-game-rules figure |
| Payne (2026), *AI Arms and Influence* | `2602.14740v1.pdf` | Machine-psychology framing; per-model "personality"; related-work "shared limitation" move |

Content/related-work sources (not structural templates): Ferraz et al. (2025),
the prosocial multi-agent simulation, inequity-aversion paper, human–AI
meta-analysis, persuasion paper, trust/agency paper.

---

## Approved Structure

### Front matter
- Title, authors, abstract, keywords.
- **Executive Summary** (½–1 page, capstone element): the question, what we did,
  headline finding (CECS), why it matters. Non-technical. *(Optional — flag if cut.)*

### 1. Introduction
- The question: strategic **coherence** vs. context-fragmentation in LLMs.
- Why it matters: LLMs increasingly deployed as agentic decision-makers.
- Contribution, stated threefold (à la Payne): (i) a cross-environment coherence
  framework, (ii) **CECS** as a novel metric, (iii) evidence across three
  structurally distinct games + opponent-identity manipulation.
- Research questions / hypotheses **H1–H4** stated here; formal operationalization
  deferred to Methods.

### 2. Literature Review
- **2.1 Behavioural games** — behavioral game theory + LLMs in strategic games:
  Akata et al., Payne & Allou-Cros (strategic fingerprints), Payne (nuclear crisis);
  classic PD / Commons / cheap-talk-signaling roots.
- **2.2 Agents** — agentic / multi-agent LLMs; prosociality & cooperation (Ferraz,
  prosocial multi-agent simulation, inequity aversion); human–AI strategic
  interaction (meta-analysis, persuasion, trust/agency).
- Closes on **the gap**: prior work tests one game or one capability; nobody
  measures *cross-environment consistency* for the same model.

### 3. Methods
- **3.1 How we collected data** — 6-model registry; experimental conditions
  (PD/CD `OPPONENT_CONDITION`: undisclosed/ai/human; Cheap-Talk's 6
  `IDENTITY_CONDITION`s); human-prior calibration condition + literature benchmarks;
  replication; SQLite (crash-safe) + CSV pipeline and schema.
- **3.2 How we prompted** — prompt design, neutrality controls (neutral objective,
  both C/D examples, hidden round total, explicit belief timing, identical prompt
  across models), JSON-only response format, prompt versioning (PD at v4.4).
  **Prompt-sensitivity controls live here** (primary validity defense).
- **3.3 How we played the games** — the three games' mechanics and **parameter
  definitions**:
  - PD (ρ conditional reciprocity, β belief calibration), payoffs T5>R3>P1>S0
  - Commons (θ exploitation threshold, β)
  - Cheap-Talk (η signal honesty, γ receiver gullibility, **β**)
  - **CECS** and **Δm** formal definitions land here.
  - **Open interpretation flag — β in Cheap-Talk:** belief was logged in this game
    too. Under sender/receiver asymmetry, "belief calibration" needs a defined
    referent (belief about partner's *intended* action vs. *realized* action). To be
    settled during analysis and documented here.

### 4. Results *(plot-heavy — Carlos: "lots of plots")*
- **4.1 PD** — cooperation curves, ρ, per-model strategic fingerprints.
- **4.2 Commons** — θ, exploitation patterns. *(Flag: partial pilot.)*
- **4.3 Cheap-Talk** — η honesty, γ gullibility, β, deception over rounds.
- **4.4 Synthesis (headline)** — **CECS** per model, **Δm**, H1–H4 verdicts.
- **4.5 Robustness checks** — battery of **10 checks** summarized in-text; full
  tables in Appendix. Candidate checks: temperature sweep (T=0.3/0.6/0.8),
  prompt-version comparison (v4.0 vs v4.1+), replication stability, parse-failure
  audit, round-1 ceiling-effect check, belief-action alignment vs. persistent
  defectors, cross-condition opponent-identity comparison, human-prior perturbation
  effect, model-family consistency, history-window sensitivity. *(Final list of 10
  to be locked during analysis.)*

### 5. What We Learn & Recommendations for Deploying Agentic AI *(replaces Discussion + Conclusion)*
- Interpretation of the coherence pattern; model "personalities" (ties to Payne's
  fingerprints).
- **Recommendations for companies** deploying agentic AI: model selection by
  strategic disposition, opponent-framing risks, safety implications.
- Limitations & future work, folded in (single-rep pilots, residual prompt
  sensitivity, Commons incompleteness, model-version drift).

### Appendices
- Full prompts (all versions), model versions, per-condition tables, **detailed
  10 robustness-check results**, all figures, reproducibility notes.

---

## Process notes

- **Cross-model sanity-checking (Carlos's instruction):** analysis steps and
  interpretations are to be double-checked with other AI models as a validation
  practice; record this as a stated method/robustness practice so it is defensible.
- **Grounding rule:** every factual claim about apparatus, data, or results traces
  to a repo artifact (script, CSV/DB, `METHODOLOGY.md`, `RESEARCH_JOURNAL.md`) or a
  cited source paper. Pull cross-branch artifacts via `git show <branch>:<path>`.
- **Data status caveat:** empirical results (CECS, per-model parameters, H1–H4
  verdicts) are not yet computed at spec time — Results sections are scaffolded and
  filled as the analysis pass completes.

## Deviations from the earlier joint draft

1. Dropped the standalone **Theoretical Framework** section — CECS/Δm now defined in
   Methods §3.3 ("how we played the games").
2. Merged **Discussion + Conclusion** into §5 (recommendations close).
3. Promoted **Robustness checks** to a first-class Results subsection (§4.5).
4. Added explicit handling of **β in Cheap-Talk** as an open interpretation item.
