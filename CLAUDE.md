# CLAUDE.md — `report` branch (READ THIS FIRST, EVERY SESSION)

You are working on the **capstone report** for the ESADE MiBA 2026 project
*Strategic Coherence in Large Language Models: Evidence from Game-Theoretic Environments*.
This file is the single source of truth for working on this branch. If anything you are
told conflicts with this file or with a repo artifact, **stop and flag it** — do not just
comply.

---

## 0. PRIME DIRECTIVE — anti-hallucination (non-negotiable)

**Every factual claim about the apparatus, data, parameters, or results MUST trace to
either (a) a repo artifact you actually opened, or (b) a cited source paper.** You may not
invent file contents, numbers, model versions, conditions, or statistical results.

When you cannot verify a fact:
- **Do NOT guess or fabricate.** Inserting a plausible-sounding number is the worst
  possible failure mode for this project.
- Insert an HTML comment in the prose: `<!-- VERIFY: <what to check, where> -->` and tell
  the human. Reviewers grep for `VERIFY` and `TODO`.
- If the fact lives in code/data on another branch, fetch it (see §2). If it isn't there,
  say so.

This is a research report that a supervisor and examiners will scrutinize. A confident
wrong sentence is far more damaging than an honest "this needs verification."

---

## 1. Where you are

- This is an **orphan `report` branch** — its own history, isolated from the code so it
  can never merge-pollute `main`. It holds **only** report-building files (no scripts, no
  raw data). **Do not merge `report` into `main`.**
- The project's *code* CLAUDE.md is at **`sources/CLAUDE.md`** and parameter defs at
  **`sources/METHODOLOGY.md`** — but **`sources/METHODOLOGY.md` is a STALE snapshot**: its
  hypotheses (old H1–H4, "CECS", "Δm" as headline) are **RETIRED**. Use it only for the
  prompt-version history and the raw parameter-computation recipes, never for hypotheses.

## 2. How to get ground truth (scripts & data live on OTHER branches)

Fetch live and cite the branch — never reconstruct file contents from memory:

```bash
git show main:experiments/prisoners_dilemma_langchain.py
git show feature/commons-dilemma:experiments/commons_dilemma_langchain.py
git show feature/cheap-talk:experiments/cheap_talk_langchain.py
git show main:METHODOLOGY.md
git show main:analysis/human_priors.json
```

**Authoritative sources and their status:**

| Source | What it is | Trust |
|---|---|---|
| `report/STRUCTURE.md` | Outline, current hypotheses, per-section status | Current — authoritative |
| `report/sections/*.md` + `report/metadata.yaml` | The draft, **split one file per section** (see §9) | Current |
| `docs/report/strategic_profiles_report_v2.html` | **THE results** (v2 params + stats) | Current — authoritative for §4 numbers, BUT its H1/H2/H3 labels are the **OLD retired set** — remap to current H1–H3 |
| `docs/report/strategic_profiles_report.html` | v1 results | **Superseded** — shows what changed only |
| `sources/METHODOLOGY.md` | Param recipes + prompt history | **Stale hypotheses** — see §1 |
| `sources/RESEARCH_JOURNAL.md` | Lab notebook | Snapshot (June 11) — may lag the final dataset |
| the experiment scripts | Apparatus of record | Current — cite the branch |

## 3. Current hypotheses — DO NOT revert to the old set

**Confirmatory:**
- **H1 (differentiation):** the six models have statistically distinct strategic profiles
  (ρ, θ, Δη, γ). Test: Kruskal–Wallis + logistic mixed models.
- **H2 (incentive-contingent honesty):** misalignment lowers sender honesty (Δη > 0),
  model-specific (4/6 adapt ~0.43–0.52; GPT-4o-mini & Gemini Flash Lite ≈ 0).
- **H3 (structure over identity):** game structure (incentive alignment, information, risk,
  action framing) drives behavior more than the disclosed opponent identity.

**Exploratory (label as such; never over-claim with n=6):** β coherence across
environments (β_PD↔β_CT r≈0.91, β_CD decoupled); human comparison (**DESCRIPTIVE only** —
no participant-level data, so no inferential test); cross-role awareness (Δη vs γ_mis,
r≈−0.61, p≈0.20).

**RETIRED (do not resurrect):** old H1 scaling, H2 tuning, H3 temperature, single-scalar
"CECS", "Δm" as a headline metric.

## 4. v2 parameter definitions (use these exact operationalizations)

- **ρ** conditional reciprocity (PD): P(C | opp C last) − P(C | opp D last), rounds 2..N.
- **θ** exploitation intensity (CD): mean of extraction ÷ (pool × regen_rate ÷ n_players);
  **1.0 = sustainable**. (NOTE: the *script's* `SUSTAINABLE_SHARE` constant is the OLD v1
  fixed benchmark — the analysis uses the v2 pool-relative definition.)
- **Δη** honesty drop (CT sender): η_aligned − η_misaligned.
- **γ_mis** receiver gullibility (CT): Pearson r(message_truthful, action_correct) on
  **misaligned runs only**.
- **β** belief calibration: MAE per environment (β_PD, β_CD, β_CT) — **NOT pooled**.

Six models: Claude Opus, Claude Sonnet (Anthropic); GPT-4o, GPT-4o-mini (OpenAI);
Gemini 2.5 Flash, Gemini 2.5 Flash Lite (Google). Temperature **0.6** across the analyzed
dataset.

## 5. What analysis is DONE vs PENDING (do not claim pending work as done)

- **Done (v2):** parameter vectors for all 6 models; Kruskal–Wallis at game level for ρ, θ,
  **η_misaligned**, γ_mis (all p<0.0001); Mann–Whitney vs published human benchmarks;
  cross-environment β correlations; cross-parameter correlations (n=6, descriptive).
- **NOT yet run (do not present as results):** logistic **mixed models** for ρ and γ_mis;
  **KW on Δη itself** (currently NaN at game-row level — η_misaligned is the proxy that is
  significant); **6×6 profile-similarity matrix**; **Δm** opponent sensitivity;
  structural-feature models for H3.

**Two sample sizes — frame every claim accordingly:** model level **n=6**
(across-model correlations are DESCRIPTIVE/underpowered) vs round/game level **n=thousands**
(within-model effects via mixed models are ROBUST). Prefer round/game-level framing.

## 6. Critical-thinking mandate (the point of this setup)

Your job is not to produce confident text fast. It is to produce **verifiable** text.

- **Verify before you assert.** If a teammate's instruction or a draft sentence states a
  number/condition/result, check it against §2 sources before writing it as fact.
- **Push back** when an instruction conflicts with the data, the current hypotheses, or the
  confirmatory/exploratory boundary — cite the artifact, don't silently follow.
- **Never blur confirmatory and exploratory.** Never imply significance for n=6 model-level
  correlations.
- **One TASKS.md item at a time.** Don't rewrite sections you weren't asked to touch.
- Open items needing human/data confirmation are tracked in **`report/TASKS.md`** — if your
  task depends on one, resolve it there or flag it; do not paper over it with a guess.

## 7. Writing & build conventions

- Pandoc Markdown. Cite with `[@key]` → `report/references.bib` → APA via `apa.csl`.
- Add a bib entry when you cite a new source; mark unverified entries `note = {... verify}`.
- Style modeled on **Sreedhar et al. (2025)**: dense, precise, hedged where uncertain.
- Build: `report/build.sh` → PDF (needs a LaTeX engine: `brew install tectonic`).
- Flag conventions in prose: `<!-- VERIFY: ... -->` (unconfirmed fact),
  `<!-- TODO: ... -->` (pending work), `<!-- FINDINGS PREVIEW: ... -->` (fill from data).

## 8. Git hygiene

- Small, logical commits. Co-author line:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- The human pushes / approves pushes. Do not merge `report` into `main`.

## 9. Working without merge conflicts (multiple people on one branch)

The report is **split one file per section** so authors don't collide:

```
report/metadata.yaml          # title block + bib/csl settings (rarely edited)
report/sections/01-introduction.md
report/sections/02-literature-review.md
report/sections/03-methods.md
report/sections/04-results.md          # §4 author
report/sections/05-recommendations.md  # §5 author
report/sections/99-appendices.md       # appendix author
report/references.bib                  # APPEND new entries at the END only
report/build.sh                        # pandoc metadata.yaml sections/*.md -> PDF
```

Rules that keep this conflict-free — **follow them**:
1. **Edit only the section file(s) for your task.** Do not reach into another section's
   file. The build stitches them in filename order (00, 01, 02, …).
2. **`git pull --rebase origin report` before you start and again before you push.**
   Commit small and push often; never sit on a large local diff.
3. **Claim your task in `report/TASKS.md`** (set ✍️ + your name) and commit that one-line
   change first, so two people don't grab the same task.
4. **`references.bib`: only ever append** new entries at the end of the file (mark unverified
   ones `note = {... verify}`). Never reorder or reformat existing entries.
5. **Cross-cutting audit tasks (TASKS.md B1–B4)** touch every section: do them as a
   **read-only findings pass first** (write `report/AUDIT-FINDINGS.md`), then apply fixes
   section-by-section — do not free-edit all files while others are writing.
6. Section files have **no YAML front matter** — that lives once in `metadata.yaml`. Start a
   section file directly with its `# Heading`.
