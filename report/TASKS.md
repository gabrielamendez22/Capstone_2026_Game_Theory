# Report Tasks — work board

Read the root `CLAUDE.md` (esp. §0 anti-hallucination and §5 analysis status) **before**
starting any task. Take **one task at a time**. Each task has a **SOURCE OF TRUTH**, a
**DONE WHEN** gate, and a **DO NOT** guardrail. If you cannot meet the gate from a verified
source, mark the task blocked and leave a `<!-- VERIFY: ... -->` flag — never fabricate.

Status: ⬜ open · ✍️ in progress · 🔒 blocked (needs human/data) · ✅ done

---

## Group A — Data-scope verification (BLOCKS finalizing §3; needs other branches/raw data)

These cannot be resolved from the `report` branch by reasoning. The analyzed dataset is
not here. Resolve by `git show`-ing the scripts on their branches **and** confirming with
the person who ran the experiment which conditions actually produced rows in the analyzed
CSV. AI working alone here **must** flag, not assume.

### A1 ⬜ Confirm the Commons info×risk design that was actually run
- **Why:** §3.3 currently describes a 2×2 (info: full/partial × risk: deterministic/
  probabilistic). The v2 results report shows a **single pooled θ** with no cell breakdown.
- **SOURCE OF TRUTH:** the person who ran CD + the raw CD CSVs (`condition` / config
  columns); `git show feature/commons-dilemma:experiments/commons_dilemma_langchain.py`.
- **DONE WHEN:** you can state exactly which of the 4 cells have data in the analyzed set.
- **DO NOT:** claim a 2×2 was executed if only the baseline cell was run. If baseline only,
  rewrite §3.3 to "single condition; factorial = planned/future work" (see the inline
  `VERIFY` comment in `sections/03-methods.md`).

### A2 ⬜ Confirm the Cheap-Talk identity conditions actually in the analyzed dataset
- **Why:** the CT script defines **six** per-role identity conditions; the v2 report's
  condition labels are the 4-value set `{undisclosed, ai, human, human_prior}`, and the
  June-11 journal only confirms `undisclosed` + aligned/misaligned pilots.
- **SOURCE OF TRUTH:** CT runner + raw CT CSVs (`identity_condition` column);
  `git show feature/cheap-talk:experiments/cheap_talk_langchain.py`.
- **DONE WHEN:** the list of CT identity conditions present in the analyzed data is fixed.
- **DO NOT:** assert "six conditions" in the report until confirmed. Resolve the `VERIFY`
  comment in `sections/03-methods.md`.

### A3 ⬜ Confirm CT round count, CD temperature, and replication count
- **CT rounds:** script says 15; journal pilots were 10. Which is in the analyzed data?
- **CD temperature bug:** journal flags OpenAI models hard-coded at T=1.0 while others were
  0.6. Confirm it was fixed before the analyzed CD runs (else GPT CD data is off-temperature).
- **Replications:** scripts default `NUM_REPLICATIONS=1`. Confirm the value used and the
  resulting **game-level n per model** (KW H-values imply many rows — verify).
- **SOURCE OF TRUTH:** raw CSVs + run log + experiment scripts.
- **DONE WHEN:** all three pinned with evidence. **DO NOT:** fill Appendix A from script
  defaults without confirming they match the analyzed dataset.

### A4 ⬜ Pin model identifiers + API settings for Appendix A
- **SOURCE OF TRUTH:** `build_model_registry()` in each script (via `git show`).
- **As configured:** `claude-opus-4-6`, `claude-sonnet-4-6`, `gpt-4o`, `gpt-4o-mini`,
  `gemini-2.5-flash`, `gemini-2.5-flash-lite`; max_tokens vary by game.
- **DONE WHEN:** Appendix A table matches the scripts that produced the analyzed data.

---

## Group B — Independent quality / anti-hallucination audit (can start now)

### B1 ⬜ Claim-tracing pass over §1–§3
- For **every factual sentence**, attach a source: repo artifact (name + branch), v2 report
  section, or `[@cite]`. Anything untraceable → `<!-- VERIFY -->` flag + list it.
- **DONE WHEN:** every apparatus/data/result claim in §1–§3 has a traced source or a flag.
- **DO NOT:** "verify" a claim by re-reading the sentence — trace it to an external source.

### B2 ⬜ Citation audit
- Check each `[@key]` actually supports its sentence. Verify the 4 newly added bib entries
  (Gneezy 2005; Dvorak & Fehrler 2024; Abatayo & Lynham 2022; Pawlick et al. 2018) —
  titles/venues/DOIs — they are marked "verify before final" in `references.bib`.
- **DONE WHEN:** no `verify` notes remain unaddressed; every in-text cite is supported.

### B3 ⬜ Number-consistency sweep
- Reconcile every number in the draft against the v2 report and scripts (payoffs 5/3/1/0,
  pool 100, regen 20%, rounds, and all ρ/θ/Δη/γ/β values).
- **Watch the human ρ anchor:** v2 uses **0.405** (≈ Dvorak T1 no-comm); the journal also
  cites T13+T14 = 0.610 and the human-prior *prompt* implies ρ≈0.49. The report must pick
  one anchor and use it consistently, stated explicitly.
- **DONE WHEN:** one documented value per quantity, matching the SOURCE OF TRUTH.

### B4 ⬜ Hypothesis-label & confirmatory/exploratory consistency
- Confirm zero leakage of retired labels (CECS / scaling / tuning / temperature / Δm-as-
  headline). Confirm confirmatory vs exploratory framing is preserved everywhere, and the
  v2 report's old H1/H2/H3 labels are remapped to the current set wherever results are used.
- **DONE WHEN:** the audit finds no retired-framing leakage and no over-claim at n=6.

---

## Group C — Drafting (after B is underway; §4 needs Group A resolved for §3 final)

### C1 ⬜ Appendices A & B
- A: model table, matchup lists, CSV schema (all from scripts via `git show`).
- B: full prompts per game/condition + prompt version history (`sources/METHODOLOGY.md`
  history table + script prompts).
- **DONE WHEN:** §3's forward-references to Appendix A/B resolve to real content.

### C2 ⬜ §4 Results
- Draft from `docs/report/strategic_profiles_report_v2.html` (authoritative numbers).
- **DO NOT:** present logistic mixed models, a Δη KW test, the profile-similarity matrix,
  or Δm as completed — they are **pending** (root `CLAUDE.md` §5). Use η_misaligned KW as
  the significant test backing H2; remap the v2 report's old hypothesis labels.

### C3 ⬜ §5 Recommendations, then Abstract (write last)
- Per `report/STRUCTURE.md`. Abstract only after §4/§5 are final.
