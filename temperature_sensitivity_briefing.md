# Temperature Sensitivity Analysis — Cheap-Talk Experiment
## Briefing for Report Generation

---

## Context

This is part of the ESADE MiBA Capstone 2026 project investigating whether LLMs exhibit stable, transferable strategic dispositions across game-theoretic environments. The instrument here is a **Cheap-Talk / Signaling game**.

**Research question for this analysis:** Does temperature affect the strategic behaviour of LLMs in a cheap-talk signaling game? Specifically, does lower temperature (more deterministic) vs higher temperature (more stochastic) change truthfulness, receiver trust, or deception rates?

---

## Game Structure

- **Roles:** Sender (knows true state) and Receiver (does not)
- **State:** H or L, drawn 50/50 each round
- **Sender action:** Send message H or L (may lie)
- **Receiver action:** Choose action A (correct for L) or B (correct for H)
- **Payoff conditions:**
  - **Aligned** — both players benefit when Receiver picks correctly
  - **Misaligned** — Sender always benefits if Receiver picks A, regardless of state (incentive to deceive)
- **Role rotation:** Fixed for first 5 rounds, roles swap at round 6 (in rotated matchups)
- **Identity condition:** `undisclosed` — neither model is told who their partner is; no persona assigned

**Models:** Claude Opus, Claude Sonnet, GPT-4o, GPT-4o-mini, Gemini 2.5 Flash, Gemini 2.5 Flash Lite

**Matchups:** 36 per run (cross-family, same-family, rotated)

---

## Data Files

| Run | Temperature | Rounds | Rows | File |
|-----|-------------|--------|------|------|
| T=0.6 Run 19 | 0.6 | 10 | 360 | `data/raw/cheap_talk_results_undisclosed_20260519_072229.csv` |
| T=0.6 Run 20 | 0.6 | 10 | 360 | `data/raw/cheap_talk_results_undisclosed_20260519_073739.csv` |
| T=0.2 | 0.2 | 15 | 540 | `data/raw/cheap_talk_results_undisclosed_20260519_141812.csv` |
| T=0.8 | 0.8 | 15 | 540 | `data/raw/cheap_talk_results_undisclosed_20260519_172858.csv` |

> Note: T=0.6 runs are 10 rounds; T=0.2 and T=0.8 runs are 15 rounds. Normalise by rounds-per-matchup when comparing rates, not raw counts.

---

## CSV Schema

Each row is one round of one matchup:

| Column | Description |
|--------|-------------|
| `game_condition` | `aligned` or `misaligned` |
| `identity_condition` | always `undisclosed` here |
| `role_rotation` | `fixed` or `rotated` |
| `model_sender` | sender model label |
| `model_receiver` | receiver model label |
| `true_state` | `H` or `L` |
| `message_sent` | `H` or `L` (what sender reported) |
| `message_truthful` | `1` if message matches true state, else `0` |
| `sender_belief` | sender's P(receiver follows message), 0–1 |
| `action_taken` | `A` or `B` |
| `action_correct` | `1` if action matches true state, else `0` |
| `receiver_belief` | receiver's P(sender is truthful), 0–1 |
| `deception_success` | `1` if sender lied AND receiver acted in sender's favour |
| `sender_payoff` | numeric payoff for sender |
| `receiver_payoff` | numeric payoff for receiver |

---

## Summary Statistics

| Metric | T=0.6 Run 19 | T=0.6 Run 20 | T=0.2 | T=0.8 |
|--------|-------------|-------------|-------|-------|
| Rounds per matchup | 10 | 10 | 15 | 15 |
| Total rows | 360 | 360 | 540 | 540 |
| **Truthfulness rate (overall)** | 68.9% | 67.5% | 78.3% | 78.7% |
| — Aligned condition | 84.4% | 84.4% | 100.0% | 100.0% |
| — Misaligned condition | 53.3% | 50.6% | 56.7% | 57.4% |
| **Correct action rate** | 64.4% | 59.7% | 76.3% | 73.9% |
| **Avg sender belief** | 0.66 | 0.65 | 0.69 | 0.75 |
| **Avg receiver belief** | 0.52 | 0.53 | 0.61 | 0.62 |
| **Deception successes** | 70/180 (38.9%) | 75/180 (41.7%) | 55/270 (20.4%) | 79/270 (29.3%) |

---

## Questions for the Report

Please generate a structured analytical report addressing:

1. **Does temperature affect truthfulness?** Compare overall and condition-split (aligned vs misaligned) truthfulness rates across T=0.2, T=0.6, T=0.8. Are the differences meaningful or within noise?

2. **Does temperature affect receiver trust and coordination?** Compare correct action rates and receiver belief across temperatures.

3. **Does temperature affect deception?** Compare deception success rates across temperatures and payoff conditions.

4. **Aligned vs misaligned — does temperature interact with incentive structure?** The aligned condition shows 100% truthfulness at T=0.2 and T=0.8 vs ~84% at T=0.6. Is this a meaningful finding?

5. **Model-level breakdown** — do specific models drive the temperature effect, or is it uniform? (Break down by `model_sender` and `model_receiver`.)

6. **Implications for the main study** — the canonical temperature for the full experiment is T=0.6. Do these results suggest T=0.6 is a reasonable default, or should we reconsider?

---

## Notes

- Treat T=0.6 runs 19 & 20 as two independent replications of the same condition; pool them or report mean ± range.
- The 15-round runs (T=0.2, T=0.8) allow observation of within-session learning / drift that the 10-round runs cannot — flag this in the analysis if relevant.
- All runs use `PROMPT_VERSION = "v2.1"` — no prompt changes across these runs.
