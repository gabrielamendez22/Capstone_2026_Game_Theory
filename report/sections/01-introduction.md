# Introduction

Large language models (LLMs) are moving from passive text generators to autonomous
agents that negotiate, allocate resources, and make consequential decisions on behalf
of people and organizations [@sun2026; @vaccaro2026]. They now mediate two-player
economic interactions [@dvorak2025], conduct large-scale agent-to-agent negotiations
[@vaccaro2026], and operate in multi-agent systems whose collective behavior is
proposed as a tool for informing policy [@sreedhar2025]. As these systems increasingly
interact with humans and with one another, their *strategic* behavior — when to
cooperate, exploit, signal, or deceive — becomes a first-order concern for both AI
safety and the management of agentic deployments.

A growing "machine behavioral game theory" shows that LLMs are competent strategic
players with distinctive, reproducible styles. @akata2025 had models play finitely
repeated 2×2 games and found behavioral signatures that were stable across robustness
checks, with GPT-4 proving notably unforgiving in the Prisoner's Dilemma family.
@payneAlloui2025 ran evolutionary iterated-Prisoner's-Dilemma tournaments spanning
roughly 32,000 decisions and identified persistent "strategic fingerprints": Google's
Gemini models played ruthlessly, OpenAI's models cooperated (often to their own
detriment in hostile environments), and Anthropic's Claude emerged as the most
forgiving reciprocator. @payne2026 extended this line to simulated nuclear crises,
observing deception, theory-of-mind reasoning, and metacognitive self-assessment.
Across these studies, each model appears to carry a recognizable strategic disposition.

Whether those signatures reflect a genuine, transferable strategic architecture or
merely prompt-level pattern matching remains contested. Analyzing roughly 400,000
Ultimatum Game decisions across 17 models, @ferraz2025 interpreted personality-linked
behavioral shifts as prompt-driven regularities rather than genuine motivational
processes. @capraro2024 found that even the strongest model they tested systematically
misestimated human prosociality, overstating altruism. And @payne2026 documented sharp
context-dependence, with a single model behaving passively under one framing and
aggressively under another. This tension motivates the question at the center of our
study: **are the strategic dispositions of LLMs coherent across structurally distinct
environments, or do they fragment when the game changes?** Prior work has largely
examined a single game family — Prisoner's Dilemma variants, the Ultimatum Game, or the
Public Goods Game — or a single capability, leaving *cross-environment* coherence
untested.

We address this gap by estimating, for each of six models from three developer families
(Anthropic, OpenAI, Google), a strategic profile across three games chosen to activate
distinct strategic mechanisms: the **Prisoner's Dilemma**, which isolates dyadic
reciprocity under direct feedback [@rapoport1965; @axelrod1984]; the **Commons
Dilemma**, which removes dyadic reciprocity and tests collective action over a shared
resource [@hardin1968; @ostrom1990]; and a **Cheap-Talk signaling game**, which
introduces communication without binding commitment under information asymmetry
[@crawford1982]. The profile comprises five behavioral parameters: conditional
reciprocity (ρ) in the Prisoner's Dilemma; exploitation intensity (θ) in the Commons;
the drop in sender honesty under incentive misalignment (Δη) and receiver gullibility
(γ) in Cheap-Talk; and belief calibration (β), elicited in all three games but — because
the games differ in response format — analyzed separately by environment rather than
pooled. Within Cheap-Talk we additionally vary whether the sender's incentive is aligned
or misaligned with the receiver's, and across all games we vary the disclosed identity
of the opponent (undisclosed, AI, or human), letting us ask whether behavior tracks the
strategic *structure* of a situation or its surface *framing* [@borthakur2025;
@dvorak2025; @vanneste2023].

Following good practice for separating planned from post hoc analysis, we distinguish
hypotheses formulated before the final model set was assembled (confirmatory) from
patterns identified afterward (exploratory), and we report all alternatives we tested.
Three confirmatory hypotheses structure the analysis:

- **H1 (differentiation).** The six models exhibit statistically distinct strategic
  profiles, differing systematically in conditional reciprocity (ρ), exploitation
  intensity (θ), honesty under misalignment, and receiver gullibility (γ).
- **H2 (incentive-contingent honesty).** Incentive misalignment reduces sender honesty
  (Δη > 0), and the magnitude of this reduction is model-specific — some models adapt
  sharply while others do not adjust at all.
- **H3 (structure over identity).** Structural features of a game — incentive alignment,
  information availability, risk structure, and the framing of available actions —
  shape model behavior more strongly than the disclosed identity of the opponent.

Three further questions are treated as **explicitly exploratory**, given the six-model
sample: whether belief calibration is *coherent* across environments — in particular,
whether it transfers between the two binary-action games but not the
continuous-extraction Commons; how model parameters compare to published human
benchmarks, which we report *descriptively* rather than as inferential tests, since
comparisons against published summary statistics lack the participant-level data needed
for valid inference; and whether models that deceive more as senders are correspondingly
less gullible as receivers. Where an exploratory pattern is strong, we note the need to
validate it on held-out runs or an extended model set.

This study makes three contributions. First, it provides a uniform, multi-game
measurement of LLM strategic behavior, showing that six models possess distinct and
statistically separable strategic profiles — "fingerprints" — rather than a single
shared disposition. Second, it isolates *incentive-contingent deception* as a
model-specific behavior: under a misaligned incentive, some models sharply reduce
honesty while others do not adapt at all. Third, it asks whether the strategic
*structure* of a game or the *stated identity* of the opponent more strongly governs
behavior, a distinction with direct consequences for deploying agentic AI — a setting in
which human–AI combinations do not reliably outperform either humans or AI alone
[@vaccaro2024]. We complement these confirmatory results with clearly demarcated
exploratory analyses of cross-environment coherence and human comparison. The remainder
of the paper reviews the relevant literature on behavioral games and agents (§2),
details our data collection, prompting, and game designs (§3), reports per-game results,
the confirmatory tests of H1–H3, and the exploratory analyses together with a battery of
robustness checks (§4), and closes with what the findings imply for the deployment of
agentic AI (§5).

<!-- FINDINGS PREVIEW: add a 2–3 sentence summary of the headline H1/H2 results here
once the confirmatory analysis is final. Do not fabricate; fill from computed data. -->
