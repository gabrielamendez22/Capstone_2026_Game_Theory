# Literature Review — Source Map

**Branch:** `report` · **Date:** 2026-06-11
**Purpose:** Map every PDF in `fonti per capstone/` to where it belongs in the
report's Literature Review (§2.1 Behavioural games / §2.2 Agents) and, where
relevant, to Methods or the Recommendations close. Grounded in each paper's
abstract + introduction (read directly). Use this to write the lit review without
re-opening every PDF.

**Supervisor note:** the persuasion paper (#13) is by **Carlos Carrasco-Farré**
— our supervisor. Cite deliberately.

**Legend for "Slot":** `2.1` = Behavioural games · `2.2` = Agents ·
`M` = also informs Methods · `D` = also informs Discussion/Recommendations.

---

## §2.1 — Behavioural Games (LLMs as strategic players)

### 1. Akata, Schulz, Coda-Forno, Oh, Bethge & Schulz (2023/2025) — *Playing Repeated Games with Large Language Models*
- **Venue/where:** arXiv (Helmholtz Munich, MPI Tübingen). One of our 3 structural templates.
- **Method:** Finitely repeated 2×2 games (6 families incl. PD, Battle of the Sexes), 10 rounds; payoff matrix rendered as text prompt; GPT-4, text-davinci-002/003, Claude 2, Llama 2.
- **Key findings:** LLMs excel at self-interested games (PD family — GPT-4 retaliates after a single defection, "unforgiving"), underperform at coordination (BoS). Behavioural signatures **stable across robustness checks**. Social-Chain-of-Thought (SCoT) prompting improves coordination and makes humans believe partner is human.
- **Use for us:** Anchor for "LLMs have stable behavioural signatures in repeated games." Methodological template for **rendering a game as a text prompt** and for **robustness checks** (our §4.5). The "unforgiving GPT-4" result contrasts with our ρ (conditional reciprocity).
- **Slot:** `2.1`, `M`

### 2. Payne & Allou-Cros (2025) — *Strategic Intelligence in LLMs: Evidence from Evolutionary Game Theory*
- **Venue/where:** arXiv (King's College London, Oxford). Structural template.
- **Method:** First evolutionary IPD tournaments mixing canonical strategies (TfT, Grim) with frontier LLMs (OpenAI, Google, Anthropic); varied termination probability ("shadow of the future"); ~32,000 decisions + prose rationales.
- **Key findings:** Persistent **"strategic fingerprints"** — Gemini ruthless/exploitative, OpenAI over-cooperative (catastrophic in hostile settings), Claude most forgiving reciprocator. Models explicitly reason about time horizon and opponent identity.
- **Use for us:** The single closest precedent — "strategic fingerprints" is the seed of our **strategic-disposition** construct. Our contribution = testing whether such fingerprints are **coherent across structurally distinct games** (CECS), which they do not test.
- **Slot:** `2.1`, `D`

### 3. Payne (2026) — *AI Arms and Influence: Frontier Models Exhibit Sophisticated Reasoning in Simulated Nuclear Crises*
- **Venue/where:** arXiv (KCL). Structural template.
- **Method:** Crisis-escalation simulation; GPT-5.2, Claude Sonnet 4, Gemini 3 Flash play opposing leaders.
- **Key findings:** Models attempt deception, signal intentions they don't hold, show theory-of-mind and metacognition; distinct strategic "personality"; strong **context-dependence** (one model passive in open scenarios, hawkish under deadline pressure).
- **Use for us:** Supports the "machine psychology" framing and the deception/signaling link to our **Cheap-Talk** game (η honesty). Context-dependence is exactly the fragility our CECS is built to detect.
- **Slot:** `2.1`, `D`

### 4. Ferraz, Olah, Sazedul, Schmidt & Schwieren (2025) — *When Artificial Minds Negotiate: Dark Personality and the Ultimatum Game in LLMs*
- **Venue/where:** AWI Discussion Paper No. 768 (KIT, Heidelberg, Debrecen, Bundesbank).
- **Method:** Ultimatum Game; Dark-Factor (D-Factor) personality prompts; **~400k decisions across 17 open-source models vs. 4,166 human benchmarks**; proposer + responder roles; temperature 0.2 vs 0.8.
- **Key findings:** Proposer fair offers decline 91%→17% across D-levels (mirrors humans but 34% steeper — hypersensitive to prompts); responders fail to reproduce reciprocity-punishment. Large **cross-model heterogeneity**. Authors read patterns as **prompt-driven regularities, not genuine motivation**. Temperature had minimal effect.
- **Use for us:** (a) Strong precedent for **personality/disposition prompting** + human benchmarking at scale; (b) the "prompt-driven not motivational" reading is the **skeptical hypothesis our CECS adjudicates**; (c) temperature-minimal-effect result is a comparison point for our H3.
- **Slot:** `2.1`, `M`, `D`

### 5. Sun & Zhang (2026) — *Persona Vectors in Games: Measuring and Steering Strategies via Activation Vectors*
- **Venue/where:** arXiv (Harvard).
- **Method:** Activation steering (ActAdd-style) to construct persona vectors for altruism, forgiveness, expectations-of-others; evaluated on canonical games (Qwen 2.5-7B).
- **Key findings:** Steering shifts both quantitative choices and natural-language justifications; **rhetoric and strategy can diverge** under steering; self-behaviour and expectations-of-others are partially **distinct internal representations**.
- **Use for us:** Mechanistic counterpart — strategic dispositions may be linear directions in activation space, i.e. **real internal traits**, lending plausibility to coherent cross-game architecture. The rhetoric≠action divergence directly motivates our **η (signal honesty)** in Cheap-Talk and **belief vs action** (β). Future-work hook.
- **Slot:** `2.1`, `D`

---

## §2.2 — Agents (prosociality, human–AI interaction, deployment)

### 6. Capraro, Di Paolo & Pizziol (2024) — *Assessing LLMs' Ability to Predict How Humans Balance Self-Interest and the Interest of Others*
- **Venue/where:** Working paper (Milan-Bicocca, IMT Lucca, Bologna).
- **Method:** Dictator Game; GPT-4, Bard, Bing predict human choice distributions across **108 experiments / 12 countries**.
- **Key findings:** Only GPT-4 captures qualitative classes (self-interested / inequity-averse / altruistic) but **underestimates self-interest and inequity-aversion, overestimates altruism**.
- **Use for us:** Justifies our **human-prior calibration** condition and warns that LLM priors are biased toward altruism — relevant when we seed `human_prior` agents and compare to literature benchmarks.
- **Slot:** `2.2`, `M`

### 7. Sreedhar & Chilton (2025) — *Simulating Cooperative Prosocial Behavior with Multi-Agent LLMs*
- **Venue/where:** IUI '25 (Columbia, Harvard, Stevens).
- **Method:** Multi-agent LLMs play the **Public Goods Game** with priming / transparency / varying-endowment treatments; tests transfer across treatments and "unbounded" real-world actions (cheating).
- **Key findings:** Multi-agent LLMs **replicate human PGG effects** (direction, not always magnitude); effects transfer across games; exhibit unbounded behaviours (collaborating, cheating). Argues for LLM simulations to inform policy.
- **Use for us:** Closest analogue to our **Commons Dilemma** (PGG = collective action, no dyadic reciprocity). Supports that cooperation effects survive in N-player public-good settings; comparison point for θ.
- **Slot:** `2.2`, `M`

### 8. Vaccaro, Almaatouq & Malone (2024) — *When Are Combinations of Humans and AI Useful? A Systematic Review and Meta-Analysis*
- **Venue/where:** arXiv (MIT Center for Collective Intelligence). Pre-registered.
- **Method:** Meta-analysis, **370 effect sizes / 106 studies** (2020–2023).
- **Key findings:** Human-AI combinations underperform the **best of either alone on average** (Hedges' g = −0.23); **losses in decision tasks, gains in content-creation**; combos help when humans beat AI alone, hurt when AI beats humans.
- **Use for us:** Backbone of the **managerial recommendations** (§5): when to delegate to agentic AI vs. keep humans in the loop. Frames the deployment stakes of strategic coherence.
- **Slot:** `2.2`, `D`

### 9. Borthakur, Diep & Plaks (2025) — *Inequity Aversion Toward AI Counterparts*
- **Venue/where:** Scientific Reports (Toronto, Waterloo).
- **Method:** **21-round Ultimatum Game** against AI vs. human counterpart; behavioural + physiological (heart-rate) + affective measures.
- **Key findings:** People reject **disadvantageous** offers more from AI than human, but reject **advantageous** offers more from humans; more negative affect toward AI; feel **less obligated** to treat AI equitably.
- **Use for us:** Direct evidence that **opponent identity (AI vs human) changes behaviour** — the human counterpart to our `OPPONENT_CONDITION` manipulation and our **Δm (opponent sensitivity)** metric.
- **Slot:** `2.2`, `D`

### 10. Vanneste & Puranam (2023) — *Artificial Intelligence, Trust, and Perceptions of Agency*
- **Venue/where:** INSEAD WP, forthcoming *Academy of Management Review* (UCL, INSEAD).
- **Method:** Conceptual/theoretical.
- **Key findings:** Perceived **agency** of AI shapes trust via three mechanisms (capability, comparative trustworthiness, betrayal aversion); more agentic-seeming AI may be trusted **more or less** — passing the Turing test can *reduce* trust.
- **Use for us:** Theoretical scaffolding for **why opponent-identity framing matters** and why "told it's a human" (deception condition) changes the strategic situation. Grounds the trust/agency angle in §2.2 and the recommendations.
- **Slot:** `2.2`, `D`

### 11. Vaccaro, Caosun, Ju, Aral & Curhan (2026) — *Advancing AI Negotiations: A Large-Scale Autonomous Negotiation Competition*
- **Venue/where:** arXiv (MIT Sloan, Johns Hopkins).
- **Method:** International AI Negotiation Competition; **>180,000 agent-agent negotiations**; NLP on full transcripts; explicitly Axelrod-tournament-inspired.
- **Key findings:** Human negotiation principles persist in AI-AI; **warmth → higher value, dominance → claiming value but more impasses**; calls for a new theory blending classic + AI-specific negotiation.
- **Use for us:** Large-scale evidence that **strategic styles** (warmth/dominance) drive outcomes among agents — a communication-game complement to Cheap-Talk and to "fingerprints." Methodological scale benchmark.
- **Slot:** `2.2`, `D`

### 12. Dvorak, Stumpf, Fehrler & Fischbacher (2025) — *Adverse Reactions to the Use of LLMs in Social Interactions*
- **Venue/where:** PNAS Nexus (Konstanz, Eawag, Bremen, CESifo). Already cited in `METHODOLOGY.md`.
- **Method:** Pre-registered online experiment, **n = 3,552**, five two-player games (**UG, Trust, PD, Stag Hunt, Coordination**); ChatGPT takes over a partner's decisions; transparent vs. opaque vs. uncertain framing.
- **Key findings:** Fairness, trust, cooperation **decrease when a partner is known to be ChatGPT**; **no adverse reaction when uncertain** whether partner is human or AI; people **cannot reliably distinguish** AI from human decisions.
- **Use for us:** Direct empirical basis for our three **PD/CD conditions** (`undisclosed` / `ai` / `human`) and the human PD benchmark. The "uncertain ≈ no penalty" result motivates the `undisclosed` baseline.
- **Slot:** `2.2`, `M`, `D`

### 13. Carrasco-Farré (2024) — *Large Language Models Are as Persuasive as Humans, But How?* **(supervisor)**
- **Venue/where:** Working paper, Toulouse Business School.
- **Method:** N = 1,251; 56 claims; compares LLM- vs human-generated arguments on cognitive effort (lexical/grammatical complexity) and moral-emotional language.
- **Key findings:** LLM arguments require **higher cognitive effort** and use **more moral language**; emotional content equivalent. Equivalence in *outcome* (persuasion) without equivalence in *process*.
- **Use for us:** Communication/**signaling** angle for **Cheap-Talk** — how LLMs construct persuasive messages bears on η (honesty) and γ (receiver gullibility). Supervisor's own work; cite in §2.2 and the Cheap-Talk framing.
- **Slot:** `2.2`

---

## Coverage check (source → report slot)

| Lit-review need | Covered by |
|---|---|
| LLMs as stable strategic players (fingerprints) | Akata; Payne & Allou-Cros; Payne 2026 |
| Personality/disposition prompting + human benchmarks | Ferraz; Capraro |
| Mechanistic basis of strategic traits | Sun & Zhang (persona vectors) |
| Collective-action / commons analogue | Sreedhar & Chilton (PGG) |
| Opponent-identity (AI vs human) effects → Δm | Borthakur; Dvorak; Vanneste & Puranam |
| Communication / signaling → Cheap-Talk | Carrasco-Farré; Advancing AI Negotiations |
| Deployment / managerial stakes | Vaccaro et al. (meta-analysis); negotiations |

## Gaps to fill from outside `fonti per capstone/` (classic anchors, not yet in folder)
- Axelrod (1984) — evolution of cooperation / IPD tournaments (cited by Payne & negotiations).
- Crawford & Sobel (1982) — foundational cheap-talk/signaling theory.
- Hardin (1968) / Ostrom (1990) — commons / collective action.
- Fehr & Schmidt (1999) — inequity aversion (theory behind Borthakur, Ferraz).
- Güth et al. (1982) — Ultimatum Game origin.
- Hagendorff (2023) — "machine psychology" coinage (used by Payne).
