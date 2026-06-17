# Literature Review

Our work sits at the intersection of two literatures. The first treats LLMs as
*strategic players* and asks how they behave inside game-theoretic environments — the
behavioral-game-theory-for-machines tradition from which the notion of stable strategic
"fingerprints" emerges. The second treats LLMs as *agents* that act on behalf of, or
alongside, humans — and asks how prosocial they are, how their behavior changes when the
counterpart is human rather than machine, and what this implies for deployment. We
review each in turn and then identify the gap our study addresses.

## Behavioural Games

The study of how agents behave in repeated strategic interaction is the province of
behavioral game theory, which departs from the assumption of purely rational,
payoff-maximizing play to examine how social preferences and bounded cognition shape
decisions [@guth1982; @fehr1999]. A rapidly growing body of work — sometimes labelled
"machine psychology" [@hagendorff2023] — applies this lens to LLMs, treating them as a
new class of decision-making agents whose behavior can be probed with the instruments of
experimental economics. @akata2025 had several LLMs play finitely repeated 2×2 games and
found that they perform well in self-interested games such as the Prisoner's Dilemma but
struggle to coordinate; crucially, the behavioral signatures they identified were
*stable across robustness checks*, and a "social chain-of-thought" prompt that asked a
model to reason about its partner improved coordination. @payneAlloui2025 pushed this
further with evolutionary iterated-Prisoner's-Dilemma tournaments spanning roughly
32,000 decisions, reporting persistent "strategic fingerprints" that differed by
developer family: Gemini models exploited cooperative opponents and retaliated against
defectors, OpenAI's models remained highly cooperative even when this was costly, and
Claude behaved as the most forgiving reciprocator. These fingerprints were not random
artifacts — the models articulated, in their written rationales, explicit reasoning
about the time horizon and the likely strategy of the opponent.

If these signatures are real, the natural question is whether they reflect a genuine
strategic disposition or a brittle response to surface prompting. The evidence is mixed.
On one hand, @payne2026 placed frontier models in simulated nuclear crises and observed
sophisticated, seemingly stable behavioral traits: models attempted deception, signalled
intentions they did not hold, reasoned about an adversary's beliefs, and reflected
metacognitively on their own capabilities. On the other hand, the same study documented
strong *context-dependence* — one model was passive in open-ended scenarios but turned
hawkish under deadline pressure — and large-scale work is more skeptical still.
@ferraz2025 prompted 17 models with validated "Dark Factor" personality profiles in the
Ultimatum Game and, across roughly 400,000 decisions benchmarked against 4,166 humans,
found systematic but *hypersensitive* shifts that they read as prompt-driven regularities
rather than genuine motivation. A complementary, mechanistic perspective comes from
@sun2026, who construct "persona vectors" for traits such as altruism and forgiveness and
show that steering a model along these activation directions shifts both its choices and
its justifications — while also finding that a model's rhetoric and its actual strategy
can diverge, and that "how I behave" and "what I expect of others" are partially separate
internal representations. Taken together, this literature establishes that LLMs exhibit
recognizable strategic styles *within* a given game, but leaves open two prior questions
that we take up: whether those styles differ *systematically across models* when measured
with a uniform instrument, and whether they persist — or fragment — when the structure of
the game changes.

## Agents

A second stream treats LLMs not as players in an abstract game but as agents whose
choices have social consequences, and asks how faithfully they capture — or replace —
human prosocial behavior. @capraro2024 tested whether leading models could predict the
distribution of human choices in the dictator game across 108 experiments in 12
countries, finding that only the strongest model recovered the qualitative classes of
behavior (self-interested, inequity-averse, altruistic) and that it systematically
*overestimated altruism* while underestimating self-interest. At the level of
interacting agents, @sreedhar2025 showed that multi-agent LLM systems reproduce
established findings from the public goods game — the effects of priming, transparency,
and unequal endowments — and even generate "unbounded" behaviors such as collaboration
and cheating that lie outside the original lab paradigm. These results matter for our
design: they justify calibrating models against human behavioral priors and motivate the
inclusion of a collective-action environment (the Commons Dilemma) in which dyadic
reciprocity is absent.

A distinct and, for our purposes, central finding is that strategic behavior depends on
*who the counterpart is believed to be*. In a pre-registered experiment with 3,552
participants spanning five two-player games, @dvorak2025 found that fairness, trust, and
cooperation declined when a partner's decisions were known to be delegated to ChatGPT,
yet — strikingly — no such penalty appeared when participants were merely uncertain
whether they faced a human or an AI, and people could not reliably tell the two apart.
@borthakur2025 reported a complementary asymmetry in a 21-round Ultimatum Game: people
rejected disadvantageous offers more often from an AI than from a human and reported
feeling *less obligated* to treat an AI counterpart equitably, consistent with classical
models of inequity aversion [@fehr1999]. @vanneste2023 supply a theoretical account of
why identity matters at all, arguing that the *perceived agency* of an AI shapes the
trust placed in it through capability, comparative-trustworthiness, and betrayal-aversion
mechanisms. Crucially, however, these identity effects are conditional rather than
universal: the cooperation penalty in @dvorak2025 appeared only when the partner was
*explicitly disclosed* as an AI and vanished under uncertainty, and the asymmetry in
@borthakur2025 reversed for advantageous offers. This mixed picture motivates both our
manipulation of disclosed opponent identity (undisclosed, AI, human) and, more centrally,
our hypothesis that the *structural* features of a game may govern model behavior more
strongly than the stated identity of the counterpart.

Finally, because two of our environments hinge on communication, we draw on work
examining how LLMs persuade and negotiate. @carrasco2024 showed that LLM-generated
arguments are as persuasive as human ones but achieve this differently — with higher
lexical and grammatical complexity and heavier use of moral language — an
*equivalence-in-outcome-without-equivalence-in-process* result that bears directly on how
models construct messages in a signaling game. At the level of full negotiations,
@vaccaro2026 analyzed more than 180,000 agent-to-agent negotiations and found that
classic human principles still apply: warmth was associated with greater value creation
while dominance produced impasses. Yet whether such communicative competence translates
into productive deployment is far from guaranteed: in a pre-registered meta-analysis of
106 studies and 370 effect sizes, @vaccaro2024 found that human–AI combinations
performed, on average, *worse* than the better of human or AI alone, with losses
concentrated in decision-making tasks. The stakes of understanding LLM strategy are
therefore practical as well as scientific.

Across both streams, prior work almost always isolates a single game family or a single
capability. Strategic fingerprints have been documented *within* the Prisoner's Dilemma,
personality effects *within* the Ultimatum Game, and prosocial replication *within* the
public goods game. Three questions remain comparatively unexamined, and we take them up
here: whether the same models, measured with a uniform instrument across structurally
distinct games, display *systematically distinct* strategic profiles (H1); whether they
adapt a specific behavior — honesty — to the *incentive structure* they face (H2); and
whether their behavior is driven more by the structural features of the environment than
by the *stated identity* of the opponent (H3). Whether these profiles additionally
*cohere across* environments — the strong form of the strategic-architecture question —
we examine as well, but, given a sample of only six models, treat as exploratory rather
than confirmatory.
