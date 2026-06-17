# Methods

We measure each model's strategic behavior with a single instrument applied uniformly
across three structurally distinct games. This section describes how the data were
collected (§3.1), how the models were prompted (§3.2), and how each game was played and
its behavioral parameters operationalized (§3.3). Throughout, we distinguish two units of
analysis that carry very different statistical weight: the *model* level, where the
sample is only six and any across-model comparison is correspondingly underpowered, and
the *round* and *game* level, where each model contributes hundreds to thousands of
decisions and within-model effects can be estimated robustly. We frame inferential claims
at the round and game level wherever the design allows, and treat across-model
correlations as descriptive.

## How We Collected Data

We study six models drawn from three developer families: Claude Opus and Claude Sonnet
(Anthropic), GPT-4o and GPT-4o-mini (OpenAI), and Gemini 2.5 Flash and Gemini 2.5 Flash
Lite (Google). Each family thus contributes a larger and a smaller model, allowing
within-family comparisons alongside cross-family ones. All six were accessed through a
single LangChain interface so that prompt construction, response handling, and logging
were identical across providers; exact model identifiers and API settings are listed in
Appendix A. Temperature was held fixed at 0.6 across all three games and is *not* treated
as an experimental factor: the consolidated dataset contains a single temperature setting,
so any behavioral variation we report reflects the game, the condition, or the model
rather than sampling stochasticity introduced by decoding. (Earlier Prisoner's Dilemma
pilot runs at other temperatures exist but are excluded from the analyzed dataset.)

The three games share a common experimental scaffold. Models play in pairs over a fixed
number of rounds (twenty in the Prisoner's Dilemma and Commons Dilemma, fifteen in
Cheap-Talk), seeing the full history of prior rounds before each decision and returning a
structured JSON response that always includes a stated *belief* about the opponent
alongside the chosen action. Pairings were chosen to span both cross-family contrasts
(large-versus-large across developers) and within-family size contrasts (each family's
larger model against its smaller sibling); the full matchup lists for each game are given
in Appendix A. Every run is written to disk twice: each round is committed immediately to
a per-game SQLite database (so that a crash never loses collected decisions), and a
complete CSV is exported at the end of the run for analysis. The schema records, for both
players in every round, the action, the stated belief, the per-round and cumulative
payoff, the raw model output, token usage, response latency, the temperature, the prompt
version, and the experimental condition (Appendix A).

We also calibrated a *simulated-human* reference agent — a Claude Sonnet instance
prompted with empirically grounded behavioral targets taken from published laboratory
studies [@dvorakFehrler2024; @abatayo2022; @gneezy2005] — to provide an internal
human-like comparison. Because our headline human comparisons in §4 are made against the
*published* summary statistics of these studies rather than against the simulated agent,
and because we have no participant-level human data, all human comparisons are reported
*descriptively* and carry no inferential test. The simulated-human runs are used only as
a supporting robustness reference (§4.5).

All behavioral parameters are estimated at the game level — one observation per complete
game per model — and then aggregated to the model level, which avoids inflating
significance through the within-game dependence of successive rounds. Between-model
differences are tested with Kruskal–Wallis at the game level, complemented by logistic
mixed models that retain the round as the unit of observation with a random intercept per
run (the design Carlos's main analysis specifies for the conditional-reciprocity and
gullibility parameters); structural-feature models for H3 likewise operate at the round
level. Profile similarity across models is summarized by a 6×6 Euclidean-distance matrix
over the standardized parameter vector. Given the six-model sample, we deliberately avoid
latent-variable techniques such as factor analysis, which that sample cannot support.

## How We Prompted

A central validity threat for this design is *prompt sensitivity*: the risk that a
model's choices shift in response to how a game is described rather than to its payoff
structure, which would render any cross-environment comparison meaningless. We therefore
treat the prompt as a fixed, controlled instrument rather than a variable. The system
prompt is identical across all six models within a game, so wording is never a
between-model confound. Within each prompt we apply a consistent set of neutrality
controls: the objective is stated neutrally (models are asked to play in their own
strategic interest, not to "maximize their score" in a way that anchors aggression, and
the Commons prompt withholds any "preserve the resource" framing); where an action format
is illustrated, *both* available actions are shown so that no single option is anchored;
and the running round number is withheld to prevent end-game horizon effects. Belief
elicitation is worded so that the stated probability is formed *before* the action,
preventing post-hoc rationalization.

The required response is a JSON object whose fields differ by game but always include the
belief. The Prisoner's Dilemma additionally requires two to three sentences of explicit
strategic reasoning before the JSON, which both unlocks adaptive play and yields a written
rationale; the Commons prompt requires JSON only; and Cheap-Talk permits a single-sentence
reasoning field. A robust parser extracts the action and belief from the model's output,
tolerating markdown fences and surrounding text, and a bounded retry re-prompts on a
malformed response before falling back to a conservative default. Prompts are versioned,
and any change to a prompt increments its recorded version so that data collected under
different wordings are never silently pooled. We do not claim a prompt-sensitivity-free
design; we claim a *controlled* one, and we report a battery of sensitivity checks among
the robustness analyses in §4.5. The full prompt text for every game and condition,
together with the version history, appears in Appendix B.

## How We Played the Games

The three games were chosen to activate distinct strategic mechanisms while sharing the
common scaffold above. Each yields one or two parameters of the strategic profile, plus a
belief-calibration measure; because the games differ in response format, belief
calibration (β) is the mean absolute error between a model's stated probability and the
realized binary outcome and is analyzed *separately* per environment (β_PD, β_CD, β_CT)
rather than pooled.

**Prisoner's Dilemma.** This game isolates dyadic reciprocity under direct feedback. Two
models repeatedly choose to cooperate or defect under the standard payoff ordering
(T = 5, R = 3, P = 1, S = 0, satisfying T > R > P > S) over twenty rounds
[@rapoport1965; @axelrod1984]. Its parameter is *conditional reciprocity* (ρ), the degree
to which a model conditions cooperation on its opponent's last move:
ρ = P(C \| opponent cooperated at t−1) − P(C \| opponent defected at t−1), computed over
rounds 2…N. A value near +1 indicates tit-for-tat-like reciprocity; a value near zero
indicates behavior insensitive to the opponent's recent action.

**Commons Dilemma.** This game removes dyadic reciprocity and tests collective action over
a shared, regenerating resource [@hardin1968; @ostrom1990]. Two players simultaneously
extract from a pool that begins at 100 units and regenerates by a fraction of its
remaining capacity each round, with over-extraction risking collapse. Crucially, the
sustainable per-capita extraction level is *not* disclosed, so models must infer it,
forcing the behavioral variation needed to estimate the parameter. That parameter is
*exploitation intensity* (θ), defined relative to the time-varying sustainable share:
θ = mean of extraction_{i,t} ÷ (pool_t × regeneration_rate ÷ n_players), so that θ = 1.0
denotes exactly sustainable play and θ > 1.0 denotes over-extraction. The Commons task is
implemented to support a 2×2 design crossing the *information* available about the
resource (the current pool level shown each round versus hidden) with the *risk* structure
of collapse (a deterministic threshold versus a collapse probability that rises as the
pool depletes) — two structural factors that bear on H3 by varying the situation's
structure while holding the resource game fixed.
<!-- VERIFY before §4/submission: the v2 analysis reports a single pooled θ per model with
no info×risk cell breakdown, and the June-11 run log does not confirm all four cells were
collected. Confirm which cells are populated in the analyzed dataset; if only the baseline
cell was run, describe this as a single condition (with the factorial as planned/future
work) rather than as an executed 2×2. -->


**Cheap-Talk signaling.** This game introduces communication without binding commitment
under information asymmetry [@crawford1982]. Each round a hidden state (High or Low) is
drawn; an informed Sender transmits a message and an uninformed Receiver chooses an action
whose correctness depends on the true state. The key structural manipulation is the
Sender's incentive: under the *aligned* payoff both players gain only when the Receiver
acts correctly, whereas under the *misaligned* payoff the Sender always profits from one
particular Receiver action regardless of the truth, creating a direct incentive to deceive.
Roles are fixed for the full game in most runs and *rotated* at the midpoint in a subset,
letting the same model be observed as both Sender and Receiver. The game yields two
parameters. On the Sender side, the *honesty drop under misalignment* (Δη) is the
difference between the fraction of truthful messages when incentives are aligned and when
they are misaligned (Δη = η_aligned − η_misaligned); a positive value marks
incentive-contingent deception. On the Receiver side, *gullibility* (γ_mis) is the
Pearson correlation between message truthfulness and action correctness on misaligned runs
only — where the message is genuinely unreliable — so that a higher value indicates a
Receiver who follows messages it should discount.

Finally, the disclosed identity of the counterpart is manipulated across all three games
to test structure against identity (H3). In the Prisoner's Dilemma and Commons Dilemma the
opponent is described as undisclosed, as another AI, or as a human (a framing manipulation
— the counterpart is the same model regardless); in Cheap-Talk, identity framing is
applied per role, since the Sender and Receiver roles are asymmetric (Appendix A).
<!-- VERIFY before §4/submission: the Cheap-Talk script defines six per-role identity
conditions, but the v2 analyzed dataset's condition labels are the 4-value set
{undisclosed, ai, human, human_prior} and the run log only confirms undisclosed +
aligned/misaligned pilots. Confirm which identity conditions are actually in the analyzed
CT dataset before claiming the full six. Also confirm CT round count (script=15; pilots=10). -->
Pitting this *identity* framing against the *structural* features
above — incentive alignment, information availability, risk, and action framing — lets us
ask, at the round level, which more strongly governs behavior.
