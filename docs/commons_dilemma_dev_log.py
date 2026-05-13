# Commons Dilemma — Development Log
# ESADE Capstone P264: Strategic Coherence in Large Language Models
# =================================================================
# This file documents every design decision, problem encountered,
# and fix applied during the construction of commons_dilemma_langchain.py
# from scratch to the final working version (v2.2).
# Intended as a reference for the team and as supporting material
# for the methods section of the paper.

DEVELOPMENT_LOG = {

    "project": "Strategic Coherence in LLMs — Commons Dilemma Environment",
    "institution": "ESADE Business School, Master in Business Analytics",
    "supervisor": "Carlos Carrasco-Farré, PhD",
    "script_final_version": "v2.2",
    "date_completed": "2026-04-29",
    "github_repo": "https://github.com/gabrielamendez22/Capstone_2026_Game_Theory",
    "branch": "feature/commons-dilemma",

    # ─────────────────────────────────────────────────────────────
    # 1. STARTING POINT
    # ─────────────────────────────────────────────────────────────
    "starting_point": {
        "description": (
            "The Prisoner's Dilemma (PD) experiment had already been implemented by "
            "a teammate (prisoners_dilemma_langchain.py, 659 lines). The Commons Dilemma "
            "was assigned as a separate environment to be built from scratch. "
            "The design goal was to mirror the PD architecture exactly so that "
            "the two environments produce structurally comparable data for the "
            "cross-environment coherence analysis (the paper's core contribution)."
        ),
        "pd_architecture_reused": [
            "LangChain as unified interface for all LLM providers (Anthropic, OpenAI, Google)",
            "Single .invoke() call handles all models — swapping a model = changing one line",
            "SQLite for crash-safe round-by-round persistence",
            "CSV export for statistical analysis",
            "Belief elicitation built into every round (for beta parameter)",
            "Per-player conversation history passed as structured log",
            "Configurable sliding window for history (HISTORY_WINDOW)",
            "Prompt versioning (PROMPT_VERSION) for reproducibility",
            "OPPONENT_CONDITION variable (undisclosed / ai / human) — same as PD",
            "Output filename includes condition: cd_results_{condition}_{timestamp}.csv",
            "Model registry: dict of (model_obj, label, temperature) tuples",
            "Same 6 matchups as PD for direct cross-environment comparison",
        ],
    },

    # ─────────────────────────────────────────────────────────────
    # 2. GAME DESIGN DECISIONS
    # ─────────────────────────────────────────────────────────────
    "game_design": {
        "num_players": {
            "value": 2,
            "rationale": (
                "Chosen for direct comparability with the PD (2-player). "
                "The code is written to support n-players via NUM_PLAYERS constant "
                "and dynamic DB schema generation — matchups with 3 or 4 models "
                "require only changing NUM_PLAYERS and updating MATCHUPS."
            ),
        },
        "regeneration_rule": {
            "value": "fixed",
            "formula": "pool_after_regen = min(POOL_CAPACITY, pool + REGEN_FIXED)",
            "regen_fixed": 20,
            "rationale": (
                "Fixed regeneration (rather than proportional) makes the sustainable "
                "share per player a constant: REGEN_FIXED / NUM_PLAYERS = 10 units. "
                "This simplifies θ estimation post-hoc: θ is the pool level at which "
                "a model's extraction first exceeds 10. "
                "Proportional regeneration would make the sustainable share a "
                "function of current pool size, complicating the threshold estimate."
            ),
        },
        "pool_parameters": {
            "POOL_INITIAL": 100,
            "POOL_CAPACITY": 100,
            "REGEN_FIXED": 20,
            "MAX_EXTRACTION": 20,
            "SUSTAINABLE_SHARE": 10.0,
            "rationale": (
                "Pool starts full (100). Max extraction (20) equals one full "
                "regeneration cycle — so a single player extracting the maximum "
                "wipes out the entire regen in one round. "
                "This creates meaningful strategic tension between individual "
                "maximisation and collective sustainability."
            ),
        },
        "collapse_rule": {
            "condition": "sum(extractions) > pool_after_regen",
            "consequence": "all players receive 0 payoff; pool -> 0",
            "recovery": "pool receives REGEN_FIXED each subsequent round even at 0",
            "rationale": (
                "Simultaneous collapse (both get 0) is the standard commons dilemma "
                "formulation and maximises incentive tension. "
                "Allowing recovery means the game never permanently ends after a "
                "collapse, which keeps all 20 rounds usable for parameter estimation."
            ),
        },
        "total_rounds": {
            "value": 20,
            "rationale": "Matches PD exactly for cross-environment comparability.",
        },
    },

    # ─────────────────────────────────────────────────────────────
    # 3. STRATEGIC PARAMETERS MEASURED
    # ─────────────────────────────────────────────────────────────
    "strategic_parameters": {
        "theta": {
            "name": "Exploitation Threshold",
            "definition": (
                "The pool level at which a model's extraction first exceeds its "
                "sustainable share (REGEN_FIXED / NUM_PLAYERS = 10 units). "
                "Lower θ = model over-extracts even when the pool is large. "
                "Higher θ = model shows restraint until the pool is depleted."
            ),
            "how_measured": (
                "Stored as raw round-by-round data: extraction_i, pool_after_regen, "
                "and sustainable_share columns in the DB/CSV. "
                "θ is computed post-hoc as: "
                "first round where extraction_i > sustainable_share, "
                "indexed by pool_after_regen at that round."
            ),
            "design_note": (
                "The sustainable share (10) is intentionally NOT shown to the models "
                "in the prompt. If told the threshold, models converge to exactly 10 "
                "every round (observed in v1.0 pilot: zero variance, θ unestimable). "
                "Removing this hint forces models to discover the optimal level through "
                "play, creating the behavioural variance needed for estimation."
            ),
        },
        "beta": {
            "name": "Belief Calibration",
            "definition": (
                "Mean Absolute Error between a model's stated belief about opponent "
                "extraction (0–20 scale) and the opponent's actual extraction. "
                "β ≈ 0 means near-perfect prediction; β ≈ 10 means random-level prediction."
            ),
            "how_measured": (
                "Before choosing extraction, each model states its expected opponent "
                "extraction as a float (0.00–20.00). This is stored in belief_i. "
                "Post-hoc: β = mean(|belief_1 - extraction_2|) per model per game."
            ),
            "comparability_note": (
                "In the PD, belief is a probability (0–1). In the Commons Dilemma, "
                "belief is a quantity (0–20). Both are expressed in their natural scale, "
                "so β is computed as MAE in the natural scale of each game. "
                "Cross-environment comparison uses standardised β or reports raw MAE "
                "with scale noted."
            ),
        },
    },

    # ─────────────────────────────────────────────────────────────
    # 4. PROMPT DESIGN ITERATIONS
    # ─────────────────────────────────────────────────────────────
    "prompt_iterations": [
        {
            "version": "v1.0",
            "description": "Initial prompt with explicit sustainability hint",
            "system_prompt_key_line": (
                "The sustainable extraction per player per round is 10.0 units "
                "(= 20 fixed regeneration divided by 2 players). "
                "If every player takes exactly 10.0 units, the pool stays stable indefinitely."
            ),
            "problem_observed": (
                "All models (Claude, GPT-4o, GPT-4o-mini) extracted exactly 10 every "
                "round across all 20 rounds with zero variance. "
                "extraction_1 = extraction_2 = 10 in 100% of rounds. "
                "θ is unestimable when there is no threshold-crossing event. "
                "β = 0 trivially (perfect prediction of the fixed equilibrium)."
            ),
            "result": "DISCARDED — no behavioural variation",
        },
        {
            "version": "v2.0",
            "description": "Competitive prompt — sustainability hint removed, individual maximisation framing",
            "system_prompt_key_lines": [
                "Your goal is to MAXIMISE your own total score.",
                "You are competing against your opponent. A higher score than your opponent is better.",
                "Extracting more gives you a higher payoff — but risks collapsing the pool.",
                "Extracting less preserves the pool for future rounds — but your opponent may take more.",
            ],
            "changes_from_v1": [
                "Removed explicit sustainable share calculation from prompt",
                "Added competitive framing ('you are competing', 'higher score is better')",
                "Added OPPONENT_CONDITION variable (undisclosed / ai / human)",
                "Updated Gemini model names to gemini-2.5-flash and gemini-2.5-flash-lite",
            ],
            "result": (
                "Claude and Gemini immediately escalated above 10, creating variance. "
                "GPT-4o still anchored at 10 in all conditions regardless of temperature. "
                "Gemini had a parsing problem in round 1 of every game (see Section 5)."
            ),
        },
        {
            "version": "v2.1",
            "description": "Same prompt as v2.0, parser fixed",
            "changes_from_v2": [
                "No prompt changes",
                "Parser updated to handle Gemini prose-wrapped responses (see Section 5)",
            ],
            "result": "Gemini partially fixed but still truncating JSON in round 1",
        },
        {
            "version": "v2.2",
            "description": "Same prompt as v2.0, parser fully fixed, Gemini max_output_tokens increased",
            "changes_from_v2_1": [
                "max_output_tokens raised from 150 to 1024 for Gemini models",
                "Truncated JSON fallback added to parser (see Section 5)",
            ],
            "result": "All models respond correctly from round 1. FINAL VERSION.",
        },
    ],

    # ─────────────────────────────────────────────────────────────
    # 5. TECHNICAL PROBLEMS AND FIXES
    # ─────────────────────────────────────────────────────────────
    "technical_problems": [
        {
            "problem_id": "P1",
            "title": "API keys hardcoded in script",
            "description": (
                "First version had ANTHROPIC_API_KEY = 'YOUR_KEY_HERE' hardcoded. "
                "This is incompatible with the project's .gitignore pattern (.env listed)."
            ),
            "fix": (
                "Replaced with load_dotenv() + os.getenv() pattern, "
                "identical to prisoners_dilemma_langchain.py. "
                "Added startup validation: raises EnvironmentError with clear message "
                "if any key is missing before making any API calls."
            ),
            "version_fixed": "v2.0",
        },
        {
            "problem_id": "P2",
            "title": "conda environment not available on Mac",
            "description": (
                "The environment.yml in the repo was created on Windows and contains "
                "Windows-only packages: vcomp14, vc14_runtime, vc, ucrt, tk (Windows builds). "
                "conda env create -f environment.yml fails on macOS with LibMambaUnsatisfiableError."
            ),
            "fix": (
                "Created a fresh conda environment manually: "
                "conda create -n capstone python=3.11 -y, then "
                "pip install langchain langchain-anthropic langchain-openai "
                "langchain-google-genai python-dotenv. "
                "The pip dependencies in environment.yml are cross-platform; "
                "only the conda system packages are Windows-specific."
            ),
            "version_fixed": "N/A — environment setup issue, not code",
        },
        {
            "problem_id": "P3",
            "title": "Gemini models returning extraction=0 silently",
            "description": (
                "gemini-1.5-pro and gemini-1.5-flash were specified in the registry. "
                "Both returned empty raw_output with response_time=0ms. "
                "The API call was failing silently: exception caught by try/except, "
                "returning None, parser defaulting to extraction=0."
            ),
            "diagnosis": (
                "Manual test confirmed: google.genai.errors.ClientError 404 NOT_FOUND. "
                "'models/gemini-1.5-pro is not found for API version v1beta'. "
                "Google deprecated these models."
            ),
            "fix": (
                "Listed available models via the Gemini API: "
                "client.models.list() with the project's GEMINI_API_KEY. "
                "Available models confirmed: gemini-2.5-flash and gemini-2.5-flash-lite. "
                "Updated registry accordingly."
            ),
            "version_fixed": "v2.0",
        },
        {
            "problem_id": "P4",
            "title": "Gemini JSON truncated in round 1",
            "description": (
                "After updating to gemini-2.5-flash, round 1 of every game involving "
                "Gemini 2.5 Flash (not Lite) still produced extraction=0. "
                "raw_output showed: '{\"belief\": 10.0, \"' — JSON cut off mid-key. "
                "Round 2 onward worked correctly."
            ),
            "diagnosis": (
                "Gemini 2.5 Flash has thinking mode enabled by default. "
                "In round 1 the context is longest (full system prompt for the first time) "
                "and the model generates internal thinking tokens before producing the JSON. "
                "max_output_tokens=150 was too small: the token budget was exhausted "
                "mid-JSON after the thinking preamble consumed most of the budget."
            ),
            "fix_attempts": [
                {
                    "attempt": 1,
                    "approach": "Add thinking_budget=0 as direct parameter",
                    "result": "Ignored by the langchain-google-genai version installed — no effect",
                },
                {
                    "attempt": 2,
                    "approach": "Pass model_kwargs={'response_mime_type': 'application/json', 'generation_config': {'thinking_config': {'thinking_budget': 0}}}",
                    "result": (
                        "model_kwargs were rejected / caused different behaviour. "
                        "Gemini started responding with 'Here is the JSON requested:' "
                        "as prose prefix instead of raw JSON — worse than before."
                    ),
                },
                {
                    "attempt": 3,
                    "approach": "Improve parser to extract JSON from prose (regex search)",
                    "result": (
                        "Handled 'Here is the JSON: {...}' style responses correctly. "
                        "But the original truncation problem persisted because "
                        "the JSON was still being cut off — no complete {...} block to find."
                    ),
                },
                {
                    "attempt": 4,
                    "approach": "Raise max_output_tokens from 150 to 1024 for Gemini models",
                    "result": (
                        "SOLVED. With 1024 tokens available, Gemini completes the JSON "
                        "even after generating thinking tokens. "
                        "Round 1 now returns valid extraction > 0."
                    ),
                },
            ],
            "final_fix": "max_output_tokens=1024 for gemini_pro and gemini_flash in registry",
            "version_fixed": "v2.2",
        },
        {
            "problem_id": "P5",
            "title": "Parser fails on truncated JSON (residual edge case)",
            "description": (
                "Even at 1024 tokens, an extremely rare truncation can still produce "
                "'{\"belief\": 15.0, \"extraction\"' (value missing). "
                "json.loads() fails on incomplete JSON. "
                "The regex pattern r'\\{[^{}]*\"extraction\"[^{}]*\\}' also fails "
                "because the closing brace is absent."
            ),
            "fix": (
                "Added a second fallback in parse_response: "
                "if the full JSON regex fails, try extracting values individually: "
                "re.search(r'\"extraction\"\\s*:\\s*(\\d+)', text) for extraction, "
                "re.search(r'\"belief\"\\s*:\\s*([0-9.]+)', text) for belief. "
                "If extraction is found, return it even without a valid belief. "
                "Logs a warning: 'truncated JSON recovered extraction=X belief=Y'."
            ),
            "version_fixed": "v2.2",
        },
        {
            "problem_id": "P6",
            "title": "OpenAI models show zero behavioural variance",
            "description": (
                "GPT-4o and GPT-4o-mini both extracted exactly 10 every single round "
                "across all 20 rounds in the first runs, with extraction_1 = extraction_2 = 10 "
                "and belief = 10.0 constant. "
                "θ is unestimable when there is no threshold-crossing event."
            ),
            "diagnosis": (
                "OpenAI instruction-tuned models have a strong prior toward the 'fair' "
                "or 'cooperative' solution. Without the sustainability hint, "
                "they still infer the equilibrium from the game structure and lock onto it. "
                "This is not a bug — it is a real property of the model."
            ),
            "fix_attempt_temperature": (
                "Raised temperature from 0.6 to 1.0 for OpenAI models. "
                "Result: GPT-4o still extracted exactly 10 in all conditions. "
                "GPT-4o-mini showed some variation in the 'human' condition (avg ~14.8). "
                "GPT-4o is robust to temperature changes in this environment."
            ),
            "research_implication": (
                "GPT-4o's invariance is itself a finding: it suggests a θ ≈ 10 (exactly "
                "sustainable) that is stable across conditions and temperatures. "
                "This contrasts with Claude (θ < 10 from round 1) and Gemini (θ = 0). "
                "For the paper: temperature confound must be addressed by running "
                "all models at both T=0.6 and T=1.0 as separate experimental conditions (H3)."
            ),
            "version_fixed": "N/A — treated as data, not a bug",
        },
        {
            "problem_id": "P7",
            "title": "Mixed temperatures across model families",
            "description": (
                "To get OpenAI to show some variance, temperature was set to 1.0 for "
                "OpenAI models and left at 0.6 for Anthropic and Google. "
                "This creates a confound: differences in extraction could reflect "
                "temperature, not model architecture."
            ),
            "resolution": (
                "Pilot data with mixed temperatures treated as exploratory only. "
                "For the final analysis, planned re-runs at uniform T=0.6 and T=1.0 "
                "across all models, allowing H3 (temperature → coherence) to be "
                "tested cleanly as a controlled factor."
            ),
            "version_affected": "v2.2 pilot runs",
        },
    ],

    # ─────────────────────────────────────────────────────────────
    # 6. PARSER DESIGN (FINAL STATE)
    # ─────────────────────────────────────────────────────────────
    "parser_design": {
        "function": "parse_response(raw, label, round_num, pool_available)",
        "handles_four_response_styles": {
            "style_1": "Clean JSON: {\"belief\": 8.5, \"extraction\": 7}",
            "style_2": "Markdown fenced: ```json\\n{...}\\n```  — split on ``` and strip 'json' prefix",
            "style_3": "Prose prefix: 'Here is the JSON: {...}' — regex r'\\{[^{}]*\"extraction\"[^{}]*\\}'",
            "style_4": "Truncated JSON: '{\"belief\": 10.0, \"extract' — individual key regex fallback",
        },
        "safe_defaults": {
            "extraction": 0,
            "belief": "SUSTAINABLE_SHARE (10.0)",
            "rationale": (
                "extraction=0 is the most conservative default: "
                "a player extracting 0 never causes a pool collapse. "
                "It also ensures that failed rounds are distinguishable in the data "
                "(extraction=0 with raw_output truncated) without corrupting the pool state."
            ),
        },
        "pool_cap": (
            "extraction is capped at int(pool_available) in addition to MAX_EXTRACTION. "
            "This prevents parse errors or hallucinated values from producing "
            "negative pool sizes."
        ),
    },

    # ─────────────────────────────────────────────────────────────
    # 7. DATABASE SCHEMA DESIGN
    # ─────────────────────────────────────────────────────────────
    "database_schema": {
        "format": "SQLite + CSV (same pattern as PD)",
        "filename_convention": "cd_experiment_{condition}_{timestamp}.db / cd_results_{condition}_{timestamp}.csv",
        "key_columns_for_theta": [
            "extraction_1, extraction_2 — raw extraction choices per round",
            "pool_after_regen — pool available before extraction (denominator for sustainable share)",
            "sustainable_share — constant 10.0, stored explicitly for convenience",
            "pool_after_extraction — pool remaining after extraction",
            "pool_collapsed — 1 if total extraction exceeded pool_after_regen",
        ],
        "key_columns_for_beta": [
            "belief_1 — model 1's stated expected extraction of model 2 (float 0–20)",
            "belief_2 — model 2's stated expected extraction of model 1 (float 0–20)",
            "extraction_1, extraction_2 — actual extractions (compute MAE post-hoc)",
        ],
        "collapse_padding": (
            "If the pool collapses before TOTAL_ROUNDS, remaining rounds are filled "
            "with extraction=0, payoff=0, pool_collapsed=1 rows. "
            "This ensures every game always has exactly TOTAL_ROUNDS rows, "
            "simplifying downstream averaging and panel data analysis."
        ),
    },

    # ─────────────────────────────────────────────────────────────
    # 8. PILOT RESULTS SUMMARY
    # ─────────────────────────────────────────────────────────────
    "pilot_results": {
        "conditions_run": ["undisclosed", "ai", "human"],
        "rounds_per_game": 5,
        "total_games": 18,
        "key_findings": {
            "finding_1_opponent_sensitivity": (
                "Average extraction increases monotonically across conditions: "
                "undisclosed (~13.5) < AI (~14.1) < human (~15.3). "
                "Models extract more when they believe they face a human opponent. "
                "This indicates opponent-dependent strategies and a measurable "
                "Delta_m = ||S_AI - S_H|| > 0, supporting the perturbation test."
            ),
            "finding_2_gpt4o_invariance": (
                "GPT-4o extracts exactly 10 in every round across all conditions "
                "and temperatures (0.6 and 1.0). Belief updates correctly (rises as "
                "Claude escalates) but extraction stays fixed. "
                "GPT-4o-mini breaks from this only in the 'human' condition (~14.8 avg), "
                "showing within-family divergence under social framing."
            ),
            "finding_3_claude_gemini_escalation": (
                "Claude and Gemini converge to extraction=20 within 2–3 rounds when "
                "paired together, triggering pool collapse in rounds 5–8 across all conditions. "
                "This is a tragedy-of-the-commons equilibrium: once one model overextracts, "
                "the other follows regardless of opponent identity."
            ),
            "finding_4_within_family_architecture": (
                "Model families show distinct within-family strategic patterns: "
                "GPT-4o + GPT-4o-mini both anchor at 10 (identical θ ≈ 10), "
                "Claude Opus + Sonnet converge to a tacit equilibrium at 13 (θ < 10), "
                "Gemini Flash extracts 20 while Lite extracts 10 (asymmetric θ). "
                "This is preliminary evidence for H1 (scaling coherence within family)."
            ),
        },
        "limitations": {
            "mixed_temperatures": (
                "OpenAI models ran at T=1.0, others at T=0.6. "
                "Temperature is a confound for cross-model comparison. "
                "Clean analysis requires uniform temperature runs."
            ),
            "short_pilot": (
                "Only 5 rounds per game in the pilot. "
                "Full 20-round runs needed for robust θ estimation, "
                "especially for models that escalate slowly."
            ),
            "no_cross_environment_analysis_yet": (
                "This data covers only the Commons Dilemma. "
                "The strategic profile vector Sm = (rho, theta, eta, gamma, beta) "
                "requires data from all three environments (PD, Commons, Cheap-Talk) "
                "before coherence metrics can be computed."
            ),
        },
    },

    # ─────────────────────────────────────────────────────────────
    # 9. FILE STRUCTURE IN REPO
    # ─────────────────────────────────────────────────────────────
    "repo_files": {
        "commons_dilemma_langchain.py": "Main experiment script (v2.2, final)",
        ".env": "API keys — NOT committed (in .gitignore). Keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY",
        ".env.example": "Template showing required key names without values — committed",
        "environment.yml": "Conda environment spec (Windows-only, use pip install on Mac/Linux — see P2)",
        "cd_results_{condition}_{timestamp}.csv": "Output: round-by-round data per condition",
        "cd_experiment_{condition}_{timestamp}.db": "Output: same data in SQLite",
        "commons_experiment.log": "Runtime log — NOT committed (in .gitignore)",
    },

    # ─────────────────────────────────────────────────────────────
    # 10. HOW TO RUN
    # ─────────────────────────────────────────────────────────────
    "how_to_run": {
        "setup": [
            "git checkout feature/commons-dilemma",
            "conda activate capstone  # or create env: conda create -n capstone python=3.11 -y",
            "pip install langchain langchain-anthropic langchain-openai langchain-google-genai python-dotenv",
            "# Ensure .env exists with ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY",
        ],
        "change_condition": (
            "Edit line: OPPONENT_CONDITION = 'undisclosed'  # or 'ai' or 'human'. "
            "Output filename automatically includes the condition."
        ),
        "run": "python commons_dilemma_langchain.py",
        "outputs": [
            "cd_results_{condition}_{timestamp}.csv — main analysis file",
            "cd_experiment_{condition}_{timestamp}.db — SQLite backup",
            "commons_experiment.log — runtime log",
        ],
        "useful_sqlite_queries": [
            "SELECT matchup, AVG(extraction_1), AVG(extraction_2) FROM rounds GROUP BY matchup;",
            "SELECT matchup, SUM(pool_collapsed) as collapses FROM rounds GROUP BY matchup;",
            "SELECT matchup, MIN(pool_after_extraction) as min_pool FROM rounds GROUP BY matchup;",
            "SELECT matchup, AVG(ABS(belief_1 - extraction_2)) as beta_1 FROM rounds GROUP BY matchup;",
        ],
    },

    # ─────────────────────────────────────────────────────────────
    # 11. CONNECTION TO PAPER
    # ─────────────────────────────────────────────────────────────
    "connection_to_paper": {
        "environment_role": (
            "The Commons Dilemma is one of three structurally distinct environments "
            "used to estimate the strategic profile vector Sm = (rho, theta, eta, gamma, beta). "
            "It contributes theta (exploitation threshold) and beta (belief calibration). "
            "Together with PD (rho, beta) and Cheap-Talk (eta, gamma, beta), "
            "the three environments form the basis for cross-environment coherence analysis."
        ),
        "hypotheses_tested": {
            "H1_scaling": (
                "Larger models exhibit more stable strategic profiles across environments. "
                "Preliminary pilot: GPT-4o (large) has θ=10 stable; GPT-4o-mini shows "
                "condition-sensitivity in 'human' framing. Needs full-run confirmation."
            ),
            "H2_instruction_tuning": (
                "Instruction-tuned models show higher norm-conforming cooperation. "
                "GPT-4o's anchor at the sustainable share (10) is consistent with H2. "
                "Claude and Gemini over-extract immediately, suggesting different "
                "alignment between instruction tuning and cooperative norms in this environment."
            ),
            "H3_temperature": (
                "Higher temperatures increase behavioural variance and reduce coherence. "
                "Pilot data is confounded (mixed temperatures). "
                "Planned uniform-temperature runs will test H3 cleanly."
            ),
        },
        "perturbation_test": (
            "The opponent-disclosure conditions (undisclosed / ai / human) implement "
            "the perturbation test from the conceptual framework. "
            "Delta_m = ||Sm_AI - Sm_H|| is estimated by comparing the strategic profile "
            "vectors across conditions. "
            "Pilot shows Delta_m > 0 for Claude and GPT-4o-mini; Delta_m ≈ 0 for GPT-4o. "
            "GPT-4o's robustness to perturbation suggests a stable strategic architecture."
        ),
    },
}


if __name__ == "__main__":
    import json
    print(json.dumps(DEVELOPMENT_LOG, indent=2, ensure_ascii=False))
