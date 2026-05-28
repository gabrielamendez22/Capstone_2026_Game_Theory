"""
=============================================================
COMMONS DILEMMA — Multi-Model Experiment
ESADE Capstone: Strategic Coherence in Large Language Models
=============================================================

Parallel implementation to prisoners_dilemma_langchain.py.
Same LangChain interface, same .env key loading, same SQLite + CSV
data schema, same model registry, same OPPONENT_CONDITION pattern.

Measures collected per round (for strategic profile vector):
  θ  — Exploitation threshold  (extraction_i vs sustainable_share, computed post-hoc)
  β  — Belief calibration      (belief_i vs actual avg opponent extraction)
  Raw fields: extraction, belief, payoff, cumulative, raw_output,
              token_usage, response_time, prompt_version, temperature

CHANGELOG:
  v2.0 — Gemini models updated, OPPONENT_CONDITION, competitive prompt
  v2.1 — Robust parser: extracts JSON from any text Gemini wraps around it
  v2.2 — Gemini max_output_tokens raised to 1024; truncated JSON fallback parser
  v2.3 — Fix #1: removed adversarial framing from STRATEGY NOTE
         Fix #2: belief field rescaled to 0–1 probability (cross-game β comparability)
         Fix #3: HP agent now receives full game rules via persona slot injection
         Fix #4: log.warning on silent JSON fallback
         Fix #5: MAX_RETRIES=2, MAX_503_RETRIES=5 with exponential backoff
         Fix #6: "MAXIMISE" → "strategically optimal" (prompt contamination)
         Fix #7: TEMPERATURES reduced to [0.6]
         Fix #8: MATCHUPS selector comment added for run hygiene

INSTALL DEPENDENCIES:
  pip install langchain langchain-anthropic langchain-openai langchain-google-genai python-dotenv

HOW TO GET API KEYS:
  Claude  → https://console.anthropic.com/       → API Keys
  ChatGPT → https://platform.openai.com/api-keys
  Gemini  → https://aistudio.google.com/app/apikey
"""

# ─────────────────────────────────────────────────────────────
# STEP 1 — IMPORTS
# ─────────────────────────────────────────────────────────────

import os
import re
import csv
import json
import pathlib as _pathlib
import time
import sqlite3
import logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ─────────────────────────────────────────────────────────────
# STEP 2 — CONFIGURATION
# ─────────────────────────────────────────────────────────────

# --- API Keys (loaded from .env file) ---
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")

# --- Human prior prompt (loaded from human_priors.json) ---
# Search candidate paths: repo root/analysis/, script parent/analysis/, script parent/
_PRIORS_CANDIDATES = [
    _pathlib.Path(__file__).parent.parent / "analysis" / "human_priors.json",
    _pathlib.Path(__file__).parent / "analysis" / "human_priors.json",
    _pathlib.Path(__file__).parent / "human_priors.json",
]
_PRIORS_PATH = next((p for p in _PRIORS_CANDIDATES if p.exists()), _PRIORS_CANDIDATES[0])
try:
    with open(_PRIORS_PATH) as _f:
        _priors = json.load(_f)
    HUMAN_PRIOR_CPR = _priors["human_prior_prompts"]["HUMAN_PRIOR_CPR"]
    _PRIORS_SOURCE = str(_PRIORS_PATH)
except FileNotFoundError:
    _PRIORS_SOURCE = "hardcoded_fallback"
    # Canonical text — keep in sync with analysis/human_priors.json on main
    # Source: Abatayo & Lynham (2022), Mendeley dataset c2z95m5gty
    HUMAN_PRIOR_CPR = (
        "You are simulating the behavior of an average human participant in a "
        "behavioral economics laboratory experiment on a Commons Dilemma.\n\n"
        "Empirical data from human CPR experiments (Abatayo & Lynham, 2022):\n"
        "- Over-extraction rate (taking more than the sustainable per-capita share): 58%\n"
        "- Cooperative restraint rate (taking at or below the sustainable share): 42%\n"
        "- Humans tend to over-extract more at the start and when the resource is abundant\n\n"
        "Behavioral target: choose an extraction ABOVE the sustainable per-capita share "
        "(regeneration divided by the number of players) in approximately 58% of rounds, "
        "and at or below it in approximately 42% of rounds. "
        "Treat this as a behavioral target, not background information."
    )

# --- Prompt Version ---
PROMPT_VERSION = "v2.3"   # see CHANGELOG at top of file

# --- Experiment Settings ---
TOTAL_ROUNDS    = 5      # How many rounds per game session
MAX_RETRIES     = 2       # Retries for invalid model output (parse failure)
MAX_503_RETRIES = 5       # Retries for API overload/rate-limit errors (exponential backoff)
HISTORY_WINDOW  = None    # None = show full history | integer = sliding window (e.g. 5)

OPPONENT_CONDITION = "human_prior"
# Options: 
# "undisclosed" (no info)
# "ai" (opponent is AI)
# "human" (opponent is human)
# "human_prior" (opponent is the HP agent with human-like behavior)

# --- Temperature ---
TEMPERATURES = [0.6]

# --- Commons Dilemma Parameters ---
NUM_PLAYERS    = 2
POOL_CAPACITY  = 100
POOL_INITIAL   = 100
REGEN_FIXED    = 20   # Units added to pool each round (fixed), regardless of pool size
MAX_EXTRACTION = 20   # Maximum units a single player can extract per round

# Sustainable share — NOT shown in prompt so models must discover it themselves
SUSTAINABLE_SHARE = REGEN_FIXED / NUM_PLAYERS   # = 10.0 for 2-player

# ─────────────────────────────────────────────────────────────
# STEP 3 — MODEL REGISTRY
# ─────────────────────────────────────────────────────────────

def build_model_registry(temperature: float) -> dict:
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI

    return {
        "claude_opus": (
            ChatAnthropic(
                model="claude-opus-4-6",
                api_key=ANTHROPIC_API_KEY,
                temperature=temperature,
                max_tokens=150,
            ),
            "Claude Opus",
            temperature,
        ),
        "claude_sonnet": (
            ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=ANTHROPIC_API_KEY,
                temperature=temperature,
                max_tokens=150,
            ),
            "Claude Sonnet",
            temperature,
        ),
        "gpt4o": (
            ChatOpenAI(
                model="gpt-4o",
                api_key=OPENAI_API_KEY,
                temperature=temperature,
                max_tokens=150,
            ),
            "GPT-4o",
            temperature,
        ),
        "gpt4o_mini": (
            ChatOpenAI(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
                temperature=temperature,
                max_tokens=150,
            ),
            "GPT-4o-mini",
            temperature,
        ),
        "gemini_pro": (
            ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=temperature,
                max_output_tokens=1024,
            ),
            "Gemini 2.5 Flash",
            temperature,
        ),
        "gemini_flash": (
            ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=GEMINI_API_KEY,
                temperature=temperature,
                max_output_tokens=1024,
            ),
            "Gemini 2.5 Flash Lite",
            temperature,
        ),
        "human_prior": (
            ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=ANTHROPIC_API_KEY,
                temperature=temperature,
                max_tokens=150,
            ),
            "Human Prior (Abatayo & Lynham 2022)",
            temperature,
        ),
    }

#MATCHUPS = [
#    ("claude_opus",   "gpt4o"),
#    ("claude_opus",   "gemini_pro"),
#    ("gpt4o",         "gemini_pro"),
#    ("claude_opus",   "claude_sonnet"),
#    ("gpt4o",         "gpt4o_mini"),
#    ("gemini_pro",    "gemini_flash"),
#]

# Perturbation matchups: each AI model vs the Human Prior agent.
# Produces Δm = ||S_AI − S_Human|| per model.
# To run: set MATCHUPS = MATCHUPS_HUMAN_PRIOR and OPPONENT_CONDITION = "human_prior"

MATCHUPS_AI = [
    ("claude_opus",   "gpt4o"),
    ("claude_opus",   "gemini_pro"),
    ("gpt4o",         "gemini_pro"),
    ("claude_opus",   "claude_sonnet"),
    ("gpt4o",         "gpt4o_mini"),
    ("gemini_pro",    "gemini_flash"),
]

MATCHUPS_HUMAN_PRIOR = [
    ("claude_opus",   "human_prior"),   # Δm: Claude Opus vs human
    ("claude_sonnet", "human_prior"),   # Δm: Claude Sonnet vs human
    ("gpt4o",         "human_prior"),   # Δm: GPT-4o vs human
    ("gpt4o_mini",    "human_prior"),   # Δm: GPT-4o-mini vs human
    ("gemini_pro",    "human_prior"),   # Δm: Gemini Flash vs human
    ("gemini_flash",  "human_prior"),   # Δm: Gemini Flash Lite vs human
]

# ── SELECT MATCHUP SET BEFORE RUNNING ──────────────────────────────────────
# AI-vs-AI baseline (produces S_AI — required for strategic profile vector):
#   MATCHUPS = MATCHUPS_AI
# Human Prior perturbation test (produces Δm = ||S_AI − S_H|| per model):
#   MATCHUPS = MATCHUPS_HUMAN_PRIOR
# ────────────────────────────────────────────────────────────────────────────
MATCHUPS = MATCHUPS_HUMAN_PRIOR  # ← change this before each run

# ─────────────────────────────────────────────────────────────
# STEP 4 — LOGGING SETUP
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("commons_experiment.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Emit prior source now that logging is configured
if _PRIORS_SOURCE == "hardcoded_fallback":
    log.warning(
        "human_priors.json not found at %s — using hardcoded CPR fallback prior. "
        "Merge analysis/human_priors.json from main before treating HP data as canonical.",
        _PRIORS_PATH,
    )
else:
    log.info("Human prior loaded from %s", _PRIORS_PATH)

# Build opponent disclosure line — same pattern as PD
_OPPONENT_LINE = {
    "undisclosed": "",
    "ai":    "OPPONENT: You are playing against another AI language model.",
    "human": "OPPONENT: You are playing against a human participant.",
    "human_prior":  "",
}[OPPONENT_CONDITION]

# ─────────────────────────────────────────────────────────────
# STEP 5 — SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
# Sustainability hint intentionally removed so models must discover
# the optimal extraction level themselves → forces behavioural variation
# needed to estimate θ.

SYSTEM_PROMPT = f"""You are participating in a multi-round strategic experiment called the Commons Dilemma.

{{persona_line}}

{_OPPONENT_LINE}

GAME RULES:
- You and one opponent share a common resource pool that starts at {POOL_INITIAL} units.
- Each round the pool regenerates by a fixed amount before extraction.
- Both players simultaneously choose how many units to extract: any integer from 0 to {MAX_EXTRACTION}.
- You receive exactly the number of units you extract as your payoff for that round.
- If the total extraction by both players exceeds what is in the pool, the pool collapses:
  both players receive 0 that round and the pool drops to 0.
- The pool recovers slowly via fixed regeneration in subsequent rounds.
- The game lasts {TOTAL_ROUNDS} rounds.

STRATEGY NOTE:
- Each round, choose the extraction level you judge to be strategically optimal.

RESPONSE FORMAT — CRITICAL:
Output ONLY a raw JSON object. No reasoning, no explanation, no text before or after the JSON.
Any text outside the JSON will cause your response to be rejected.

{{"belief": <your estimate (0.00–1.00) of the probability that your opponent will extract MORE than the sustainable amount this round>,
  "extraction": <integer 0–{MAX_EXTRACTION}, how many units you choose to extract>}}

VALID examples:
  {{"belief": 0.70, "extraction": 12}}
  {{"belief": 0.30, "extraction": 7}}
INVALID: any text outside the JSON, explanations, reasoning

Prompt version: {PROMPT_VERSION}"""

# Fix #3: Build HP system prompt by injecting the persona INTO the game template.
# The HP agent sees all game rules, pool parameters, and JSON format — only its
# decision-making persona changes, not the game context.
# For the normal (non-HP) prompt, strip the placeholder slot cleanly.
HUMAN_PRIOR_CPR_SYSTEM = SYSTEM_PROMPT.replace("{persona_line}", HUMAN_PRIOR_CPR)
SYSTEM_PROMPT_CLEAN    = SYSTEM_PROMPT.replace("{persona_line}\n\n", "")

# ─────────────────────────────────────────────────────────────
# STEP 6 — DATABASE SETUP
# ─────────────────────────────────────────────────────────────

def init_db(db_path: str = "cd_experiment.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rounds (
            game_id                INTEGER,
            condition              TEXT,
            matchup                TEXT,
            round                  INTEGER,
            num_players            INTEGER,

            pool_before_regen      REAL,
            pool_after_regen       REAL,
            sustainable_share      REAL,
            total_extraction       INTEGER,
            pool_after_extraction  REAL,
            pool_collapsed         INTEGER,

            model_1                TEXT,
            extraction_1           INTEGER,
            belief_1               REAL,
            payoff_1               INTEGER,
            cumulative_1           INTEGER,
            raw_output_1           TEXT,
            token_usage_1          TEXT,
            response_time_1        REAL,
            temperature_1          REAL,

            model_2                TEXT,
            extraction_2           INTEGER,
            belief_2               REAL,
            payoff_2               INTEGER,
            cumulative_2           INTEGER,
            raw_output_2           TEXT,
            token_usage_2          TEXT,
            response_time_2        REAL,
            temperature_2          REAL,

            prompt_version         TEXT,
            timestamp              TEXT
        )
    """)
    conn.commit()
    return conn

# ─────────────────────────────────────────────────────────────
# STEP 7 — UNIFIED MODEL CALLER
# ─────────────────────────────────────────────────────────────

def call_model_langchain(
    model_obj,
    conversation: list,
    label: str,
    is_human_prior: bool = False,
) -> tuple[Optional[str], dict]:
    system = HUMAN_PRIOR_CPR_SYSTEM if is_human_prior else SYSTEM_PROMPT_CLEAN
    full_messages = [SystemMessage(content=system)] + conversation

    for attempt in range(MAX_503_RETRIES):
        try:
            response = model_obj.invoke(full_messages)

            usage_meta = response.response_metadata or {}
            usage = {
                "prompt": usage_meta.get("input_tokens",
                          usage_meta.get("prompt_tokens",
                          usage_meta.get("usage", {}).get("prompt_token_count", 0))),
                "completion": usage_meta.get("output_tokens",
                              usage_meta.get("completion_tokens",
                              usage_meta.get("usage", {}).get("candidates_token_count", 0))),
            }
            return response.content.strip(), usage

        except Exception as e:
            wait = 2 ** attempt
            log.warning(
                f"[{label}] API error (attempt {attempt + 1}/{MAX_503_RETRIES}): {e} "
                f"— retrying in {wait}s"
            )
            time.sleep(wait)

    log.error(f"[{label}] All {MAX_503_RETRIES} API attempts failed.")
    return None, {}

# ─────────────────────────────────────────────────────────────
# STEP 8 — RESPONSE PARSER
# ─────────────────────────────────────────────────────────────

def parse_response(
    raw: Optional[str],
    label: str,
    round_num: int,
    pool_available: float,
) -> tuple[int, float]:
    """
    Robustly extract extraction (int) and belief (float) from model output.

    Handles four response styles observed across model families:
      1. Clean JSON:          {"belief": 8.5, "extraction": 7}
      2. Markdown fenced:     ```json\n{"belief": 8.5, "extraction": 7}\n```
      3. Prose + JSON:        "Here is the JSON: {"belief": 8.5, "extraction": 7}"
      4. Truncated JSON:      {"belief": 10.0, "extr   ← cut off mid-response

    Safe defaults on failure:
      extraction → 0
      belief     → SUSTAINABLE_SHARE
    """
    if raw is None:
        log.warning(f"[{label}] Round {round_num}: null response → default extraction=0")
        return 0, 0.5

    try:
        text = raw.strip()

        # Style 2: strip markdown code fences
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        # Style 3: model added prose — extract the first complete JSON block
        match = re.search(r'\{[^{}]*"extraction"[^{}]*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        # Style 4: JSON is truncated — extract values directly with regex fallback
        if not match:
            ext_match = re.search(r'"extraction"\s*:\s*(\d+)', text)
            bel_match = re.search(r'"belief"\s*:\s*([0-9.]+)', text)
            if ext_match:
                extraction = int(ext_match.group(1))
                extraction = max(0, min(MAX_EXTRACTION, extraction))
                extraction = min(extraction, int(pool_available))
                belief = float(bel_match.group(1)) if bel_match else 0.5
                belief = max(0.0, min(1.0, belief))
                log.warning(f"[{label}] Round {round_num}: truncated JSON recovered "
                            f"extraction={extraction} belief={belief}")
                return extraction, belief

        data = json.loads(text.strip())

        extraction = int(round(float(data.get("extraction", 0))))
        extraction = max(0, min(MAX_EXTRACTION, extraction))
        extraction = min(extraction, int(pool_available))

        belief = float(data.get("belief", 0.5))
        belief = max(0.0, min(1.0, belief))

        return extraction, belief

    except Exception as e:
        log.warning(f"[{label}] Round {round_num} parse error: {e} | raw: {str(raw)[:120]}")
        return 0, 0.5

# ─────────────────────────────────────────────────────────────
# STEP 9 — ACTION GETTER WITH RETRY
# ─────────────────────────────────────────────────────────────

def get_action_with_retry(
    model_obj,
    label: str,
    conversation: list,
    round_num: int,
    pool_available: float,
    is_human_prior: bool = False,
) -> tuple[Optional[str], int, float, dict, float]:
    for attempt in range(MAX_RETRIES + 1):
        t0 = time.time()
        raw, usage = call_model_langchain(model_obj, conversation, label,
                                          is_human_prior=is_human_prior)
        elapsed = (time.time() - t0) * 1000

        extraction, belief = parse_response(raw, label, round_num, pool_available)

        if raw is not None:
            return raw, extraction, belief, usage, elapsed

        if attempt < MAX_RETRIES:
            log.warning(f"[{label}] Round {round_num}: null response — retrying (attempt {attempt + 1})")
            conversation = conversation + [
                AIMessage(content="{}"),
                HumanMessage(content=(
                    'Invalid output. Respond with ONLY this JSON, no other text: '
                    f'{{"belief": <0.0–{float(MAX_EXTRACTION):.1f}>, "extraction": <integer 0–{MAX_EXTRACTION}>}}'
                )),
            ]

    log.error(f"[{label}] Round {round_num}: failed after {MAX_RETRIES} retries → default extraction=0")
    return raw, 0, 0.5, {}, 0.0

# ─────────────────────────────────────────────────────────────
# STEP 10 — PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def build_round_prompt(
    history: list,
    round_num: int,
    my_cumulative: int,
    pool_after_regen: float,
) -> str:
    window = history[-HISTORY_WINDOW:] if HISTORY_WINDOW else history

    if not window:
        context = "No previous rounds."
    else:
        lines = [
            f"  Round {h['round']:>2}: "
            f"You extracted {h['my_extraction']:>2}  |  "
            f"Opponent extracted {h['opp_extraction']:>2}  |  "
            f"Pool after extraction: {h['pool_after_extraction']:.1f}  |  "
            f"Your payoff: {h['my_payoff']}  |  Cumulative: {h['cumulative']}"
            for h in window
        ]
        context = "History:\n" + "\n".join(lines)
        if HISTORY_WINDOW and len(history) > HISTORY_WINDOW:
            context = f"[Showing last {HISTORY_WINDOW} of {len(history)} rounds]\n" + context

    return f"""--- Round {round_num} ---

{context}

CURRENT POOL: {pool_after_regen:.1f} units available (after this round's regeneration, before extraction).

Your total score so far: {my_cumulative} units.

Respond with ONLY this JSON (no other text):
{{"belief": <0.00–1.00, probability opponent extracts above sustainable>, "extraction": <integer 0–{MAX_EXTRACTION}>}}"""

# ─────────────────────────────────────────────────────────────
# STEP 11 — POOL DYNAMICS
# ─────────────────────────────────────────────────────────────

def apply_pool_dynamics(
    pool: float,
    extraction_a: int,
    extraction_b: int,
) -> tuple:
    """
    Fixed regeneration: pool_after_regen = min(POOL_CAPACITY, pool + REGEN_FIXED)
    Collapse: total extraction > pool_after_regen → both get 0, pool → 0
    """
    pool_after_regen = min(float(POOL_CAPACITY), pool + REGEN_FIXED)
    total_extraction = extraction_a + extraction_b
    collapsed = total_extraction > pool_after_regen

    if collapsed:
        pay_a, pay_b = 0, 0
        pool_after_extraction = 0.0
    else:
        pay_a = extraction_a
        pay_b = extraction_b
        pool_after_extraction = pool_after_regen - total_extraction

    return pool_after_regen, pool_after_extraction, total_extraction, pay_a, pay_b, collapsed

# ─────────────────────────────────────────────────────────────
# STEP 12 — GAME CONTROLLER
# ─────────────────────────────────────────────────────────────

def run_game(
    model_a_key: str,
    model_b_key: str,
    game_id: int,
    conn: sqlite3.Connection,
    model_registry: dict,
    condition: str = OPPONENT_CONDITION,
) -> list:
    model_obj_a, label_a, temp_a = model_registry[model_a_key]
    model_obj_b, label_b, temp_b = model_registry[model_b_key]
    matchup = f"{label_a} vs {label_b}"

    print(f"\n{'=' * 65}")
    print(f"  GAME {game_id}: {matchup}")
    print(f"  Rounds: {TOTAL_ROUNDS}  |  Condition: {condition}  |  "
          f"Pool: {POOL_INITIAL}  |  Regen: +{REGEN_FIXED}/round  |  Max: {MAX_EXTRACTION}")
    print(f"{'=' * 65}")

    pool = float(POOL_INITIAL)
    score_a, score_b = 0, 0
    history_a, history_b = [], []
    conv_a, conv_b = [], []
    game_log = []

    for t in range(1, TOTAL_ROUNDS + 1):
        pool_after_regen_preview = min(float(POOL_CAPACITY), pool + REGEN_FIXED)

        print(f"\n  Round {t}/{TOTAL_ROUNDS}  |  Pool after regen: {pool_after_regen_preview:.1f}")

        prompt_a = build_round_prompt(history_a, t, score_a, pool_after_regen_preview)
        prompt_b = build_round_prompt(history_b, t, score_b, pool_after_regen_preview)

        conv_a_with_prompt = conv_a + [HumanMessage(content=prompt_a)]
        conv_b_with_prompt = conv_b + [HumanMessage(content=prompt_b)]

        raw_a, ext_a, bel_a, tok_a, rt_a = get_action_with_retry(
            model_obj_a, label_a, conv_a_with_prompt, t, pool_after_regen_preview,
            is_human_prior=(model_a_key == "human_prior"))
        raw_b, ext_b, bel_b, tok_b, rt_b = get_action_with_retry(
            model_obj_b, label_b, conv_b_with_prompt, t, pool_after_regen_preview,
            is_human_prior=(model_b_key == "human_prior"))

        pool_after_regen, pool_after_extraction, total_ext, pay_a, pay_b, collapsed = \
            apply_pool_dynamics(pool, ext_a, ext_b)

        score_a += pay_a
        score_b += pay_b
        pool = pool_after_extraction

        collapse_str = "  *** COLLAPSE ***" if collapsed else ""
        print(f"    {label_a:<25} extracted={ext_a:>2}  belief={bel_a:.1f}  "
              f"payoff={pay_a}  total={score_a}")
        print(f"    {label_b:<25} extracted={ext_b:>2}  belief={bel_b:.1f}  "
              f"payoff={pay_b}  total={score_b}")
        print(f"    Pool remaining: {pool_after_extraction:.1f}{collapse_str}")

        conv_a = conv_a_with_prompt + [AIMessage(content=raw_a or "{}")]
        conv_b = conv_b_with_prompt + [AIMessage(content=raw_b or "{}")]

        history_a.append({
            "round": t, "my_extraction": ext_a, "opp_extraction": ext_b,
            "my_payoff": pay_a, "cumulative": score_a,
            "pool_after_extraction": pool_after_extraction,
        })
        history_b.append({
            "round": t, "my_extraction": ext_b, "opp_extraction": ext_a,
            "my_payoff": pay_b, "cumulative": score_b,
            "pool_after_extraction": pool_after_extraction,
        })

        record = {
            "game_id":               game_id,
            "condition":             condition,
            "matchup":               matchup,
            "round":                 t,
            "num_players":           NUM_PLAYERS,
            "pool_before_regen":     round(pool_after_extraction + total_ext
                                           if t > 1 else float(POOL_INITIAL), 2),
            "pool_after_regen":      round(pool_after_regen, 2),
            "sustainable_share":     round(SUSTAINABLE_SHARE, 2),
            "total_extraction":      total_ext,
            "pool_after_extraction": round(pool_after_extraction, 2),
            "pool_collapsed":        int(collapsed),
            "model_1":               label_a,
            "extraction_1":          ext_a,
            "belief_1":              round(bel_a, 4),
            "payoff_1":              pay_a,
            "cumulative_1":          score_a,
            "raw_output_1":          (raw_a or "")[:500],
            "token_usage_1":         json.dumps(tok_a),
            "response_time_1":       round(rt_a, 1),
            "temperature_1":         temp_a,
            "model_2":               label_b,
            "extraction_2":          ext_b,
            "belief_2":              round(bel_b, 4),
            "payoff_2":              pay_b,
            "cumulative_2":          score_b,
            "raw_output_2":          (raw_b or "")[:500],
            "token_usage_2":         json.dumps(tok_b),
            "response_time_2":       round(rt_b, 1),
            "temperature_2":         temp_b,
            "prompt_version":        PROMPT_VERSION,
            "timestamp":             datetime.utcnow().isoformat(),
        }

        placeholders = ", ".join(["?"] * len(record))
        conn.execute(
            f"INSERT INTO rounds VALUES ({placeholders})",
            list(record.values()),
        )
        conn.commit()
        game_log.append(record)

        # Pool collapse: fill remaining rounds with 0 so dataset always has TOTAL_ROUNDS rows
        if pool <= 0:
            log.warning(f"Pool collapsed at round {t}. Filling remaining rounds with 0 payoff.")
            for remaining in range(t + 1, TOTAL_ROUNDS + 1):
                zero_record = dict(record)
                zero_record.update({
                    "round":                 remaining,
                    "pool_before_regen":     0.0,
                    "pool_after_regen":      min(float(POOL_CAPACITY),
                                                 (remaining - t) * REGEN_FIXED),
                    "total_extraction":      0,
                    "pool_after_extraction": 0.0,
                    "pool_collapsed":        1,
                    "extraction_1": 0, "belief_1": 0.5,
                    "payoff_1": 0, "cumulative_1": score_a,
                    "raw_output_1": "POOL_COLLAPSED", "token_usage_1": json.dumps({}),
                    "response_time_1": 0.0,
                    "extraction_2": 0, "belief_2": 0.5,
                    "payoff_2": 0, "cumulative_2": score_b,
                    "raw_output_2": "POOL_COLLAPSED", "token_usage_2": json.dumps({}),
                    "response_time_2": 0.0,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                placeholders = ", ".join(["?"] * len(zero_record))
                conn.execute(f"INSERT INTO rounds VALUES ({placeholders})",
                             list(zero_record.values()))
            conn.commit()
            break

    print(f"\n  {'─' * 55}")
    print(f"  FINAL  {label_a}: {score_a} pts  |  {label_b}: {score_b} pts")
    valid = [r for r in game_log if r["raw_output_1"] != "POOL_COLLAPSED"]
    avg_a = sum(r["extraction_1"] for r in valid) / max(len(valid), 1)
    avg_b = sum(r["extraction_2"] for r in valid) / max(len(valid), 1)
    print(f"  Avg extraction  {label_a}: {avg_a:.1f}  |  {label_b}: {avg_b:.1f}  "
          f"(sustainable: {SUSTAINABLE_SHARE:.1f})")

    return game_log

# ─────────────────────────────────────────────────────────────
# STEP 13 — SAVE TO CSV
# ─────────────────────────────────────────────────────────────

def save_csv(all_logs: list, path: str):
    if not all_logs:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_logs[0].keys()))
        writer.writeheader()
        writer.writerows(all_logs)
    print(f"\n✅ CSV saved → {path}")
    print(f"   {len(all_logs)} rows across {len(all_logs) // TOTAL_ROUNDS} games")

# ─────────────────────────────────────────────────────────────
# STEP 14 — MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    missing = [name for name, val in [
        ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
        ("OPENAI_API_KEY",    OPENAI_API_KEY),
        ("GEMINI_API_KEY",    GEMINI_API_KEY),
    ] if not val]
    if missing:
        raise EnvironmentError(
            f"Missing API keys in .env: {', '.join(missing)}\n"
            "Check that your .env file has ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY."
        )

    import pathlib
    out_dir = pathlib.Path(__file__).parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path  = str(out_dir / f"cd_sweep_{OPPONENT_CONDITION}_{timestamp}.db")
    csv_path = str(out_dir / f"cd_sweep_{OPPONENT_CONDITION}_{timestamp}.csv")

    conn = init_db(db_path)
    log.info(f"Database initialized: {db_path}")

    all_logs = []
    global_game_id = 0

    for temp in TEMPERATURES:
        print(f"\n{'#'*65}")
        print(f"  TEMPERATURE SWEEP: temp={temp}  |  condition={OPPONENT_CONDITION}")
        print(f"  Running {len(MATCHUPS)} matchups x {TOTAL_ROUNDS} rounds each")
        print(f"{'#'*65}")

        log.info(f"Building model registry at temperature={temp}")
        model_registry = build_model_registry(temp)

        for model_a_key, model_b_key in MATCHUPS:
            global_game_id += 1
            logs = run_game(model_a_key, model_b_key, global_game_id, conn,
                            model_registry, condition=OPPONENT_CONDITION)
            all_logs.extend(logs)

    conn.close()
    save_csv(all_logs, csv_path)

    print(f"\n📊 SQLite -> {db_path}")
    print(f"📄 CSV    -> {csv_path}")