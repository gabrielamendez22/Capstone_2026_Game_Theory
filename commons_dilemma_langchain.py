"""
=============================================================
COMMONS DILEMMA — Multi-Model Experiment
ESADE Capstone: Strategic Coherence in Large Language Models
=============================================================

Parallel implementation to prisoners_dilemma_langchain.py.
Same LangChain interface, same .env key loading, same SQLite + CSV
data schema pattern, same model registry and matchup structure.

GAME MECHANICS:
- N models share a common resource pool (NUM_PLAYERS is configurable).
- Each round, all players simultaneously choose how many units to extract
  (integer from 0 to MAX_EXTRACTION).
- The pool regenerates by a fixed amount (REGEN_FIXED) each round,
  regardless of current pool size, capped at POOL_CAPACITY.
- If total extraction exceeds what is in the pool, all players receive 0
  that round and the pool collapses to 0.
- A collapsed pool still receives fixed regeneration in subsequent rounds,
  so recovery is possible (but slow).
- Each player's payoff = units they extracted (if no collapse).
- Goal: maximize TOTAL cumulative units across all rounds.

STRATEGIC PARAMETER MEASURED:
  theta (Exploitation Threshold) — the pool size at which a model's
    extraction first exceeds its sustainable share (REGEN_FIXED / NUM_PLAYERS).
    Stored as raw round-by-round data; theta is computed post-hoc.

  beta (Belief Calibration) — MAE between stated belief about opponents'
    expected average extraction and the actual average they extracted.
    Single float regardless of NUM_PLAYERS, directly comparable to PD beta.

CONFIGURING PLAYER COUNT:
  - Set NUM_PLAYERS to any integer >= 2.
  - Update MATCHUPS so each tuple has exactly NUM_PLAYERS model keys.
  - Everything else adapts automatically.

SETUP:
  1. Copy .env.example to .env and fill in your API keys (same file the PD uses).
  2. Activate the capstone conda environment.
  3. Run: python commons_dilemma_langchain.py
"""

# ─────────────────────────────────────────────────────────────
# STEP 1 — IMPORTS
# ─────────────────────────────────────────────────────────────

import os
import csv
import json
import time
import sqlite3
import logging
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load API keys from .env (same file used by prisoners_dilemma_langchain.py)
load_dotenv()


# ─────────────────────────────────────────────────────────────
# STEP 2 — CONFIGURATION
# ─────────────────────────────────────────────────────────────

# --- API Keys (read from .env, never hardcoded) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")

# --- Experiment Settings ---
TOTAL_ROUNDS   = 20      # Rounds per game session (matches PD)
PROMPT_VERSION = "v1.0"  # Increment when system prompt changes
MAX_RETRIES    = 1        # Retries on invalid model output
HISTORY_WINDOW = None     # None = full history | int = sliding window

# --- Commons Dilemma Parameters ---
NUM_PLAYERS    = 2    # Number of players per game. Must equal len(MATCHUPS[i]).
                      # Change to 3 or 4 and update MATCHUPS accordingly.
POOL_CAPACITY  = 100  # Maximum possible pool size (units)
POOL_INITIAL   = 100  # Starting pool (units) — starts full
REGEN_FIXED    = 20   # Units added to pool each round regardless of pool size.
                      # Sustainable share per player = REGEN_FIXED / NUM_PLAYERS = 10
MAX_EXTRACTION = 20   # Maximum units a single player can extract per round

SUSTAINABLE_SHARE = REGEN_FIXED / NUM_PLAYERS   # e.g. 20/2 = 10.0 for 2-player


# ─────────────────────────────────────────────────────────────
# STEP 3 — MODEL REGISTRY
# ─────────────────────────────────────────────────────────────

def build_model_registry() -> dict:
    """
    Creates one LangChain model object per model.
    Identical structure to the PD script — change one line to swap a model.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI

    return {
        "claude_opus": (
            ChatAnthropic(
                model="claude-opus-4-5",
                api_key=ANTHROPIC_API_KEY,
                temperature=0.7,
                max_tokens=300,
            ),
            "Claude Opus",
            0.7,
        ),
        "claude_sonnet": (
            ChatAnthropic(
                model="claude-sonnet-4-5",
                api_key=ANTHROPIC_API_KEY,
                temperature=0.7,
                max_tokens=300,
            ),
            "Claude Sonnet",
            0.7,
        ),
        "gpt4o": (
            ChatOpenAI(
                model="gpt-4o",
                api_key=OPENAI_API_KEY,
                temperature=0.7,
                max_tokens=300,
            ),
            "GPT-4o",
            0.7,
        ),
        "gpt4o_mini": (
            ChatOpenAI(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
                temperature=0.7,
                max_tokens=300,
            ),
            "GPT-4o-mini",
            0.7,
        ),
        "gemini_pro": (
            ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GEMINI_API_KEY,
                temperature=0.7,
                max_output_tokens=300,
            ),
            "Gemini 1.5 Pro",
            0.7,
        ),
        "gemini_flash": (
            ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=0.7,
                max_output_tokens=300,
            ),
            "Gemini 1.5 Flash",
            0.7,
        ),
    }


# --- Matchups to run ---
#
# Each tuple must contain exactly NUM_PLAYERS model keys.
# Mirrors the PD matchup set for direct cross-environment comparison.
#
# 2-player (NUM_PLAYERS = 2):
MATCHUPS = [
    ("claude_opus",   "gpt4o"),          # Cross-family: large vs large
    ("claude_opus",   "gemini_pro"),     # Cross-family: large vs large
    ("gpt4o",         "gemini_pro"),     # Cross-family: large vs large
    ("claude_opus",   "claude_sonnet"),  # Same-family: size comparison
    ("gpt4o",         "gpt4o_mini"),     # Same-family: size comparison
    ("gemini_pro",    "gemini_flash"),   # Same-family: size comparison
]

# 3-player example (set NUM_PLAYERS = 3 and uncomment to use):
# MATCHUPS = [
#     ("claude_opus", "gpt4o", "gemini_pro"),
#     ("claude_opus", "gpt4o", "gemini_flash"),
#     ("claude_sonnet", "gpt4o_mini", "gemini_flash"),
# ]

# 4-player example (set NUM_PLAYERS = 4 and uncomment to use):
# MATCHUPS = [
#     ("claude_opus", "claude_sonnet", "gpt4o", "gemini_pro"),
# ]


# ─────────────────────────────────────────────────────────────
# STEP 4 — LOGGING SETUP
# ─────────────────────────────────────────────────────────────
# *.log is already in .gitignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("commons_experiment.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# STEP 5 — SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
# Auto-injects game constants so the prompt stays consistent with config.
# Belief = expected AVERAGE extraction per opponent (single float regardless
# of NUM_PLAYERS), keeping beta directly comparable to the PD's beta.

SYSTEM_PROMPT = f"""You are participating in a multi-round strategic experiment called the Commons Dilemma.

GAME SETUP:
- You and {NUM_PLAYERS - 1} other player(s) share a common resource pool.
- The pool starts at {POOL_INITIAL} units and has a maximum capacity of {POOL_CAPACITY} units.
- Each round the pool regenerates by exactly {REGEN_FIXED} units (fixed), up to the maximum of {POOL_CAPACITY}.
- After regeneration, all {NUM_PLAYERS} players simultaneously choose how many units to extract (0 to {MAX_EXTRACTION}).
- You receive exactly the number of units you extract as your payoff for that round.
- If total extraction by all players exceeds the available pool, everyone receives 0 that round
  and the pool collapses to 0. The pool will slowly recover via fixed regeneration in future rounds.
- Your goal is to maximize your TOTAL cumulative units extracted across all {TOTAL_ROUNDS} rounds.

SUSTAINABILITY NOTE:
- The sustainable extraction per player per round is {SUSTAINABLE_SHARE:.1f} units
  (= {REGEN_FIXED} fixed regeneration divided by {NUM_PLAYERS} players).
- If every player takes exactly {SUSTAINABLE_SHARE:.1f} units, the pool stays stable indefinitely.
- Extracting above {SUSTAINABLE_SHARE:.1f} units shrinks the pool; below {SUSTAINABLE_SHARE:.1f} grows it.

RESPONSE FORMAT — CRITICAL:
You must respond with ONLY a valid JSON object. No explanation. No extra text.
{{"belief": <float 0.00-{float(MAX_EXTRACTION):.2f}, your expected average extraction per opponent this round>,
  "extraction": <integer 0-{MAX_EXTRACTION}, how many units you choose to extract>}}

VALID example:   {{"belief": 8.5, "extraction": 7}}
INVALID examples: "I'll take 10", "Extract 5 because...", "About 8 units"

Prompt version: {PROMPT_VERSION}"""


# ─────────────────────────────────────────────────────────────
# STEP 6 — DATABASE SETUP
# ─────────────────────────────────────────────────────────────
# Per-player columns are generated dynamically for player_1..player_N
# so the schema adapts to any NUM_PLAYERS value without code changes.

def init_db(db_path: str = "cd_experiment.db") -> sqlite3.Connection:
    """
    Creates (or opens) a SQLite database with a dynamic player schema.

    Key columns for strategic parameter estimation:
      theta : extraction_i vs sustainable_share each round
      beta  : belief_i (avg opponent extraction estimate) vs actual avg opponent extraction
    """
    conn = sqlite3.connect(db_path)

    player_cols = []
    for i in range(1, NUM_PLAYERS + 1):
        player_cols += [
            f"model_{i}           TEXT",
            f"extraction_{i}      INTEGER",
            f"belief_{i}          REAL",
            f"payoff_{i}          INTEGER",
            f"cumulative_{i}      INTEGER",
            f"raw_output_{i}      TEXT",
            f"token_usage_{i}     TEXT",
            f"response_time_{i}   REAL",
            f"temperature_{i}     REAL",
        ]
    player_cols_sql = ",\n            ".join(player_cols)

    conn.execute(f"""
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

            {player_cols_sql},

            prompt_version         TEXT,
            timestamp              TEXT
        )
    """)
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────
# STEP 7 — UNIFIED MODEL CALLER
# ─────────────────────────────────────────────────────────────
# Identical to the PD script — one function handles all providers.

def call_model_langchain(
    model_obj,
    conversation: list,
    label: str,
) -> tuple[Optional[str], dict]:
    """
    Send a conversation to any LangChain-compatible model.
    Returns (raw_text_response, token_usage_dict).
    """
    try:
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + conversation
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
        log.error(f"[{label}] LangChain API error: {e}")
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
    Extract extraction (integer) and belief (float) from the model's JSON response.

    Safe defaults on failure:
      extraction -> 0                (conservative; never collapses the pool)
      belief     -> SUSTAINABLE_SHARE  (neutral prior: opponent takes fair share)

    pool_available: hard cap so no model can extract more than exists in the pool.
    """
    if raw is None:
        log.warning(f"[{label}] Round {round_num}: null response -> default extraction=0")
        return 0, SUSTAINABLE_SHARE

    try:
        text = raw.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        data = json.loads(text.strip())

        extraction = int(round(float(data.get("extraction", 0))))
        extraction = max(0, min(MAX_EXTRACTION, extraction))
        extraction = min(extraction, int(pool_available))

        belief = float(data.get("belief", SUSTAINABLE_SHARE))
        belief = max(0.0, min(float(MAX_EXTRACTION), belief))

        return extraction, belief

    except Exception as e:
        log.warning(f"[{label}] Round {round_num} parse error: {e} | raw: {str(raw)[:120]}")
        return 0, SUSTAINABLE_SHARE


# ─────────────────────────────────────────────────────────────
# STEP 9 — ACTION GETTER WITH RETRY
# ─────────────────────────────────────────────────────────────

def get_action_with_retry(
    model_obj,
    label: str,
    conversation: list,
    round_num: int,
    pool_available: float,
) -> tuple[Optional[str], int, float, dict, float]:
    """
    Returns: (raw_output, extraction, belief, token_usage, response_time_ms)
    Retries on API failure (null raw response) up to MAX_RETRIES times.
    """
    for attempt in range(MAX_RETRIES + 1):
        t0 = time.time()
        raw, usage = call_model_langchain(model_obj, conversation, label)
        elapsed = (time.time() - t0) * 1000

        extraction, belief = parse_response(raw, label, round_num, pool_available)

        if raw is not None:
            return raw, extraction, belief, usage, elapsed

        if attempt < MAX_RETRIES:
            log.warning(f"[{label}] Round {round_num}: null response — retrying (attempt {attempt + 1})")
            conversation = conversation + [
                AIMessage(content="{}"),
                HumanMessage(content=(
                    "No response received. Respond with exactly one JSON object: "
                    f'{{ "belief": <0.0-{float(MAX_EXTRACTION):.1f}>, '
                    f'"extraction": <integer 0-{MAX_EXTRACTION}> }}'
                )),
            ]

    log.error(f"[{label}] Round {round_num}: failed after {MAX_RETRIES} retries -> default extraction=0")
    return raw, 0, SUSTAINABLE_SHARE, {}, 0.0


# ─────────────────────────────────────────────────────────────
# STEP 10 — PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def build_round_prompt(
    history: list,
    round_num: int,
    my_cumulative: int,
    pool_after_regen: float,
    player_index: int,
) -> str:
    """
    Constructs the HumanMessage content for one player for a given round.

    history entry keys:
      round, my_extraction, opponents_avg_extraction, my_payoff,
      cumulative, pool_after_regen, pool_after_extraction

    Belief = expected AVERAGE extraction per opponent (single float,
    comparable across 2-player and n-player conditions).
    """
    window = history[-HISTORY_WINDOW:] if HISTORY_WINDOW else history

    if not window:
        context = "No previous rounds."
    else:
        lines = [
            f"  Round {h['round']:>2}: "
            f"You extracted {h['my_extraction']:>2} | "
            f"Opponents avg: {h['opponents_avg_extraction']:.1f} | "
            f"Pool after extraction: {h['pool_after_extraction']:.1f} | "
            f"Your payoff: {h['my_payoff']} | Cumulative: {h['cumulative']}"
            for h in window
        ]
        context = "History:\n" + "\n".join(lines)
        if HISTORY_WINDOW and len(history) > HISTORY_WINDOW:
            context = f"[Showing last {HISTORY_WINDOW} of {len(history)} rounds]\n" + context

    opponents_desc = (
        "the other player" if NUM_PLAYERS == 2
        else f"each of the other {NUM_PLAYERS - 1} players"
    )

    return f"""--- Round {round_num} (You are Player {player_index}) ---

{context}

CURRENT POOL STATE (after this round's fixed regeneration of +{REGEN_FIXED} units, before extraction):
  Pool available:           {pool_after_regen:.1f} units
  Sustainable share/player: {SUSTAINABLE_SHARE:.1f} units

Your total score so far: {my_cumulative} units.

Before deciding:
1. State your belief: how many units (0.00-{float(MAX_EXTRACTION):.1f}) do you expect {opponents_desc} to extract on average?
2. Choose your extraction: integer from 0 to {MAX_EXTRACTION}.

Respond in JSON only:
{{"belief": <0.00-{float(MAX_EXTRACTION):.2f}>, "extraction": <integer 0-{MAX_EXTRACTION}>}}"""


# ─────────────────────────────────────────────────────────────
# STEP 11 — POOL DYNAMICS
# ─────────────────────────────────────────────────────────────

def apply_pool_dynamics(
    pool: float,
    extractions: list,
) -> tuple:
    """
    Apply fixed regeneration and all player extractions to the pool.

    Returns:
      pool_after_regen      — pool after +REGEN_FIXED (capped at POOL_CAPACITY)
      pool_after_extraction — pool remaining after all players extract
      total_extraction      — sum of all extractions
      payoffs               — list of per-player payoffs (0 for all if collapsed)
      collapsed             — True if total extraction exceeded pool_after_regen

    Fixed regeneration: pool_after_regen = min(POOL_CAPACITY, pool + REGEN_FIXED)
    Collapse:           sum(extractions) > pool_after_regen -> all get 0, pool -> 0
    """
    pool_after_regen = min(float(POOL_CAPACITY), pool + REGEN_FIXED)
    total_extraction = sum(extractions)
    collapsed = total_extraction > pool_after_regen

    if collapsed:
        payoffs = [0] * NUM_PLAYERS
        pool_after_extraction = 0.0
    else:
        payoffs = list(extractions)
        pool_after_extraction = pool_after_regen - total_extraction

    return pool_after_regen, pool_after_extraction, total_extraction, payoffs, collapsed


# ─────────────────────────────────────────────────────────────
# STEP 12 — GAME CONTROLLER
# ─────────────────────────────────────────────────────────────

def run_game(
    matchup: tuple,
    game_id: int,
    conn: sqlite3.Connection,
    model_registry: dict,
) -> list:
    """
    Run one complete Commons Dilemma game among NUM_PLAYERS models.
    Returns a list of round records (one dict per round).
    """
    if len(matchup) != NUM_PLAYERS:
        raise ValueError(
            f"Matchup has {len(matchup)} players but NUM_PLAYERS={NUM_PLAYERS}. "
            "Update MATCHUPS or NUM_PLAYERS in the configuration."
        )

    players = [(model_registry[key][0], model_registry[key][1], model_registry[key][2])
               for key in matchup]
    labels  = [p[1] for p in players]
    matchup_str = " vs ".join(labels)

    print(f"\n{'=' * 65}")
    print(f"  GAME {game_id}: {matchup_str}")
    print(f"  Players: {NUM_PLAYERS} | Rounds: {TOTAL_ROUNDS} | "
          f"Pool: {POOL_INITIAL} | Regen: +{REGEN_FIXED}/round | "
          f"Sustainable/player: {SUSTAINABLE_SHARE:.1f} | Max extraction: {MAX_EXTRACTION}")
    print(f"{'=' * 65}")

    pool      = float(POOL_INITIAL)
    scores    = [0] * NUM_PLAYERS
    histories = [[] for _ in range(NUM_PLAYERS)]
    convs     = [[] for _ in range(NUM_PLAYERS)]
    game_log  = []

    for t in range(1, TOTAL_ROUNDS + 1):

        # Pool after fixed regeneration (shown in prompt before extraction decisions)
        pool_after_regen_preview = min(float(POOL_CAPACITY), pool + REGEN_FIXED)

        print(f"\n  Round {t}/{TOTAL_ROUNDS} | Pool after regen: {pool_after_regen_preview:.1f} "
              f"| Sustainable/player: {SUSTAINABLE_SHARE:.1f}")

        # Collect decisions from all models (logically simultaneous)
        raw_outputs  = []
        extractions  = []
        beliefs      = []
        token_usages = []
        resp_times   = []

        for i, (model_obj, label, _) in enumerate(players):
            prompt = build_round_prompt(
                histories[i], t, scores[i], pool_after_regen_preview, player_index=i + 1
            )
            conv_with_prompt = convs[i] + [HumanMessage(content=prompt)]
            raw, ext, bel, tok, rt = get_action_with_retry(
                model_obj, label, conv_with_prompt, t, pool_after_regen_preview
            )
            raw_outputs.append(raw)
            extractions.append(ext)
            beliefs.append(bel)
            token_usages.append(tok)
            resp_times.append(rt)
            convs[i] = conv_with_prompt + [AIMessage(content=raw or "{}")]

        # Apply pool dynamics with all extractions
        pool_after_regen, pool_after_extraction, total_ext, payoffs, collapsed = \
            apply_pool_dynamics(pool, extractions)

        for i in range(NUM_PLAYERS):
            scores[i] += payoffs[i]
        pool = pool_after_extraction

        # Print round summary
        collapse_str = "  *** COLLAPSE ***" if collapsed else ""
        for i, label in enumerate(labels):
            print(f"  {label:<22} extracted={extractions[i]:>2}  "
                  f"belief={beliefs[i]:.1f}  payoff={payoffs[i]}  total={scores[i]}")
        print(f"  Pool remaining: {pool_after_extraction:.1f}{collapse_str}")

        # Update per-player prompt histories
        for i in range(NUM_PLAYERS):
            opp_exts = [extractions[j] for j in range(NUM_PLAYERS) if j != i]
            opp_avg  = sum(opp_exts) / len(opp_exts) if opp_exts else 0.0
            histories[i].append({
                "round":                    t,
                "my_extraction":            extractions[i],
                "opponents_avg_extraction": round(opp_avg, 2),
                "my_payoff":                payoffs[i],
                "cumulative":               scores[i],
                "pool_after_regen":         round(pool_after_regen, 2),
                "pool_after_extraction":    round(pool_after_extraction, 2),
            })

        # Build round record (fixed columns + dynamic per-player columns)
        record = {
            "game_id":               game_id,
            "condition":             "AI-AI",
            "matchup":               matchup_str,
            "round":                 t,
            "num_players":           NUM_PLAYERS,
            "pool_before_regen":     round(pool_after_extraction + total_ext
                                           if t > 1 else float(POOL_INITIAL), 2),
            "pool_after_regen":      round(pool_after_regen, 2),
            "sustainable_share":     round(SUSTAINABLE_SHARE, 2),
            "total_extraction":      total_ext,
            "pool_after_extraction": round(pool_after_extraction, 2),
            "pool_collapsed":        int(collapsed),
        }
        for i in range(NUM_PLAYERS):
            n = i + 1
            record[f"model_{n}"]         = labels[i]
            record[f"extraction_{n}"]    = extractions[i]
            record[f"belief_{n}"]        = round(beliefs[i], 4)
            record[f"payoff_{n}"]        = payoffs[i]
            record[f"cumulative_{n}"]    = scores[i]
            record[f"raw_output_{n}"]    = (raw_outputs[i] or "")[:500]
            record[f"token_usage_{n}"]   = json.dumps(token_usages[i])
            record[f"response_time_{n}"] = round(resp_times[i], 1)
            record[f"temperature_{n}"]   = players[i][2]

        record["prompt_version"] = PROMPT_VERSION
        record["timestamp"]      = datetime.utcnow().isoformat()

        # Write to SQLite immediately (crash-safe, same pattern as PD)
        placeholders = ", ".join(["?"] * len(record))
        conn.execute(
            f"INSERT INTO rounds VALUES ({placeholders})",
            list(record.values()),
        )
        conn.commit()
        game_log.append(record)

        # Pool collapse: fill remaining rounds with zero records so the dataset
        # always has exactly TOTAL_ROUNDS rows per game (mirrors PD behaviour)
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
                    "timestamp":             datetime.utcnow().isoformat(),
                })
                for i in range(NUM_PLAYERS):
                    n = i + 1
                    zero_record[f"extraction_{n}"]    = 0
                    zero_record[f"belief_{n}"]        = SUSTAINABLE_SHARE
                    zero_record[f"payoff_{n}"]        = 0
                    zero_record[f"cumulative_{n}"]    = scores[i]
                    zero_record[f"raw_output_{n}"]    = "POOL_COLLAPSED"
                    zero_record[f"token_usage_{n}"]   = json.dumps({})
                    zero_record[f"response_time_{n}"] = 0.0

                placeholders = ", ".join(["?"] * len(zero_record))
                conn.execute(f"INSERT INTO rounds VALUES ({placeholders})",
                             list(zero_record.values()))
            conn.commit()
            break

    # End-of-game summary
    valid_rounds = [r for r in game_log if r.get("raw_output_1") != "POOL_COLLAPSED"]
    print(f"\n  {'─' * 55}")
    for i, label in enumerate(labels):
        avg_ext = (sum(r[f"extraction_{i+1}"] for r in valid_rounds)
                   / max(len(valid_rounds), 1))
        print(f"  FINAL  {label}: {scores[i]} pts | avg extraction: {avg_ext:.1f} "
              f"(sustainable: {SUSTAINABLE_SHARE:.1f})")
    print(f"  Pool remaining: {pool:.1f} / {POOL_CAPACITY}")

    return game_log


# ─────────────────────────────────────────────────────────────
# STEP 13 — SAVE TO CSV
# ─────────────────────────────────────────────────────────────

def save_csv(all_logs: list, path: str):
    """Save all round records to a CSV file."""
    if not all_logs:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_logs[0].keys()))
        writer.writeheader()
        writer.writerows(all_logs)
    print(f"\n✅ CSV saved -> {path}")
    print(f"   {len(all_logs)} rows across "
          f"{len(set(r['game_id'] for r in all_logs))} games")


# ─────────────────────────────────────────────────────────────
# STEP 14 — MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Validate API keys before spending time building the registry
    missing = [name for name, val in [
        ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
        ("OPENAI_API_KEY",    OPENAI_API_KEY),
        ("GEMINI_API_KEY",    GEMINI_API_KEY),
    ] if not val]
    if missing:
        raise EnvironmentError(
            f"Missing API keys in .env: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your keys."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path  = f"cd_experiment_{timestamp}.db"
    csv_path = f"cd_results_{timestamp}.csv"

    log.info("Initializing LangChain model registry...")
    model_registry = build_model_registry()

    conn = init_db(db_path)
    log.info(f"Database initialized: {db_path}")

    all_logs = []

    for game_id, matchup in enumerate(MATCHUPS, start=1):
        logs = run_game(matchup, game_id, conn, model_registry)
        all_logs.extend(logs)

    conn.close()
    save_csv(all_logs, csv_path)

    print(f"\n📊 SQLite database -> {db_path}")
    print("   Query examples:")
    print("   SELECT matchup, AVG(extraction_1) FROM rounds GROUP BY matchup;")
    print("   SELECT matchup, SUM(pool_collapsed) as collapses FROM rounds GROUP BY matchup;")
    print("   SELECT matchup, MIN(pool_after_extraction) as min_pool FROM rounds GROUP BY matchup;")
