"""
=============================================================
ITERATED PRISONER'S DILEMMA — Multi-Model Experiment
ESADE Capstone: Strategic Coherence in Large Language Models
=============================================================

LangChain refactor of prisoners_dilemma_v1.py
  - Single unified interface for all LLM providers
  - Same experiment logic, payoffs, and data schema
  - Simpler model-swapping: change one line per matchup

Measures collected per round (for strategic profile vector):
  ρ  — Conditional reciprocity  (computed post-hoc from action history)
  β  — Belief calibration       (belief_X vs actual opponent action)
  Raw fields: action, belief, payoff, cumulative, raw_output,
              token_usage, response_time, prompt_version, temperature

INSTALL DEPENDENCIES:
  pip install langchain langchain-anthropic langchain-openai langchain-google-genai

HOW TO GET API KEYS (all have free tiers):
  Claude  → https://console.anthropic.com/       → API Keys
  ChatGPT → https://platform.openai.com/api-keys
  Gemini  → https://aistudio.google.com/app/apikey
"""

# ─────────────────────────────────────────────────────────────
# STEP 1 — IMPORTS
# ─────────────────────────────────────────────────────────────
# Standard Python libraries for data storage, timing, and logging.
# LangChain is imported per-provider only when that provider is needed.

import os
import csv
import json
import time
import sqlite3
import logging
from datetime import datetime
from typing import Optional

# LangChain: these are the unified message objects used by ALL providers.
# Instead of writing {"role": "user", "content": "..."} for OpenAI
# and a different format for Anthropic, LangChain gives us one format.
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ─────────────────────────────────────────────────────────────
# STEP 2 — CONFIGURATION
# ─────────────────────────────────────────────────────────────
# All experiment settings in one place.
# Change these values to adjust the experiment without touching
# any other part of the code.

# --- API Keys ---
# Paste your keys here. All three providers can have free-tier keys.
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_KEY_HERE"
OPENAI_API_KEY    = "YOUR_OPENAI_KEY_HERE"
GEMINI_API_KEY    = "YOUR_GEMINI_KEY_HERE"

# --- Experiment Settings ---
TOTAL_ROUNDS   = 20       # How many rounds per game session
PROMPT_VERSION = "v3.0"  # Increment this whenever you change the system prompt
MAX_RETRIES    = 1        # If a model gives an invalid answer, retry this many times
HISTORY_WINDOW = None     # None = show full history | integer = sliding window (e.g. 5)

# --- Payoff Matrix (must satisfy T > R > P > S) ---
# T = Temptation to defect (highest reward, but only if opponent cooperates)
# R = Reward for mutual cooperation
# P = Punishment for mutual defection
# S = Sucker's payoff (you cooperated but opponent defected)
T, R, P, S = 5, 3, 1, 0

PAYOFFS = {
    ("C", "C"): (R, R),   # Both cooperate       → (3, 3)
    ("C", "D"): (S, T),   # A cooperates, B defects → (0, 5)
    ("D", "C"): (T, S),   # A defects, B cooperates → (5, 0)
    ("D", "D"): (P, P),   # Both defect          → (1, 1)
}

# ─────────────────────────────────────────────────────────────
# STEP 3 — MODEL REGISTRY
# ─────────────────────────────────────────────────────────────
# This is where LangChain shines.
#
# In the original file we had three separate functions:
#   call_claude(), call_openai(), call_gemini()
#
# With LangChain, all three are replaced by one unified object.
# The only difference is which class you import:
#   ChatAnthropic, ChatOpenAI, ChatGoogleGenerativeAI
#
# Each entry in AVAILABLE_MODELS maps a short nickname to:
#   (LangChain model object, human-readable label, temperature)
#
# To add a new model (e.g. Mistral), you just install
# langchain-mistralai and add one line here — zero other changes.

def build_model_registry() -> dict:
    """
    Creates one LangChain model object per model.
    Each object already knows its API key, model name, and temperature.
    The rest of the code never calls any provider-specific API directly.
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
# Each tuple is (model_a_key, model_b_key) using the keys above.
MATCHUPS = [
    ("claude_opus",   "gpt4o"),         # Cross-family: large vs large
    ("claude_opus",   "gemini_pro"),    # Cross-family: large vs large
    ("gpt4o",         "gemini_pro"),    # Cross-family: large vs large
    ("claude_opus",   "claude_sonnet"), # Same-family: size comparison
    ("gpt4o",         "gpt4o_mini"),    # Same-family: size comparison
    ("gemini_pro",    "gemini_flash"),  # Same-family: size comparison
]

# ─────────────────────────────────────────────────────────────
# STEP 4 — LOGGING SETUP
# ─────────────────────────────────────────────────────────────
# Logs go to both a file (experiment.log) and the console simultaneously.
# Useful for monitoring long runs without losing output.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# STEP 5 — SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
# This is the instruction given to the model at the START of every game.
# It explains the rules and — critically — forces the model to respond
# in JSON format, which makes parsing reliable.
#
# The f-string inserts the actual payoff values (R, S, T, P) so that
# if you change the payoff matrix above, the prompt updates automatically.

SYSTEM_PROMPT = f"""You are participating in a multi-round strategic decision experiment called the Iterated Prisoner's Dilemma.

GAME RULES:
- Each round you and your opponent simultaneously choose: COOPERATE or DEFECT.
- Payoff matrix (T > R > P > S):
    Both cooperate             → you get {R} points, opponent gets {R} points
    You cooperate, they defect → you get {S} points, opponent gets {T} points
    You defect, they cooperate → you get {T} points, opponent gets {S} points
    Both defect                → you each get {P} point

- The game lasts multiple rounds. Each round matters.
- Your goal is to maximize your TOTAL cumulative score.
- You will see the history of all previous rounds before deciding.

RESPONSE FORMAT — CRITICAL:
You must respond with ONLY a valid JSON object. No explanation. No extra text.

{{"belief": <float 0.00–1.00, your probability opponent cooperates this round>,
  "action": "<exactly COOPERATE or DEFECT>"}}

VALID example:   {{"belief": 0.72, "action": "COOPERATE"}}
INVALID examples: "I choose cooperate", "Probably cooperate", "It depends"

Prompt version: {PROMPT_VERSION}"""

# ─────────────────────────────────────────────────────────────
# STEP 6 — DATABASE SETUP
# ─────────────────────────────────────────────────────────────
# Creates a SQLite database on disk.
# Each row = one round of one game.
# Writing to the database after every round means that if the
# experiment crashes halfway through, you keep all data up to that point.

def init_db(db_path: str = "pd_experiment.db") -> sqlite3.Connection:
    """
    Creates (or opens) a SQLite database and ensures the rounds table exists.
    The schema captures everything needed for the strategic profile analysis:
      - ρ (conditional reciprocity): derived from action_a, action_b columns
      - β (belief calibration): belief_a vs actual action_b, and vice-versa
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rounds (
            -- Game identifiers
            game_id         INTEGER,
            condition       TEXT,        -- AI-AI / Human-AI / Human-Human
            matchup         TEXT,        -- e.g. 'Claude Opus vs GPT-4o'
            round           INTEGER,

            -- Player A
            model_a         TEXT,
            action_a        TEXT,        -- C or D  (used for ρ)
            belief_a        REAL,        -- P(opponent cooperates) stated by A (used for β)
            payoff_a        INTEGER,
            cumulative_a    INTEGER,
            raw_output_a    TEXT,        -- unprocessed model string
            token_usage_a   TEXT,        -- JSON: {prompt, completion}
            response_time_a REAL,        -- milliseconds
            temperature_a   REAL,
            prompt_version  TEXT,

            -- Player B
            model_b         TEXT,
            action_b        TEXT,
            belief_b        REAL,
            payoff_b        INTEGER,
            cumulative_b    INTEGER,
            raw_output_b    TEXT,
            token_usage_b   TEXT,
            response_time_b REAL,
            temperature_b   REAL,

            -- Round metadata
            timestamp       TEXT
        )
    """)
    conn.commit()
    return conn

# ─────────────────────────────────────────────────────────────
# STEP 7 — UNIFIED MODEL CALLER  (the key LangChain upgrade)
# ─────────────────────────────────────────────────────────────
# In the original file, calling each provider required separate functions
# and separate message formats.
#
# Here, ONE function handles ALL providers.
# The model object (ChatAnthropic, ChatOpenAI, ChatGoogleGenerativeAI)
# is passed in as an argument — all three share the same .invoke() method.
#
# LangChain message types used:
#   SystemMessage  → the game rules (sent once at the start)
#   HumanMessage   → the round prompt (sent each round)
#   AIMessage      → the model's previous response (for multi-turn memory)

def call_model_langchain(
    model_obj,          # A LangChain chat model (any provider)
    conversation: list, # List of LangChain message objects (SystemMessage, HumanMessage, AIMessage)
    label: str,         # Human-readable model name for logging
) -> tuple[Optional[str], dict]:
    """
    Send a conversation to any LangChain-compatible model.
    Returns (raw_text_response, token_usage_dict).

    The conversation list already includes:
      - SystemMessage with game rules (prepended automatically below)
      - All previous HumanMessage / AIMessage pairs (the game history)
      - The new HumanMessage with this round's prompt

    Token usage is extracted where available (OpenAI and Anthropic expose it;
    Gemini may return zeros in some configurations).
    """
    try:
        # Prepend the system prompt to the full conversation.
        # SystemMessage tells the model who it is and how to respond.
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + conversation

        # .invoke() is the unified call — works identically for all providers.
        response = model_obj.invoke(full_messages)

        # Extract token usage from response metadata (provider-dependent field names).
        usage_meta = response.response_metadata or {}
        usage = {
            "prompt":     usage_meta.get("input_tokens",
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
# Models are asked to respond in JSON, but sometimes they wrap it in
# markdown code fences (```json ... ```) or include extra text.
# This function handles all those cases and always returns a valid
# (action, belief) pair — defaulting to ("D", 0.5) on failure.

def parse_response(raw: Optional[str], label: str, round_num: int) -> tuple[str, float]:
    """
    Extract action (C or D) and belief (float 0–1) from the model's JSON response.

    Safe defaults on failure:
      action → "D"  (DEFECT — conservative default)
      belief → 0.5  (maximum uncertainty)
    """
    if raw is None:
        log.warning(f"[{label}] Round {round_num}: null response → defaulting DEFECT / 0.5")
        return "D", 0.5
    try:
        text = raw.strip()

        # Strip markdown code fences if present (e.g. ```json { ... } ```)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        data = json.loads(text.strip())

        # Parse action — accept "COOPERATE", "cooperate", "C", etc.
        action_raw = str(data.get("action", "DEFECT")).upper()
        action = "C" if "COOPERATE" in action_raw else "D"

        # Parse belief — clamp to valid range [0, 1]
        belief = float(data.get("belief", 0.5))
        belief = max(0.0, min(1.0, belief))

        return action, belief

    except Exception as e:
        log.warning(f"[{label}] Round {round_num} parse error: {e} | raw: {raw[:120]}")
        return "D", 0.5

# ─────────────────────────────────────────────────────────────
# STEP 9 — ACTION GETTER WITH RETRY
# ─────────────────────────────────────────────────────────────
# Wraps the model call + parser in a retry loop.
# If the model returns an invalid action (not C or D), we send it
# a correction message and try again (up to MAX_RETRIES times).
# If it still fails, we log the error and default to DEFECT.

def get_action_with_retry(
    model_obj,          # LangChain model object
    label: str,         # Human-readable name for logging
    conversation: list, # Current conversation history (LangChain messages)
    round_num: int,     # Current round number (for logging)
) -> tuple[Optional[str], str, float, dict, float]:
    """
    Returns: (raw_output, action, belief, token_usage, response_time_ms)
    """
    for attempt in range(MAX_RETRIES + 1):
        t0 = time.time()
        raw, usage = call_model_langchain(model_obj, conversation, label)
        elapsed = (time.time() - t0) * 1000  # convert to milliseconds

        action, belief = parse_response(raw, label, round_num)

        # Valid action received — return immediately
        if action in ("C", "D"):
            return raw, action, belief, usage, elapsed

        # Invalid — add a correction message and retry
        if attempt < MAX_RETRIES:
            log.warning(f"[{label}] Round {round_num}: invalid action '{action}' — retrying")
            conversation = conversation + [
                AIMessage(content=raw or "{}"),
                HumanMessage(
                    content='Invalid output. Respond with exactly one JSON object: '
                            '{"belief": <0-1>, "action": "COOPERATE or DEFECT"}'
                ),
            ]

    # All retries exhausted — log and return safe default
    log.error(f"[{label}] Round {round_num}: failed after {MAX_RETRIES} retries → defaulting DEFECT")
    return raw, "D", 0.5, {}, 0.0

# ─────────────────────────────────────────────────────────────
# STEP 10 — PROMPT BUILDER
# ─────────────────────────────────────────────────────────────
# Builds the per-round user message sent to each model.
# This includes the full game history (or a sliding window of it)
# and asks the model to state its belief and then make a decision.

def build_round_prompt(history: list, round_num: int, my_cumulative: int) -> str:
    """
    Constructs the HumanMessage content for a given round.

    history: list of dicts, one per completed round, from this player's perspective.
    round_num: the round we are about to play.
    my_cumulative: this player's total score so far.

    If HISTORY_WINDOW is set (e.g. 5), only the last 5 rounds are shown.
    This simulates limited memory and tests whether models behave differently
    with truncated vs. full context.
    """
    # Apply sliding window if configured
    window = history[-HISTORY_WINDOW:] if HISTORY_WINDOW else history

    if not window:
        context = "No previous rounds."
    else:
        lines = [
            f"  Round {h['round']:>2}: You → {h['my_action']}  |  "
            f"Opponent → {h['opp_action']}  |  "
            f"Your payoff: {h['my_payoff']}  |  Cumulative: {h['cumulative']}"
            for h in window
        ]
        context = "History:\n" + "\n".join(lines)
        if HISTORY_WINDOW and len(history) > HISTORY_WINDOW:
            context = f"[Showing last {HISTORY_WINDOW} of {len(history)} rounds]\n" + context

    return f"""--- Round {round_num} ---

{context}

Your total score so far: {my_cumulative} points.

Before deciding:
1. State your belief: probability (0.00–1.00) that opponent cooperates this round.
2. Choose your action: COOPERATE or DEFECT.

Respond in JSON only:
{{"belief": <0.00–1.00>, "action": "<COOPERATE or DEFECT>"}}"""

# ─────────────────────────────────────────────────────────────
# STEP 11 — GAME CONTROLLER
# ─────────────────────────────────────────────────────────────
# Runs one complete game between two models.
# Each model maintains its own:
#   - conversation history (LangChain messages)
#   - game history (human-readable round records)
#   - cumulative score
#
# Both models are called independently each round (simulating
# simultaneous decisions — neither sees the other's choice until
# after both have responded).

def run_game(
    model_a_key: str,
    model_b_key: str,
    game_id: int,
    conn: sqlite3.Connection,
    model_registry: dict,
) -> list:
    """
    Run one full game between model A and model B.
    Returns a list of round records (one dict per round).
    """
    model_obj_a, label_a, temp_a = model_registry[model_a_key]
    model_obj_b, label_b, temp_b = model_registry[model_b_key]
    matchup = f"{label_a} vs {label_b}"

    print(f"\n{'='*65}")
    print(f"  GAME {game_id}: {matchup}")
    print(f"  Rounds: {TOTAL_ROUNDS}  |  History window: {HISTORY_WINDOW or 'full'}")
    print(f"{'='*65}")

    # Scores and histories — maintained separately for each player
    score_a, score_b = 0, 0
    history_a, history_b = [], []    # Round-by-round summaries (for prompt building)
    conv_a, conv_b = [], []          # LangChain message objects (for API calls)
    game_log = []

    for t in range(1, TOTAL_ROUNDS + 1):
        print(f"\n  Round {t}/{TOTAL_ROUNDS}")

        # Build this round's prompt for each player
        prompt_a = build_round_prompt(history_a, t, score_a)
        prompt_b = build_round_prompt(history_b, t, score_b)

        # Add the new round prompt to each player's conversation
        # HumanMessage is the LangChain equivalent of {"role": "user", "content": ...}
        conv_a_with_prompt = conv_a + [HumanMessage(content=prompt_a)]
        conv_b_with_prompt = conv_b + [HumanMessage(content=prompt_b)]

        # Get decisions from both models (sequential here, but logically simultaneous
        # — neither model knows the other's choice when making its decision)
        raw_a, act_a, bel_a, tok_a, rt_a = get_action_with_retry(
            model_obj_a, label_a, conv_a_with_prompt, t)
        raw_b, act_b, bel_b, tok_b, rt_b = get_action_with_retry(
            model_obj_b, label_b, conv_b_with_prompt, t)

        # Determine payoffs from the payoff matrix
        pay_a, pay_b = PAYOFFS[(act_a, act_b)]

        # Update cumulative scores
        score_a += pay_a
        score_b += pay_b

        print(f"    {label_a:<22} {act_a}  belief={bel_a:.2f}  payoff={pay_a}  total={score_a}")
        print(f"    {label_b:<22} {act_b}  belief={bel_b:.2f}  payoff={pay_b}  total={score_b}")

        # Update each player's conversation with their own response
        # AIMessage is the LangChain equivalent of {"role": "assistant", "content": ...}
        conv_a = conv_a_with_prompt + [AIMessage(content=raw_a or "{}")]
        conv_b = conv_b_with_prompt + [AIMessage(content=raw_b or "{}")]

        # Update each player's game history (used in next round's prompt)
        history_a.append({
            "round": t, "my_action": act_a, "opp_action": act_b,
            "my_payoff": pay_a, "cumulative": score_a,
        })
        history_b.append({
            "round": t, "my_action": act_b, "opp_action": act_a,
            "my_payoff": pay_b, "cumulative": score_b,
        })

        # Build the full record for this round
        record = {
            "game_id":         game_id,
            "condition":       "AI-AI",
            "matchup":         matchup,
            "round":           t,
            "model_a":         label_a,
            "action_a":        act_a,
            "belief_a":        round(bel_a, 4),
            "payoff_a":        pay_a,
            "cumulative_a":    score_a,
            "raw_output_a":    (raw_a or "")[:500],
            "token_usage_a":   json.dumps(tok_a),
            "response_time_a": round(rt_a, 1),
            "temperature_a":   temp_a,
            "prompt_version":  PROMPT_VERSION,
            "model_b":         label_b,
            "action_b":        act_b,
            "belief_b":        round(bel_b, 4),
            "payoff_b":        pay_b,
            "cumulative_b":    score_b,
            "raw_output_b":    (raw_b or "")[:500],
            "token_usage_b":   json.dumps(tok_b),
            "response_time_b": round(rt_b, 1),
            "temperature_b":   temp_b,
            "timestamp":       datetime.utcnow().isoformat(),
        }

        # Write to SQLite immediately (crash-safe: data is never lost mid-game)
        placeholders = ", ".join(["?"] * len(record))
        conn.execute(
            f"INSERT INTO rounds VALUES ({placeholders})",
            list(record.values()),
        )
        conn.commit()
        game_log.append(record)

    # Print end-of-game summary
    print(f"\n  {'─'*55}")
    print(f"  FINAL  {label_a}: {score_a} pts  |  {label_b}: {score_b} pts")
    coop_a = sum(1 for r in game_log if r["action_a"] == "C") / TOTAL_ROUNDS
    coop_b = sum(1 for r in game_log if r["action_b"] == "C") / TOTAL_ROUNDS
    print(f"  Cooperation rate  {label_a}: {coop_a:.0%}  |  {label_b}: {coop_b:.0%}")

    return game_log

# ─────────────────────────────────────────────────────────────
# STEP 12 — SAVE TO CSV
# ─────────────────────────────────────────────────────────────
# Writes all round records to a CSV file at the end of the experiment.
# This is the file you will import into R or Python for statistical analysis.
# SQLite gives you the same data in a queryable database format.

def save_csv(all_logs: list, path: str):
    """Save all round records to a CSV file."""
    if not all_logs:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_logs[0].keys()))
        writer.writeheader()
        writer.writerows(all_logs)
    print(f"\n✅ CSV saved → {path}")
    print(f"   {len(all_logs)} rows across {len(all_logs) // TOTAL_ROUNDS} games")

# ─────────────────────────────────────────────────────────────
# STEP 13 — MAIN
# ─────────────────────────────────────────────────────────────
# Entry point. Builds the model registry, initializes the database,
# runs all matchups in sequence, and saves results.
#
# Output files are timestamped so repeated runs never overwrite each other.

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path  = f"pd_experiment_{timestamp}.db"
    csv_path = f"pd_results_{timestamp}.csv"

    # Build all LangChain model objects (one per model in the registry)
    log.info("Initializing LangChain model registry...")
    model_registry = build_model_registry()

    # Initialize the SQLite database
    conn = init_db(db_path)
    log.info(f"Database initialized: {db_path}")

    all_logs = []

    # Run each matchup in sequence
    for game_id, (model_a_key, model_b_key) in enumerate(MATCHUPS, start=1):
        logs = run_game(model_a_key, model_b_key, game_id, conn, model_registry)
        all_logs.extend(logs)

    conn.close()
    save_csv(all_logs, csv_path)

    print(f"\n📊 SQLite database → {db_path}")
    print("   Query example:")
    print("   SELECT matchup, AVG(action_a='C') as coop_rate FROM rounds GROUP BY matchup;")
