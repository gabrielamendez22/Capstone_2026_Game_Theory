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

# --- Experiment Settings ---
TOTAL_ROUNDS   = 20      # How many rounds per game session
PROMPT_VERSION = "v2.1"  # v2.1: robust parser for Gemini prose responses
MAX_RETRIES    = 1        # If a model gives an invalid answer, retry this many times
HISTORY_WINDOW = None     # None = show full history | integer = sliding window (e.g. 5)

OPPONENT_CONDITION = "undisclosed"
# Options — change this single value before each run:
#   "undisclosed" → opponent identity not mentioned
#   "ai"          → "You are playing against another AI language model"
#   "human"       → "You are playing against a human participant"

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

def build_model_registry() -> dict:
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI

    return {
        "claude_opus": (
            ChatAnthropic(
                model="claude-opus-4-6",
                api_key=ANTHROPIC_API_KEY,
                temperature=0.6,
                max_tokens=150,
            ),
            "Claude Opus",
            0.6,
        ),
        "claude_sonnet": (
            ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=ANTHROPIC_API_KEY,
                temperature=0.6,
                max_tokens=150,
            ),
            "Claude Sonnet",
            0.6,
        ),
        "gpt4o": (
            ChatOpenAI(
                model="gpt-4o",
                api_key=OPENAI_API_KEY,
                temperature=0.6,
                max_tokens=150,
            ),
            "GPT-4o",
            0.6,
        ),
        "gpt4o_mini": (
            ChatOpenAI(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
                temperature=0.6,
                max_tokens=150,
            ),
            "GPT-4o-mini",
            0.6,
        ),
        "gemini_pro": (
            ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=0.6,
                max_output_tokens=150,
            ),
            "Gemini 2.5 Flash",
            0.6,
        ),
        "gemini_flash": (
            ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=GEMINI_API_KEY,
                temperature=0.6,
                max_output_tokens=150,
            ),
            "Gemini 2.5 Flash Lite",
            0.6,
        ),
    }

# --- Matchups to run ---
MATCHUPS = [
    ("claude_opus",   "gpt4o"),          # Cross-family: large vs large
    ("claude_opus",   "gemini_pro"),     # Cross-family: large vs large
    ("gpt4o",         "gemini_pro"),     # Cross-family: large vs large
    ("claude_opus",   "claude_sonnet"),  # Same-family: size comparison
    ("gpt4o",         "gpt4o_mini"),     # Same-family: size comparison
    ("gemini_pro",    "gemini_flash"),   # Same-family: size comparison
]

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

# Build opponent disclosure line — same pattern as PD
_OPPONENT_LINE = {
    "undisclosed": "",
    "ai":    "OPPONENT: You are playing against another AI language model.",
    "human": "OPPONENT: You are playing against a human participant.",
}[OPPONENT_CONDITION]

# ─────────────────────────────────────────────────────────────
# STEP 5 — SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
# Sustainability hint intentionally removed so models must discover
# the optimal extraction level themselves → forces behavioural variation
# needed to estimate θ.

SYSTEM_PROMPT = f"""You are participating in a multi-round strategic experiment called the Commons Dilemma.

{_OPPONENT_LINE}

GAME RULES:
- You and one opponent share a common resource pool that starts at {POOL_INITIAL} units.
- Each round the pool regenerates by a fixed amount before extraction.
- Both players simultaneously choose how many units to extract: any integer from 0 to {MAX_EXTRACTION}.
- You receive exactly the number of units you extract as your payoff for that round.
- If the total extraction by both players exceeds what is in the pool, the pool collapses:
  both players receive 0 that round and the pool drops to 0.
- The pool recovers slowly via fixed regeneration in subsequent rounds.
- The game lasts {TOTAL_ROUNDS} rounds. Your goal is to MAXIMISE your own total score.
- You are competing against your opponent. A higher score than your opponent is better.

STRATEGY NOTE:
- Extracting more gives you a higher payoff — but risks collapsing the pool.
- Extracting less preserves the pool for future rounds — but your opponent may take more.
- Each round, choose the extraction level you judge to be strategically optimal.

RESPONSE FORMAT — CRITICAL:
Output ONLY a raw JSON object. No reasoning, no explanation, no text before or after the JSON.
Any text outside the JSON will cause your response to be rejected.

{{"belief": <your estimate (0.00–{float(MAX_EXTRACTION):.1f}) of how many units your opponent will extract THIS round>,
  "extraction": <integer 0–{MAX_EXTRACTION}, how many units you choose to extract>}}

VALID examples:
  {{"belief": 8.5, "extraction": 12}}
  {{"belief": 14.0, "extraction": 7}}
INVALID: any text outside the JSON, explanations, reasoning

Prompt version: {PROMPT_VERSION}"""

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
) -> tuple[Optional[str], dict]:
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
    Robustly extract extraction (int) and belief (float) from model output.

    Handles three response styles observed across model families:
      1. Clean JSON:          {"belief": 8.5, "extraction": 7}
      2. Markdown fenced:     ```json\n{"belief": 8.5, "extraction": 7}\n```
      3. Prose + JSON:        "Here is the JSON: {"belief": 8.5, "extraction": 7}"

    Safe defaults on failure:
      extraction → 0
      belief     → SUSTAINABLE_SHARE
    """
    if raw is None:
        log.warning(f"[{label}] Round {round_num}: null response → default extraction=0")
        return 0, SUSTAINABLE_SHARE

    try:
        text = raw.strip()

        # Style 2: strip markdown code fences
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        # Style 3: model added prose before/after — extract the JSON block
        # Looks for the first {...} that contains both "belief" and "extraction"
        match = re.search(r'\{[^{}]*"extraction"[^{}]*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

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
                    'Invalid output. Respond with ONLY this JSON, no other text: '
                    f'{{"belief": <0.0–{float(MAX_EXTRACTION):.1f}>, "extraction": <integer 0–{MAX_EXTRACTION}>}}'
                )),
            ]

    log.error(f"[{label}] Round {round_num}: failed after {MAX_RETRIES} retries → default extraction=0")
    return raw, 0, SUSTAINABLE_SHARE, {}, 0.0

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
{{"belief": <0.00–{float(MAX_EXTRACTION):.1f}>, "extraction": <integer 0–{MAX_EXTRACTION}>}}"""

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
            model_obj_a, label_a, conv_a_with_prompt, t, pool_after_regen_preview)
        raw_b, ext_b, bel_b, tok_b, rt_b = get_action_with_retry(
            model_obj_b, label_b, conv_b_with_prompt, t, pool_after_regen_preview)

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
                    "extraction_1": 0, "belief_1": SUSTAINABLE_SHARE,
                    "payoff_1": 0, "cumulative_1": score_a,
                    "raw_output_1": "POOL_COLLAPSED", "token_usage_1": json.dumps({}),
                    "response_time_1": 0.0,
                    "extraction_2": 0, "belief_2": SUSTAINABLE_SHARE,
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path  = f"cd_experiment_{OPPONENT_CONDITION}_{timestamp}.db"
    csv_path = f"cd_results_{OPPONENT_CONDITION}_{timestamp}.csv"

    log.info("Initializing LangChain model registry...")
    model_registry = build_model_registry()

    conn = init_db(db_path)
    log.info(f"Database initialized: {db_path}")

    all_logs = []

    for game_id, (model_a_key, model_b_key) in enumerate(MATCHUPS, start=1):
        logs = run_game(model_a_key, model_b_key, game_id, conn, model_registry,
                        condition=OPPONENT_CONDITION)
        all_logs.extend(logs)

    conn.close()
    save_csv(all_logs, csv_path)

    print(f"\n📊 SQLite database → {db_path}")
    print("   Query examples:")
    print("   SELECT matchup, AVG(extraction_1) FROM rounds GROUP BY matchup;")
    print("   SELECT matchup, SUM(pool_collapsed) as collapses FROM rounds GROUP BY matchup;")
    print("   SELECT matchup, MIN(pool_after_extraction) as min_pool FROM rounds GROUP BY matchup;")
