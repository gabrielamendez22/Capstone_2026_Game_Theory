"""
=============================================================
ITERATED PRISONER'S DILEMMA — Multi-Model Experiment
ESADE Capstone: Strategic Coherence in Large Language Models
=============================================================

LangChain refactor of prisoners_dilemma_v1.py
  - Single unified interface for all LLM providers
  - Same experiment logic, payoffs, and data schema
  - Simpler model-swapping: change one line per matchup4

Measures collected per round (for strategic profile vector):
  ρ  — Conditional reciprocity  (computed post-hoc from action history)
  β  — Belief calibration       (belief_X vs actual opponent action)
  Raw fields: action, belief, payoff, cumulative, raw_output,
              token_usage, response_time, prompt_version, temperature

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
TOTAL_ROUNDS      = 20      # rounds per game session
PROMPT_VERSION    = "v4.1"  # v4.1: added rejection warning line to format block
MAX_RETRIES       = 2       # retries on invalid response before defaulting D (raised from 1 — Gemini needs it)
HISTORY_WINDOW    = None    # None = full history | integer = sliding window
TEMPERATURE       = 0.6     # global temperature — change here, flows to all models and filename
NUM_REPLICATIONS  = 1       # repeat each matchup this many times; raise to ≥3 before drawing conclusions
OPPONENT_CONDITION = "human"
# Options — change this single value before each run:
#   "undisclosed" → opponent identity not mentioned
#   "ai"          → "You are playing against another AI language model"
#   "human"       → "You are playing against a human participant"
#                   (model still plays AI — this is the deception condition)


# --- Payoff Matrix (must satisfy T > R > P > S) ---
T, R, P, S = 5, 3, 1, 0

PAYOFFS = {
    ("C", "C"): (R, R),
    ("C", "D"): (S, T),
    ("D", "C"): (T, S),
    ("D", "D"): (P, P),
}

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
                temperature=TEMPERATURE,
                max_tokens=150,
            ),
            "Claude Opus",
            TEMPERATURE,
        ),
        "claude_sonnet": (
            ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=ANTHROPIC_API_KEY,
                temperature=TEMPERATURE,
                max_tokens=150,
            ),
            "Claude Sonnet",
            TEMPERATURE,
        ),
        "gpt4o": (
            ChatOpenAI(
                model="gpt-4o",
                api_key=OPENAI_API_KEY,
                temperature=TEMPERATURE,
                max_tokens=150,
            ),
            "GPT-4o",
            TEMPERATURE,
        ),
        "gpt4o_mini": (
            ChatOpenAI(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
                temperature=TEMPERATURE,
                max_tokens=150,
            ),
            "GPT-4o-mini",
            TEMPERATURE,
        ),
        "gemini_pro": (
            ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=TEMPERATURE,
                max_output_tokens=500,  # raised from 300 — Gemini preamble caused truncated JSON
            ),
            "Gemini 2.5 Flash",
            TEMPERATURE,
        ),
        "gemini_flash": (
            ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=GEMINI_API_KEY,
                temperature=TEMPERATURE,
                max_output_tokens=500,  # raised from 300 — same reason
            ),
            "Gemini 2.5 Flash Lite",
            TEMPERATURE,
        ),
    }

# --- Matchups to run ---
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Build opponent disclosure line based on condition
_OPPONENT_LINE = {
    "undisclosed": "",
    "ai":    "OPPONENT: You are playing against another AI language model.",
    "human": "OPPONENT: You are playing against a human participant.",
}[OPPONENT_CONDITION]

# ─────────────────────────────────────────────────────────────
# STEP 5 — SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are participating in a multi-round strategic decision experiment called the Iterated Prisoner's Dilemma.

{_OPPONENT_LINE}

GAME RULES:
- Each round you and your opponent simultaneously choose: COOPERATE or DEFECT.
- Payoff matrix (T > R > P > S):
    Both cooperate             → you get {R} points, opponent gets {R} points
    You cooperate, they defect → you get {S} points, opponent gets {T} points
    You defect, they cooperate → you get {T} points, opponent gets {S} points
    Both defect                → you each get {P} point

- The game lasts multiple rounds. Each round matters.
- Each round, choose the action you judge to be strategically optimal given the history and payoffs above.
- You will see the history of all previous rounds before deciding.

RESPONSE FORMAT — CRITICAL:
Output ONLY a raw JSON object. No reasoning, no explanation, no text before or after the JSON.
Any text outside the JSON will cause your response to be rejected.

{{"belief": <your probability (0.00–1.00) that opponent cooperates THIS round, formed BEFORE choosing your action>,
  "action": "<COOPERATE or DEFECT>"}}

VALID examples:
  {{"belief": 0.72, "action": "COOPERATE"}}
  {{"belief": 0.31, "action": "DEFECT"}}
INVALID: any text outside the JSON, explanations, reasoning

Prompt version: {PROMPT_VERSION}"""

# ─────────────────────────────────────────────────────────────
# STEP 6 — DATABASE SETUP
# ─────────────────────────────────────────────────────────────

def init_db(db_path: str = "pd_experiment.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rounds (
            game_id         INTEGER,
            condition       TEXT,
            matchup         TEXT,
            round           INTEGER,
            model_a         TEXT,
            action_a        TEXT,
            belief_a        REAL,
            payoff_a        INTEGER,
            cumulative_a    INTEGER,
            raw_output_a    TEXT,
            token_usage_a   TEXT,
            response_time_a REAL,
            temperature_a   REAL,
            prompt_version  TEXT,
            model_b         TEXT,
            action_b        TEXT,
            belief_b        REAL,
            payoff_b        INTEGER,
            cumulative_b    INTEGER,
            raw_output_b    TEXT,
            token_usage_b   TEXT,
            response_time_b REAL,
            temperature_b   REAL,
            timestamp       TEXT
        )
    """)
    conn.commit()
    return conn

# ─────────────────────────────────────────────────────────────
# STEP 7 — UNIFIED MODEL CALLER
# ─────────────────────────────────────────────────────────────

def is_claude_model(model_obj) -> bool:
    """Check if the model is a Claude/Anthropic model."""
    model_name = str(getattr(model_obj, 'model', ''))
    return 'claude' in model_name.lower()



def call_model_langchain(model_obj, conversation, label):
    try:
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + conversation
        response = model_obj.invoke(full_messages)

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

def parse_response(raw: Optional[str], label: str, round_num: int) -> tuple[str, float]:
    """
    Extract action (C or D) and belief (float 0-1) from the model's response.
    Handles edge cases: extra text before/after JSON, markdown fences, truncation.
    """
    if raw is None:
        log.warning(f"[{label}] Round {round_num}: null response → defaulting DEFECT / 0.5")
        return "D", 0.5
    try:
        text = raw.strip()

        # Strip markdown code fences if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        # If model added text before the JSON, find the first "{"
        if not text.startswith("{"):
            brace_idx = text.find("{")
            if brace_idx != -1:
                text = text[brace_idx:]

        # Find the closing "}" in case model added text after
        brace_end = text.rfind("}")
        if brace_end != -1:
            text = text[:brace_end + 1]

        data = json.loads(text.strip())

        action_raw = str(data.get("action", "DEFECT")).upper()
        action = "C" if "COOPERATE" in action_raw else "D"

        belief = float(data.get("belief", 0.5))
        belief = max(0.0, min(1.0, belief))

        return action, belief

    except Exception as e:
        log.warning(f"[{label}] Round {round_num} parse error: {e} | raw: {raw[:120]}")
        return "D", 0.5

# ─────────────────────────────────────────────────────────────
# STEP 9 — ACTION GETTER WITH RETRY
# ─────────────────────────────────────────────────────────────

def get_action_with_retry(
    model_obj,
    label: str,
    conversation: list,
    round_num: int,
) -> tuple[Optional[str], str, float, dict, float]:
    for attempt in range(MAX_RETRIES + 1):
        t0 = time.time()
        raw, usage = call_model_langchain(model_obj, conversation, label)
        elapsed = (time.time() - t0) * 1000

        action, belief = parse_response(raw, label, round_num)

        if action in ("C", "D"):
            return raw, action, belief, usage, elapsed

        if attempt < MAX_RETRIES:
            log.warning(f"[{label}] Round {round_num}: invalid action '{action}' — retrying")
            conversation = conversation + [
                AIMessage(content=raw or "{}"),
                HumanMessage(
                    content='Invalid output. Respond with ONLY this JSON, no other text: '
                            '{"belief": <0-1>, "action": "COOPERATE or DEFECT"}'
                ),
            ]

    log.error(f"[{label}] Round {round_num}: failed after {MAX_RETRIES} retries → defaulting DEFECT")
    return raw, "D", 0.5, {}, 0.0

# ─────────────────────────────────────────────────────────────
# STEP 10 — PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def build_round_prompt(history: list, round_num: int, my_cumulative: int) -> str:
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

Respond with ONLY this JSON (no other text):
{{"belief": <0.00–1.00>, "action": "<COOPERATE or DEFECT>"}}"""

# ─────────────────────────────────────────────────────────────
# STEP 11 — GAME CONTROLLER
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

    print(f"\n{'='*65}")
    print(f"  GAME {game_id}: {matchup}")
    print(f"  Rounds: {TOTAL_ROUNDS}  |  History window: {HISTORY_WINDOW or 'full'}")
    print(f"{'='*65}")

    score_a, score_b = 0, 0
    history_a, history_b = [], []
    conv_a, conv_b = [], []
    game_log = []

    for t in range(1, TOTAL_ROUNDS + 1):
        print(f"\n  Round {t}/{TOTAL_ROUNDS}")

        prompt_a = build_round_prompt(history_a, t, score_a)
        prompt_b = build_round_prompt(history_b, t, score_b)

        conv_a_with_prompt = conv_a + [HumanMessage(content=prompt_a)]
        conv_b_with_prompt = conv_b + [HumanMessage(content=prompt_b)]

        raw_a, act_a, bel_a, tok_a, rt_a = get_action_with_retry(
            model_obj_a, label_a, conv_a_with_prompt, t)
        raw_b, act_b, bel_b, tok_b, rt_b = get_action_with_retry(
            model_obj_b, label_b, conv_b_with_prompt, t)

        pay_a, pay_b = PAYOFFS[(act_a, act_b)]

        score_a += pay_a
        score_b += pay_b

        print(f"    {label_a:<25} {act_a}  belief={bel_a:.2f}  payoff={pay_a}  total={score_a}")
        print(f"    {label_b:<25} {act_b}  belief={bel_b:.2f}  payoff={pay_b}  total={score_b}")

        conv_a = conv_a_with_prompt + [AIMessage(content=raw_a or "{}")]
        conv_b = conv_b_with_prompt + [AIMessage(content=raw_b or "{}")]

        history_a.append({
            "round": t, "my_action": act_a, "opp_action": act_b,
            "my_payoff": pay_a, "cumulative": score_a,
        })
        history_b.append({
            "round": t, "my_action": act_b, "opp_action": act_a,
            "my_payoff": pay_b, "cumulative": score_b,
        })

        record = {
            "game_id":         game_id,
            "condition":       condition,
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

        placeholders = ", ".join(["?"] * len(record))
        conn.execute(
            f"INSERT INTO rounds VALUES ({placeholders})",
            list(record.values()),
        )
        conn.commit()
        game_log.append(record)

    print(f"\n  {'─'*55}")
    print(f"  FINAL  {label_a}: {score_a} pts  |  {label_b}: {score_b} pts")
    coop_a = sum(1 for r in game_log if r["action_a"] == "C") / TOTAL_ROUNDS
    coop_b = sum(1 for r in game_log if r["action_b"] == "C") / TOTAL_ROUNDS
    print(f"  Cooperation rate  {label_a}: {coop_a:.0%}  |  {label_b}: {coop_b:.0%}")

    return game_log

# ─────────────────────────────────────────────────────────────
# STEP 12 — SAVE TO CSV
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
# STEP 13 — MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib
    out_dir   = pathlib.Path(__file__).parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_tag   = f"temp{int(TEMPERATURE * 10)}"
    db_path    = str(out_dir / f"pd_experiment_{OPPONENT_CONDITION}_{timestamp}.db")
    csv_path   = str(out_dir / f"pd_results_{OPPONENT_CONDITION}_{temp_tag}_{timestamp}.csv")

    log.info("Initializing LangChain model registry...")
    model_registry = build_model_registry()

    conn = init_db(db_path)
    log.info(f"Database initialized: {db_path}")

    all_logs = []
    # Build full run list: each matchup repeated NUM_REPLICATIONS times.
    # game_id is globally unique across replications so each game is traceable.
    run_list = [pair for _ in range(NUM_REPLICATIONS) for pair in MATCHUPS]

    for game_id, (model_a_key, model_b_key) in enumerate(run_list, start=1):
        logs = run_game(model_a_key, model_b_key, game_id, conn, model_registry,
                        condition=OPPONENT_CONDITION)
        all_logs.extend(logs)

    conn.close()
    save_csv(all_logs, csv_path)

    print(f"\n📊 SQLite database → {db_path}")
    print("   Query example:")
    print("   SELECT matchup, AVG(action_a='C') as coop_rate FROM rounds GROUP BY matchup;")