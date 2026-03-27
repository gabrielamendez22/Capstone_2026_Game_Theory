"""
=============================================================
ITERATED PRISONER'S DILEMMA — Multi-Model Experiment
ESADE Capstone 2026: Strategic Coherence in Large Language Models
=============================================================

Based on professor's TwoBotConversation structure.
Saves results to JSON files (no database needed).

INSTALL:
    pip install langchain langchain-anthropic langchain-google-genai langchain-openai

API KEYS:
    Gemini  → https://aistudio.google.com/app/apikey
    Claude  → https://console.anthropic.com/
    OpenAI  → https://platform.openai.com/api-keys
"""

import json
import time
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI                         # ← NEW
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os


# ─────────────────────────────────────────────────────────────
# CONFIGURATION — change settings here
# ─────────────────────────────────────────────────────────────

load_dotenv()

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")               # ← NEW

TOTAL_ROUNDS   = 20
TEMPERATURE    = 0.7
PROMPT_VERSION = "v1.0"

# Payoff matrix: T > R > P > S
T, R, P, S = 5, 3, 1, 0
PAYOFFS = {
    ("C", "C"): (R, R),   # Both cooperate          → (3, 3)
    ("C", "D"): (S, T),   # A cooperates, B defects → (0, 5)
    ("D", "C"): (T, S),   # A defects, B cooperates → (5, 0)
    ("D", "D"): (P, P),   # Both defect             → (1, 1)
}

# ─────────────────────────────────────────────────────────────
# MATCHUPS — add or remove pairs as needed
# Available model keys:
#   Gemini  : "gemini_flash", "gemini_pro"
#   Claude  : "claude_sonnet", "claude_opus"
#   OpenAI  : "gpt4o", "gpt4o_mini"
# ─────────────────────────────────────────────────────────────

MATCHUPS = [
    # ("gemini_flash",  "gemini_flash"),    # Gemini Flash vs Gemini Flash
    # ("gemini_pro",    "gemini_flash"),    # Gemini Pro vs Gemini Flash
      ("gemini_flash",  "claude_sonnet"),  # Gemini Flash vs Claude Sonnet
      ("gemini_flash",  "gpt4o_mini"),     # Gemini Flash vs GPT-4o Mini
    # ("claude_sonnet", "claude_sonnet"),  # Claude Sonnet vs Claude Sonnet
    ("claude_sonnet", "gpt4o"),          # Claude Sonnet vs GPT-4o
    #("gpt4o",         "gpt4o"),          # GPT-4o vs GPT-4o
    ("gpt4o_mini",    "gpt4o_mini"),     # GPT-4o Mini vs GPT-4o Mini
    ("claude_opus",   "gpt4o"),          # Claude Opus vs GPT-4o
]


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT — the rules given to each model at the start
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are participating in a multi-round experiment called the Iterated Prisoner's Dilemma.

RULES:
- Each round you and your opponent simultaneously choose: COOPERATE or DEFECT.
- Payoffs:
    Both cooperate             → you get {R} points, opponent gets {R} points
    You cooperate, they defect → you get {S} points, opponent gets {T} points
    You defect, they cooperate → you get {T} points, opponent gets {S} points
    Both defect                → you each get {P} point

- Your goal is to maximize your TOTAL score across all rounds.
- You will see what happened in previous rounds before each decision.

CRITICAL INSTRUCTION: You must respond with ONLY a JSON object. 
Do NOT include any explanation, reasoning, or text before or after the JSON.
Do NOT say "I'll analyze" or any other text.
Your ENTIRE response must be ONLY this JSON, nothing else:
{{"belief": <number between 0.00 and 1.00 = your probability that opponent cooperates>,
  "action": "<COOPERATE or DEFECT>"}}

Example of a valid response (the entire response, nothing else):
{{"belief": 0.72, "action": "COOPERATE"}}

If you include ANY text outside the JSON, the experiment will fail."""

# ─────────────────────────────────────────────────────────────
# PLAYER CLASS — based on professor's TwoBotConversation
# ─────────────────────────────────────────────────────────────

class Player:

    def __init__(self, name, model_obj):
        self.name = name
        self.model = model_obj
        self.score = 0
        self.conversation = []
        self.history = []

    def decide(self, round_num):
        prompt = self._build_prompt(round_num)
        self.conversation.append(HumanMessage(content=prompt))
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + self.conversation

        try:
            time.sleep(3)
            t0 = time.time()
            response = self.model.invoke(full_messages)
            elapsed = round((time.time() - t0) * 1000, 1)
            raw = response.content.strip()
        except Exception as e:
            print(f"  ⚠ [{self.name}] API error: {e}")
            raw = '{"belief": 0.5, "action": "DEFECT"}'
            elapsed = 0

        self.conversation.append(AIMessage(content=raw))
        action, belief = self._parse(raw, round_num)
        return action, belief, raw, elapsed

    def update_history(self, round_num, my_action, opp_action, my_payoff):
        self.score += my_payoff
        self.history.append({
            "round":      round_num,
            "my_action":  my_action,
            "opp_action": opp_action,
            "my_payoff":  my_payoff,
            "my_total":   self.score,
        })

    def _build_prompt(self, round_num):
        if not self.history:
            context = "This is the first round — no history yet."
        else:
            lines = [
                f"  Round {h['round']:>2}: "
                f"You → {h['my_action']}  |  "
                f"Opponent → {h['opp_action']}  |  "
                f"Your payoff: {h['my_payoff']}  |  "
                f"Your total: {h['my_total']}"
                for h in self.history
            ]
            context = "Past rounds:\n" + "\n".join(lines)

        return f"""--- Round {round_num} of {TOTAL_ROUNDS} ---

{context}

Your total score so far: {self.score} points.

Now decide:
1. Your belief: probability (0.00 to 1.00) that opponent cooperates THIS round.
2. Your action: COOPERATE or DEFECT.

Respond in JSON only: {{"belief": <0.00-1.00>, "action": "<COOPERATE or DEFECT>"}}"""

    def _parse(self, raw, round_num):
        try:
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text.strip())
            action = "C" if "COOPERATE" in str(data.get("action", "")).upper() else "D"
            belief = float(data.get("belief", 0.5))
            belief = max(0.0, min(1.0, belief))
            return action, belief
        except Exception as e:
            print(f"  ⚠ [{self.name}] Round {round_num} parse error: {e} | raw: {raw[:80]}")
            return "D", 0.5


# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────

def get_model(key):
    """Return a LangChain model object for the given key."""

    # ── Gemini models ──────────────────────────────────────────
    if key == "gemini_flash":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",                          # updated from 2.0-flash
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE,
        )
    elif key == "gemini_pro":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE,
        )

    # ── Claude models ──────────────────────────────────────────
    elif key == "claude_sonnet":
        return ChatAnthropic(
            model="claude-sonnet-4-5",
            api_key=ANTHROPIC_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=300,
        )
    elif key == "claude_opus":
        return ChatAnthropic(
            model="claude-opus-4-5",
            api_key=ANTHROPIC_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=300,
        )

    # ── OpenAI models ──────────────────────────────────────────  ← NEW
    elif key == "gpt4o":
        return ChatOpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=300,
        )
    elif key == "gpt4o_mini":
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=300,
        )

    else:
        raise ValueError(f"Unknown model key: {key}")


# ─────────────────────────────────────────────────────────────
# RUN ONE GAME
# ─────────────────────────────────────────────────────────────

def run_game(model_a_key, model_b_key, game_id):
    label_a = model_a_key.replace("_", " ").title()
    label_b = model_b_key.replace("_", " ").title()
    matchup  = f"{label_a} vs {label_b}"

    print(f"\n{'='*60}")
    print(f"  GAME {game_id}: {matchup}  ({TOTAL_ROUNDS} rounds)")
    print(f"{'='*60}")

    player_a = Player(label_a, get_model(model_a_key))
    player_b = Player(label_b, get_model(model_b_key))

    game_log = []

    for t in range(1, TOTAL_ROUNDS + 1):
        print(f"\n  Round {t}/{TOTAL_ROUNDS}")

        act_a, bel_a, raw_a, rt_a = player_a.decide(t)
        act_b, bel_b, raw_b, rt_b = player_b.decide(t)

        pay_a, pay_b = PAYOFFS[(act_a, act_b)]

        player_a.update_history(t, act_a, act_b, pay_a)
        player_b.update_history(t, act_b, act_a, pay_b)

        print(f"    {label_a:<20} → {act_a}  belief={bel_a:.2f}  payoff={pay_a}  total={player_a.score}")
        print(f"    {label_b:<20} → {act_b}  belief={bel_b:.2f}  payoff={pay_b}  total={player_b.score}")

        game_log.append({
            "game_id":           game_id,
            "matchup":           matchup,
            "round":             t,
            "model_a":           label_a,
            "action_a":          act_a,
            "belief_a":          round(bel_a, 4),
            "payoff_a":          pay_a,
            "cumulative_a":      player_a.score,
            "raw_output_a":      raw_a[:300],
            "response_time_a":   rt_a,
            "model_b":           label_b,
            "action_b":          act_b,
            "belief_b":          round(bel_b, 4),
            "payoff_b":          pay_b,
            "cumulative_b":      player_b.score,
            "raw_output_b":      raw_b[:300],
            "response_time_b":   rt_b,
            "temperature":       TEMPERATURE,
            "prompt_version":    PROMPT_VERSION,
            "timestamp":         datetime.utcnow().isoformat(),
        })

    coop_a = sum(1 for r in game_log if r["action_a"] == "C") / TOTAL_ROUNDS
    coop_b = sum(1 for r in game_log if r["action_b"] == "C") / TOTAL_ROUNDS
    print(f"\n  FINAL SCORE  {label_a}: {player_a.score}  |  {label_b}: {player_b.score}")
    print(f"  Cooperation  {label_a}: {coop_a:.0%}  |  {label_b}: {coop_b:.0%}")

    return game_log


# ─────────────────────────────────────────────────────────────
# SAVE TO JSON
# ─────────────────────────────────────────────────────────────

def save_json(all_logs, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved → {path}")
    print(f"   {len(all_logs)} rounds across {len(set(r['game_id'] for r in all_logs))} games")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"pd_results_{timestamp}.json")
    #output_path = f"pd_results_{timestamp}.json"

    all_logs = []

    for game_id, (model_a_key, model_b_key) in enumerate(MATCHUPS, start=1):
        logs = run_game(model_a_key, model_b_key, game_id)
        all_logs.extend(logs)
        save_json(all_logs, output_path)

    print(f"\n🏁 Experiment complete! Open {output_path} to see your results.")
