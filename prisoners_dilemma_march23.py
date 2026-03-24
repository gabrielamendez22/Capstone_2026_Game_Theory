"""
=============================================================
ITERATED PRISONER'S DILEMMA — Multi-Model Experiment
ESADE Capstone 2026: Strategic Coherence in Large Language Models
=============================================================

Based on professor's TwoBotConversation structure.
Saves results to JSON files (no database needed).

INSTALL:
    pip install langchain langchain-anthropic langchain-google-genai

API KEYS (free tier available):
    Gemini  → https://aistudio.google.com/app/apikey    ← FREE
    Claude  → https://console.anthropic.com/            ← PAID (~$5 min)
"""

import json
import time
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os


# ─────────────────────────────────────────────────────────────
# CONFIGURATION — change settings here
# ─────────────────────────────────────────────────────────────

load_dotenv()
#GEMINI_API_KEY    = "YOUR_GEMINI_KEY_HERE"
#ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_KEY_HERE"

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

TOTAL_ROUNDS   = 20
TEMPERATURE    = 0.7
PROMPT_VERSION = "v1.0"

# Payoff matrix: T > R > P > S
T, R, P, S = 5, 3, 1, 0
PAYOFFS = {
    ("C", "C"): (R, R),   # Both cooperate       → (3, 3)
    ("C", "D"): (S, T),   # A cooperates, B defects → (0, 5)
    ("D", "C"): (T, S),   # A defects, B cooperates → (5, 0)
    ("D", "D"): (P, P),   # Both defect          → (1, 1)
}

# Matchups to run — add or remove pairs as needed
MATCHUPS = [
    ("gemini_flash", "gemini_flash"),   # FREE: Gemini vs Gemini (good for testing)
    ("gemini_pro",   "gemini_flash"),   # FREE: Gemini Pro vs Gemini Flash
    # ("gemini_pro", "claude_sonnet"), # PAID: uncomment when ready
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

RESPONSE FORMAT — you must respond with ONLY this JSON, nothing else:
{{"belief": <number between 0.00 and 1.00 = your probability that opponent cooperates>,
  "action": "<COOPERATE or DEFECT>"}}

Example: {{"belief": 0.72, "action": "COOPERATE"}}"""


# ─────────────────────────────────────────────────────────────
# PLAYER CLASS — based on professor's TwoBotConversation
# ─────────────────────────────────────────────────────────────
# Each player has:
#   - their own LangChain model object
#   - their own conversation history (what they remember)
#   - their own game history (past rounds, for building prompts)
#
# Crucially: Player A and Player B have SEPARATE conversation histories.
# Player A never sees Player B's internal "thinking" — only the
# outcome of each round (what B chose, after both have decided).

class Player:

    def __init__(self, name, model_obj):
        self.name = name
        self.model = model_obj
        self.score = 0

        # Conversation history: list of LangChain message objects
        # Starts empty — SystemMessage is prepended on every API call
        self.conversation = []

        # Game history: list of round summaries (used to build prompts)
        self.history = []

    def decide(self, round_num):
        """
        Ask this player to make a decision for the current round.
        They see all previous rounds but NOT the opponent's current decision.
        Returns (action, belief, raw_response).
        """

        # Build the prompt for this round
        prompt = self._build_prompt(round_num)

        # Add it to this player's conversation as a user message
        self.conversation.append(HumanMessage(content=prompt))

        # Call the model — prepend SystemMessage with game rules
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + self.conversation

        try:
            time.sleep(3)  # wait 3 seconds between calls
            t0 = time.time()
            response = self.model.invoke(full_messages)
            elapsed = round((time.time() - t0) * 1000, 1)
            raw = response.content.strip()
        except Exception as e:
            print(f"  ⚠ [{self.name}] API error: {e}")
            raw = '{"belief": 0.5, "action": "DEFECT"}'
            elapsed = 0

        # Add model's response to conversation memory
        self.conversation.append(AIMessage(content=raw))

        # Parse the JSON response
        action, belief = self._parse(raw, round_num)

        return action, belief, raw, elapsed

    def update_history(self, round_num, my_action, opp_action, my_payoff):
        """
        After BOTH players have decided, update this player's history
        with the outcome. This is the only moment a player learns
        what the opponent chose.
        """
        self.score += my_payoff
        self.history.append({
            "round":      round_num,
            "my_action":  my_action,
            "opp_action": opp_action,
            "my_payoff":  my_payoff,
            "my_total":   self.score,
        })

    def _build_prompt(self, round_num):
        """Build the round prompt showing past history."""
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
        """Extract action and belief from the model's JSON response."""
        try:
            # Remove markdown fences if present
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
            return "D", 0.5  # safe default


# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY — add models here
# ─────────────────────────────────────────────────────────────

def get_model(key):
    """Return a LangChain model object for the given key."""
    if key == "gemini_flash":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE,
        )
    elif key == "gemini_pro":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE,
        )
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
    else:
        raise ValueError(f"Unknown model key: {key}")


# ─────────────────────────────────────────────────────────────
# RUN ONE GAME
# ─────────────────────────────────────────────────────────────

def run_game(model_a_key, model_b_key, game_id):
    """
    Run one full game between two models.
    Returns a list of round records (one dict per round).

    SIMULTANEITY GUARANTEE:
    Each round, Player A decides first — but at that moment,
    Player A's conversation only contains history up to round (t-1).
    Player B then decides — same situation, only sees history up to (t-1).
    Only AFTER both have decided do we call update_history() on both,
    so neither player ever sees the opponent's current-round choice
    before making their own.
    """

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

        # ── Both players decide independently ──────────────────
        # Player A decides (sees history up to round t-1 only)
        act_a, bel_a, raw_a, rt_a = player_a.decide(t)

        # Player B decides (sees history up to round t-1 only)
        # Player B does NOT know what A just chose
        act_b, bel_b, raw_b, rt_b = player_b.decide(t)

        # ── Outcome is revealed to both ─────────────────────────
        pay_a, pay_b = PAYOFFS[(act_a, act_b)]

        # Now update both players with the result
        player_a.update_history(t, act_a, act_b, pay_a)
        player_b.update_history(t, act_b, act_a, pay_b)

        print(f"    {label_a:<20} → {act_a}  belief={bel_a:.2f}  payoff={pay_a}  total={player_a.score}")
        print(f"    {label_b:<20} → {act_b}  belief={bel_b:.2f}  payoff={pay_b}  total={player_b.score}")

        # Save this round's record
        game_log.append({
            "game_id":        game_id,
            "matchup":        matchup,
            "round":          t,
            "model_a":        label_a,
            "action_a":       act_a,
            "belief_a":       round(bel_a, 4),
            "payoff_a":       pay_a,
            "cumulative_a":   player_a.score,
            "raw_output_a":   raw_a[:300],
            "response_time_a": rt_a,
            "model_b":        label_b,
            "action_b":       act_b,
            "belief_b":       round(bel_b, 4),
            "payoff_b":       pay_b,
            "cumulative_b":   player_b.score,
            "raw_output_b":   raw_b[:300],
            "response_time_b": rt_b,
            "temperature":    TEMPERATURE,
            "prompt_version": PROMPT_VERSION,
            "timestamp":      datetime.utcnow().isoformat(),
        })

    # End of game summary
    coop_a = sum(1 for r in game_log if r["action_a"] == "C") / TOTAL_ROUNDS
    coop_b = sum(1 for r in game_log if r["action_b"] == "C") / TOTAL_ROUNDS
    print(f"\n  FINAL SCORE  {label_a}: {player_a.score}  |  {label_b}: {player_b.score}")
    print(f"  Cooperation  {label_a}: {coop_a:.0%}  |  {label_b}: {coop_b:.0%}")

    return game_log


# ─────────────────────────────────────────────────────────────
# SAVE TO JSON
# ─────────────────────────────────────────────────────────────

def save_json(all_logs, path):
    """Save all round records to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved → {path}")
    print(f"   {len(all_logs)} rounds across {len(set(r['game_id'] for r in all_logs))} games")


# ─────────────────────────────────────────────────────────────
# MAIN — runs all matchups and saves results
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"pd_results_{timestamp}.json"

    all_logs = []

    for game_id, (model_a_key, model_b_key) in enumerate(MATCHUPS, start=1):
        logs = run_game(model_a_key, model_b_key, game_id)
        all_logs.extend(logs)

        # Save after every game — so data is never lost if script crashes
        save_json(all_logs, output_path)

    print(f"\n🏁 Experiment complete! Open {output_path} to see your results.")
