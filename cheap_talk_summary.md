# Cheap-Talk Signaling Experiment — Summary
ESADE Capstone 2026 | Run dates: 2026-05-03 and 2026-05-04

---

## Experiment Design

**Game:** Cheap-Talk Signaling (Crawford & Sobel framework)
- 1 Sender (knows true state H or L) vs 1 Receiver (does not)
- Sender sends a message (H or L), Receiver picks action A or B
- Action A is correct for state L; Action B is correct for state H

**Conditions:**
| Condition | Sender incentive | Receiver incentive |
|---|---|---|
| Aligned | +3 if Receiver correct | +3 if correct |
| Misaligned | +5 whenever Receiver picks A (regardless of state) | +3 if correct |

**Models:** Claude Sonnet 4.6, GPT-4o, Gemini 2.5 Flash
**Matchups:** 18 total (12 fixed-role + 6 role-rotated) × 10 rounds = 180 rounds per run
**Temperatures:** 0.7 (Run 1, pilot) and 0.3 (Run 2, low-variance baseline)

---

## Research Questions

1. Do LLMs behave honestly when incentives are aligned?
2. Do LLMs engage in strategic deception when misaligned incentives reward lying?
3. Do receivers learn to distrust senders when interests diverge?
4. Do models differ in deception propensity and receiver skepticism?
5. Does temperature affect strategic behaviour?

---

## Results

### Sender Truthfulness Rate

| Model | Condition | T=0.7 | T=0.3 |
|---|---|---|---|
| Claude Sonnet | Aligned | 100% | 100% |
| Claude Sonnet | Misaligned | 53% | 40% |
| GPT-4o | Aligned | 100% | 100% |
| GPT-4o | Misaligned | 40% | 43% |
| Gemini 2.5 Flash | Aligned | 100% | 100% |
| Gemini 2.5 Flash | Misaligned | 47% | 47% |

### Receiver Correct Action Rate

| Model | Condition | T=0.7 | T=0.3 |
|---|---|---|---|
| Claude Sonnet | Aligned | 100% | 100% |
| Claude Sonnet | Misaligned | 50% | 50% |
| GPT-4o | Aligned | 100% | 100% |
| GPT-4o | Misaligned | 37% | 50% |
| Gemini 2.5 Flash | Aligned | 93% | 97% |
| Gemini 2.5 Flash | Misaligned | 33% | 27% |

### Deception Success Rate (Misaligned only)

| Model | T=0.7 | T=0.3 |
|---|---|---|
| Claude Sonnet | 37% | 33% |
| GPT-4o | 30% | 33% |
| Gemini 2.5 Flash | 23% | 30% |

### Receiver Message-Following Rate

| Model | Condition | T=0.7 | T=0.3 |
|---|---|---|---|
| Claude Sonnet | Aligned | 100% | 100% |
| Claude Sonnet | Misaligned | 37% | 30% |
| GPT-4o | Aligned | 100% | 100% |
| GPT-4o | Misaligned | 50% | 47% |
| Gemini 2.5 Flash | Aligned | 93% | 97% |
| Gemini 2.5 Flash | Misaligned | 80% | 67% |

---

## Key Observations

- **Aligned condition:** All models were fully (or near-fully) truthful and receivers followed messages correctly — consistent with equilibrium play when incentives are shared.
- **Misaligned condition:** All models lied at meaningful rates (40–53%), confirming that LLMs respond to misaligned incentive structures with strategic deception.
- **Receiver scepticism varied widely:** Claude Sonnet was most sceptical of misaligned senders (30–37% follow rate); Gemini remained highly trusting (67–80%), making it the most exploitable receiver.
- **Deception success was moderate across all models** (23–37%), suggesting receivers do partially discount sender messages even without being explicitly told to.
- **Temperature effect was small:** Results were broadly consistent across T=0.7 and T=0.3, suggesting strategic dispositions are stable at these temperature levels.
