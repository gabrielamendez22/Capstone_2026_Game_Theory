# Prosociality with AI Through Game Theory

**ESADE Master in Business Analytics — Capstone Project 2026 (P264)**

Supervised by: Carlos Carrasco-Farré, PhD.

---

## Overview

This project investigates whether large language models (LLMs) exhibit **stable, transferable strategic dispositions** across diverse game-theoretic environments — or whether their behavior is context-dependent and reactive. We test six models across three classic games from behavioral economics and game theory to estimate a multi-dimensional strategic profile for each.

### Core Question

Do LLMs possess coherent strategic architectures — behavioral parameters that respond predictably to the structural features of strategic environments — or does their behavior reflect prompt-level pattern matching that collapses under structural variation?

---

## Models Tested

- Claude Opus
- Claude Sonnet  
- GPT-4o
- GPT-4o-mini
- Gemini 2.5 Flash
- Gemini 2.5 Flash Lite

---

## The Three Games

### 1. Repeated Prisoner's Dilemma (PD)
Measures **conditional reciprocity (ρ)**: the tendency to cooperate when opponents cooperate, and defect when they defect. Tests whether models learn and respond to tit-for-tat dynamics.

**Parameter:** ρ = P(C at t | opponent C at t−1) − P(C at t | opponent D at t−1)

### 2. Commons Dilemma (CD)
Measures **extraction behavior (θ)** and resource sustainability. Tests whether models calibrate their consumption to the regeneration rate of a shared pool.

**Parameter:** θ = Pool size at which model's extraction first exceeds per-capita regeneration rate

### 3. Cheap-Talk Signaling Game
Measures **honesty (η)** and **receiver susceptibility (γ)**. Tests whether models communicate truthfully under costless signaling and whether receivers believe the signals.

**Parameters:**
- η = Proportion of rounds where sent message matched intended action
- γ = P(receiver cooperates | message="cooperate") − P(receiver cooperates | message="defect")

---

## The Strategic Profile Vector

For each model, we estimate a comprehensive strategic profile:

**Sm = (ρ, θ, η, γ, β)**

| Parameter | Name | Environment | Meaning |
|---|---|---|---|
| **ρ** | Reciprocity coefficient | Prisoner's Dilemma | Tendency to mirror opponent behavior |
| **θ** | Extraction weight | Commons Dilemma | How aggressively the model extracts from shared resources |
| **η** | Honesty rate | Cheap-Talk | Proportion of truthful signals |
| **γ** | Receiver gullibility | Cheap-Talk | Susceptibility to incoming signals |
| **β** | Belief calibration | All environments | Accuracy of stated beliefs vs. actual outcomes |

---

## Repository Structure

```
.
├── experiments/                          # Core experiment runners
│   ├── cheap_talk_langchain.py          # Cheap-Talk experiment (LangChain-based)
│   ├── prisoners_dilemma_langchain.py   # Prisoner's Dilemma experiment
│   ├── commons_dilemma_langchain.py     # Commons Dilemma experiment
│   └── prisoners_dilemma_langchain_HE.py # PD with human expert baseline
│
├── analysis/                             # Post-experiment analysis
│   ├── human_prior_builder.py           # Constructs human benchmark baseline
│   ├── json_to_csv.py                   # Converts raw output to tabular format
│   └── human_priors.json                # Human experimental benchmarks
│
├── data/                                 # Data organization
│   ├── raw/                             # Raw experiment outputs (SQLite + CSV)
│   ├── processed/                       # Cleaned, aggregated results
│   └── human_benchmarks/                # Human behavioral data (Gneezy 2005, etc.)
│
├── dashboard/                            # Interactive Dash visualization
│   ├── app.py                           # Main Dash application
│   ├── data_loader.py                   # Data pipeline for dashboard
│   ├── pages/                           # Multi-page layout
│   ├── tabs/                            # Tab-based visualization modules
│   ├── assets/                          # CSS and static assets
│   └── requirements.txt                 # Dash dependencies
│
├── all_raw/                             # Raw experiment data (large)
│   ├── cd_*.csv / .db                   # Commons Dilemma raw results
│   ├── pd_*.csv / .db                   # Prisoner's Dilemma raw results
│   └── cheap_talk_*.csv                 # Cheap-Talk raw results
│
├── METHODOLOGY.md                       # Formal definitions of all parameters
├── CHANGELOG.md                         # Development history
├── RESEARCH_JOURNAL.md                  # Detailed findings & notes
├── requirements.txt                     # Python dependencies
├── environment.yml                      # Conda environment spec
├── build_strategic_profiles.py          # Aggregates raw results into Sm vectors
└── README.md                            # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- Conda (recommended) or pip
- API keys for LLM providers (OpenAI, Anthropic, Google)

### Steps

1. **Clone and navigate:**
   ```bash
   cd Capstone_2026_Game_Theory
   ```

2. **Create environment:**
   ```bash
   conda env create -f environment.yml
   conda activate capstone-2026
   ```
   
   Or with pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys:**
   Create a `.env` file (or set environment variables):
   ```
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=...
   ```

4. **Verify installation:**
   ```bash
   python -c "import langchain; print('LangChain installed')"
   ```
---

## Running Experiments

Each game is run independently via LangChain-based orchestration. All experiments log to SQLite for crash recovery and export to CSV for analysis.

### Cheap-Talk Signaling Game
```bash
python experiments/cheap_talk_langchain.py \
  --models "claude-opus" "gpt-4o" \
  --rounds 50 \
  --conditions "aligned_incentives" "misaligned_incentives" "human_prior"
```

### Prisoner's Dilemma
```bash
python experiments/prisoners_dilemma_langchain.py \
  --models "claude-sonnet" "gemini-2.5-flash" \
  --rounds 20 \
  --iterations 10
```

### Commons Dilemma
```bash
python experiments/commons_dilemma_langchain.py \
  --models "gpt-4o-mini" \
  --pool_size 100 \
  --rounds 30
```

### Human Expert Baseline (PD Only)
```bash
python experiments/prisoners_dilemma_langchain_HE.py \
  --iterations 10
```

---

## Analyzing Results

### Build Strategic Profiles
Aggregates raw per-round data into model-level strategic vectors:

```bash
python build_strategic_profiles.py \
  --input_dir all_raw/ \
  --output_file data/processed/strategic_profiles.csv
```

Output: CSV with columns [model, ρ, θ, η, γ, β, ρ_ci_lower, ρ_ci_upper, ...]

### Convert JSON Logs to CSV
```bash
python analysis/json_to_csv.py --input cheap_talk_experiment.log
```

### Build Human Benchmarks
```bash
python analysis/human_prior_builder.py --output analysis/human_priors.json
```

---

## Interactive Dashboard

Visualize and explore results in real time.

```bash
cd dashboard
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:8050` in your browser.

**Features:**
- **Overview** — summary statistics and model rankings
- **Explorer** — per-model, per-game drilldown
- **Metrics** — distribution plots and confidence intervals
- **Fingerprints** — radar charts of strategic profiles
- **Simulation** — what-if analysis (coming soon)

---

## Key Findings

### Honesty Under Misalignment
Models exhibit **incentive-contingent honesty**: aligned conditions show ~95% η, while misaligned conditions drop to ~40–50%, validating the strategic nature of deception.

### Receiver Gullibility
Large variance in γ across models suggests no universal "receiver sophistication" — different models rely on different cues (message text, length, repetition).

### Reciprocity Patterns
Weak PD reciprocity ρ across most models indicates limited ability to learn opponent strategies over 20 rounds. Claude models show marginally higher ρ.

### Cross-Game Consistency
Spearman rank correlation of strategic profiles across PD, CD, and CT is ~0.3–0.5, suggesting **partial transferability** of strategic dispositions rather than full architectural coherence.

---

## Data & Files

### Raw Experiments (`all_raw/`)
- **Format:** CSV (human-readable) + SQLite (crash recovery)
- **Naming:** `[game]_[condition]_[timestamp].{csv|db}`
- **Examples:** 
  - `cheap_talk_misaligned_incentives_20260520_123456.csv`
  - `pd_undisclosed_temp6_20260603_101245_5.csv`

### Processed Results (`data/processed/`)
- **strategic_profiles.csv** — aggregated Sm vectors with CIs
- **cross_game_correlations.csv** — Spearman ρ by model pair
- **summary_statistics.json** — benchmark comparisons vs. human

### Human Benchmarks (`data/human_benchmarks/`)
- Gneezy (2005) cheap-talk data
- Experimental cheapening rates
- Baseline deception rates

---

## Documentation

- **METHODOLOGY.md** — formal definitions, parameter specs, validity checks
- **RESEARCH_JOURNAL.md** — detailed field notes, iterative findings, debugging logs
- **CHANGELOG.md** — experiment runs, version history, decisions
- **CLAUDE.md** — Claude-specific prompts and insights

---

## Team

- **Giorgio Fiorentino**
- **Gabriela Méndez**
- **María Mora**
- **Andrés Ramírez**
- **Samreen Siddique** 

**Supervisor:** Carlos Carrasco-Farré, PhD., ESADE Business School

---

## Dependencies

**Core:**
- `langchain>=0.1.0` — LLM orchestration
- `anthropic`, `openai`, `google-generativeai` — LLM APIs
- `pandas`, `numpy` — data processing
- `scipy` — statistical inference (Wilson CIs)

**Analysis:**
- `matplotlib`, `seaborn` — visualization
- `plotly`, `dash` — interactive dashboard

**Experimentation:**
- `sqlite3` — crash-safe logging
- `python-dotenv` — API key management

See `requirements.txt` and `environment.yml` for complete specs.

---

## Methodology References

- Axelrod, R. (1984). *The Evolution of Cooperation*. Basic Books.
- Gneezy, U. (2005). Deception: The role of consequences. *American Economic Review*, 95(1), 384–394.
- Crawford, V. P., & Sobel, J. (1982). Strategic information transmission. *Econometrica*, 50(6), 1431–1451.
- Hardin, G. (1968). The tragedy of the commons. *Science*, 162(3859), 1243–1248.

---

## License

This project is part of the ESADE 2026 Capstone Program. For academic use only.

---

## Feedback & Questions

For technical issues, data integrity concerns, or experimental questions, see `RESEARCH_JOURNAL.md` or contact the team via the project repository.
