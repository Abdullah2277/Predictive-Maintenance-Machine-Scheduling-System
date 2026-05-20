# P16 — Predictive Maintenance & Machine Scheduling System

> **Course:** Reinforcement Learning — Final Project  
> **Algorithm:** Deep Q-Network (DQN) + LSTM Failure Predictor  
> **Environment:** Custom 5-machine manufacturing simulator  

---

## Problem Statement

Unplanned equipment downtime costs manufacturers **$50 billion/year**. Preventive maintenance wastes resources by replacing parts too early; reactive maintenance causes costly production halts. This project implements an RL agent that learns a **joint policy** for machine scheduling and maintenance scheduling — deciding which jobs to run when and when to service machines — to minimise downtime while maximising production output.

---

## System Architecture

```
Sensor Data (temp, vibration, wear)
        ↓
LSTM Failure Predictor  →  Time-to-Failure estimate
        ↓
Factory State Vector (33 dims)
        ↓
DQN Agent  →  Actions for 5 machines
        ↓
Factory Environment  →  Reward + Next State
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| Machine Simulator | `simulator/machine.py` | Individual machine with degradation model |
| Factory Environment | `simulator/factory.py` | 5-machine RL environment |
| LSTM Predictor | `lstm/trainLstm.py` | Time-to-failure prediction (~2h error) |
| DQN Agent | `rlAgent/dqnAgent.py` | Single DQN for all 5 machines |
| Training Script | `trainDqn.py` | Full training pipeline with logging |
| Baselines | `baselines.py` | Random, Preventive, ML-only (RF), RL |
| Dashboard | `dashboard/app.py` | Streamlit web interface |

---

## Formal Reward Function

```
R = α·P − β·D − γ·M − δ·L + B

α = 10   (production value multiplier)
β = 150  (downtime penalty — unplanned > planned cost)
γ = 20/35 (maintenance / part replacement cost)
δ = 1    (deferral risk penalty)
B = 30   (production target completion bonus)
```

**Justification:** β=150 > γ=20 reflects the industry principle that unplanned downtime costs 3–5× more than scheduled maintenance.

---

## Job Scheduling Policy

The agent performs implicit job scheduling through action selection:

| Machine Health | Job Type | Production Value |
|---------------|----------|-----------------|
| ≥ 70% | Urgent | 25 units/cycle |
| 40–70% | Priority | 15 units/cycle |
| < 40% | Standard | 8 units/cycle |

Keeping machines healthy unlocks high-value urgent jobs — directly incentivising joint maintenance and production optimisation.

---

## Action Space

| Action | Description | Reward Effect |
|--------|-------------|---------------|
| 0 — Continue Production | Run assigned job | +α·P |
| 1 — Maintenance Now | Planned service | −γ=20 |
| 2 — Defer 24h | Keep running, small risk | −δ=1 |
| 3 — Replace Part | Near-full health restore | −γ=35 |

---

## Results

| Policy | Avg Reward | Avg Downtime | Avg Cost |
|--------|-----------|-------------|----------|
| Random Policy | 122,645 | 0.28h | $1,056 |
| Preventive (Fixed Interval) | 265,154 | 3.00h | $230 |
| Predictive (ML Only — RF) | 159,425 | 1.08h | $758 |
| **RL Agent (DQN)** | **296,955** | **1.76h** | **$726** |

**RL Agent achieves the highest reward — +12% over Preventive, +86% over ML-only.**

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip

### Install Dependencies

```bash
pip install torch numpy pandas scikit-learn streamlit plotly
```

### Project Structure

```
RL_Project/
├── simulator/
│   ├── machine.py          # Machine degradation model
│   └── factory.py          # 5-machine RL environment
├── lstm/
│   ├── generateData.py     # Synthetic training data generation
│   ├── trainLstm.py        # LSTM training
│   └── predictor.py        # LSTM inference
├── rlAgent/
│   └── dqnAgent.py         # DQN implementation
├── dashboard/
│   └── app.py              # Streamlit web interface
├── data/                   # Generated datasets and results
├── models/                 # Saved model weights
├── trainDqn.py             # RL training script
├── baselines.py            # Baseline comparison
└── README.md
```

---

## How to Run

### Option 1 — Run the Dashboard (Pre-trained models included)

```bash
streamlit run dashboard/app.py
```

### Option 2 — Reproduce Training from Scratch

**Step 1 — Generate LSTM training data:**
```bash
python -m lstm.generateData
```

**Step 2 — Train LSTM failure predictor:**
```bash
python -m lstm.trainLstm
```

**Step 3 — Train DQN agent:**
```bash
python trainDqn.py
```

**Step 4 — Run baseline comparison:**
```bash
python baselines.py
```

**Step 5 — Launch dashboard:**
```bash
streamlit run dashboard/app.py
```

### Reproducing Results

All random seeds are fixed:
- Factory environment: `seed=42`
- Baseline evaluation: `seed=200`
- LSTM data generation: deterministic per machine ID

---

## LSTM Failure Predictor

- **Architecture:** 2-layer LSTM, hidden size 64
- **Input:** 20-step sensor sequences (temperature, vibration, wear, health, hours since maintenance)
- **Output:** Time-to-failure estimate in hours
- **Validation error:** ~2.0 hours
- **Training data:** 61,311 samples from 200 simulated machine runs

---

## DQN Implementation Details

- **State space:** 33 dimensions (5 machines × 6 features + 3 system features)
- **Action space:** 4 discrete actions per machine (joint action for all 5)
- **Network:** 3-layer MLP (33 → 128 → 128 → 20)
- **Replay buffer:** 10,000 transitions
- **Target network:** Updated every 10 learning steps
- **Training:** 500 episodes, epsilon-greedy exploration (ε: 1.0 → 0.05)
- **Optimizer:** Adam, lr=0.0005

---

## Simulator Design

The manufacturing simulator uses a physics-inspired degradation model:

- Health degrades at base rate 0.3 units/cycle, accelerating with machine age
- Sensor readings deteriorate proportionally to the degradation factor
- Machine lifetimes (~300–350 cycles) are comparable to NASA CMAPSS FD001 benchmark (128–362 cycles), validating physical plausibility
- Sim-to-real methodology: standard in RL research where real-world training is impractical

**Reference:** Saxena & Goebel (2008), NASA CMAPSS Turbofan Engine Degradation Dataset.

---

## Simulator Justification

| Parameter | Our Simulator | NASA CMAPSS FD001 |
|-----------|--------------|-------------------|
| Mean lifetime | ~325 cycles | 206 cycles |
| Sensor count | 3 (temp, vib, wear) | 21 sensors |
| Degradation pattern | Nonlinear, age-accelerated | Nonlinear |
| Failure mode | Health → 0 | RUL → 0 |

---

## License

Academic project — NED University of Engineering & Technology
