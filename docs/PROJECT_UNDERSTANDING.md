# AI-IDS — Complete Project Understanding

## What This Project Actually Does

Think of it like a **security camera system for computer networks**, but instead of watching
for physical intruders, it watches network traffic and uses AI to detect and predict cyber attacks.

---

## How It Works — Step by Step

```
Real Network Traffic (NSL-KDD / CICIDS datasets)
         │
         ▼
┌─────────────────────┐
│   PREPROCESSING     │  → cleans data, encodes categories,
│   preprocess.py     │    normalises numbers, splits train/test
└─────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                  THREE LAYERS OF AI                  │
│                                                      │
│  Layer 1 – Baseline ML        Layer 2 – Deep Learning│
│  ┌────────────────┐            ┌────────────────────┐│
│  │ RandomForest   │            │ MLP (neural net)   ││
│  │ XGBoost        │            │ CNN-1D             ││
│  │ SVM            │            │ LSTM               ││
│  └────────────────┘            │ Transformer        ││
│                                └────────────────────┘│
│                                                      │
│  Layer 3 – Forecasting                               │
│  ┌──────────────────────────────┐                    │
│  │ LSTM Forecaster + Prophet    │                    │
│  │ (predicts FUTURE attacks)    │                    │
│  └──────────────────────────────┘                    │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   EVALUATION        │  → Accuracy, F1, ROC-AUC,
│   metrics.py        │    confusion matrix, ROC curve plots
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   RISK SCORING      │  → Labels each connection as
│                     │    LOW / MEDIUM / HIGH threat
└─────────────────────┘
```

---

## What Each Part Does in Plain English

### The Data (NSL-KDD & CICIDS)

These are recordings of real network connections — each row is one connection with ~41 features like:
- How long it lasted
- Which protocol (TCP/UDP/ICMP)
- How many bytes were sent
- Whether login failed
- Port scan patterns

Each row is labelled `normal` or one of 20+ attack types (DoS, port scan, probe, R2L, U2R).

### Layer 1 — Baseline ML

Fast, interpretable models. RandomForest and XGBoost achieve **ROC-AUC ~0.96** on NSL-KDD
in seconds. These are your production-ready detectors.

### Layer 2 — Deep Learning

Learns more complex patterns in the traffic. The **Transformer** model hit **0.959 ROC-AUC**
even in fast mode with only 5 epochs — it is the strongest model in the project.

### Layer 3 — Forecasting (what makes this unique)

Most IDS tools just say *"attack detected right now."* This project goes further:

- Aggregates attack counts over time into a time series
- LSTM learns the pattern (e.g., attacks spike on weekends, or escalate over hours)
- Predicts the **next 5 time windows** of attack volume
- Prophet adds trend + seasonality decomposition

**The difference:** reactive detection → proactive defense.
You can alert the security team *before* the attack wave peaks.

### Risk Scoring

Every network connection gets a probability score and a severity label:
- `< 0.5`    → **Low**    (probably normal)
- `0.5–0.8`  → **Medium** (suspicious, monitor)
- `> 0.8`    → **High**   (block / alert immediately)

---

## Real-World Use Cases

| Scenario | How AI-IDS Helps |
|---|---|
| Enterprise network monitoring | Deploy as a sidecar to a firewall — score every connection in real time |
| SOC (Security Operations Centre) | Feed High-risk alerts to analysts, auto-suppress Low-risk noise |
| Cloud infrastructure | Detect lateral movement, port scans, brute-force attempts |
| Research / academic | Benchmark new IDS algorithms against NSL-KDD/CICIDS baselines |
| IoT networks | Lightweight MLP model can run on edge devices |

---

## Advanced Project Views — What This Can Become

### 1. Real-Time Streaming IDS

```
Network tap / Kafka stream
        ↓
  AI-IDS inference engine  (model loaded, scoring each packet live)
        ↓
  SIEM dashboard (Splunk / Elastic / Grafana)
        ↓
  Auto-block rule pushed to firewall
```

The models are already saved to disk (`saved_models/`). The next step is wrapping them in a
Flask/FastAPI server that accepts a JSON payload of features and returns a risk score in milliseconds.

### 2. Explainable AI Layer

Right now the model says *"High risk."* Security analysts need to know *why*.
Adding SHAP values would give output like:

> "High risk because: `src_bytes=9999` (+0.42), `serror_rate=1.0` (+0.38), `flag=S0` (+0.31)"

That maps directly to actionable intelligence.

### 3. Federated / Distributed IDS

Multiple organisations share model updates without sharing raw traffic data.
Each site trains locally, gradients are aggregated centrally — solves the privacy problem
that stops real-world IDS data sharing.

### 4. Adversarial Robustness

Attackers are starting to craft traffic specifically to fool ML-based IDS (adversarial examples).
Adding adversarial training would make the models robust to evasion attacks.

### 5. Full Research Paper Pipeline

The project is already structured for this:

```
Experiment run → results dict → benchmark table → LaTeX export
```

You have baseline + deep learning + forecasting all benchmarked on the same dataset
with the same metrics. That is a publishable comparison table.

---

## Current Results Summary

| Model | Accuracy | ROC-AUC | Speed |
|---|---|---|---|
| RandomForest | 0.771 | 0.946 | 10s |
| XGBoost | 0.800 | 0.969 | 3s |
| SVM | 0.784 | 0.927 | 36s |
| MLP | 0.782 | 0.929 | fast-mode |
| CNN-1D | 0.778 | 0.877 | fast-mode |
| LSTM | 0.749 | 0.922 | fast-mode |
| **Transformer** | **0.825** | **0.959** | fast-mode |

XGBoost wins on speed + accuracy for production.
Transformer wins on accuracy for research.
The DL numbers will improve significantly with full training (15 epochs, all 125k rows).

---

## Project File Map

```
AI-IDS/
│
├── main.py                        ← Run the full pipeline from here
│   │                                python main.py --dataset nslkdd --mode baseline
│   │                                python main.py --dataset nslkdd --mode deep --fast
│   │                                python main.py --dataset nslkdd --mode forecast
│
├── data/
│   ├── NSL-KDD-Train.csv          ← 125,973 labelled network connections (training)
│   └── NSL-KDD-Test.csv           ← 22,544 labelled connections (testing)
│
├── src/
│   ├── data/
│   │   └── preprocess.py          ← Load → clean → encode → scale → split
│   │
│   ├── models/
│   │   ├── baseline.py            ← RandomForest, XGBoost, SVM
│   │   ├── deep_ids.py            ← MLP, CNN-1D, LSTM, Transformer (PyTorch)
│   │   └── forecasting.py         ← LSTM time-series + Prophet
│   │
│   ├── evaluation/
│   │   └── metrics.py             ← Accuracy/F1/ROC-AUC, risk scoring, plots
│   │
│   └── utils/
│       ├── config.py              ← All hyperparams and paths in one place
│       └── helpers.py             ← Logger, seed, Timer
│
├── saved_models/                  ← Trained models saved here (--save flag)
│
├── outputs/                       ← Confusion matrices + ROC curves saved here
│
├── notebooks/
│   └── EDA.ipynb                  ← Exploratory data analysis
│
├── tests/
│   ├── test_models.py             ← Unit tests: baseline ML + metrics
│   ├── test_preprocess.py         ← Unit tests: data pipeline
│   └── test_deep_ids.py           ← Unit tests: all 4 DL models + forecaster
│
└── docs/
    ├── research_notes.md          ← Academic methodology and references
    └── PROJECT_UNDERSTANDING.md   ← This file
```

---

## How to Run (Quick Reference)

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline ML only (fast, ~1 minute)
python main.py --dataset nslkdd --mode baseline

# Run all DL models quickly on CPU (--fast = 5 epochs, 20k rows)
python main.py --dataset nslkdd --mode deep --fast

# Run full DL training (15 epochs, 125k rows, ~20 min on CPU)
python main.py --dataset nslkdd --mode deep

# Run attack forecasting
python main.py --dataset nslkdd --mode forecast

# Run everything and save models to disk
python main.py --dataset nslkdd --mode all --save

# Run all unit tests
pytest tests/ -v
```

---

## What "Positive (attack) rate: 46.54%" Means

When you run the pipeline you see this log line. It means 46.54% of the training
connections are attacks, and 53.46% are normal. This is a fairly balanced dataset —
good for training. The test set is 56.92% attacks (slightly harder, more real-world).

---

## The One-Line Summary

**AI-IDS is a research-grade intrusion detection system that does not just detect attacks —
it learns attack patterns, scores their severity, and forecasts when the next wave is coming.**
