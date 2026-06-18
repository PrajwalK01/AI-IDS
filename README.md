# AI-IDS — AI-Based Intrusion Detection & Prediction System

A **hybrid Intrusion Detection System** that combines classical ML, deep learning, and time-series forecasting to move from *reactive detection* to *proactive cyber-threat prediction*.

---

## What's inside

| Layer | Models |
|---|---|
| Baseline ML | RandomForest, XGBoost, SVM |
| Deep Learning | MLP, CNN-1D, LSTM, Transformer |
| Forecasting | LSTM Forecaster, Facebook Prophet |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, Risk Scoring |

---

## Project structure

```
AI-IDS/
├── main.py                  ← pipeline entrypoint (run this)
├── requirements.txt
├── README.md
├── src/
│   ├── data/
│   │   └── preprocess.py    ← load, encode, normalise NSL-KDD / CICIDS
│   ├── models/
│   │   ├── baseline.py      ← RandomForest, XGBoost, SVM
│   │   ├── deep_ids.py      ← MLP, CNN-1D, LSTM, Transformer (PyTorch)
│   │   └── forecasting.py   ← LSTM Forecaster + Prophet wrapper
│   ├── evaluation/
│   │   └── metrics.py       ← metrics, risk scoring, ROC/CM plots
│   └── utils/
│       ├── config.py        ← all hyperparams & paths in one place
│       └── helpers.py       ← logger, seed, Timer
├── notebooks/
│   └── EDA.ipynb
├── docs/
│   └── research_notes.md
└── tests/
    ├── test_models.py
    ├── test_preprocess.py
    └── test_deep_ids.py
```

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets
Place files in the `data/` folder:
- **NSL-KDD**: [download here](https://www.unb.ca/cic/datasets/nsl.html)
  - `data/NSL-KDD-Train.csv`
  - `data/NSL-KDD-Test.csv`
- **CICIDS 2017**: [download here](https://www.unb.ca/cic/datasets/ids-2017.html)
  - `data/CICIDS2017.csv`

### 3. Run the pipeline
```bash
# Baseline ML on NSL-KDD
python main.py --dataset nslkdd --mode baseline

# Deep learning models
python main.py --dataset nslkdd --mode deep

# All models + save to disk
python main.py --dataset nslkdd --mode all --save

# Attack forecasting
python main.py --dataset nslkdd --mode forecast

# CICIDS dataset
python main.py --dataset cicids --mode baseline
```

### 4. Run tests
```bash
pytest tests/ -v
```

Outputs (confusion matrices, ROC curves) are saved to `outputs/`.  
Saved models land in `saved_models/`.

---

## Roadmap

- [x] Data preprocessing — encoding, scaling, NSL-KDD & CICIDS loaders
- [x] Baseline ML — RandomForest, XGBoost, SVM
- [x] Deep Learning IDS — MLP, CNN-1D, LSTM, Transformer
- [x] Attack Forecasting — LSTM time-series + Prophet
- [x] Risk scoring — Low / Medium / High severity per sample
- [x] Full evaluation suite — ROC-AUC, confusion matrix, benchmark table
- [x] Config system — all hyperparams in one file
- [x] Unit tests
- [ ] Real-time streaming inference (Kafka / socket)
- [ ] SHAP explainability for model decisions
- [ ] SIEM integration (Splunk / Elastic)
- [ ] IoT / Cloud attack scenario experiments

---

## Datasets
- [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
- [CICIDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)
