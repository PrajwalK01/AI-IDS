# AI-IDS
# AI-Based Intrusion Detection & Prediction System 🚀
Hybrid Intrusion Detection &amp; Prediction System using ML + Deep Learning (NSL-KDD, CICIDS datasets).

## Overview
A hybrid **Intrusion Detection System (IDS)** enhanced with **cyber attack prediction**.  
It combines anomaly detection, risk scoring, and time-series forecasting to move from **reactive detection → proactive defense**.

## Features
- Baseline IDS: ML classifiers (RandomForest, SVM, XGBoost).
- Deep IDS: CNN/RNN/Transformer-based anomaly detection.
- Hybrid IDS + Prediction: Forecast future threats using LSTMs/Prophet.
- Risk Scoring: Severity levels for detected anomalies.
- Benchmarking: NSL-KDD and CICIDS datasets.

## Repo Structure (Planned)
AI-IDS/
│── README.md
│── requirements.txt
│── src/
│   ├── data/            # Preprocessing + loaders
│   ├── models/          # ML, DL, Hybrid IDS
│   ├── evaluation/      # Metrics + risk scoring
│   └── utils/           # Helpers + configs
│── notebooks/           # EDA + experiments
│── docs/                # Research notes
│── tests/               # Unit tests


## Datasets
- [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
- [CICIDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)

## Roadmap
1. Data preprocessing + loaders
2. Baseline ML models
3. Deep learning IDS
4. Hybrid IDS + forecasting
5. Publishable research notes
