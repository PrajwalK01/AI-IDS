# Research Notes: AI-Based Intrusion Detection & Prediction System

## Abstract
Intrusion Detection Systems (IDS) are critical for modern cybersecurity.  
This project extends traditional IDS by integrating **predictive modeling** to forecast potential attacks, moving from reactive detection to proactive defense.

## Motivation
- Cyber attacks are evolving rapidly, requiring adaptive defense.
- Current IDS solutions focus on detection but lack forecasting.
- Combining anomaly detection + risk scoring + time-series prediction creates a hybrid IDS with real-world impact.

## Methodology
1. **Datasets**
   - NSL-KDD (benchmark for IDS research)
   - CICIDS 2017 (modern traffic dataset with diverse attack types)

2. **Models**
   - Baseline ML: RandomForest, SVM, XGBoost
   - Deep Learning IDS: CNN, RNN, Transformer
   - Hybrid IDS + Prediction: LSTM, Prophet for time-series forecasting

3. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
   - Risk scoring: Severity levels for anomalies
   - Comparative benchmarking across datasets

## Results (Planned)
- Baseline ML performance on NSL-KDD
- Deep IDS improvements on CICIDS
- Hybrid IDS forecasting attack trends
- Publishable tables/graphs for academic review

## Future Work
- Real-time IDS deployment with streaming data
- Integration with SIEM tools
- Explainable AI for IDS decisions
- Expansion to IoT/Cloud attack scenarios

## References
- Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set"
- Sharafaldin et al., "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"
- Recent IEEE/ACM papers on IDS + forecasting
