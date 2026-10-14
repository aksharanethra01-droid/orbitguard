# Resume and demo notes

## Resume project entry (use after you have run the experiments)
**OrbitGuard — Adaptive Satellite Telemetry Anomaly Detection | Python, scikit-learn, FastAPI, React, Docker**
- Developed a full-stack research prototype for anomaly detection on historical ESA OPS-SAT telemetry, with CSV validation, leakage-aware preprocessing, and reproducible held-out evaluation.
- Benchmarked a robust statistical baseline, Isolation Forest, One-Class SVM, and a tuned supervised Random Forest using precision, recall, F1, ROC-AUC, PR-AUC, MCC, and false-alarm/missed-anomaly analysis.
- Built a React telemetry dashboard and FastAPI inference service; containerized the app with Docker Compose and recorded experiment results as JSON Lines.

Add a numeric performance claim only after running the official dataset and retaining the actual experiment log.

## Demo flow (3–5 minutes)
1. Explain the real problem and state that this is retrospective telemetry analysis, not live satellite control.
2. Open Dataset Explorer; upload the official tabular feature CSV from Zenodo; show row/feature counts, missing cells, label and official split.
3. Open Model Lab; explain baseline, unsupervised models, supervised comparator, and validation-only threshold selection; run the experiment.
4. Return to Overview; explain model comparison and confusion matrix, then feature importance and false alarm/missed anomaly trade-off.
5. Open Alert Review; explain that model-flagged fragments are candidates for human investigation.
6. Prefill the single-fragment inference form from the first dataset row and run the API prediction. Clarify that a single row is only scored by the trained research model and is not a diagnosis.
7. Show `/docs` API page and Docker Compose setup; discuss limitations and future work.
