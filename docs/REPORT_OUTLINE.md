# OrbitGuard report outline (IEEE style, maximum 10 pages)

## Title
OrbitGuard: An Adaptive and Evaluated Machine Learning Framework for Satellite Telemetry Anomaly Detection and Alert Prioritization

## Abstract (≤200 words)
Write this after experiments. State problem, official dataset, methods, evaluation protocol, actual best result, and limitations. Do not invent results.

## I. Introduction
- Why telemetry anomaly monitoring matters; human monitoring and fixed-threshold limitations.
- Scope: retrospective analysis of OPS-SAT telemetry fragments, not live spacecraft diagnosis.
- Research question: Does validation-based threshold calibration and alert prioritization improve useful detection compared with a simple baseline?

## II. Literature Review
- Ruszczak et al., “The OPS-SAT benchmark for detecting anomalies in satellite telemetry,” Scientific Data, vol. 12, art. 710, 2025, doi: 10.1038/s41597-025-05035-3.
- Review anomaly detection baselines and the official benchmark protocol.
- State the gap precisely: test the chosen pipeline/calibration/ablation combination; do not claim that satellite anomaly detection itself is new.

## III. Dataset and Methodology
- Dataset record/version, downloaded filenames, label encoding, features, official train flag and class distribution.
- Data flow: CSV → schema checks → training-only imputation/scaling → models → validation threshold → held-out test evaluation → dashboard.
- Define Isolation Forest, One-Class SVM, robust statistical baseline, and supervised Random Forest.
- Document hyperparameter grid and random seeds.
- Include a block diagram.

## IV. Experiments
- Environment: OS, CPU/GPU, Python and package versions.
- Exact split protocol and any deviations from benchmark.
- Metrics: precision, recall, F1, ROC-AUC, PR-AUC, MCC, false alarms, missed anomalies.
- Five or more charts: data/label distribution, model comparison, confusion matrix, ROC/PR curves, anomaly-score distribution/threshold, feature importance, and error trade-off.
- Ablation: fixed versus calibrated threshold; with versus without ranking.

## V. Results and Comparative Analysis
- Tables with actual measured values.
- Compare with published baseline only if split and metrics are compatible.
- Inspect false positives and false negatives; discuss consequences of each.
- Explain where the model performs poorly and why.

## VI. Application and Deployment
- React UI, FastAPI endpoints, validation, Docker build/run, deployment evidence and screenshots.
- Security and operational limitations.

## VII. Conclusion and Future Work
- Answer the research question from measured results.
- Future work: persistence, temporal model/autoencoder, continuous telemetry support, calibration drift, human feedback.

## References
Use IEEE numbered style. Verify bibliographic details and cite dataset DOI plus the benchmark paper.
