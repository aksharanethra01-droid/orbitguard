# CAT II rubric coverage checklist

Use this as a completion checklist; do not claim a criterion is complete until evidence exists.

## 1. Data engineering & visualisation — 8 marks
- [x] Real, domain-specific labelled benchmark selected: OPSSAT-AD (official Zenodo source).
- [x] CSV schema inspection, missingness count, numeric feature selection.
- [x] Training-only median imputation and scaling.
- [ ] Inspect channel distributions, label balance, missingness patterns and split integrity.
- [ ] Add at least five evidence-backed visualizations to the report/dashboard (model comparison, confusion matrix, score distribution/threshold, PR/ROC curves, feature importance, error trade-off).
- [ ] Record any additional dimensionality reduction (PCA/UMAP) and explain why it helps; do not add it just to tick a box.

## 2. Advanced modelling & pipeline optimization — 10 marks
- [x] Modular scikit-learn pipeline.
- [x] Statistical baseline, Isolation Forest, One-Class SVM and supervised Random Forest comparator.
- [x] Optional GridSearchCV for Random Forest.
- [x] Validation-only threshold selection; test set not used for threshold calibration.
- [ ] Save a full experiment log and chosen hyperparameters with every run; currently API keeps recent runs in memory.
- [ ] Consider an autoencoder only after baselines are stable and data shape is understood.

## 3. Benchmarking & comparative analysis — 10 marks
- [x] Metrics: precision, recall, F1, accuracy, ROC-AUC, PR-AUC, MCC, confusion matrix, false alarms and missed anomalies.
- [ ] Reproduce the official split exactly and compare with the paper's baseline table only where protocol/metrics match.
- [ ] Run ablations: calibrated vs fixed threshold; alert ranking on/off; random split vs official split where applicable.
- [ ] Document failure cases and limitations.

## 4. Deployment & containerization — 8 marks
- [x] FastAPI input validation and API docs.
- [x] Dockerfiles and Docker Compose for local containerized deployment.
- [ ] Build and test containers locally.
- [ ] Deploy API and UI to a public host (e.g. Hugging Face Spaces-compatible setup or cloud VM) and record actual live URL. No deployment is claimed yet.
- [ ] Configure production CORS, upload limits, authentication/rate limits and secrets.

## 5. Documentation & report quality — 4 marks
- [x] Modular code and README setup instructions.
- [x] IEEE-style report outline in `REPORT_OUTLINE.md`.
- [ ] Complete a max-10-page report with real experimental results and verified IEEE references.
- [ ] Add screenshots, environment versions, GitHub URL and public deployment URL.

## Resume-safe project claim
"Built a research prototype for anomaly detection on historical ESA OPS-SAT telemetry, comparing statistical and machine-learning detectors with leakage-aware evaluation, validation-calibrated thresholds, error analysis, and a React/FastAPI dashboard." Add quantitative performance only after running the official dataset and recording the actual results.
