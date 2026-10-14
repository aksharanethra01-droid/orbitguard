# OrbitGuard
### Adaptive, explainable anomaly detection for satellite telemetry

OrbitGuard is a research prototype for analysing historical OPS-SAT telemetry fragments. It compares a simple statistical baseline with machine-learning detectors, calibrates an alert threshold on validation data, and presents evidence for human review. It does **not** connect to a live satellite or diagnose spacecraft failures.

## What is included
- React + Vite dashboard with overview, dataset explorer, model lab, alert review, and API-backed inference.
- FastAPI service for CSV upload, data validation, training/evaluation, prediction, and experiment history.
- Modular preprocessing/model pipeline: median imputation, scaling, Isolation Forest, One-Class SVM, supervised Random Forest when labels exist, threshold calibration, metrics, and confusion matrix.
- Experiment JSON/CSV artifacts, Dockerfiles, Compose, tests, report outline, and dataset instructions.

## Dataset (required for real results)
Use the official [OPSSAT-AD Zenodo dataset](https://zenodo.org/records/15108715) and its [benchmark paper](https://doi.org/10.1038/s41597-025-05035-3). The benchmark contains 2,123 labelled telemetry fragments and a recommended train/test split. Download the **tabular handcrafted-feature CSV** from the record and upload it through the UI, or place it at `data/dataset.csv` and call the API. Keep the paper's `train` split for a comparable benchmark. The app attempts to infer label and train columns; if your downloaded CSV uses different names, choose them in the UI/API. Do not report results until the dataset is loaded and experiments have actually run.

## Run locally (Windows PowerShell)
1. Install Python 3.11+ and Node.js 20+.
2. Backend:
   ```powershell
   cd backend
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   uvicorn app.main:app --reload --port 8000
   ```
3. Frontend in a second terminal:
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```
4. Open the Vite URL printed in the terminal (normally `http://localhost:5173`). API docs: `http://localhost:8000/docs`.

## Run with Docker
From the project root: `docker compose up --build`. UI: `http://localhost:8080`; API: `http://localhost:8000/docs`.

## Deploy to Render
This project works well as two Render web services:

1. Backend service (`backend/`): Python + FastAPI
   - Service type: Web
   - Root directory: `backend`
   - Build command: `pip install --no-cache-dir -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Env var: `CORS_ORIGINS=https://<your-frontend-domain>`
2. Frontend service (`frontend/`): static React build
   - Service type: Web (static site)
   - Root directory: `frontend`
   - Build command: `npm install && npm run build`
   - Publish directory: `dist`
   - Env var: `VITE_API_URL=https://<your-backend-domain>`

A ready-to-use Render config is included at `render.yaml`.

## Expected dataset columns
- One binary label column (auto-detected from `label`, `anomaly`, `is_anomaly`, or `target`; values 0=normal, 1=anomaly).
- Optional official split column (`train`, boolean/string or 0/1). If available, train/test follows it.
- All other numeric columns are features. Non-numeric IDs/metadata are excluded automatically. Rows with missing features are imputed using training data only.

## Research protocol
1. Confirm dataset version, file, columns, split and label encoding.
2. Run a statistical baseline and at least two ML models.
3. Calibrate threshold on validation data only; never tune on the test set.
4. Report precision, recall, F1, ROC-AUC (when score/labels support it), PR-AUC, MCC, confusion matrix and false-alarm/missed-anomaly counts.
5. Compare against the OPSSAT-AD paper only when split, preprocessing and metrics are comparable. Record differences.
6. Perform ablations (fixed vs calibrated threshold; with vs without alert ranking) and error analysis.

## Important
- Dashboard metrics stay empty until experiments run. No fake benchmark scores are bundled.
- Uploaded CSVs are processed in memory; this starter does not persist raw uploads.
- Alerts are decision-support flags, not verified failure diagnoses.
- For public deployment, set CORS to the deployed frontend origin and add authentication/rate limits before accepting untrusted uploads.

## Repository structure
`backend/` API and ML service · `frontend/` dashboard · `ml/` research scripts · `docs/` report plan and rubric checklist · `data/` place dataset here (not committed).
