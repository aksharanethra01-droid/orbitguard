from __future__ import annotations
import io, json, logging, os, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .pipeline import inspect_dataset, run_experiments, _find_column, LABEL_CANDIDATES, predict_with_runtime

app = FastAPI(title="OrbitGuard API", version="1.0.0", description="Research prototype for satellite telemetry anomaly detection.")
logger = logging.getLogger(__name__)
origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in origins], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

DATA: pd.DataFrame | None = None
DATA_NAME = ""
LAST_EXPERIMENT: dict | None = None
RUNTIME: dict | None = None
HISTORY: list[dict] = []
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_UPLOAD_ROWS = 100000

class TrainRequest(BaseModel):
    label_column: Optional[str] = None
    split_column: Optional[str] = None
    contamination: float = Field(default=0.2, ge=0.01, le=0.5)
    tune: bool = True

class PredictRequest(BaseModel):
    model: str = "Isolation Forest"
    features: dict[str, float]

@app.get("/api/health")
def health():
    return {"status":"ok","service":"OrbitGuard API","dataset_loaded":DATA is not None,"dataset_name":DATA_NAME,"experiments_run":len(HISTORY)}

@app.get("/api/dataset")
def dataset_info():
    if DATA is None: return {"loaded":False,"message":"Upload the official OPSSAT-AD feature CSV to start real experiments."}
    info = inspect_dataset(DATA); info.update({"loaded":True,"name":DATA_NAME,"preview":DATA.head(8).where(pd.notna(DATA.head(8)),None).to_dict(orient="records")})
    return info

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    global DATA, DATA_NAME, LAST_EXPERIMENT, RUNTIME
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400,"Please upload a CSV file.")
    raw = bytearray()
    while len(raw) <= MAX_UPLOAD_BYTES:
        chunk = await file.read(min(UPLOAD_CHUNK_BYTES, MAX_UPLOAD_BYTES + 1 - len(raw)))
        if not chunk:
            break
        raw.extend(chunk)
    if len(raw) > MAX_UPLOAD_BYTES: raise HTTPException(413,"CSV exceeds 30 MB limit.")
    try:
        # Stop parsing as soon as the row limit is exceeded instead of loading
        # the rest of an otherwise valid, size-bounded CSV into memory.
        df = pd.read_csv(io.BytesIO(raw), nrows=MAX_UPLOAD_ROWS + 1)
        if df.empty: raise ValueError("CSV contains no rows.")
        if len(df) > MAX_UPLOAD_ROWS: raise ValueError(f"For this prototype, limit uploads to {MAX_UPLOAD_ROWS:,} rows.")
    except Exception as e: raise HTTPException(400,f"Could not read CSV: {e}")
    DATA, DATA_NAME, LAST_EXPERIMENT, RUNTIME = df, file.filename, None, None
    return {"message":"Dataset loaded in memory.","dataset":dataset_info()}

@app.post("/api/experiments/run")
def train(req: TrainRequest):
    global LAST_EXPERIMENT, HISTORY, RUNTIME
    if DATA is None: raise HTTPException(400,"Upload a labelled CSV first.")
    try: result = run_experiments(DATA, req.label_column, req.split_column, req.contamination, req.tune)
    except ValueError as e: raise HTTPException(400,str(e)) from e
    except Exception as e:
        logger.exception("Experiment run failed")
        raise HTTPException(500,"Experiment run failed. Check server logs for details.") from e
    RUNTIME = result.pop("_runtime", None)
    result["id"] = str(uuid.uuid4())[:8]; result["dataset_name"] = DATA_NAME
    LAST_EXPERIMENT = result; HISTORY = [result] + HISTORY[:9]
    # Custom experiment tracking: append full metric/parameter records as JSON Lines.
    log_path = Path(os.getenv("ORBITGUARD_LOG_PATH", "artifacts/experiments.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f: f.write(json.dumps(result, allow_nan=False) + "\n")
    return result

@app.get("/api/experiments/latest")
def latest():
    if LAST_EXPERIMENT is None: return {"available":False,"message":"Run an experiment to see genuine evaluation metrics."}
    return {"available":True,**LAST_EXPERIMENT}

@app.get("/api/experiments/history")
def history():
    return {"experiments":[{"id":x["id"],"created_at":x["created_at"],"dataset_name":x["dataset_name"],"best_model":x["best_model"],"test_rows":x["dataset"]["test_rows"],"metrics":x["metrics"]} for x in HISTORY]}

@app.post("/api/predict")
def predict(req: PredictRequest):
    if LAST_EXPERIMENT is None or RUNTIME is None:
        raise HTTPException(400, "Run experiments after loading a labelled dataset before requesting model inference.")
    if req.model not in RUNTIME["thresholds"]:
        raise HTTPException(400, "Unknown model. Select a model listed in the latest experiment.")
    try:
        result = predict_with_runtime(req.model, req.features, RUNTIME)
        return {**result, "disclaimer": "A model flag is not a diagnosis. Verify telemetry context and mission operating conditions."}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(500, "Prediction failed. Check server logs for details.") from e

@app.get("/api/alerts")
def alerts(limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0), model: str | None = None):
    if LAST_EXPERIMENT is None:
        return {"available":False,"model":model,"total_candidates":0,"offset":offset,"limit":limit,"has_more":False,"alerts":[]}
    result=LAST_EXPERIMENT; model=model or result["best_model"]
    candidates = _alert_candidates(result, model)
    page = candidates[offset:offset + limit]
    return {"available":True,"model":model,"total_candidates":len(candidates),"offset":offset,"limit":limit,"has_more":offset + len(page) < len(candidates),"alerts":page}

def _alert_candidates(result: dict, model: str) -> list[dict]:
    scored=result["predictions"].get(model)
    if not scored: raise HTTPException(400,"Model not found in latest experiment.")
    threshold = next((metric.get("threshold") for metric in result.get("metrics", []) if metric.get("model") == model), None)
    alerts=[]
    for i,(p,s) in enumerate(zip(scored["predictions"],scored["scores"])):
        if p:
            alerts.append({"fragment_id":f"TEST-{i+1:04d}","score":float(s),"threshold":threshold,"prediction":"Anomaly candidate","is_anomaly":True,"severity":"High" if s>=float(pd.Series(scored["scores"]).quantile(.9)) else "Review","model":model,"index_in_test":i,"note":"Severity is a score-ranking label; the prediction is the threshold-based anomaly flag. Inspect telemetry context; this is not a diagnosis."})
    return sorted(alerts,key=lambda x:x["score"],reverse=True)

@app.get("/api/alerts/{fragment_id}")
def alert_detail(fragment_id: str, model: str | None = None):
    if LAST_EXPERIMENT is None: raise HTTPException(404,"No experiment is available for candidate details.")
    model = model or LAST_EXPERIMENT["best_model"]
    candidate = next(
        (item for item in _alert_candidates(LAST_EXPERIMENT, model) if item["fragment_id"] == fragment_id),
        None,
    )
    if candidate is None: raise HTTPException(404,"Candidate not found in the selected detector's results.")
    runtime = RUNTIME or {}
    test_features = runtime.get("test_features")
    index = candidate["index_in_test"]
    features = (
        {
            str(column): None if pd.isna(value) else float(value)
            for column, value in test_features.iloc[index].items()
        }
        if isinstance(test_features, pd.DataFrame) and index < len(test_features)
        else None
    )
    thresholds = runtime.get("thresholds", {})
    threshold = thresholds.get(model)
    if threshold is None:
        threshold = next(
            (metric.get("threshold") for metric in LAST_EXPERIMENT.get("metrics", []) if metric.get("model") == model),
            None,
        )
    return {
        **candidate,
        "threshold": threshold,
        "features": features,
    }

@app.get("/api/features")
def feature_schema():
    if LAST_EXPERIMENT is None: return {"available":False,"features":[]}
    return {"available":True,"features":LAST_EXPERIMENT["dataset"]["feature_columns"]}
