"""Leakage-aware baseline and anomaly detection pipeline for tabular telemetry features."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

LABEL_CANDIDATES = ["label", "anomaly", "is_anomaly", "target", "class", "y"]
SPLIT_CANDIDATES = ["train", "split", "subset", "is_train"]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    low = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in low: return low[candidate]
    return None


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    label_col = _find_column(df, LABEL_CANDIDATES)
    split_col = _find_column(df, SPLIT_CANDIDATES)
    features = [c for c in df.columns if c not in {label_col, split_col} and pd.api.types.is_numeric_dtype(df[c])]
    return {"rows": int(len(df)), "columns": int(len(df.columns)), "label_column": str(label_col) if label_col is not None else None,
            "split_column": str(split_col) if split_col is not None else None, "feature_columns": [str(c) for c in features],
            "numeric_features": len(features), "missing_cells": int(df[features].isna().sum().sum()) if features else 0,
            "label_distribution": ({str(k): int(v) for k,v in df[label_col].value_counts(dropna=False).items()} if label_col is not None else {}),
            "columns": [str(c) for c in df.columns]}


def _binary_labels(series: pd.Series) -> np.ndarray:
    if series.isna().any():
        missing = int(series.isna().sum())
        raise ValueError(f"Label column contains {missing} missing value(s); remove or correct them before training.")
    if pd.api.types.is_numeric_dtype(series):
        vals = sorted(pd.Series(series.dropna().unique()).tolist())
        if len(vals) != 2: raise ValueError(f"Label column must have exactly two classes; found {vals[:10]}")
        # Conventional 0/1 is preserved; otherwise the larger sorted class is anomaly.
        return (series.astype(float).to_numpy() == max(vals)).astype(int)
    s = series.astype(str).str.strip().str.lower()
    normal = {"0", "normal", "false", "no", "nominal", "benign"}
    anomaly = {"1", "anomaly", "anomalous", "true", "yes", "abnormal", "outlier"}
    unknown = set(s.unique()) - normal - anomaly
    if unknown: raise ValueError(f"Unrecognised labels: {list(unknown)[:8]}. Map labels to normal/0 and anomaly/1 first.")
    return s.isin(anomaly).astype(int).to_numpy()


def _train_mask(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    return s.isin({"1", "true", "train", "training", "yes"}).to_numpy()


def _metrics(y: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y, pred, labels=[0,1])
    out = {"accuracy": float(accuracy_score(y,pred)), "precision": float(precision_score(y,pred,zero_division=0)),
           "recall": float(recall_score(y,pred,zero_division=0)), "f1": float(f1_score(y,pred,zero_division=0)),
           "mcc": float(matthews_corrcoef(y,pred)), "confusion_matrix": cm.tolist(),
           "false_alarms": int(cm[0,1]), "missed_anomalies": int(cm[1,0]),
           "support": {"normal": int((y==0).sum()), "anomaly": int((y==1).sum())}}
    try: out["roc_auc"] = float(roc_auc_score(y, scores))
    except Exception: out["roc_auc"] = None
    try: out["pr_auc"] = float(average_precision_score(y, scores))
    except Exception: out["pr_auc"] = None
    return out


def _best_model_from_validation(validation_f1: dict[str, float]) -> str:
    """Select the dashboard default using validation results only."""
    if not validation_f1:
        raise ValueError("No validation scores are available for model selection.")
    return max(validation_f1, key=validation_f1.get)


def _score_model(model_name: str, transformed: np.ndarray, runtime: dict[str, Any]) -> float:
    if model_name == "Robust statistical baseline":
        med = np.asarray(runtime["stat_median"], dtype=float)
        mad = np.asarray(runtime["stat_mad"], dtype=float)
        mad = np.where(np.isfinite(mad) & (mad > 0), mad, 1.0)
        score = np.mean(np.abs(transformed - med) / mad, axis=1)
        return float(score[0])

    model = runtime["models"].get(model_name)
    if model is None:
        raise ValueError(f"Unknown model '{model_name}' in runtime metadata.")

    if model_name == "Random Forest (supervised)":
        return float(model.predict_proba(transformed)[0, 1])

    if hasattr(model, "decision_function"):
        return float(-model.decision_function(transformed)[0])
    raise ValueError(f"Model '{model_name}' does not expose a compatible anomaly score.")


def predict_with_runtime(model_name: str, feature_values: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    columns = list(runtime["feature_columns"])
    if not columns:
        raise ValueError("No feature columns were recorded in the runtime metadata.")
    row: dict[str, float] = {}
    for column in columns:
        if column not in feature_values:
            raise ValueError(f"Missing required feature '{column}'.")
        raw = feature_values[column]
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            raise ValueError(f"Feature '{column}' is missing or empty.")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{column}' must be numeric.")
        if not np.isfinite(value):
            raise ValueError(f"Feature '{column}' must be finite; got {raw!r}.")
        row[column] = value

    frame = pd.DataFrame([row], columns=columns)
    transformed = runtime["preprocessor"].transform(frame)
    score = _score_model(model_name, transformed, runtime)
    threshold = float(runtime["thresholds"][model_name])
    is_anomaly = bool(score >= threshold)
    return {
        "model": model_name,
        "score": float(score),
        "threshold": float(threshold),
        "prediction": "Anomaly candidate" if is_anomaly else "Within learned range",
        "is_anomaly": is_anomaly,
    }


def run_experiments(df: pd.DataFrame, label_column: str | None = None, split_column: str | None = None,
                    contamination: float = 0.2, tune: bool = True) -> dict[str, Any]:
    if not 0.01 <= contamination <= 0.5: raise ValueError("contamination must be between 0.01 and 0.5")
    label_column = label_column or _find_column(df, LABEL_CANDIDATES)
    split_column = split_column or _find_column(df, SPLIT_CANDIDATES)
    if not label_column or label_column not in df.columns: raise ValueError("A binary ground-truth label column is required for evaluation. Expected e.g. label/anomaly/is_anomaly/target.")
    y_all = _binary_labels(df[label_column])
    excluded = {label_column}
    if split_column and split_column in df.columns: excluded.add(split_column)
    feature_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if len(feature_cols) < 2: raise ValueError("At least two numeric feature columns are required.")
    X_all = df[feature_cols].replace([np.inf,-np.inf], np.nan)
    if split_column and split_column in df.columns:
        tr_mask = _train_mask(df[split_column])
        if tr_mask.sum() < 20 or (~tr_mask).sum() < 10: raise ValueError("Official split column was found but does not yield enough train/test rows. Check split values.")
    else:
        # Reproducible stratified holdout when official split is unavailable.
        idx = np.arange(len(df)); tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=42, stratify=y_all)
        tr_mask = np.zeros(len(df), dtype=bool); tr_mask[tr_idx] = True
    X_train, X_test = X_all.loc[tr_mask], X_all.loc[~tr_mask]
    y_train, y_test = y_all[tr_mask], y_all[~tr_mask]
    # Hold out validation from training data for threshold selection; test data is not used to calibrate.
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    pre = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())])
    X_fit_t = pre.fit_transform(X_fit); X_val_t = pre.transform(X_val); X_test_t = pre.transform(X_test)
    records: list[dict[str, Any]] = []; predictions: dict[str, Any] = {}; models: dict[str, Any] = {}; thresholds: dict[str, float] = {}; validation_f1: dict[str, float] = {}

    # Fit unsupervised normality profiles using nominal training rows only.
    X_normal = X_fit_t[y_fit == 0]
    if len(X_normal) < 10: raise ValueError("Too few normal training rows to fit unsupervised detectors.")
    # Statistical baseline: robust distance from normal-training median, threshold calibrated on validation scores.
    med = np.nanmedian(X_normal, axis=0); mad = np.nanmedian(np.abs(X_normal-med), axis=0); mad[mad < 1e-8] = 1.0
    val_stat = np.mean(np.abs(X_val_t-med)/mad, axis=1); test_stat = np.mean(np.abs(X_test_t-med)/mad, axis=1)
    stat_threshold = float(np.quantile(val_stat, 1-contamination))
    validation_f1["Robust statistical baseline"] = float(f1_score(y_val, val_stat >= stat_threshold, zero_division=0))
    stat_pred = (test_stat >= stat_threshold).astype(int)
    records.append({"model":"Robust statistical baseline","kind":"baseline","threshold":stat_threshold,"validation_f1":validation_f1["Robust statistical baseline"], **_metrics(y_test,stat_pred,test_stat)})
    thresholds["Robust statistical baseline"] = stat_threshold
    predictions["Robust statistical baseline"] = {"predictions":stat_pred.tolist(),"scores":test_stat.tolist()}

    models_to_fit = [
        ("Isolation Forest", Pipeline([("model", IsolationForest(contamination=contamination, random_state=42, n_estimators=300))]), "anomaly"),
        ("One-Class SVM", Pipeline([("model", OneClassSVM(kernel="rbf", nu=contamination, gamma="scale"))]), "anomaly")]
    # Supervised comparator is valid because labels are available; clearly mark it as supervised.
    if len(np.unique(y_fit)) == 2:
        rf = Pipeline([("model", RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=300, min_samples_leaf=2))])
        if tune:
            search = GridSearchCV(rf, {"model__max_depth":[None,8,16],"model__max_features":["sqrt",0.8]}, cv=3, scoring="f1", n_jobs=-1)
            search.fit(X_fit_t,y_fit); rf = search.best_estimator_; best_params = search.best_params_
        else:
            rf.fit(X_fit_t,y_fit); best_params = {}
        proba_val = rf.predict_proba(X_val_t)[:,1]; proba_test = rf.predict_proba(X_test_t)[:,1]
        # Validation threshold is selected by F1, with deterministic tie-break toward threshold nearest 0.5.
        threshold_candidates = np.unique(np.r_[np.linspace(0.1,0.9,81),0.5])
        scored = [(f1_score(y_val,proba_val>=t,zero_division=0), -abs(t-0.5), float(t)) for t in threshold_candidates]
        threshold = max(scored)[2]; pred = (proba_test>=threshold).astype(int)
        validation_f1["Random Forest (supervised)"] = float(f1_score(y_val, proba_val >= threshold, zero_division=0))
        rec = {"model":"Random Forest (supervised)","kind":"supervised","threshold":threshold,"validation_f1":validation_f1["Random Forest (supervised)"],"best_params":best_params,**_metrics(y_test,pred,proba_test)}
        records.append(rec); predictions[rec["model"]]={"predictions":pred.tolist(),"scores":proba_test.tolist()}; models[rec["model"]]=rf; thresholds[rec["model"]]=threshold

    for name, pipe, _ in models_to_fit:
        pipe.fit(X_normal)
        # decision_function is higher for normal observations in sklearn; invert to make higher = more anomalous.
        val_score = -pipe.decision_function(X_val_t); test_score = -pipe.decision_function(X_test_t)
        threshold = float(np.quantile(val_score, 1-contamination))
        validation_f1[name] = float(f1_score(y_val, val_score >= threshold, zero_division=0))
        pred = (test_score >= threshold).astype(int)
        rec = {"model":name,"kind":"unsupervised","threshold":threshold,"validation_f1":validation_f1[name],**_metrics(y_test,pred,test_score)}
        records.append(rec); predictions[name]={"predictions":pred.tolist(),"scores":test_score.tolist()}; models[name]=pipe; thresholds[name]=threshold

    # Actual feature importance from supervised model, when trained.
    importance = []
    if "Random Forest (supervised)" in models:
        rf_model = models["Random Forest (supervised)"].named_steps["model"]
        names = list(feature_cols)
        # imputer indicators add columns, so use transformed feature names when available.
        try: names = list(pre.get_feature_names_out())
        except Exception: pass
        vals = rf_model.feature_importances_
        importance = [{"feature":str(n),"importance":float(v)} for n,v in sorted(zip(names,vals),key=lambda z:z[1],reverse=True)[:12]]
    best_model = _best_model_from_validation(validation_f1)
    return {"created_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"dataset":{"rows":int(len(df)),"train_rows":int(len(X_train)),"test_rows":int(len(X_test)),"feature_count":len(feature_cols),"feature_columns":[str(c) for c in feature_cols],"label_column":str(label_column),"split_column":str(split_column) if split_column else None,"split_method":"provided official split" if split_column else "stratified 75/25 holdout (no official split found)","missing_cells":int(X_all.isna().sum().sum())},"metrics":records,"predictions":predictions,"feature_importance":importance,"best_model":best_model,"model_selection_metric":"validation F1","note":"Per-model metrics are computed on the held-out test set. The dashboard default model is selected by validation F1, and thresholds are selected using training/validation data only. Do not compare with published values unless the split and protocol match.", "_runtime":{"preprocessor":pre,"models":models,"thresholds":thresholds,"feature_columns":list(feature_cols),"test_features":X_test,"stat_median":med,"stat_mad":mad}}
