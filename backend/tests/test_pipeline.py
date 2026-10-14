import numpy as np, pandas as pd
from fastapi.testclient import TestClient
from app import main as api_main
from app.pipeline import _best_model_from_validation, _binary_labels, inspect_dataset, run_experiments, predict_with_runtime

def make_data(n=160):
    rng=np.random.default_rng(3); y=np.r_[np.zeros(n-20,dtype=int),np.ones(20,dtype=int)]
    return pd.DataFrame({"f1":rng.normal(y*2,1),"f2":rng.normal(y*1.5,1),"f3":rng.normal(size=n),"label":y})

def test_inspect():
    info=inspect_dataset(make_data())
    assert info["rows"]==160 and info["numeric_features"]==3 and info["label_column"]=="label"

def test_pipeline_returns_metrics():
    result=run_experiments(make_data(),tune=False)
    assert result["dataset"]["test_rows"]>0
    assert len(result["metrics"])>=3
    assert all("f1" in row and "confusion_matrix" in row for row in result["metrics"])
    assert result["best_model"] == max(result["metrics"], key=lambda row: row["validation_f1"])["model"]
    assert result["model_selection_metric"] == "validation F1"

def test_best_model_selection_uses_validation_f1_not_test_f1():
    # The first model scores better on the held-out test set, while the second
    # wins on validation; selection must follow validation only.
    metrics = [
        {"model": "test-set winner", "validation_f1": 0.2, "f1": 0.99},
        {"model": "validation winner", "validation_f1": 0.8, "f1": 0.1},
    ]

    selected = _best_model_from_validation({row["model"]: row["validation_f1"] for row in metrics})

    assert selected == "validation winner"

def test_numeric_binary_labels_reject_missing_values():
    try:
        _binary_labels(pd.Series([0, 1, np.nan]))
        assert False, "Expected missing labels to be rejected"
    except ValueError as error:
        assert "missing value" in str(error)

def test_prediction_uses_runtime_threshold_and_model_score():
    data = make_data(220)
    runtime = run_experiments(data, tune=False, contamination=0.2)
    model_name = runtime["best_model"]
    feature_values = {col: float(value) for col, value in zip(data.columns, data.iloc[0].tolist()) if col != "label"}
    result = predict_with_runtime(model_name, feature_values, runtime["_runtime"])
    threshold = float(runtime["_runtime"]["thresholds"][model_name])
    assert result["threshold"] == threshold
    assert result["prediction"] in {"Anomaly candidate", "Within learned range"}
    assert result["is_anomaly"] == (result["score"] >= threshold)


def test_prediction_rejects_non_numeric_feature_values():
    data = make_data(180)
    runtime = run_experiments(data, tune=False, contamination=0.2)
    try:
        predict_with_runtime("Isolation Forest", {"f1": "bad", "f2": 1.0, "f3": 2.0}, runtime["_runtime"])
        assert False, "Expected a validation error for bad input"
    except ValueError:
        pass

def test_alert_detail_returns_candidate_features_and_detector_threshold(monkeypatch):
    result = run_experiments(make_data(220), tune=False, contamination=0.2)
    runtime = result.pop("_runtime")
    model = next(
        name for name, scored in result["predictions"].items()
        if any(scored["predictions"])
    )
    index = result["predictions"][model]["predictions"].index(1)
    monkeypatch.setattr(api_main, "LAST_EXPERIMENT", result)
    monkeypatch.setattr(api_main, "RUNTIME", runtime)

    response = TestClient(api_main.app).get(
        f"/api/alerts/TEST-{index + 1:04d}",
        params={"model": model},
    )

    assert response.status_code == 200
    detail = response.json()
    assert detail["fragment_id"] == f"TEST-{index + 1:04d}"
    assert detail["index_in_test"] == index
    assert detail["model"] == model
    assert detail["score"] == result["predictions"][model]["scores"][index]
    assert detail["threshold"] == runtime["thresholds"][model]
    assert detail["prediction"] == "Anomaly candidate"
    expected_features = {
        str(column): None if pd.isna(value) else float(value)
        for column, value in runtime["test_features"].iloc[index].items()
    }
    assert detail["features"] == expected_features

def test_alert_detail_returns_404_for_an_unflagged_test_fragment(monkeypatch):
    result = run_experiments(make_data(220), tune=False, contamination=0.2)
    runtime = result.pop("_runtime")
    model = next(
        name for name, scored in result["predictions"].items()
        if 0 in scored["predictions"]
    )
    index = result["predictions"][model]["predictions"].index(0)
    monkeypatch.setattr(api_main, "LAST_EXPERIMENT", result)
    monkeypatch.setattr(api_main, "RUNTIME", runtime)

    response = TestClient(api_main.app).get(
        f"/api/alerts/TEST-{index + 1:04d}",
        params={"model": model},
    )

    assert response.status_code == 404

def test_alert_list_reports_total_and_paginates_with_threshold_metadata(monkeypatch):
    scores = list(map(float, range(260)))
    result = {
        "best_model": "Test detector",
        "metrics": [{"model": "Test detector", "threshold": 160.0}],
        "predictions": {
            "Test detector": {
                "predictions": [int(score >= 160) for score in scores],
                "scores": scores,
            }
        },
    }
    monkeypatch.setattr(api_main, "LAST_EXPERIMENT", result)
    client = TestClient(api_main.app)

    first = client.get("/api/alerts", params={"model": "Test detector", "limit": 40, "offset": 0}).json()
    second = client.get("/api/alerts", params={"model": "Test detector", "limit": 40, "offset": 40}).json()

    assert first["total_candidates"] == second["total_candidates"] == 100
    assert first["has_more"] is True and second["has_more"] is True
    assert len(first["alerts"]) == len(second["alerts"]) == 40
    assert first["alerts"][0]["fragment_id"] == "TEST-0260"
    candidate = second["alerts"][0]
    assert candidate["fragment_id"] == "TEST-0220"
    assert candidate["score"] == 219.0
    assert candidate["threshold"] == 160.0
    assert candidate["prediction"] == "Anomaly candidate"
    assert candidate["is_anomaly"] is True
    assert not set(item["fragment_id"] for item in first["alerts"]) & set(item["fragment_id"] for item in second["alerts"])

def test_upload_rejects_oversized_csv_while_reading_in_chunks(monkeypatch):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 5)
    monkeypatch.setattr(api_main, "UPLOAD_CHUNK_BYTES", 3)

    response = TestClient(api_main.app).post(
        "/api/dataset/upload",
        files={"file": ("large.csv", b"abcdef", "text/csv")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "CSV exceeds 30 MB limit."

def test_experiment_internal_error_is_logged_but_not_returned_to_client(monkeypatch):
    monkeypatch.setattr(api_main, "DATA", make_data())

    def fail_experiment(*args, **kwargs):
        raise RuntimeError("internal filesystem path")

    monkeypatch.setattr(api_main, "run_experiments", fail_experiment)
    response = TestClient(api_main.app).post("/api/experiments/run", json={})

    assert response.status_code == 500
    assert response.json()["detail"] == "Experiment run failed. Check server logs for details."
    assert "internal filesystem path" not in response.text

def test_prediction_internal_error_is_logged_but_not_returned_to_client(monkeypatch):
    monkeypatch.setattr(api_main, "LAST_EXPERIMENT", {"available": True})
    monkeypatch.setattr(api_main, "RUNTIME", {"thresholds": {"Test detector": 0.5}})

    def fail_prediction(*args, **kwargs):
        raise RuntimeError("internal filesystem path")

    monkeypatch.setattr(api_main, "predict_with_runtime", fail_prediction)
    response = TestClient(api_main.app).post(
        "/api/predict",
        json={"model": "Test detector", "features": {"f1": 1.0}},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed. Check server logs for details."
    assert "internal filesystem path" not in response.text

def test_alert_detail_handles_unavailable_telemetry_runtime(monkeypatch):
    result = run_experiments(make_data(220), tune=False, contamination=0.2)
    model = next(
        name for name, scored in result["predictions"].items()
        if any(scored["predictions"])
    )
    index = result["predictions"][model]["predictions"].index(1)
    threshold = next(metric["threshold"] for metric in result["metrics"] if metric["model"] == model)
    monkeypatch.setattr(api_main, "LAST_EXPERIMENT", result)
    monkeypatch.setattr(api_main, "RUNTIME", None)

    response = TestClient(api_main.app).get(
        f"/api/alerts/TEST-{index + 1:04d}",
        params={"model": model},
    )

    assert response.status_code == 200
    assert response.json()["threshold"] == threshold
    assert response.json()["features"] is None

def test_upload_rejects_non_csv_file():
    response = TestClient(api_main.app).post(
        "/api/dataset/upload",
        files={"file": ("telemetry.json", b"{}", "application/json")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Please upload a CSV file."

def test_upload_reports_empty_csv_as_client_error():
    response = TestClient(api_main.app).post(
        "/api/dataset/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Could not read CSV:")

def test_upload_rejects_rows_past_configured_limit(monkeypatch):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_ROWS", 2)
    csv_data = b"f1,f2,label\n1,2,0\n3,4,0\n5,6,1\n"

    response = TestClient(api_main.app).post(
        "/api/dataset/upload",
        files={"file": ("telemetry.csv", csv_data, "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Could not read CSV: For this prototype, limit uploads to 2 rows."
