# API examples

After starting the backend, open `http://localhost:8000/docs` for interactive Swagger documentation.

## Health
`GET /api/health`

## Upload dataset
`POST /api/dataset/upload` as multipart form field `file` with a labelled CSV.

## Run experiment
`POST /api/experiments/run`
```json
{"label_column":"label","split_column":"train","contamination":0.2,"tune":true}
```
Use the actual column names from the uploaded dataset. Omit `split_column` only when the source has no official split.

## Single-fragment inference
After a run, call `GET /api/features` to get required numeric feature names, then send values to `POST /api/predict`:
```json
{"model":"Isolation Forest","features":{"feature_a":0.1,"feature_b":0.2}}
```
The example keys above are placeholders; send every feature returned by `/api/features` with actual measurements. This endpoint returns a model score and threshold, not a failure diagnosis.
