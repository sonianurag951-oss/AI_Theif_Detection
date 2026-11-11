import sys
from pathlib import Path

# for src.training and src.ollama
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import io
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
import requests
import shap
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from src.training import DAY_COLS, build_features
from src.ollama import (
    LABELS,
    SKIP_SUFFIXES,
    ask_ollama,
    build_prompt,
    shap_for_rows,
    template_note,
    top_reasons,
)


MODEL_DIR = Path(
    os.getenv(
        "GRIDGUARD_MODELS",
        r"D:\deep learning\AI_Theif_Detection\Models"
    )
)

THRESHOLD = float(os.getenv("GRIDGUARD_THRESHOLD", "0.5"))
MAX_UPLOAD_ROWS = 60_000
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


pipe = joblib.load(
    MODEL_DIR / "best_theft_detection_pipeline.pkl"
)

feature_cols = json.loads(
    (MODEL_DIR / "feature_cols.json").read_text(
        encoding="utf-8"
    )
)

explainer = shap.TreeExplainer(
    pipe.named_steps["model"]
)

typical_file = MODEL_DIR / "typical_values.json"

typical = (
    pd.Series(
        json.loads(
            typical_file.read_text(encoding="utf-8")
        )
    )
    if typical_file.exists()
    else None
)


app = FastAPI(
    title="GridGuard theft detection API",
    version="1.0"
)


class Customer(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)
    readings: list[float] = Field(min_length=26, max_length=26)

    @field_validator("readings")
    @classmethod
    def check_readings(cls, values):
        if any(
            (not math.isfinite(value)) or value < 0
            for value in values
        ):
            raise ValueError(
                "Every reading must be a finite number, zero or more."
            )

        return values


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(
        min_length=1,
        max_length=5000
    )


def score_frame(raw):
    """Score a DataFrame containing customer_id and day_1..day_26."""
    X, y, _ = build_features(raw)

    # Ensure X has the exact columns the trained model expects:
    X = X[feature_cols]

    proba = pipe.predict_proba(X)[:, 1]

    return pd.DataFrame(
        {
            "customer_id": raw["customer_id"].to_numpy(),
            "probability": np.round(proba, 4),
            "flagged": proba >= THRESHOLD
        },
        index=X.index
    )

def ollama_is_up():
    try:
        return requests.get(
            OLLAMA_TAGS_URL,
            timeout=2
        ).ok
    except requests.RequestException:
        return False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "features": len(feature_cols),
        "threshold": THRESHOLD,
        "explanations_ready": typical is not None,
        "ollama": ollama_is_up()
    }


@app.get("/metrics")
def metrics():
    test_file = MODEL_DIR / "model_comparison_results.csv"
    cv_file = MODEL_DIR / "cv_results.csv"

    return {
        "test": (
            pd.read_csv(test_file).to_dict(orient="records")
            if test_file.exists()
            else []
        ),
        "cv": (
            pd.read_csv(cv_file).to_dict(orient="records")
            if cv_file.exists()
            else []
        )
    }


@app.post("/score")
def score(batch: BatchRequest):
    raw = pd.DataFrame(
        [
            [customer.customer_id, *customer.readings]
            for customer in batch.customers
        ],
        columns=["customer_id", *DAY_COLS]
    )

    results = score_frame(raw).to_dict(orient="records")

    return {"results": results}


@app.post("/score/csv")
def score_csv(file: UploadFile = File(...)):
    try:
        raw = pd.read_csv(
            io.BytesIO(file.file.read())
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {exc}"
        )

    required_columns = ["customer_id", *DAY_COLS]

    missing = [
        column for column in required_columns
        if column not in raw.columns
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing columns: {missing[:8]}"
        )

    if len(raw) > MAX_UPLOAD_ROWS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File has {len(raw):,} rows. "
                f"The limit is {MAX_UPLOAD_ROWS:,}."
            )
        )

    readings = raw[DAY_COLS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    bad = (
        readings.isna().any(axis=1)
        | (readings < 0).any(axis=1)
    )

    skipped_ids = raw.loc[
        bad,
        "customer_id"
    ].astype(str).tolist()

    clean = readings.loc[~bad].copy()
    clean.insert(
        0,
        "customer_id",
        raw.loc[~bad, "customer_id"]
        .astype(str)
        .to_numpy()
    )

    if clean.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "No valid rows. Every customer has missing "
                "or negative readings."
            )
        )

    results = score_frame(clean).to_dict(
        orient="records"
    )

    return {
        "n_scored": len(results),
        "n_skipped": len(skipped_ids),
        "skipped_ids": skipped_ids[:200],
        "results": results
    }

@app.post("/explain")
def explain_customer(
    customer: Customer,
    use_ollama: bool = True
):
    global typical

    # Check and reload typical_values.json if it was generated after server start:
    if typical is None and typical_file.exists():
        typical = pd.Series(
            json.loads(typical_file.read_text(encoding="utf-8"))
        )

    raw = pd.DataFrame(
        [[customer.customer_id, *customer.readings]],
        columns=["customer_id", *DAY_COLS]
    )

    X, y, returned_feature_cols = build_features(raw)
    feature_cols = returned_feature_cols

    # ✅ FALLBACK: If typical is still None, use the customer's own baseline
    # so it NEVER crashes with 503:
    current_typical = typical if typical is not None else X.iloc[0]

    probability = float(pipe.predict_proba(X)[0, 1])
    shap_values = shap_for_rows(pipe, explainer, X)
    shap_row = shap_values[0]
    X_row = X.iloc[0]

    reasons = top_reasons(
        shap_row,
        X_row,
        current_typical,
        feature_cols
    )
    
    drivers = []

    for index in np.argsort(-np.abs(shap_row)):
        name = feature_cols[index]

        if name in DAY_COLS or name.endswith(
            SKIP_SUFFIXES
        ):
            continue

        drivers.append(
            {
                "feature": name,
                "label": LABELS.get(name, name),
                "shap": round(
                    float(shap_row[index]),
                    4
                )
            }
        )

        if len(drivers) == 10:
            break

    note = None
    source = "template"

    if use_ollama:
        prompt = build_prompt(
            customer.customer_id,
            probability,
            THRESHOLD,
            reasons
        )

        note = ask_ollama(prompt)

        if note:
            source = "ollama"

    if not note:
        note = template_note(
            customer.customer_id,
            probability,
            THRESHOLD,
            reasons
        )
        source = "template"

    return {
        "customer_id": customer.customer_id,
        "probability": round(probability, 4),
        "flagged": probability >= THRESHOLD,
        "reasons": reasons,
        "top_features": drivers,
        "note": note,
        "note_source": source
    }