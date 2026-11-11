
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
import shap
from .training import DAY_COLS, build_features
DATASET_DIR = Path(r"D:\deep learning\AI_Theif_Detection\Dataset")
MODEL_DIR = Path(r"D:\deep learning\AI_Theif_Detection\Models")
RESULT_DIR = Path(r"D:\deep learning\AI_Theif_Detection\Result")

INPUT_FILE = DATASET_DIR / "gridguard_random_consumption.csv"

THRESHOLD = 0.5     
TOP_N = 20          
TOP_REASONS = 3

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"



LABELS = {
    "longest_zero_streak": "longest run of zero-reading days",
    "zero_consumption_ratio": "share of days with zero reading",
    "zero_consumption_days": "number of zero-reading days",
    "low_consumption_days": "days far below the customer's average",
    "sudden_decrease_days": "number of sharp day-to-day drops",
    "sudden_increase_days": "number of sharp day-to-day jumps",
    "max_daily_decrease": "largest single-day drop",
    "daily_change_std": "day-to-day variability",
    "consumption_cv": "variability of daily consumption",
    "consumption_mean": "average daily consumption",
    "consumption_trend": "overall consumption trend",
    "half_period_change_ratio": "change between first and second half",
    "total_outliers": "number of outlier days",
    "peak_to_average_ratio": "peak day compared with average",
    "high_deviation_days": "days far from the customer's own average",
}

# Per-day columns are too granular to explain in a note
SKIP_SUFFIXES = ("_outlier", "_relative_deviation")


def positive_class_shap(values):
    """Return a 2-D array (rows x features) for the theft class."""
    if hasattr(values, "values"):          # shap.Explanation object
        values = values.values
    if isinstance(values, list):           # one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                   # (rows, features, classes)
        values = values[:, :, 1]
    return values


def shap_for_rows(pipe, explainer, X_rows):
    """SHAP values (rows x features) for the theft class."""
    transformed = pipe.named_steps["preprocess"].transform(X_rows)
    return positive_class_shap(explainer.shap_values(transformed))

def top_reasons(shap_row, X_row, typical, feature_cols, k=TOP_REASONS):
    """Features that pushed this customer's risk score up the most."""
    reasons = []
    for i in np.argsort(shap_row)[::-1]:
        if len(reasons) == k or shap_row[i] <= 0:
            break
        name = feature_cols[i]
        if name in DAY_COLS or name.endswith(SKIP_SUFFIXES):
            continue
        reasons.append({
            "feature": name,
            "label": LABELS.get(name, name),
            "value": round(float(X_row[name]), 2),
            "typical": round(float(typical[name]), 2),
            "shap": round(float(shap_row[i]), 3),
        })
    return reasons

# OLLAMA

def build_prompt(customer_id, probability, threshold, reasons):
    if reasons:
        facts = "\n".join(
            f"- {r['label']}: {r['value']} (typical customer: {r['typical']})"
            for r in reasons
        )
    else:
        facts = "- no strong individual signals"

    return (
        "You write short inspection notes for an electricity utility.\n"
        "Rules:\n"
        "- Use only the facts listed below.\n"
        "- Do not invent causes, dates, amounts, or numbers that are not listed.\n"
        "- Describe the score as a risk indicator, not proof of theft.\n"
        "- Write exactly three sentences: the risk level, the main signals, "
        "and the recommended next step.\n\n"
        f"Customer ID: {customer_id}\n"
        f"Theft risk score: {probability:.0%} (inspection threshold {threshold:.0%})\n"
        f"Signals that raised the score most:\n{facts}\n"
    )


def ask_ollama(prompt):
    """Return the model's text, or None if Ollama cannot be reached."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 120},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.RequestException:
        return None


def template_note(customer_id, probability, threshold, reasons):
    status = "flagged for inspection" if probability >= threshold else "not flagged"
    note = f"Customer {customer_id} is {status} (risk score {probability:.0%})."
    if reasons:
        signals = ", ".join(
            f"{r['label']} = {r['value']} (typical {r['typical']})"
            for r in reasons[:3]
        )
        note += f" Main signals: {signals}."
    return note


# ONE CUSTOMER

def explain_one(customer_id, probability, shap_row, X_row, typical, feature_cols,
                threshold=THRESHOLD):
    reasons = top_reasons(shap_row, X_row, typical, feature_cols)
    prompt = build_prompt(customer_id, probability, threshold, reasons)

    note = ask_ollama(prompt)
    source = "ollama"
    if note is None:
        note = template_note(customer_id, probability, threshold, reasons)
        source = "template"

    return {
        "customer_id": customer_id,
        "probability": round(float(probability), 4),
        "flagged": probability >= threshold,
        "reasons": reasons,
        "note": note,
        "note_source": source,
    }


def main():
    pipe = joblib.load(
        MODEL_DIR / "best_theft_detection_pipeline.pkl"
    )

    feature_cols = json.loads(
        (MODEL_DIR / "feature_cols.json").read_text(encoding="utf-8")
    )

    df = pd.read_csv(INPUT_FILE)

    X, y, returned_feature_cols = build_features(df)

    # Use the feature order produced by the same feature-engineering function
    feature_cols = returned_feature_cols

    probability = pipe.predict_proba(X)[:, 1]

    scored = pd.DataFrame(
        {
            "customer_id": df["customer_id"].to_numpy(),
            "probability": probability
        },
        index=X.index
    )

    flagged = scored[scored["probability"] >= THRESHOLD]
    top = flagged.sort_values(
        by="probability",
        ascending=False
    ).head(TOP_N)

    print(
        f"Flagged customers: {len(flagged)} | "
        f"writing notes for: {len(top)}"
    )

    # Typical values across all customers
    typical = X.median(numeric_only=True)

    typical_values_file = MODEL_DIR / "typical_values.json"

    typical_values_file.write_text(
        json.dumps(typical.to_dict(), indent=4),
        encoding="utf-8"
    )

    explainer = shap.TreeExplainer(pipe.named_steps["model"])

    transformed = pipe.named_steps["preprocess"].transform(X)

    shap_values = positive_class_shap(
        explainer.shap_values(transformed)
    )

    print("SHAP shape:", shap_values.shape)

    shap_values_df = pd.DataFrame(
        shap_values,
        columns=feature_cols,
        index=X.index
    )

    # Make sure the output folder exists
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for position, idx in enumerate(top.index):
        result = explain_one(
            customer_id=top.loc[idx, "customer_id"],
            probability=top.loc[idx, "probability"],
            shap_row=shap_values[position],
            X_row=X.loc[idx],
            typical=typical,
            feature_cols=feature_cols,
        )

        results.append(result)

        print(
            f"[{result['customer_id']}] "
            f"{result['note_source']}: "
            f"{result['note'][:90]}..."
        )

    out = pd.DataFrame(results)
    out["reasons"] = out["reasons"].apply(json.dumps)

    output_file = RESULT_DIR / "inspection_notes.csv"
    out.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")