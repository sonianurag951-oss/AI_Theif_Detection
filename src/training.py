
import os
from pathlib import Path
# both for warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import shap
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*tf\\.betainc.*deprecated.*",
    category=Warning
)

DATASET_DIR = Path(r"D:\deep learning\AI_Theif_Detection\Dataset")
MODEL_DIR = Path(r"D:\deep learning\AI_Theif_Detection\Models")

INPUT_FILE = DATASET_DIR / "gridguard_random_consumption.csv"
TRACKING_DB = MODEL_DIR / "mlflow.db"
EXPERIMENT_NAME = "gridguard-theft-detection"

SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15
N_FOLDS = 5

DAY_COLS = [f"day_{i}" for i in range(1, 27)]

EPOCHS = 20
PATIENCE = 8
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


tf.random.set_seed(SEED)
rng = np.random.default_rng(42)
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
mlflow.set_experiment("THEIF_DETECTION")


def build_features(df: pd.DataFrame):
    """Create engineered features and return X and y."""

    new_features = {}

    new_features["consumption_mean"] = df[DAY_COLS].mean(axis=1)
    new_features["consumption_median"] = df[DAY_COLS].median(axis=1)
    new_features["consumption_std"] = df[DAY_COLS].std(axis=1)
    new_features["consumption_min"] = df[DAY_COLS].min(axis=1)
    new_features["consumption_max"] = df[DAY_COLS].max(axis=1)

    new_features["consumption_range"] = (
        new_features["consumption_max"]
        - new_features["consumption_min"]
    )

    new_features["consumption_total"] = df[DAY_COLS].sum(axis=1)

    new_features["consumption_iqr"] = (
        df[DAY_COLS].quantile(0.75, axis=1)
        - df[DAY_COLS].quantile(0.25, axis=1)
    )

    safe_mean = new_features["consumption_mean"].replace(0, np.nan)
    safe_std = new_features["consumption_std"].replace(0, np.nan)

    new_features["consumption_cv"] = safe_std / safe_mean
    new_features["peak_to_average_ratio"] = (
        new_features["consumption_max"] / safe_mean
    )

    outlier_cols = []

    for day in DAY_COLS:
        q1 = df[day].quantile(0.25)
        q3 = df[day].quantile(0.75)
        iqr = q3 - q1

        lower_bound = max(q1 - 1.5 * iqr, 0)
        upper_bound = q3 + 1.5 * iqr

        outlier_col = f"{day}_outlier"
        outlier_cols.append(outlier_col)

        df[outlier_col] = (
            (df[day] < lower_bound) | (df[day] > upper_bound)
        ).astype(int)

    new_features["total_outliers"] = df[outlier_cols].sum(axis=1)
    new_features["has_outlier"] = (
        new_features["total_outliers"] > 0
    ).astype(int)
    new_features["outlier_ratio"] = (
        new_features["total_outliers"] / len(DAY_COLS)
    )
    new_features["high_outlier_customer"] = (
        new_features["outlier_ratio"] >= 0.5
    ).astype(int)

    relative_deviation_features = {}

    for day in DAY_COLS:
        relative_deviation_features[f"{day}_relative_deviation"] = (
            (df[day] - safe_mean) / safe_mean
        )

    relative_deviation_df = pd.DataFrame(
        relative_deviation_features,
        index=df.index
    )

    new_features["high_deviation_days"] = (
        relative_deviation_df.abs().gt(0.5).sum(axis=1)
    )
    new_features["has_high_deviation"] = (
        new_features["high_deviation_days"] > 0
    ).astype(int)
    new_features["max_relative_deviation"] = (
        relative_deviation_df.abs().max(axis=1)
    )
    new_features["mean_absolute_deviation"] = (
        relative_deviation_df.abs().mean(axis=1)
    )

    daily_change = df[DAY_COLS].diff(axis=1)

    new_features["average_daily_change"] = daily_change.mean(axis=1)
    new_features["max_daily_increase"] = daily_change.max(axis=1)
    new_features["max_daily_decrease"] = daily_change.min(axis=1)
    new_features["daily_change_std"] = daily_change.std(axis=1)

    new_features["sudden_increase_days"] = daily_change.gt(
        0.5 * safe_mean,
        axis=0
    ).sum(axis=1)

    new_features["sudden_decrease_days"] = daily_change.lt(
        -0.5 * safe_mean,
        axis=0
    ).sum(axis=1)

    new_features["zero_consumption_days"] = (
        df[DAY_COLS] == 0
    ).sum(axis=1)

    new_features["zero_consumption_ratio"] = (
        new_features["zero_consumption_days"] / len(DAY_COLS)
    )

    def longest_zero_streak(row):
        longest = 0
        current = 0

        for value in row:
            if value == 0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        return longest

    new_features["longest_zero_streak"] = df[DAY_COLS].apply(
        longest_zero_streak,
        axis=1
    )

    new_features["low_consumption_days"] = df[DAY_COLS].lt(
        0.25 * safe_mean,
        axis=0
    ).sum(axis=1)

    first_half = DAY_COLS[:13]
    second_half = DAY_COLS[13:]

    new_features["first_half_mean"] = df[first_half].mean(axis=1)
    new_features["second_half_mean"] = df[second_half].mean(axis=1)

    new_features["half_period_change"] = (
        new_features["second_half_mean"]
        - new_features["first_half_mean"]
    )

    new_features["half_period_change_ratio"] = (
        new_features["half_period_change"]
        / new_features["first_half_mean"].replace(0, np.nan)
    )

    day_numbers = np.arange(1, 27)

    def calculate_trend(row):
        return np.polyfit(day_numbers, row, 1)[0]

    new_features["consumption_trend"] = df[DAY_COLS].apply(
        calculate_trend,
        axis=1
    )

    new_features_df = pd.DataFrame(new_features, index=df.index)

    df = pd.concat(
        [df, relative_deviation_df, new_features_df],
        axis=1
    )

    exclude_cols = ["customer_id", "meter_id", "FLAG"]

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    feature_cols = list(dict.fromkeys(feature_cols))


    X = df[feature_cols]
    y = df["FLAG"].astype(int) if "FLAG" in df.columns else None
    return X, y, feature_cols

if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    print("\nDataset shape:", df.shape)
    print(df.dtypes)
    print("\nFirst five rows:")
    print(df.head())
    
    X, y, feature_cols = build_features(df)
    feature_cols_file = MODEL_DIR / "feature_cols.json"
    with feature_cols_file.open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=4)

    print(f"Saved feature columns: {feature_cols_file}")

    print("\nNumber of features:", X.shape[1])
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    print(f" {X_train.shape}")
    print(f"{X_test.shape}")
    print(f"Theft customers in training set: {pos}")
    print(f"Class imbalance weight: {neg / pos:.2f}")

    # CROSS-VALIDATION SETUP
    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    cv_rows = []
    test_rows = []
    fitted = {}

    # CLASSICAL MODEL PREPROCESSING
    preprocess = ColumnTransformer(
        transformers=[(
            "numeric", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler())
            ]),
            feature_cols
        )]
    )

    # CLASSICAL MODELS
    classical_models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg / pos,
            eval_metric="aucpr",
            random_state=SEED,
            n_jobs=-1
        )
    }

    # TRAIN AND EVALUATE CLASSICAL MODELS
    for name, model in classical_models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", model)
            ]
        )

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)

            scores = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring=[
                    "average_precision",
                    "roc_auc",
                    "f1"
                ]
            )

            cv_pr = scores["test_average_precision"].mean()
            cv_pr_std = scores["test_average_precision"].std()

            mlflow.log_metric("cv_pr_auc", cv_pr)
            mlflow.log_metric("cv_pr_auc_std", cv_pr_std)

            cv_rows.append({
                "model": name,
                "cv_pr_auc": cv_pr,
                "cv_pr_auc_std": cv_pr_std
            })

            pipe.fit(X_train, y_train)
            fitted[name] = pipe

            y_prob = pipe.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics = {
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "pr_auc": average_precision_score(y_test, y_prob)
            }

            mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
            mlflow.log_text(classification_report(y_test, y_pred, zero_division=0), "test_report.txt")

            mlflow.sklearn.log_model(
                pipe,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                    "sklearn.tree._tree.Tree"
                ]
            )

            test_rows.append({"model": name, **metrics})

            print(f"[{name}] CV PR-AUC: {cv_pr:.4f} | Test PR-AUC: {metrics['pr_auc']:.4f}")

    # CNN MODEL
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y_train
    )

    def shape_input(frame):
        readings = frame[DAY_COLS].to_numpy(dtype="float32")
        own_mean = readings.mean(axis=1, keepdims=True) + 1e-6
        return np.clip(readings / own_mean, 0, 10)[:, :, None]

    shape_fit = shape_input(X_fit)
    shape_val = shape_input(X_val)
    shape_test = shape_input(X_test)

    eng_cols = [col for col in feature_cols if col not in DAY_COLS]
    print(f" {len(eng_cols)}")

    if len(eng_cols) == 0:
        raise ValueError("No engineered features found. ")

    scaler = RobustScaler().fit(X_fit[eng_cols].to_numpy(dtype="float32"))
    eng_fit = np.clip(scaler.transform(X_fit[eng_cols]), -10, 10).astype("float32")
    eng_val = np.clip(scaler.transform(X_val[eng_cols]), -10, 10).astype("float32")
    eng_test = np.clip(scaler.transform(X_test[eng_cols]), -10, 10).astype("float32")

    hybrid_weight = {
        0: 1.0,
        1: float((y_fit == 0).sum() / (y_fit == 1).sum())
    }

    shape_in = tf.keras.Input(shape=(26, 1), name="shape")
    s = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(shape_in)
    s = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(s)
    s = tf.keras.layers.GlobalMaxPooling1D()(s)

    eng_in = tf.keras.Input(shape=(len(eng_cols),), name="engineered")
    e = tf.keras.layers.Dense(64, activation="relu")(eng_in)
    e = tf.keras.layers.Dropout(0.2)(e)

    x = tf.keras.layers.Concatenate()([s, e])
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    hybrid = tf.keras.Model(inputs=[shape_in, eng_in], outputs=out, name="hybrid_cnn")
    hybrid.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=PATIENCE,
        restore_best_weights=True
    )

    with mlflow.start_run(run_name="HybridCNN"):
        mlflow.log_params({
            "model": "HybridCNN",
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "class_weight_theft": round(hybrid_weight[1], 2)
        })

        history = hybrid.fit(
            [shape_fit, eng_fit],
            y_fit.to_numpy(),
            validation_data=([shape_val, eng_val], y_val.to_numpy()),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=hybrid_weight,
            callbacks=[early_stop],
            verbose=2
        )

        for epoch in range(len(history.history["loss"])):
            mlflow.log_metrics({key: value[epoch] for key, value in history.history.items()}, step=epoch + 1)

        best_epoch = int(np.argmax(history.history["val_pr_auc"])) + 1
        best_val_pr_auc = max(history.history["val_pr_auc"])

        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_metric("best_val_pr_auc", best_val_pr_auc)

        y_prob = hybrid.predict([shape_test, eng_test], verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        hybrid_metrics = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob)
        }

        mlflow.log_metrics({f"test_{key}": value for key, value in hybrid_metrics.items()})

        hybrid_model_file = MODEL_DIR / "hybrid_cnn_theft_model.keras"
        hybrid.save(hybrid_model_file)

        try:
            mlflow.tensorflow.log_model(hybrid, name="model")
        except AttributeError:
            mlflow.keras.log_model(hybrid, "model")

    test_rows.append({"model": "HybridCNN", **hybrid_metrics})
    print(f"\n[HybridCNN] Best epoch: {best_epoch} | Test PR-AUC: {hybrid_metrics['pr_auc']:.4f}")

    cv_df = pd.DataFrame(cv_rows).sort_values(by="cv_pr_auc", ascending=False)
    cv_df.to_csv(MODEL_DIR / "cv_results.csv", index=False)

    test_df = pd.DataFrame(test_rows).sort_values(by="pr_auc", ascending=False)
    test_df.to_csv(MODEL_DIR / "model_comparison_results.csv", index=False)

    print("\nCross-validation results:")
    print(cv_df.round(4).to_string(index=False))

    print("\nTest-set comparison:")
    print(test_df.round(4).to_string(index=False))

    best_classical = cv_df.iloc[0]["model"]
    joblib.dump(fitted[best_classical], MODEL_DIR / "best_theft_detection_pipeline.pkl")

    print(f"{best_classical}")
    print(MODEL_DIR / "best_theft_detection_pipeline.pkl")
    print(MODEL_DIR / "hybrid_cnn_theft_model.keras")

    shap_dir = MODEL_DIR / "shap_explanations"
    shap_dir.mkdir(parents=True, exist_ok=True)

    best_pipe = fitted[best_classical]
    preprocessor = best_pipe.named_steps["preprocess"]
    best_model = best_pipe.named_steps["model"]

    X_test_transformed = preprocessor.transform(X_test)
    sample_size = min(500, X_test_transformed.shape[0])
    X_shap_sample = X_test_transformed[:sample_size]

    try:
        transformed_feature_names = preprocessor.get_feature_names_out()
    except Exception:
        transformed_feature_names = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]

    X_shap_sample = np.asarray(X_shap_sample)
    if hasattr(X_shap_sample, "toarray"):
        X_shap_sample = X_shap_sample.toarray()

    if best_classical == "Random Forest":
        print("\nUsing TreeExplainer for Random Forest...")
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_shap_sample)

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        elif isinstance(shap_values, list):
            shap_values = shap_values[1]

        print("SHAP values shape:", np.asarray(shap_values).shape)

        plt.figure()
        shap.summary_plot(shap_values, X_shap_sample, feature_names=transformed_feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(shap_dir / "random_forest_shap_global_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, X_shap_sample, feature_names=transformed_feature_names, show=False)
        plt.tight_layout()
        plt.savefig(shap_dir / "random_forest_shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()

        shap_values_df = pd.DataFrame(shap_values, columns=transformed_feature_names)
        shap_values_df.to_csv(shap_dir / "random_forest_shap_values.csv", index=False)

        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_importance_df = pd.DataFrame({
            "feature": transformed_feature_names,
            "mean_abs_shap": mean_shap
        }).sort_values(by="mean_abs_shap", ascending=False)
        shap_importance_df.to_csv(shap_dir / "random_forest_shap_feature_importance.csv", index=False)

        print(shap_importance_df.head(20).round(4).to_string(index=False))

        with mlflow.start_run(run_name="SHAP-RandomForest-Explainability"):
            mlflow.log_artifact(str(shap_dir / "random_forest_shap_global_importance.png"))
            mlflow.log_artifact(str(shap_dir / "random_forest_shap_beeswarm.png"))
            mlflow.log_artifact(str(shap_dir / "random_forest_shap_values.csv"))
            mlflow.log_artifact(str(shap_dir / "random_forest_shap_feature_importance.csv"))