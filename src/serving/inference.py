"""
Inference helpers for the Telco churn model.

This module loads a trained MLflow model plus the feature column list saved by
the training pipeline, then applies the same basic transformations expected by
the model before prediction.
"""

from pathlib import Path

import mlflow
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_MODEL_DIR = Path("/app/model")
BUNDLED_RUN_DIR = PROJECT_ROOT / "src" / "serving" / "model" / "a6c6ae3778c64b38ab62c92448aeb2ce"

MODEL = None
MODEL_DIR = None
FEATURE_COLS = []

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _candidate_artifact_dirs() -> list[Path]:
    """Return model artifact directories from most production-like to local."""
    candidates = [CONTAINER_MODEL_DIR, BUNDLED_RUN_DIR / "artifacts" / "model"]

    mlruns_dir = PROJECT_ROOT / "mlruns"
    if mlruns_dir.exists():
        candidates.extend(
            sorted(
                mlruns_dir.glob("*/*/artifacts/model"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )

    return candidates


def _feature_file_for(model_dir: Path) -> Path:
    """Locate feature_columns.txt for both flat Docker and MLflow layouts."""
    flat_file = model_dir / "feature_columns.txt"
    if flat_file.exists():
        return flat_file

    sibling_file = model_dir.parent / "feature_columns.txt"
    if sibling_file.exists():
        return sibling_file

    raise FileNotFoundError(f"feature_columns.txt not found near {model_dir}")


def _load_model_and_features():
    errors = []

    for model_dir in _candidate_artifact_dirs():
        if not model_dir.exists():
            continue

        try:
            model = mlflow.pyfunc.load_model(str(model_dir))
            feature_file = _feature_file_for(model_dir)
            with open(feature_file, encoding="utf-8") as f:
                feature_cols = [line.strip() for line in f if line.strip()]

            if not feature_cols:
                raise ValueError(f"No feature columns found in {feature_file}")

            return model, model_dir, feature_cols
        except Exception as exc:
            errors.append(f"{model_dir}: {exc}")

    details = "\n".join(errors) if errors else "No candidate model directories found."
    raise RuntimeError(f"Failed to load model artifacts.\n{details}")


def _get_model():
    global MODEL, MODEL_DIR, FEATURE_COLS

    if MODEL is None:
        MODEL, MODEL_DIR, FEATURE_COLS = _load_model_and_features()
        print(f"Loaded model from {MODEL_DIR} with {len(FEATURE_COLS)} features")

    return MODEL


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col, mapping in BINARY_MAP.items():
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    return df.reindex(columns=FEATURE_COLS, fill_value=0)


def predict(input_dict: dict) -> str:
    if not isinstance(input_dict, dict):
        raise TypeError("input_dict must be a dictionary of customer features")

    model = _get_model()
    raw_df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(raw_df)

    try:
        preds = model.predict(df_enc)
    except Exception as exc:
        raise RuntimeError(f"Model prediction failed: {exc}") from exc

    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    result = preds[0] if isinstance(preds, (list, tuple)) and preds else preds
    return "Likely to churn" if int(result) == 1 else "Not likely to churn"
