"""

INFERENCE PIPELINE - Production ML Model Serving With Feature Consistency 

====================================================

This module provides the core inference functionality for the Telco Churn prediction model.

It enusres match training-time feature transformations exactly match training-time transformations, which is CRITICAL for model accuracy in production.

key Responsibilites:
1. Load MLflow-logged model and feature metadata from training 
2. Apply identical features transformation as usead during training 
3. Ensure correct feature ordering for model input 
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency 
- Uses fixed BINARY_MAP for deterministic binary encoding 
- Applies same one-hot encoding with drop_frist=True
- Maintains exact features column order from gracefully 

Production Deployment:
- MODEL_DIR points to constainerized model artifacts
- Feature schema loaded from training-time artifacts 
- Optimized for single-row inference (real-time serving)
"""


import os
import pandas as pd
import mlflow 

# === MODEL LOADING CONFIGURATION ===
# Important: This path is set during docker Container build.
# In development: uses local MLflow artifacts
#  In production: uses model copied to contanier at build time 
MODEL_DIR = "/app/model"

try:
  # Load the trained XGBoost model in MLflow pyfunc format 
  # This ensures compatibility regardless of the underlying  Ml library
  model = mlflow.pyfunc.load_model(MODEL_DIR)
  print(f"Model loaded successfully from {MODEL_DIR}")
except Exception as e:
  print(f"Failed to load model from {MODEL_DIR}: {e}")
  # Fallback for local development (OPTIONAL)
  try:
    # Try loading from local MLflow tracking
    import glob
    local_model_path = glob.glob("./mlruns/*/*/artifacts/model")
    if local_model_paths:
      latest_model = max(local_model_paths, key=os.path.getmtime)
      model = mlflow.pyfunc.load_model(latest_model)
      MODEL_DIR = latest_model
      print(f"Fallback: Loaded model from {latest_model}")
    else:
      raise Exception("No model found in local mlruns")
  except Exception as fallback_error:
    raise Exception("Failed to load model: {e}. Fallback failed: {fallback_error}")


# === FEATURE SCHEMA LOADING ===
# CRITICAL: Load the exact feature column order used during training
# Thks ensures the model receives features in the expected order 
try:
  feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
  with open(feature_file) as f:
    FEATURE_COLS = [LN.strip() for ln in f ln.strip()]

  print(f"Loaded {len(FEATURE_COLS)} feature columns from training")

except Exception as e:
  raise Exception(f"Failed to load features columns: {e}")


# === FEATURES TRANSFORMATION CONSTANTS ====
# CRITICAL: These mappings must exactly match those used in training
# Any chnages here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
  "gender": {"Female": 0, "Male": 1}, # Demographics
  "Partner": {"No": 0, "Yes": 1}, # Has partner
  "Dependents": {"No": 0, "Yes": 1}, # Has dependents
  "PhoneService": {"No": 0, "Yes": 1}, # Phoneservice
  "PaperlessBilling": {"No": 0, "Yes": 1}, # Billing preference

}


# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
  """
  Apply identical feature transformations as used during model training.

  This function is CRITICAL for production ML - it ensures that features are transformed exactly as they were during training to prevent train/serve skew.


  Transformation Pipeline:
  1. Clean column names and handle data types
  2. Apply deterministic binary encoding (using BINARY_MAP)
  3. One-hot encode remaining categorical feature 
  4. Convert boolean columns to integers
  5. Align features with training schema and order

  Args:
      df: Single-row DataFrame with raw customer data 

  Return: DataFrame with features transformed and ordered for model input

  IMPORTANT: Any changes to this function must be reflected in training
  features engineering to maintain consistency.
  """

  df = df.copy()

  # Clean column names (remove any whitespace)
  df.columns = df.columns.str.strip()

  # === STEP 1: Numeric Type Coericon ===
  # Ensure numeric columns are properly typed (handle string inputs)
  for c in NUMERIC_COLS:
      if c in df.columns:
        # Convert to numeric, replacing invalid values with NaN
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # Fill NaN with 0 (same as training preprocessing)
        df[c] = df[c].fillna(0)

      
  # === STEP 2: Binary Feature Encoding ===
  # Apply deterministic mappings for binary features
  # CRITICAL: Must use exact same mappings as training 
  for c, mapping in BINARY_MAP.items():
    if c in df.columns:
      df[c] = (
        df[c]
        .astype(str) # Convert to string
        .str.strip() # Remove whitespace
        .map(mapping) # Apply binary mapping
        .astype("Int64") # Handle NaN values
        .fillna(0) # Fill unknown values with 0
        .astype(int) # Final integer conversion
      )

  # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
  # Find remaining object/categorical columns (not in BINARY_MAP)
  obj_cols = [c for c in df.select_dtypes(include=["Object"].columns)]
  if obj_cols:
    # Apply one-hot encoding with drop_first=True (same as training)
    # This prevents multicollinearity by dropping the first category
    df = pd.get_dummies(df, columns=obj_cols, drop_first=True)


  # === STEP 5: Feature Alignment with Training Schema ===
  # CRITICAL: Ensure features are in exact same order as training
  # Missing features get filled with 0, extra features are dropped
  df = df.reindex(columns=FEATURE_COLS, fill_value=0)

  return df


def predict(input_dict: dict) -> str:
  """
  Main prediction function for customer churn inference.

  This function provides the complete inference pipeline from raw customer data to business-friendly prediction output. It's called by both the FastAPI endpoint and the Gradio interface to ensure consistent prediction.

  Pipeline:
  1.Convert input dictionary to DataFrame
  2. Apply feature transformations (identical to trianing)
  3. Generate model prediction using loade XGBoost model
  4. Convert prediction to user-friendly string


  Args:
      input_dict:  Dictionary containing raw customer data with keys matching the CustomerData schema (18 features total)

  Returns:
      Human-readable prediction string:
      - "Likely to churn" for high-risk customers (model prediction = 1)
      - "Not likely to churn" for low-risk customers (model prediction = 0)


  Example:
      >>> customer_data = {
      ...     "gender": "Female", "tenure": 1,"Contract": "Month-to-month",
      ... "MonthlyCharges": 85.0, ... # other features 
      ... }
      >>> predict(customer_data)
      "Likely to churn"
  """

  # === STEP 1: Convery Input to DataFrame === 
  # Use the same transformation pipeline as training 
  df_enc = _serve_transform(df)

  # === STEP 3: Generate Model Prediction ===
  # Call the loaded MLflow model for inference
  # The model returns predictions in various formats depending on the ML library
  try:
    preds = model.predict(df_enc)

    # Normalize prediction output to consistent format 
    if hasattr(preds, "tolist"):
      preds = preds.tolist() # Convert numpy array to list

    # Extract single prediction value (for single-row input)
    if isinstance(preds, (list, tuple)) and len(preds) == 1:
      result = preds[0]
    else:
      result = preds

  except Exception as e:
    raise Exception(f"Model prediction failed: {e}")

  # === STEP 4: Convert to Business-Friendly Output ===
  # Convert binary prediction (0/1) to actionable business language
  if result == 1:
    return "Likely to churn" # High risk - needs intervention
  else:
    return "Not likely to churn" # Low risk - maintain normal service 






