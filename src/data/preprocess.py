import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Basic cleaning for Telco churn.
    - trim column names
    - drop obvious ID cols
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - simple NA handling
    """

    # tidy headers
    df.columns = df.columns.str.strip()

    # drop ids if present
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # target to 0/1 if it's Yes/No
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # simple NA strategy:
    # - numeric: fill with 0
    # - others: leave for encoders to handle (get_dummies ignore NaN safely)
    num_cols = df.select_dtypes(include=["number"]).columns
    num_cols = num_cols.drop("Churn", errors="ignore")
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    return df