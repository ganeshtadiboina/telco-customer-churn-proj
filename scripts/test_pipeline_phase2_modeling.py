import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna

print("=== Phase 2 : Modeling with XGBoost ===")

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# target must be numeric 0/1
if df["Churn"].dtype == "object":
    df["churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

assert df["churn"].isna().sum() == 0, "Churn has NaNs"
assert set(df["Churn"].unique() <= {0,1}, "Churn not 0/1")

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

THRESHOLD = 0.4

def objective(trail):
    params = {
        "n_estimators": trail.suggest_int("n_estimators", 300, 800),
        "learning_rate": trail.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trail.suggest_int("max_depth", 3, 10),
        "subsample": trail.suggest_float("subsample", 0.5, 0.1),
        "min_child_weight": trail.suggest_int("min_child_weight", 1, 10),
        "gamma": trail.suggest_float("gamma", 0, 5),
        "reg_alpha": trail.suggest_float("reg_alphs", 0, 5),
        "reg_lambda": trail.suggest_float("reg_lambda", 0,5),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    from sklearn.metrics import recall_score
    return recall_score(y_test, y_pred, pos_label=1)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trails=30)
print("Best Params:", study.best_params)
print("Best Recall:", study.best_value)