"""
Stage 5: Hyperparameter Tuning
--------------------------------
Uses GridSearchCV to find the best XGBoost hyperparameters.
Tests every combination of parameters using 5-fold cross validation.
Saves the best tuned model to models/best_tuned_model.pkl
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.impute           import SimpleImputer
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import accuracy_score, f1_score
import xgboost as xgb

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "../models")

FEATURE_COLS = [
    "home_rank", "away_rank", "rank_diff",
    "home_points", "away_points", "points_diff",
    "home_elo", "away_elo", "elo_diff",
    "home_form_wr", "away_form_wr",
    "home_form_gd", "away_form_gd",
    "h2h_win_rate",
    "tournament_weight",
    "neutral",
    "home_goals_pg",
    "away_goals_pg",
    "goals_pg_diff",
]

TARGET_COL = "result"


# ─── 1. Load & Split ──────────────────────────────────────────────────────────

def load_and_split():
    df = pd.read_csv(
        os.path.join(PROCESSED_DIR, "features.csv"),
        parse_dates=["date"]
    )
    df = df.sort_values("date").reset_index(drop=True)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    print(f"[Split] Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ─── 2. Encode Labels ─────────────────────────────────────────────────────────

def encode_labels(y_train, y_test):
    """XGBoost needs labels 0, 1, 2 not -1, 0, 1"""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    return y_train_enc, y_test_enc, le


# ─── 3. Build Pipeline ────────────────────────────────────────────────────────

def build_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model",  xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ))
    ])


# ─── 4. Define Grid ───────────────────────────────────────────────────────────

def get_param_grid():
    return {
        "model__n_estimators":  [100, 200, 300],
        "model__max_depth":     [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
    }


# ─── 5. Run GridSearchCV ──────────────────────────────────────────────────────

def run_grid_search(X_train, y_train_enc):
    pipeline  = build_pipeline()
    param_grid = get_param_grid()

    cv = StratifiedKFold(n_splits=5, shuffle=False)

    grid_search = GridSearchCV(
        estimator  = pipeline,
        param_grid = param_grid,
        cv         = cv,
        scoring    = "accuracy",
        verbose    = 2,
        n_jobs     = -1
    )

    print(f"\n[GridSearch] Testing {len(param_grid['model__n_estimators']) * len(param_grid['model__max_depth']) * len(param_grid['model__learning_rate'])} combinations × 5 folds...")
    print("[GridSearch] This will take a few minutes...\n")

    grid_search.fit(X_train, y_train_enc)

    return grid_search


# ─── 6. Evaluate Best Model ───────────────────────────────────────────────────

def evaluate_best(grid_search, X_test, y_test, y_test_enc, le):
    best_model = grid_search.best_estimator_

    y_pred_enc = best_model.predict(X_test)
    y_pred     = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 55)
    print("  Best Parameters Found")
    print("=" * 55)
    for param, value in grid_search.best_params_.items():
        clean_name = param.replace("model__", "")
        print(f"  {clean_name:<20} → {value}")

    print(f"\n  CV Best Score : {grid_search.best_score_:.4f}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test F1 Macro : {f1:.4f}")

    return best_model, acc, f1


# ─── 7. Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Stage 5: XGBoost Hyperparameter Tuning")
    print("=" * 55)

    X_train, X_test, y_train, y_test = load_and_split()
    y_train_enc, y_test_enc, le      = encode_labels(y_train, y_test)

    grid_search = run_grid_search(X_train, y_train_enc)

    best_model, acc, f1 = evaluate_best(
        grid_search, X_test, y_test, y_test_enc, le
    )

    # Save tuned model
    model_path = os.path.join(MODELS_DIR, "best_tuned_model.pkl")
    joblib.dump({
        "model":         best_model,
        "name":          "XGBoost (Tuned)",
        "features":      FEATURE_COLS,
        "label_encoder": le,
        "best_params":   grid_search.best_params_
    }, model_path)

    print(f"\n[Saved] {model_path}")
    print("\n[Top 5 parameter combinations:]")
    results_df = pd.DataFrame(grid_search.cv_results_)
    top5 = results_df.nlargest(5, "mean_test_score")[
        ["param_model__n_estimators",
         "param_model__max_depth",
         "param_model__learning_rate",
         "mean_test_score",
         "std_test_score"]
    ]
    top5.columns = ["n_estimators", "max_depth", "learning_rate", "mean_acc", "std_acc"]
    top5["mean_acc"] = top5["mean_acc"].round(4)
    top5["std_acc"]  = top5["std_acc"].round(4)
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()