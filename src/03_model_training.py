"""
Stage 3: Model Training & Evaluation
--------------------------------------
Trains and compares multiple classifiers:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost (best classical)
  4. MLP Neural Network (DL)

Evaluation:
  - Accuracy, F1 (macro), Log Loss
  - Confusion matrix plots
  - Feature importance (RF + XGBoost)
  - Saves best model to models/

Split strategy: time-based (train on older matches, test on recent ones)
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (accuracy_score, f1_score, log_loss,
                                     confusion_matrix, classification_report)
from sklearn.impute          import SimpleImputer
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "../models")
OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "../outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

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
LABEL_MAP  = {1: "Home Win", 0: "Draw", -1: "Away Win"}


# ─── 1. Load & Split ──────────────────────────────────────────────────────────

def load_and_split(test_frac=0.2):
    """Time-based train/test split (no shuffle — respects temporal order)."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    split_idx = int(len(df) * (1 - test_frac))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[Split] Train: {len(X_train):,} | Test: {len(X_test):,} rows")
    print(f"        Train period: {df['date'].iloc[0].date()} → {df['date'].iloc[split_idx-1].date()}")
    print(f"        Test  period: {df['date'].iloc[split_idx].date()} → {df['date'].iloc[-1].date()}")
    return X_train, X_test, y_train, y_test, df


# ─── 2. Define Models ─────────────────────────────────────────────────────────

def build_models():
    """Returns a dict of named model pipelines."""
    
    imputer = SimpleImputer(strategy="median")

    return {
        "Logistic Regression": Pipeline([
            ("impute",  SimpleImputer(strategy="median")),
            ("scale",   StandardScaler()),
            ("model",   LogisticRegression(max_iter=1000, C=1.0,
                                           solver="lbfgs", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("impute",  SimpleImputer(strategy="median")),
            ("model",   RandomForestClassifier(n_estimators=200, max_depth=8,
                                               min_samples_leaf=5, random_state=42,
                                               n_jobs=-1))
        ]),
        "XGBoost": Pipeline([
            ("impute",  SimpleImputer(strategy="median")),
            ("model",   xgb.XGBClassifier(n_estimators=200, max_depth=5,
                                           learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, use_label_encoder=False,
                                           eval_metric="mlogloss", random_state=42,
                                           n_jobs=-1))
        ]),
        "MLP (Neural Net)": Pipeline([
            ("impute",  SimpleImputer(strategy="median")),
            ("scale",   StandardScaler()),
            ("model",   MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                      activation="relu", solver="adam",
                                      max_iter=300, learning_rate_init=0.001,
                                      early_stopping=True, validation_fraction=0.1,
                                      random_state=42))
        ])
    }


# ─── 3. Evaluate ──────────────────────────────────────────────────────────────

def evaluate(name, model, X_test, y_test):
    """Returns dict of metrics for one model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return {
        "Model":    name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "F1 Macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "Log Loss": round(log_loss(y_test, y_prob), 4),
        "_y_pred":  y_pred,
        "_model":   model
    }


# ─── 4. Plots ─────────────────────────────────────────────────────────────────

def plot_confusion(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
    labels = ["Away Win", "Draw", "Home Win"]
    
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fname = name.replace(" ", "_").replace("(", "").replace(")", "") + "_cm.png"
    plt.savefig(os.path.join(OUTPUTS_DIR, fname), dpi=120)
    plt.close()
    print(f"  → Saved confusion matrix: {fname}")


def plot_feature_importance(model, model_name):
    """Plot feature importance for tree-based models."""
    try:
        clf = model.named_steps["model"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            fi = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            fi.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title(f"Feature Importance — {model_name}")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            fname = model_name.replace(" ", "_") + "_feature_importance.png"
            plt.savefig(os.path.join(OUTPUTS_DIR, fname), dpi=120)
            plt.close()
            print(f"  → Saved feature importance: {fname}")
    except Exception:
        pass  # Not all models have feature_importances_


def plot_comparison(results_df):
    """Bar chart comparing model metrics side by side."""
    metrics = ["Accuracy", "F1 Macro"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for i, metric in enumerate(metrics):
        results_df.set_index("Model")[metric].sort_values().plot(
            kind="barh", ax=axes[i], color="coral")
        axes[i].set_title(metric)
        axes[i].set_xlim(0, 1)
        for bar in axes[i].patches:
            axes[i].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                         f"{bar.get_width():.3f}", va="center", fontsize=9)
    
    plt.suptitle("Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "model_comparison.png"), dpi=120)
    plt.close()
    print("  → Saved model_comparison.png")


# ─── 5. XGBoost label encoding ────────────────────────────────────────────────

def encode_labels(y_train, y_test):
    """XGBoost needs labels 0, 1, 2 not -1, 0, 1."""
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    return y_train_enc, y_test_enc, le


# ─── 6. Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Stage 3: Model Training & Evaluation")
    print("=" * 55)

    X_train, X_test, y_train, y_test, df = load_and_split(test_frac=0.2)

    # XGBoost requires 0-indexed labels
    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)

    models   = build_models()
    results  = []
    best_acc = 0
    best_name = None
    best_model = None

    print()
    for name, pipeline in models.items():
        print(f"[Training] {name}...")

        # XGBoost gets encoded labels
        if "XGBoost" in name:
            pipeline.fit(X_train, y_train_enc)
            y_pred     = le.inverse_transform(pipeline.predict(X_test))
            y_pred_enc = pipeline.predict(X_test)
            y_prob     = pipeline.predict_proba(X_test)

            metrics = {
                "Model":    name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "F1 Macro": round(f1_score(y_test, y_pred, average="macro"), 4),
                "Log Loss": round(log_loss(y_test_enc, y_prob), 4),
                "_y_pred":  y_pred,
                "_model":   pipeline
            }
        else:
            pipeline.fit(X_train, y_train)
            metrics = evaluate(name, pipeline, X_test, y_test)

        print(f"         Accuracy={metrics['Accuracy']:.4f} | "
              f"F1={metrics['F1 Macro']:.4f} | "
              f"LogLoss={metrics['Log Loss']:.4f}")

        # Confusion matrix
        plot_confusion(name, y_test, metrics["_y_pred"])

        # Feature importance (tree models)
        plot_feature_importance(metrics["_model"], name)

        # Track best
        if metrics["Accuracy"] > best_acc:
            best_acc   = metrics["Accuracy"]
            best_name  = name
            best_model = metrics["_model"]

        results.append({k: v for k, v in metrics.items() if not k.startswith("_")})

    # ── Summary table
    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    print("\n" + "=" * 55)
    print("  Results Summary")
    print("=" * 55)
    print(results_df.to_string(index=False))

    # ── Comparison plot
    plot_comparison(results_df)

    # ── Save best model
    print(f"\n[Best Model] {best_name} (Accuracy={best_acc:.4f})")
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump({"model": best_model, "name": best_name,
                 "features": FEATURE_COLS, "label_encoder": le if "XGBoost" in best_name else None},
                model_path)
    print(f"[Saved] {model_path}")

    # Save results table
    results_df.to_csv(os.path.join(OUTPUTS_DIR, "model_results.csv"), index=False)
    print(f"[Saved] outputs/model_results.csv")


if __name__ == "__main__":
    main()
