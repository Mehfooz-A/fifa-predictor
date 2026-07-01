"""
Stage 4: Predict New Match Outcomes
-------------------------------------
Loads the trained best model and predicts match outcomes
for user-specified team pairs.

Usage (from terminal):
    python src/04_predict.py

Or import and call predict_match() from another script/notebook.
"""

import pandas as pd
import numpy as np
import os
import joblib

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "../models")

LABEL_MAP = {1: "Home Win", 0: "Draw", -1: "Away Win"}


# ─── 1. Load model & data ─────────────────────────────────────────────────────

def load_artifacts():
    bundle   = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    features = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"), parse_dates=["date"])
    rankings = pd.read_csv(os.path.join(PROCESSED_DIR, "rankings_clean.csv"), parse_dates=["date"])
    return bundle, features, rankings


# ─── 2. Build a single match feature vector ───────────────────────────────────

def get_latest_rank(team, rankings):
    """Get most recent rank and points for a team."""
    df = rankings[rankings["team"] == team].sort_values("date")
    if df.empty:
        return None, None, None
    row = df.iloc[-1]
    return row["rank"], row["total_points"], row["confederation"]


def get_last_elo(team, features):
    """Get most recent Elo rating from the feature table."""
    home_df = features[features["home_team"] == team][["date", "home_elo"]].rename(
        columns={"home_elo": "elo"})
    away_df = features[features["away_team"] == team][["date", "away_elo"]].rename(
        columns={"away_elo": "elo"})
    combined = pd.concat([home_df, away_df]).sort_values("date")
    if combined.empty:
        return 1500.0
    return combined.iloc[-1]["elo"]


def get_recent_form(team, features, n=5):
    """Get last-n form stats for a team from the feature table."""
    home_form = features[features["home_team"] == team][
        ["date", "home_form_wr", "home_form_gd"]
    ].rename(columns={"home_form_wr": "form_wr", "home_form_gd": "form_gd"})
    
    away_form = features[features["away_team"] == team][
        ["date", "away_form_wr", "away_form_gd"]
    ].rename(columns={"away_form_wr": "form_wr", "away_form_gd": "form_gd"})
    
    combined = pd.concat([home_form, away_form]).sort_values("date")
    if combined.empty:
        return 0.4, 0.0
    recent = combined.tail(n)
    return recent["form_wr"].mean(), recent["form_gd"].mean()


def get_h2h(home_team, away_team, features):
    """Compute historical H2H win rate for home_team vs away_team."""
    mask = (
        ((features["home_team"] == home_team) & (features["away_team"] == away_team)) |
        ((features["home_team"] == away_team) & (features["away_team"] == home_team))
    )
    past = features[mask]
    if past.empty:
        return 0.5

    wins = 0
    for _, row in past.iterrows():
        if row["home_team"] == home_team and row["result"] == 1:
            wins += 1
        elif row["away_team"] == home_team and row["result"] == -1:
            wins += 1
    return wins / len(past)


TOURNAMENT_WEIGHTS = {
    "world_cup": 5,
    "qualifier": 4,
    "confederation": 3,
    "nations_league": 2,
    "friendly": 1
}


def build_feature_vector(home_team, away_team, tournament="friendly",
                          neutral=False, features_df=None, rankings=None):
    """Build a single-row feature vector for a new match."""
    
    h_rank, h_pts, _ = get_latest_rank(home_team, rankings)
    a_rank, a_pts, _ = get_latest_rank(away_team, rankings)
    h_elo            = get_last_elo(home_team, features_df)
    a_elo            = get_last_elo(away_team, features_df)
    h_wr, h_gd       = get_recent_form(home_team, features_df)
    a_wr, a_gd       = get_recent_form(away_team, features_df)
    h2h_wr           = get_h2h(home_team, away_team, features_df)
    t_weight         = TOURNAMENT_WEIGHTS.get(tournament.lower(), 1)

    # Fallback if team not found in rankings
    h_rank = h_rank or 80
    a_rank = a_rank or 80
    h_pts  = h_pts  or 500
    a_pts  = a_pts  or 500

    vector = {
        "home_rank":        h_rank,
        "away_rank":        a_rank,
        "rank_diff":        a_rank - h_rank,
        "home_points":      h_pts,
        "away_points":      a_pts,
        "points_diff":      h_pts - a_pts,
        "home_elo":         h_elo,
        "away_elo":         a_elo,
        "elo_diff":         h_elo - a_elo,
        "home_form_wr":     h_wr,
        "away_form_wr":     a_wr,
        "home_form_gd":     h_gd,
        "away_form_gd":     a_gd,
        "h2h_win_rate":     h2h_wr,
        "tournament_weight": t_weight,
        "neutral":          int(neutral)
    }

    return pd.DataFrame([vector])


# ─── 3. Predict ───────────────────────────────────────────────────────────────

def predict_match(home_team, away_team,
                  tournament="world_cup", neutral=False,
                  verbose=True):
    """
    Predict outcome of a match between home_team and away_team.
    
    Parameters
    ----------
    home_team  : str  e.g. "Brazil"
    away_team  : str  e.g. "Germany"
    tournament : str  one of: world_cup, qualifier, confederation, nations_league, friendly
    neutral    : bool True if played at neutral venue
    verbose    : bool Print breakdown
    
    Returns
    -------
    dict with prediction and probabilities
    """
    bundle, features_df, rankings = load_artifacts()
    model    = bundle["model"]
    le       = bundle["label_encoder"]
    feat_cols = bundle["features"]

    X = build_feature_vector(home_team, away_team, tournament, neutral,
                             features_df, rankings)

    proba = model.predict_proba(X)[0]
    classes = model.classes_

    # Map encoded classes back to -1/0/1 if XGBoost with label encoder
    if le is not None:
        classes = le.inverse_transform(classes)

    # Build probability dict
    class_probs = dict(zip(classes, proba))
    
    # Predicted class
    pred_class = classes[np.argmax(proba)]
    pred_label = LABEL_MAP.get(pred_class, str(pred_class))

    if verbose:
        print("\n" + "═" * 45)
        print(f"  🏟  {home_team}  vs  {away_team}")
        print(f"  📍  Tournament: {tournament} | Neutral: {neutral}")
        print("═" * 45)
        print(f"  ✅  Prediction: {pred_label}")
        print()
        print(f"  Home Win  → {class_probs.get(1, 0)*100:.1f}%")
        print(f"  Draw      → {class_probs.get(0, 0)*100:.1f}%")
        print(f"  Away Win  → {class_probs.get(-1, 0)*100:.1f}%")
        print("═" * 45 + "\n")

    return {
        "home_team":   home_team,
        "away_team":   away_team,
        "prediction":  pred_label,
        "home_win_p":  round(class_probs.get(1, 0), 3),
        "draw_p":      round(class_probs.get(0, 0), 3),
        "away_win_p":  round(class_probs.get(-1, 0), 3)
    }


# ─── 4. Simulate a World Cup bracket ──────────────────────────────────────────

def simulate_bracket(matches, tournament="world_cup"):
    """
    Predict outcomes for a list of (home, away) tuples.
    
    Example:
        simulate_bracket([
            ("Brazil", "Germany"),
            ("France", "Argentina"),
            ("Spain", "England")
        ])
    """
    print("\n" + "=" * 55)
    print("  World Cup Match Predictions")
    print("=" * 55)
    results = []
    for home, away in matches:
        r = predict_match(home, away, tournament=tournament, verbose=True)
        results.append(r)
    return pd.DataFrame(results)


# ─── 5. Main (demo) ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Example: Predict individual match
    predict_match("Brazil", "Germany", tournament="world_cup", neutral=True)
    predict_match("France", "Argentina", tournament="world_cup", neutral=False)
    predict_match("Spain", "England", tournament="qualifier", neutral=False)

    # ── Example: Simulate WC 2018-era group stage
    wc_matches = [
        ("France",    "Australia"),
        ("Argentina", "Iceland"),
        ("Brazil",    "Switzerland"),
        ("Germany",   "Mexico"),
        ("Spain",     "Portugal"),
    ]
    df = simulate_bracket(wc_matches)
    print("\nSummary Table:")
    print(df[["home_team", "away_team", "prediction",
              "home_win_p", "draw_p", "away_win_p"]].to_string(index=False))
