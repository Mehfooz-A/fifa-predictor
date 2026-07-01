"""
Stage 2: Feature Engineering
------------------------------
For every match, we build features that represent team strength and form
BEFORE the match date (no data leakage).

Features built:
  - FIFA rank of each team (nearest rank_date before match)
  - Elo rating at time of match (computed from scratch)
  - Recent form (win rate over last N matches)
  - Recent avg goal difference (last N matches)
  - Head-to-head win rate (historical)
  - Tournament type encoding
  - Neutral venue flag
  - Confederation of each team
"""

import pandas as pd
import numpy as np
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")


# ─── 1. FIFA Rank Lookup ──────────────────────────────────────────────────────

def get_rank_at_date(team, date, rankings_df):
    """
    Get a team's FIFA rank on or just before the match date.
    Returns None if no ranking found within 12 months before the match.
    """
    df = rankings_df[rankings_df["team"] == team]
    df = df[df["date"] <= date]
    if df.empty:
        return None
    return df.iloc[-1]["rank"]


def add_rankings(results, rankings):
    """
    Vectorized-ish ranking lookup using merge_asof.
    Joins ranking snapshot closest to (but not after) the match date.
    """
    print("[Features] Adding FIFA rankings...")
    
    rankings = rankings.sort_values("date")
    results  = results.sort_values("date")

    # Home team rank
    home_rank = pd.merge_asof(
        results[["date", "home_team"]].rename(columns={"home_team": "team"}),
        rankings[["date", "team", "rank", "total_points", "confederation"]].rename(
            columns={"rank": "home_rank", "total_points": "home_points",
                     "confederation": "home_conf"}),
        on="date", by="team", direction="backward", tolerance=pd.Timedelta("365D")
    )

    # Away team rank
    away_rank = pd.merge_asof(
        results[["date", "away_team"]].rename(columns={"away_team": "team"}),
        rankings[["date", "team", "rank", "total_points", "confederation"]].rename(
            columns={"rank": "away_rank", "total_points": "away_points",
                     "confederation": "away_conf"}),
        on="date", by="team", direction="backward", tolerance=pd.Timedelta("365D")
    )

    results = results.copy()
    results["home_rank"]   = home_rank["home_rank"].values
    results["home_points"] = home_rank["home_points"].values
    results["home_conf"]   = home_rank["home_conf"].values
    results["away_rank"]   = away_rank["away_rank"].values
    results["away_points"] = away_rank["away_points"].values
    results["away_conf"]   = away_rank["away_conf"].values

    # Rank difference (positive = home is ranked higher = lower rank number)
    results["rank_diff"] = results["away_rank"] - results["home_rank"]
    results["points_diff"] = results["home_points"] - results["away_points"]

    return results


# ─── 2. Elo Rating ────────────────────────────────────────────────────────────

ELO_K = 30          # sensitivity of rating update
ELO_START = 1500    # default starting Elo for all teams

def compute_elo(results):
    """
    Compute Elo ratings from scratch, match by match.
    Each team starts at 1500. After each match, ratings are updated
    using the standard Elo update formula.
    
    Returns results df with home_elo_before and away_elo_before columns.
    """
    print("[Features] Computing Elo ratings...")
    
    elo = {}  # team → current Elo

    home_elos, away_elos = [], []

    for _, row in results.iterrows():
        h = row["home_team"]
        a = row["away_team"]

        # Get current ratings (default 1500)
        elo_h = elo.get(h, ELO_START)
        elo_a = elo.get(a, ELO_START)

        home_elos.append(elo_h)
        away_elos.append(elo_a)

        # Expected scores
        exp_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        exp_a = 1 - exp_h

        # Actual scores
        if row["result"] == 1:     # Home win
            s_h, s_a = 1.0, 0.0
        elif row["result"] == 0:   # Draw
            s_h, s_a = 0.5, 0.5
        else:                      # Away win
            s_h, s_a = 0.0, 1.0

        # Update
        elo[h] = elo_h + ELO_K * (s_h - exp_h)
        elo[a] = elo_a + ELO_K * (s_a - exp_a)

    results = results.copy()
    results["home_elo"] = home_elos
    results["away_elo"] = away_elos
    results["elo_diff"] = results["home_elo"] - results["away_elo"]

    return results


# ─── 3. Recent Form ───────────────────────────────────────────────────────────

def compute_form(results, n=5):
    """
    For each match row, compute each team's recent form over the last n matches.
    
    - form_win_rate: fraction of last n matches won
    - form_goal_diff: average goal difference in last n matches
    
    These are computed BEFORE the current match (no leakage).
    """
    print(f"[Features] Computing last-{n} form stats...")
    
    results = results.copy().reset_index(drop=True)
    results["match_idx"] = results.index

    # Build a team-centric history: one row per team per match
    home_rows = results[["match_idx", "date", "home_team", "home_score", "away_score", "result"]].copy()
    home_rows.columns = ["match_idx", "date", "team", "goals_for", "goals_against", "result_raw"]
    home_rows["won"] = (home_rows["result_raw"] == 1).astype(int)
    home_rows["goal_diff"] = home_rows["goals_for"] - home_rows["goals_against"]

    away_rows = results[["match_idx", "date", "away_team", "away_score", "home_score", "result"]].copy()
    away_rows.columns = ["match_idx", "date", "team", "goals_for", "goals_against", "result_raw"]
    away_rows["won"] = (away_rows["result_raw"] == -1).astype(int)
    away_rows["goal_diff"] = away_rows["goals_for"] - away_rows["goals_against"]

    history = pd.concat([home_rows, away_rows]).sort_values(["team", "date", "match_idx"])

    # Rolling stats (shift 1 so current match not included)
    history["form_win_rate"] = (
        history.groupby("team")["won"]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    )
    history["form_goal_diff"] = (
        history.groupby("team")["goal_diff"]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    )

    # Split back into home/away perspectives
    h_form = history[history["match_idx"].isin(home_rows["match_idx"])][
        ["match_idx", "form_win_rate", "form_goal_diff"]
    ].rename(columns={"form_win_rate": "home_form_wr", "form_goal_diff": "home_form_gd"})

    a_form = history[history["match_idx"].isin(away_rows["match_idx"])][
        ["match_idx", "form_win_rate", "form_goal_diff"]
    ].rename(columns={"form_win_rate": "away_form_wr", "form_goal_diff": "away_form_gd"})

    # Deduplicate (each match_idx appears twice in history)
    h_form = h_form.drop_duplicates("match_idx").set_index("match_idx")
    a_form = a_form.drop_duplicates("match_idx").set_index("match_idx")

    results = results.join(h_form, on="match_idx")
    results = results.join(a_form, on="match_idx")

    return results


# ─── 4. Head-to-Head ──────────────────────────────────────────────────────────

def compute_h2h(results):
    """
    For each match, compute home team's historical win rate vs this specific away team.
    Only uses matches before the current match (no leakage).
    """
    print("[Features] Computing head-to-head stats...")
    
    results = results.copy().reset_index(drop=True)
    h2h_rates = []

    for i, row in results.iterrows():
        # All previous matches between these two teams
        prev = results.iloc[:i]
        mask = (
            ((prev["home_team"] == row["home_team"]) & (prev["away_team"] == row["away_team"])) |
            ((prev["home_team"] == row["away_team"]) & (prev["away_team"] == row["home_team"]))
        )
        past = prev[mask]

        if past.empty:
            h2h_rates.append(0.5)  # No history → neutral
            continue

        # Count wins for home_team (row perspective)
        wins = 0
        for _, p in past.iterrows():
            if p["home_team"] == row["home_team"] and p["result"] == 1:
                wins += 1
            elif p["away_team"] == row["home_team"] and p["result"] == -1:
                wins += 1

        h2h_rates.append(wins / len(past))

    results["h2h_win_rate"] = h2h_rates
    return results

# ─── . compute_form() ──────────────────────────────────────────────────────────

def compute_scoring_features(results, goals_df):
    print("[Features] Computing scoring features...")
    
    goals_df = goals_df[goals_df["own_goal"] == False].copy()
    goals_df["date"] = pd.to_datetime(goals_df["date"])
    
    team_goals = goals_df.groupby(["date", "team"]).size().reset_index(name="goals")
    team_goals = team_goals.sort_values(["team", "date"]).reset_index(drop=True)
    
    team_goals["goals_avg"] = (
        team_goals.groupby("team")["goals"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    
    team_goals_dict = team_goals.set_index(["team", "date"])["goals_avg"].to_dict()
    
    def get_scoring_avg(team, date):
        subset = team_goals[
            (team_goals["team"] == team) & 
            (team_goals["date"] < date)
        ]
        if subset.empty:
            return 1.5
        return subset.iloc[-1]["goals_avg"]
    
    results = results.copy()
    results["home_goals_pg"] = results.apply(
        lambda r: get_scoring_avg(r["home_team"], r["date"]), axis=1)
    results["away_goals_pg"] = results.apply(
        lambda r: get_scoring_avg(r["away_team"], r["date"]), axis=1)
    results["goals_pg_diff"] = results["home_goals_pg"] - results["away_goals_pg"]
    
    print("[Features] Done.")
    return results


# ─── 5. Tournament Encoding ───────────────────────────────────────────────────

def encode_tournament(results):
    """Encode tournament type into ordinal importance score."""
    tournament_priority = {
        "FIFA World Cup": 5,
        "FIFA World Cup qualification": 4,
        "Confederations Cup": 3,
        "UEFA Euro": 3,
        "Copa América": 3,
        "African Cup of Nations": 3,
        "UEFA Nations League": 2,
        "Friendly": 1,
    }
    results = results.copy()
    results["tournament_weight"] = results["tournament"].map(
        lambda t: next((v for k, v in tournament_priority.items() if k in str(t)), 1)
    )
    return results


# ─── 6. Final Feature Set ─────────────────────────────────────────────────────

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


# ─── 7. Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Stage 2: Feature Engineering")
    print("=" * 55)

    rankings = pd.read_csv(os.path.join(PROCESSED_DIR, "rankings_clean.csv"), parse_dates=["date"])
    results  = pd.read_csv(os.path.join(PROCESSED_DIR, "results_clean.csv"),  parse_dates=["date"])

    print(f"\nLoaded {len(results):,} matches and {len(rankings):,} ranking entries.\n")

    results = add_rankings(results, rankings)
    results = compute_elo(results)
    results = compute_form(results, n=5)
    results = compute_h2h(results)
    goals = pd.read_csv(os.path.join(RAW_DIR, "goalscorers.csv"), parse_dates=["date"])
    results = compute_scoring_features(results, goals)
    results = encode_tournament(results)

    # Cast neutral to int if bool
    results["neutral"] = results["neutral"].astype(int)

    # Drop rows with missing core features
    before = len(results)
    results = results.dropna(subset=["home_rank", "away_rank", "home_elo", "away_elo"])
    print(f"\n[Clean] Dropped {before - len(results)} rows with missing rank/elo → {len(results):,} remain")

    # Save
    out_path = os.path.join(PROCESSED_DIR, "features.csv")
    results.to_csv(out_path, index=False)
    print(f"\n[Done] Saved features.csv → {len(results):,} rows × {len(results.columns)} cols")
    print(f"\nFeature columns: {FEATURE_COLS}")
    print(f"\nSample row:\n{results[FEATURE_COLS].iloc[0]}")


if __name__ == "__main__":
    main()
