"""
Stage 1: Data Loading & Cleaning
---------------------------------
- Loads FIFA rankings and former country names
- Downloads/loads match results (from Kaggle)
- Harmonizes country names using former_names mapping
- Filters to World Cup era matches (1993+)
- Saves cleaned data to data/processed/
"""

import pandas as pd
import numpy as np
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ─── 1. Load FIFA Rankings ────────────────────────────────────────────────────

def load_rankings():
    path = os.path.join(RAW_DIR, "fifa_ranking.csv")
    df = pd.read_csv(path, parse_dates=["rank_date"])
    
    # Keep only the columns we need
    df = df[["country_full", "rank", "total_points", "confederation", "rank_date"]]
    df = df.rename(columns={"country_full": "team", "rank_date": "date"})
    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    
    print(f"[Rankings] Loaded {len(df):,} rows | {df['team'].nunique()} teams | "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ─── 2. Load Former Names Mapping ─────────────────────────────────────────────

def load_name_map():
    path = os.path.join(RAW_DIR, "former_names.csv")
    df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    print(f"[Names] Loaded {len(df)} former name mappings")
    return df


def resolve_team_name(team, match_date, name_map):
    """
    Maps a former country name to its current name based on the match date.
    E.g. 'Zaire' in 1974 → 'DR Congo'
    """
    match_date = pd.Timestamp(match_date)
    for _, row in name_map.iterrows():
        if (row["former"] == team and
                row["start_date"] <= match_date <= row["end_date"]):
            return row["current"]
    return team


# ─── 3. Load Match Results ────────────────────────────────────────────────────

def load_results(path=None):
    """
    Load international match results.
    
    If you have results.csv from Kaggle, place it in data/raw/ and it will
    be picked up automatically. Otherwise a sample is created for testing.
    
    Kaggle dataset: 
    https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
    """
    if path is None:
        path = os.path.join(RAW_DIR, "results.csv")

    if not os.path.exists(path):
        print("[Results] results.csv not found — generating sample data for pipeline testing.")
        print("          → Download from Kaggle and place at data/raw/results.csv")
        return _generate_sample_results()

    df = pd.read_csv(path, parse_dates=["date"])
    print(f"[Results] Loaded {len(df):,} matches | "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


def _generate_sample_results():
    """Creates a minimal sample dataset for pipeline testing without Kaggle data."""
    np.random.seed(42)
    teams = [
        "Brazil", "Germany", "France", "Spain", "Argentina",
        "Italy", "England", "Netherlands", "Portugal", "Belgium",
        "Uruguay", "Croatia", "Mexico", "Colombia", "Chile"
    ]
    tournaments = ["FIFA World Cup", "FIFA World Cup qualification", "Friendly"]
    records = []
    
    for _ in range(500):
        home, away = np.random.choice(teams, 2, replace=False)
        hg = np.random.randint(0, 5)
        ag = np.random.randint(0, 5)
        records.append({
            "date": pd.Timestamp("1993-01-01") + pd.Timedelta(days=np.random.randint(0, 9000)),
            "home_team": home,
            "away_team": away,
            "home_score": hg,
            "away_score": ag,
            "tournament": np.random.choice(tournaments, p=[0.15, 0.45, 0.4]),
            "city": "Sample City",
            "country": "Sample Country",
            "neutral": np.random.choice([True, False])
        })
    
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"[Results] Sample data: {len(df)} matches created for testing.")
    return df


# ─── 4. Harmonize Names ───────────────────────────────────────────────────────

def harmonize_names(results, name_map):
    """Apply former-name resolution to all teams in results."""
    print("[Names] Resolving former country names...")
    results = results.copy()
    
    # Vectorized approach: build a lookup for speed
    # For each (team, date) check against name_map
    def resolve(row, col):
        return resolve_team_name(row[col], row["date"], name_map)
    
    results["home_team"] = results.apply(lambda r: resolve(r, "home_team"), axis=1)
    results["away_team"] = results.apply(lambda r: resolve(r, "away_team"), axis=1)
    
    print("[Names] Done.")
    return results


# ─── 5. Filter & Label ────────────────────────────────────────────────────────

def add_target(df):
    """Add match result label: 1=Home Win, 0=Draw, -1=Away Win"""
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
        df["home_score"] < df["away_score"]
    ]
    df["result"] = np.select(conditions, [1, 0, -1])
    return df


def filter_wc_era(df, start_year=1993):
    """Keep only matches from the FIFA ranking era onwards."""
    mask = df["date"].dt.year >= start_year
    filtered = df[mask].copy().reset_index(drop=True)
    print(f"[Filter] {len(df):,} → {len(filtered):,} matches (from {start_year})")
    return filtered


# ─── 6. Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Stage 1: Data Loading & Cleaning")
    print("=" * 55)

    rankings = load_rankings()
    name_map = load_name_map()
    results  = load_results()

    results  = filter_wc_era(results)
    results  = harmonize_names(results, name_map)
    results  = add_target(results)

    # Save
    rankings.to_csv(os.path.join(PROCESSED_DIR, "rankings_clean.csv"), index=False)
    results.to_csv(os.path.join(PROCESSED_DIR, "results_clean.csv"), index=False)

    print("\n[Done] Saved to data/processed/")
    print(f"  - rankings_clean.csv  → {len(rankings):,} rows")
    print(f"  - results_clean.csv   → {len(results):,} rows")
    print(f"\nResult distribution:")
    print(results["result"].map({1: "Home Win", 0: "Draw", -1: "Away Win"}).value_counts())


if __name__ == "__main__":
    main()
