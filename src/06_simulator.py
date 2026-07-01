"""
Stage 6: 2026 FIFA World Cup Bracket Simulator (Optimized)
-----------------------------------------------------------
Optimizations:
  1. Pre-computed probability table (Method 3)
     → All 48x48 matchup probabilities computed once upfront
     → Simulations use instant dictionary lookup instead of model calls

  2. Parallel simulations (Method 2)  
     → Uses multiprocessing to run simulations across all CPU cores
     → 8 cores running simultaneously = ~8x speedup
"""

import pandas as pd
import numpy as np
import joblib
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count

BASE_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")

FEATURE_COLS = [
    "home_rank", "away_rank", "rank_diff",
    "home_points", "away_points", "points_diff",
    "home_elo", "away_elo", "elo_diff",
    "home_form_wr", "away_form_wr",
    "home_form_gd", "away_form_gd",
    "h2h_win_rate", "tournament_weight", "neutral",
    "home_goals_pg", "away_goals_pg", "goals_pg_diff",
]

GROUPS = {
    "A": ["Argentina", "Ecuador", "Canada", "Slovenia"],
    "B": ["France", "Morocco", "Croatia", "Paraguay"],
    "C": ["Spain", "Chile", "Senegal", "Albania"],
    "D": ["England", "Colombia", "Japan", "Panama"],
    "E": ["Brazil", "Netherlands", "Egypt", "Bolivia"],
    "F": ["Germany", "Portugal", "Ghana", "Honduras"],
    "G": ["Belgium", "Mexico", "Saudi Arabia", "New Zealand"],
    "H": ["Uruguay", "Denmark", "Cameroon", "Jamaica"],
    "I": ["Italy", "United States", "Australia", "Jordan"],
    "J": ["Switzerland", "South Korea", "Nigeria", "Venezuela"],
    "K": ["Turkey", "Iran", "Uzbekistan", "Costa Rica"],
    "L": ["Serbia", "Peru", "South Africa", "Greece"],
}

NAME_MAP = {
    "United States": "United States",
    "South Korea":   "South Korea",
    "Iran":          "Iran",
    "DR Congo":      "Congo",
}


# ─── Load Artifacts ───────────────────────────────────────────────────────────

def load_artifacts():
    bundle   = joblib.load(os.path.join(MODELS_DIR, "best_tuned_model.pkl"))
    features = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"),
                           parse_dates=["date"])
    rankings = pd.read_csv(os.path.join(PROCESSED_DIR, "rankings_clean.csv"),
                           parse_dates=["date"])
    return bundle["model"], bundle["label_encoder"], features, rankings


# ─── Team Stats ───────────────────────────────────────────────────────────────

def get_team_stats(team, features, rankings):
    lookup = NAME_MAP.get(team, team)

    r      = rankings[rankings["team"] == lookup].sort_values("date")
    rank   = float(r.iloc[-1]["rank"])          if not r.empty else 80.0
    points = float(r.iloc[-1]["total_points"])  if not r.empty else 400.0

    h_elo  = features[features["home_team"] == lookup][["date","home_elo"]].rename(columns={"home_elo":"elo"})
    a_elo  = features[features["away_team"] == lookup][["date","away_elo"]].rename(columns={"away_elo":"elo"})
    elo_df = pd.concat([h_elo, a_elo]).sort_values("date")
    elo    = float(elo_df.iloc[-1]["elo"]) if not elo_df.empty else 1500.0

    h_form  = features[features["home_team"] == lookup][["date","home_form_wr","home_form_gd"]].rename(columns={"home_form_wr":"wr","home_form_gd":"gd"})
    a_form  = features[features["away_team"] == lookup][["date","away_form_wr","away_form_gd"]].rename(columns={"away_form_wr":"wr","away_form_gd":"gd"})
    form_df = pd.concat([h_form, a_form]).sort_values("date")
    wr = float(form_df.tail(5)["wr"].mean()) if not form_df.empty else 0.4
    gd = float(form_df.tail(5)["gd"].mean()) if not form_df.empty else 0.0

    h_gpg  = features[features["home_team"] == lookup][["date","home_goals_pg"]].rename(columns={"home_goals_pg":"gpg"})
    a_gpg  = features[features["away_team"] == lookup][["date","away_goals_pg"]].rename(columns={"away_goals_pg":"gpg"})
    gpg_df = pd.concat([h_gpg, a_gpg]).sort_values("date")
    gpg    = float(gpg_df.tail(5)["gpg"].mean()) if not gpg_df.empty else 1.5

    return {"rank": rank, "points": points, "elo": elo,
            "wr": wr, "gd": gd, "gpg": gpg}


def get_h2h(home, away, features):
    lh   = NAME_MAP.get(home, home)
    la   = NAME_MAP.get(away, away)
    mask = (
        ((features["home_team"] == lh) & (features["away_team"] == la)) |
        ((features["home_team"] == la) & (features["away_team"] == lh))
    )
    past = features[mask]
    if past.empty:
        return 0.5
    wins = (((past["home_team"] == lh) & (past["result"] == 1)).sum() +
            ((past["away_team"] == lh) & (past["result"] == -1)).sum())
    return wins / len(past)


# ─── Method 3: Pre-compute Probability Table ──────────────────────────────────

def build_prob_table(all_teams, stats, features, model, le):
    """
    Pre-compute win probabilities for every possible matchup.
    
    Instead of calling model.predict_proba() 103,000 times during simulation,
    we call it once for all 48x48=2304 matchups upfront.
    
    Result: a dictionary {(home, away): (home_p, draw_p, away_p)}
    Lookup during simulation is O(1) — instant.
    """
    print("[Optimization] Pre-computing probability table for all matchups...")

    # Build ALL feature vectors at once (one row per matchup)
    rows  = []
    pairs = []

    for home in all_teams:
        for away in all_teams:
            if home == away:
                continue
            hs  = stats[home]
            as_ = stats[away]
            h2h = get_h2h(home, away, features)

            rows.append({
                "home_rank":         hs["rank"],
                "away_rank":         as_["rank"],
                "rank_diff":         as_["rank"] - hs["rank"],
                "home_points":       hs["points"],
                "away_points":       as_["points"],
                "points_diff":       hs["points"] - as_["points"],
                "home_elo":          hs["elo"],
                "away_elo":          as_["elo"],
                "elo_diff":          hs["elo"] - as_["elo"],
                "home_form_wr":      hs["wr"],
                "away_form_wr":      as_["wr"],
                "home_form_gd":      hs["gd"],
                "away_form_gd":      as_["gd"],
                "h2h_win_rate":      h2h,
                "tournament_weight": 5,
                "neutral":           1,
                "home_goals_pg":     hs["gpg"],
                "away_goals_pg":     as_["gpg"],
                "goals_pg_diff":     hs["gpg"] - as_["gpg"],
            })
            pairs.append((home, away))

    # Single batch prediction — replaces 103,000 individual calls
    X      = pd.DataFrame(rows)
    probas = model.predict_proba(X)
    classes = le.inverse_transform(model.classes_)

    # Map class indices to home/draw/away probabilities
    c_map  = {c: i for i, c in enumerate(classes)}
    home_i = c_map.get(1,  0)
    draw_i = c_map.get(0,  1)
    away_i = c_map.get(-1, 2)

    prob_table = {}
    for i, (home, away) in enumerate(pairs):
        hp = float(probas[i][home_i])
        dp = float(probas[i][draw_i])
        ap = float(probas[i][away_i])
        prob_table[(home, away)] = (hp, dp, ap)

    n_matchups = len(pairs)
    print(f"[Optimization] Done — {n_matchups} matchups pre-computed in one batch call.")
    return prob_table


# ─── Match Simulation (using lookup table) ────────────────────────────────────

def simulate_match_fast(home, away, prob_table, knockout=False):
    """
    Instant match simulation using pre-computed probability table.
    No model call needed — just a dictionary lookup.
    """
    hp, dp, ap = prob_table[(home, away)]

    if knockout:
        hp += dp / 2
        ap += dp / 2
        dp  = 0

    probs = np.array([hp, dp, ap], dtype=float)
    probs /= probs.sum()
    outcome = np.random.choice(["home", "draw", "away"], p=probs)

    if outcome == "home":
        return home
    elif outcome == "away":
        return away
    else:
        return None  # draw in group stage


# ─── Group Stage ──────────────────────────────────────────────────────────────

def simulate_group_fast(group_teams, prob_table, stats):
    points = defaultdict(int)
    gd     = defaultdict(float)

    for i in range(len(group_teams)):
        for j in range(i+1, len(group_teams)):
            home   = group_teams[i]
            away   = group_teams[j]
            winner = simulate_match_fast(home, away, prob_table)

            if winner is None:
                points[home] += 1
                points[away] += 1
            elif winner == home:
                points[home] += 3
                elo_diff = stats[home]["elo"] - stats[away]["elo"]
                gd[home] += max(1, round(elo_diff / 200))
                gd[away] -= max(1, round(elo_diff / 200))
            else:
                points[away] += 3
                elo_diff = stats[away]["elo"] - stats[home]["elo"]
                gd[away] += max(1, round(elo_diff / 200))
                gd[home] -= max(1, round(elo_diff / 200))

    standings = sorted(group_teams,
                       key=lambda t: (points[t], gd[t]),
                       reverse=True)
    return standings, points, gd


# ─── Full Tournament (one simulation) ─────────────────────────────────────────

def simulate_tournament_fast(prob_table, stats):
    qualified   = []
    third_place = []

    for group_name, teams in GROUPS.items():
        standings, pts, gd = simulate_group_fast(teams, prob_table, stats)
        qualified.append(standings[0])
        qualified.append(standings[1])
        third_place.append((standings[2], pts[standings[2]], gd[standings[2]]))

    third_place.sort(key=lambda x: (x[1], x[2]), reverse=True)
    qualified += [t[0] for t in third_place[:8]]

    np.random.shuffle(qualified)

    round_teams = qualified
    while len(round_teams) > 1:
        next_round = []
        for i in range(0, len(round_teams), 2):
            winner = simulate_match_fast(
                round_teams[i], round_teams[i+1],
                prob_table, knockout=True
            )
            next_round.append(winner)
        round_teams = next_round

    return round_teams[0]


# ─── Method 2: Parallel Worker ────────────────────────────────────────────────

def run_batch(args):
    """
    Worker function for parallel processing.
    Each CPU core runs this function independently.
    
    args = (n_sims, prob_table, stats, seed)
    Returns a list of winners from this batch.
    """
    n_sims, prob_table, stats, seed = args
    np.random.seed(seed)  # different seed per core → different random outcomes
    winners = []
    for _ in range(n_sims):
        winner = simulate_tournament_fast(prob_table, stats)
        winners.append(winner)
    return winners


# ─── Main Simulation ──────────────────────────────────────────────────────────

def run_simulations(n=1000):
    import time
    start = time.time()

    print("=" * 55)
    print("  2026 FIFA World Cup Simulator (Optimized)")
    print("=" * 55)

    model, le, features, rankings = load_artifacts()

    all_teams = list(set(t for g in GROUPS.values() for t in g))
    print(f"\n[Setup] Computing stats for {len(all_teams)} teams...")
    stats = {team: get_team_stats(team, features, rankings)
             for team in all_teams}

    # Method 3: Pre-compute all matchup probabilities
    prob_table = build_prob_table(all_teams, stats, features, model, le)

    # Method 2: Parallel processing
    n_cores = min(8, cpu_count())
    batch_size = n // n_cores
    remainder  = n % n_cores

    print(f"\n[Parallel] Using {n_cores} CPU cores")
    print(f"[Parallel] {n} simulations split into {n_cores} batches of ~{batch_size} each")
    print(f"[Simulation] Running {n} simulations in parallel...\n")

    # Each core gets a batch of simulations with a unique random seed
    batches = []
    for i in range(n_cores):
        size = batch_size + (1 if i < remainder else 0)
        batches.append((size, prob_table, stats, i * 1000))

    # Run all batches simultaneously across CPU cores
    with Pool(processes=n_cores) as pool:
        batch_results = pool.map(run_batch, batches)

    # Combine results from all cores
    all_winners = [w for batch in batch_results for w in batch]

    # Count wins
    win_counts = defaultdict(int)
    for winner in all_winners:
        win_counts[winner] += 1

    elapsed = time.time() - start

    # Results
    print("=" * 55)
    print("  2026 World Cup Win Probabilities")
    print(f"  ({n} simulations | {elapsed:.1f} seconds)")
    print("=" * 55)

    results = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Team':<25} {'Wins':<8} {'Probability'}")
    print("-" * 55)
    for rank, (team, wins) in enumerate(results, 1):
        prob = wins / n * 100
        bar  = "█" * int(prob / 2)
        print(f"{rank:<6} {team:<25} {wins:<8} {prob:.1f}%  {bar}")

    print(f"\n🏆 Most likely winner: {results[0][0]} ({results[0][1]/n*100:.1f}%)")
    print(f"⚡ Total time: {elapsed:.1f} seconds")

    return dict(win_counts)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_simulations(n=1000)