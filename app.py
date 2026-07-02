"""
FIFA Match Outcome Predictor — Streamlit App
---------------------------------------------
Run with: streamlit run app.py

Pages:
  1. Match Predictor  — predict outcome of any match
  2. WC Simulator     — simulate the 2026 World Cup bracket
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from collections import defaultdict

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FIFA Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# ─── Load Model & Data ────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    base     = os.path.dirname(os.path.abspath(__file__))
    bundle   = joblib.load(os.path.join(base, "models/best_tuned_model.pkl"))
    features = pd.read_csv(os.path.join(base, "data/processed/features.csv"),
                           parse_dates=["date"])
    rankings = pd.read_csv(os.path.join(base, "data/processed/rankings_clean.csv"),
                           parse_dates=["date"])
    return bundle, features, rankings


# ─── Constants ────────────────────────────────────────────────────────────────

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup":      5,
    "World Cup Qualifier": 4,
    "Continental Cup":     3,
    "Nations League":      2,
    "Friendly":            1,
}

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


# ─── Feature Building Helpers ─────────────────────────────────────────────────

def get_latest_rank(team, rankings):
    df = rankings[rankings["team"] == team].sort_values("date")
    if df.empty:
        return 80, 500
    row = df.iloc[-1]
    return row["rank"], row["total_points"]


def get_last_elo(team, features):
    home = features[features["home_team"] == team][["date","home_elo"]].rename(columns={"home_elo":"elo"})
    away = features[features["away_team"] == team][["date","away_elo"]].rename(columns={"away_elo":"elo"})
    combined = pd.concat([home, away]).sort_values("date")
    if combined.empty:
        return 1500.0
    return combined.iloc[-1]["elo"]


def get_recent_form(team, features, n=5):
    home = features[features["home_team"] == team][["date","home_form_wr","home_form_gd"]].rename(
        columns={"home_form_wr":"wr","home_form_gd":"gd"})
    away = features[features["away_team"] == team][["date","away_form_wr","away_form_gd"]].rename(
        columns={"away_form_wr":"wr","away_form_gd":"gd"})
    combined = pd.concat([home, away]).sort_values("date")
    if combined.empty:
        return 0.4, 0.0
    recent = combined.tail(n)
    return recent["wr"].mean(), recent["gd"].mean()


def get_goals_pg(team, features, n=5):
    home = features[features["home_team"] == team][["date","home_goals_pg"]].rename(
        columns={"home_goals_pg":"gpg"})
    away = features[features["away_team"] == team][["date","away_goals_pg"]].rename(
        columns={"away_goals_pg":"gpg"})
    combined = pd.concat([home, away]).sort_values("date")
    if combined.empty:
        return 1.5
    return combined.tail(n)["gpg"].mean()


def get_h2h(home_team, away_team, features):
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


def build_features(home_team, away_team, tournament, neutral, features_df, rankings):
    h_rank, h_pts = get_latest_rank(home_team, rankings)
    a_rank, a_pts = get_latest_rank(away_team, rankings)
    h_elo         = get_last_elo(home_team, features_df)
    a_elo         = get_last_elo(away_team, features_df)
    h_wr, h_gd    = get_recent_form(home_team, features_df)
    a_wr, a_gd    = get_recent_form(away_team, features_df)
    h_gpg         = get_goals_pg(home_team, features_df)
    a_gpg         = get_goals_pg(away_team, features_df)
    h2h           = get_h2h(home_team, away_team, features_df)
    t_weight      = TOURNAMENT_WEIGHTS.get(tournament, 1)

    return pd.DataFrame([{
        "home_rank":         h_rank,
        "away_rank":         a_rank,
        "rank_diff":         a_rank - h_rank,
        "home_points":       h_pts,
        "away_points":       a_pts,
        "points_diff":       h_pts - a_pts,
        "home_elo":          h_elo,
        "away_elo":          a_elo,
        "elo_diff":          h_elo - a_elo,
        "home_form_wr":      h_wr,
        "away_form_wr":      a_wr,
        "home_form_gd":      h_gd,
        "away_form_gd":      a_gd,
        "h2h_win_rate":      h2h,
        "tournament_weight": t_weight,
        "neutral":           int(neutral),
        "home_goals_pg":     h_gpg,
        "away_goals_pg":     a_gpg,
        "goals_pg_diff":     h_gpg - a_gpg,
    }])


# ─── Page 1: Match Predictor ──────────────────────────────────────────────────

def show_predictor(model, le, features_df, rankings):
    st.title("⚽ FIFA Match Outcome Predictor")
    st.markdown("Predict the outcome of any international football match using ML.")
    st.divider()

    all_teams = sorted(set(
        features_df["home_team"].tolist() +
        features_df["away_team"].tolist()
    ))

    st.subheader("🏟 Match Setup")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", all_teams,
                                  index=all_teams.index("Brazil") if "Brazil" in all_teams else 0)
    with col2:
        away_team = st.selectbox("Away Team", all_teams,
                                  index=all_teams.index("Argentina") if "Argentina" in all_teams else 1)

    col3, col4 = st.columns(2)
    with col3:
        tournament = st.selectbox("Tournament", list(TOURNAMENT_WEIGHTS.keys()))
    with col4:
        neutral = st.checkbox("Neutral Venue", value=False)

    st.divider()

    if st.button("🔮 Predict Outcome", use_container_width=True):
        if home_team == away_team:
            st.error("Home and Away teams must be different.")
            return

        X = build_features(home_team, away_team, tournament, neutral,
                           features_df, rankings)

        proba     = model.predict_proba(X)[0]
        classes   = le.inverse_transform(model.classes_)
        class_map = dict(zip(classes, proba))

        home_p = class_map.get(1,  0)
        draw_p = class_map.get(0,  0)
        away_p = class_map.get(-1, 0)

        pred       = max(class_map, key=class_map.get)
        pred_label = {
            1:  f"🏆 {home_team} Win",
            0:  "🤝 Draw",
            -1: f"🏆 {away_team} Win"
        }[pred]

        st.subheader("📊 Prediction")
        st.success(f"**Predicted Outcome: {pred_label}**")

        st.markdown("#### Win Probabilities")
        col_h, col_d, col_a = st.columns(3)
        with col_h:
            st.metric(f"{home_team} Win", f"{home_p*100:.1f}%")
            st.progress(float(home_p))
        with col_d:
            st.metric("Draw", f"{draw_p*100:.1f}%")
            st.progress(float(draw_p))
        with col_a:
            st.metric(f"{away_team} Win", f"{away_p*100:.1f}%")
            st.progress(float(away_p))

        st.divider()
        st.subheader("📈 Team Stats Comparison")

        h_rank, _ = get_latest_rank(home_team, rankings)
        a_rank, _ = get_latest_rank(away_team, rankings)
        h_elo     = get_last_elo(home_team, features_df)
        a_elo     = get_last_elo(away_team, features_df)
        h_wr, _   = get_recent_form(home_team, features_df)
        a_wr, _   = get_recent_form(away_team, features_df)
        h_gpg     = get_goals_pg(home_team, features_df)
        a_gpg     = get_goals_pg(away_team, features_df)
        h2h       = get_h2h(home_team, away_team, features_df)

        stats_df = pd.DataFrame({
            "Stat":    ["FIFA Rank", "Elo Rating", "Form Win Rate",
                        "Goals/Game", "H2H Win Rate"],
            home_team: [f"#{int(h_rank)}", f"{h_elo:.0f}",
                        f"{h_wr*100:.0f}%", f"{h_gpg:.2f}",
                        f"{h2h*100:.0f}%"],
            away_team: [f"#{int(a_rank)}", f"{a_elo:.0f}",
                        f"{a_wr*100:.0f}%", f"{a_gpg:.2f}",
                        f"{(1-h2h)*100:.0f}%"],
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


# ─── Page 2: WC Simulator ─────────────────────────────────────────────────────

def show_simulator(model, le, features_df, rankings):
    st.title("🏆 2026 FIFA World Cup Simulator")
    st.markdown("Simulate the entire 2026 World Cup bracket using ML predictions.")
    st.divider()

    # Show groups
    st.subheader("📋 2026 World Cup Groups")
    group_items = list(GROUPS.items())
    cols = st.columns(4)
    for i, (group, teams) in enumerate(group_items):
        with cols[i % 4]:
            st.markdown(f"**Group {group}**")
            for t in teams:
                st.markdown(f"- {t}")

    st.divider()

    # Settings
    st.subheader("⚙️ Simulation Settings")
    col1, col2 = st.columns(2)
    with col1:
        n_sims = st.slider("Number of Simulations",
                           min_value=100, max_value=5000,
                           value=1000, step=100)
    with col2:
        st.metric("CPU Cores Available", 8)
        st.caption("Simulations run in parallel automatically")

    st.divider()

    if st.button("🚀 Run World Cup Simulation", use_container_width=True):
        import time

        progress = st.progress(0)
        status   = st.empty()

        # ── Step 1: Team stats
        status.text("Step 1/3 — Computing team stats...")

        def get_stats(team):
            lookup = NAME_MAP.get(team, team)
            r      = rankings[rankings["team"] == lookup].sort_values("date")
            rank   = float(r.iloc[-1]["rank"])         if not r.empty else 80.0
            points = float(r.iloc[-1]["total_points"]) if not r.empty else 400.0
            h_elo  = features_df[features_df["home_team"] == lookup][["date","home_elo"]].rename(columns={"home_elo":"elo"})
            a_elo  = features_df[features_df["away_team"] == lookup][["date","away_elo"]].rename(columns={"away_elo":"elo"})
            elo_df = pd.concat([h_elo, a_elo]).sort_values("date")
            elo    = float(elo_df.iloc[-1]["elo"]) if not elo_df.empty else 1500.0
            h_form = features_df[features_df["home_team"] == lookup][["date","home_form_wr","home_form_gd"]].rename(columns={"home_form_wr":"wr","home_form_gd":"gd"})
            a_form = features_df[features_df["away_team"] == lookup][["date","away_form_wr","away_form_gd"]].rename(columns={"away_form_wr":"wr","away_form_gd":"gd"})
            form_df = pd.concat([h_form, a_form]).sort_values("date")
            wr  = float(form_df.tail(5)["wr"].mean()) if not form_df.empty else 0.4
            gd  = float(form_df.tail(5)["gd"].mean()) if not form_df.empty else 0.0
            h_gpg  = features_df[features_df["home_team"] == lookup][["date","home_goals_pg"]].rename(columns={"home_goals_pg":"gpg"})
            a_gpg  = features_df[features_df["away_team"] == lookup][["date","away_goals_pg"]].rename(columns={"away_goals_pg":"gpg"})
            gpg_df = pd.concat([h_gpg, a_gpg]).sort_values("date")
            gpg    = float(gpg_df.tail(5)["gpg"].mean()) if not gpg_df.empty else 1.5
            return {"rank": rank, "points": points, "elo": elo,
                    "wr": wr, "gd": gd, "gpg": gpg}

        all_teams = list(set(t for g in GROUPS.values() for t in g))
        stats     = {team: get_stats(team) for team in all_teams}
        progress.progress(20)

        # ── Step 2: Pre-compute probability table
        status.text("Step 2/3 — Pre-computing match probabilities...")

        rows, pairs = [], []
        for home in all_teams:
            for away in all_teams:
                if home == away:
                    continue
                hs  = stats[home]
                as_ = stats[away]
                lh  = NAME_MAP.get(home, home)
                la  = NAME_MAP.get(away, away)
                mask = (
                    ((features_df["home_team"] == lh) & (features_df["away_team"] == la)) |
                    ((features_df["home_team"] == la) & (features_df["away_team"] == lh))
                )
                past = features_df[mask]
                if past.empty:
                    h2h = 0.5
                else:
                    wins = (((past["home_team"] == lh) & (past["result"] == 1)).sum() +
                            ((past["away_team"] == lh) & (past["result"] == -1)).sum())
                    h2h = wins / len(past)

                rows.append({
                    "home_rank": hs["rank"], "away_rank": as_["rank"],
                    "rank_diff": as_["rank"] - hs["rank"],
                    "home_points": hs["points"], "away_points": as_["points"],
                    "points_diff": hs["points"] - as_["points"],
                    "home_elo": hs["elo"], "away_elo": as_["elo"],
                    "elo_diff": hs["elo"] - as_["elo"],
                    "home_form_wr": hs["wr"], "away_form_wr": as_["wr"],
                    "home_form_gd": hs["gd"], "away_form_gd": as_["gd"],
                    "h2h_win_rate": h2h, "tournament_weight": 5, "neutral": 1,
                    "home_goals_pg": hs["gpg"], "away_goals_pg": as_["gpg"],
                    "goals_pg_diff": hs["gpg"] - as_["gpg"],
                })
                pairs.append((home, away))

        X       = pd.DataFrame(rows)
        probas  = model.predict_proba(X)
        classes = le.inverse_transform(model.classes_)
        c_map   = {c: i for i, c in enumerate(classes)}

        prob_table = {}
        for i, (home, away) in enumerate(pairs):
            prob_table[(home, away)] = (
                float(probas[i][c_map.get(1,  0)]),
                float(probas[i][c_map.get(0,  1)]),
                float(probas[i][c_map.get(-1, 2)])
            )
        progress.progress(50)

        # ── Step 3: Run simulations
        status.text(f"Step 3/3 — Running {n_sims} simulations...")

        def sim_match(home, away, knockout=False):
            hp, dp, ap = prob_table[(home, away)]
            if knockout:
                hp += dp/2; ap += dp/2; dp = 0
            probs = np.array([hp, dp, ap], dtype=float)
            probs /= probs.sum()
            outcome = np.random.choice(["home","draw","away"], p=probs)
            return home if outcome=="home" else (away if outcome=="away" else None)

        def sim_group(teams):
            pts = defaultdict(int)
            gd  = defaultdict(float)
            for i in range(len(teams)):
                for j in range(i+1, len(teams)):
                    w = sim_match(teams[i], teams[j])
                    if w is None:
                        pts[teams[i]] += 1; pts[teams[j]] += 1
                    elif w == teams[i]:
                        pts[teams[i]] += 3
                        d = stats[teams[i]]["elo"] - stats[teams[j]]["elo"]
                        gd[teams[i]] += max(1, round(d/200))
                        gd[teams[j]] -= max(1, round(d/200))
                    else:
                        pts[teams[j]] += 3
                        d = stats[teams[j]]["elo"] - stats[teams[i]]["elo"]
                        gd[teams[j]] += max(1, round(d/200))
                        gd[teams[i]] -= max(1, round(d/200))
            return sorted(teams, key=lambda t: (pts[t], gd[t]), reverse=True), pts, gd

        def sim_tournament():
            qualified = []
            third     = []
            for g, teams in GROUPS.items():
                s, pts, gd = sim_group(teams)
                qualified.append(s[0]); qualified.append(s[1])
                third.append((s[2], pts[s[2]], gd[s[2]]))
            third.sort(key=lambda x: (x[1], x[2]), reverse=True)
            qualified += [t[0] for t in third[:8]]
            np.random.shuffle(qualified)
            while len(qualified) > 1:
                qualified = [sim_match(qualified[i], qualified[i+1], knockout=True)
                             for i in range(0, len(qualified), 2)]
            return qualified[0]

        start      = time.time()
        win_counts = defaultdict(int)
        for _ in range(n_sims):
            win_counts[sim_tournament()] += 1
        elapsed = time.time() - start

        progress.progress(100)
        status.text(f"✅ Done! {n_sims} simulations completed in {elapsed:.1f} seconds")

        # ── Results
        st.divider()
        st.subheader("🏆 World Cup Win Probabilities")

        results = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
        winner  = results[0][0]
        st.success(f"🏆 Most likely 2026 World Cup winner: **{winner}** "
                   f"({results[0][1]/n_sims*100:.1f}% of simulations)")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("#### 📊 Top 10 Contenders")
            top10 = pd.DataFrame(results[:10], columns=["Team", "Wins"])
            top10["Win %"] = (top10["Wins"] / n_sims * 100).round(1)
            st.bar_chart(top10.set_index("Team")["Win %"])

        with col_right:
            st.markdown("#### 📋 Full Rankings")
            df_results = pd.DataFrame([
                {
                    "Rank":  i+1,
                    "Team":  team,
                    "Win %": f"{wins/n_sims*100:.1f}%"
                }
                for i, (team, wins) in enumerate(results)
            ])
            st.dataframe(df_results, use_container_width=True, hide_index=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.sidebar.title("⚽ FIFA Predictor")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["🔮 Match Predictor", "🏆 2026 WC Simulator"]
    )

    bundle, features_df, rankings = load_artifacts()
    model = bundle["model"]
    le    = bundle["label_encoder"]

    if page == "🔮 Match Predictor":
        show_predictor(model, le, features_df, rankings)
    else:
        show_simulator(model, le, features_df, rankings)


if __name__ == "__main__":
    main()