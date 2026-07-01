# ⚽ FIFA World Cup Match Outcome Predictor

I built this project because I wanted to answer a simple question — *can machine learning actually predict football matches better than a coin flip?*

Turns out, yes. But football humbles you fast.

This is a full end-to-end ML pipeline that predicts international football match outcomes and simulates the 2026 FIFA World Cup bracket. It started as a learning project and grew into something I'm genuinely proud of.

## 🎯 What it does

- Predicts **Win / Draw / Loss** for any international football match
- Simulates the **2026 World Cup bracket** 1000+ times using Monte Carlo simulation
- Trains and compares **4 ML models** head to head
- Runs as a **live web app** — no code needed to use it

## 🧠 The thinking behind it

Football prediction is hard. Really hard. Even the best bookmakers in the world barely crack 60% accuracy. So I focused on building features that capture what actually matters before a match:

- **Elo rating** — borrowed from chess, updated after every match based on who you beat and how strong they were. Turned out to be the single most predictive feature.
- **FIFA rank** — official ranking at the exact date of the match, not today's ranking
- **Recent form** — win rate and goal difference over the last 5 matches
- **Head to head history** — does one team historically dominate this specific opponent?
- **Goals per game** — recent scoring rate, not career averages
- **Tournament weight** — a World Cup match means more than a friendly

One rule I was strict about throughout: **no data leakage**. Every feature is built using only information available before the match. No peeking at the future.

## 📊 Results

Trained on **14,506 matches (1993–2014)**, tested on **3,627 matches (2014–2019)**:

| Model | Accuracy | F1 Macro |
|---|---|---|
| **XGBoost (Tuned)** | **58.0%** | **0.446** |
| Logistic Regression | 57.9% | 0.427 |
| Random Forest | 57.8% | 0.422 |
| MLP Neural Net | 57.5% | 0.419 |

58% might not sound exciting until you compare it to the baselines:
- Random guessing → **33%**
- Always predicting home win → **48%**
- Our model → **58%**

The most surprising finding: **Logistic Regression nearly matched XGBoost**. The simplest model almost won. Football has too much randomness for complexity to shine.

## 🏆 2026 World Cup Simulation

Based on 1000 Monte Carlo simulations of the full bracket:

| Rank | Team | Win Probability |
|---|---|---|
| 1 | Brazil | 18.3% |
| 2 | France | 12.7% |
| 3 | Belgium | 8.6% |
| 4 | Spain | 7.8% |
| 5 | England | 6.7% |
| 6 | Argentina | 6.2% |
| 7 | Portugal | 5.6% |
| 8 | Germany | 5.3% |

## 🗂 Project Structure
fifa_predictor/

├── app.py                         ← Streamlit web app

├── requirements.txt

├── data/

│   └── raw/                       ← Download from Kaggle (see below)

├── models/

│   ├── best_model.pkl

│   └── best_tuned_model.pkl

└── src/

├── 01_data_loading.py         ← Clean and label 30,769 matches

├── 02_feature_engineering.py  ← Build 19 features

├── 03_model_training.py       ← Train and evaluate 4 models

├── 04_predict.py              ← Predict any matchup

├── 05_tuning.py               ← GridSearchCV hyperparameter tuning

└── 06_simulator.py            ← 2026 WC Monte Carlo simulator

## 🚀 Run it yourself

**1. Clone the repo**
```bash
git clone https://github.com/Mehfooz-A/fifa-predictor.git
cd fifa-predictor
```

**2. Set up environment**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**3. Get the data**

Download and place in `data/raw/`:
- [International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) — `results.csv`, `goalscorers.csv`, `former_names.csv`
- [FIFA World Rankings](https://www.kaggle.com/datasets/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now) — `fifa_ranking.csv`

**4. Run the pipeline**
```bash
python src/01_data_loading.py
python src/02_feature_engineering.py
python src/03_model_training.py
```

**5. Launch the app**
```bash
streamlit run app.py
```

## 🛠 Stack

Python 3.11 · pandas · numpy · scikit-learn · XGBoost · Streamlit · matplotlib · seaborn · joblib

## 💡 What I learned

**Elo beats FIFA rankings.** The official ranking is slow to update and uses a fixed formula. Elo reacts to every match and rewards beating strong opponents. It was the #1 feature by a wide margin.

**Home advantage is real but smaller than expected.** Once you control for team strength, the home field effect shrinks significantly.

**Parallel processing matters at scale.** The simulator originally took 15 minutes to run 1000 simulations. After pre-computing a probability lookup table and parallelizing across 8 CPU cores, it runs in 8 seconds.

**Football is random.** No model will consistently predict draws. They're rare, contextual, and often decided by moments that no dataset captures. That's part of what makes the sport beautiful.

## 🔮 What's next

- Update FIFA rankings data beyond 2018
- Add player availability and injury signals
- Deploy on Streamlit Cloud for public access
- Add LSTM using sequential match history per team

## 👤 About me

**Mehfooz Alam** — MS Data Science, Kent State University

I'm interested in applying ML to real-world prediction problems, particularly in sports analytics, finance, and healthcare.

[LinkedIn](https://www.linkedin.com/in/mehfooz-mehboob-alam-4699ab190) · [GitHub](https://github.com/Mehfooz-A)