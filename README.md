---

## 💡 What I learned

**Elo beats FIFA rankings.** The official ranking is slow to update and uses a fixed formula. Elo reacts to every match and rewards beating strong opponents. It was the #1 feature by a wide margin.

**Home advantage is real but smaller than expected.** Once you control for team strength, the home field effect shrinks significantly.

**Parallel processing matters at scale.** The simulator originally took 15 minutes to run 1000 simulations. After pre-computing a probability lookup table and parallelizing across 8 CPU cores, it runs in 8 seconds.

**Football is random.** No model will consistently predict draws. They're rare, contextual, and often decided by moments that no dataset captures. That's part of what makes the sport beautiful.

---

## 🔮 What's next

- Update FIFA rankings data beyond 2018
- Add player availability and injury signals
- Deploy on Streamlit Cloud for public access
- Add LSTM using sequential match history per team

---

## 👤 About me

**Mehfooz Alam** — MS Data Science, Kent State University

I'm interested in applying ML to real-world prediction problems, particularly in sports analytics, finance, and healthcare.

[LinkedIn](https://www.linkedin.com/in/mehfooz-mehboob-alam-4699ab190) · 
[GitHub](https://github.com/Mehfooz-A)