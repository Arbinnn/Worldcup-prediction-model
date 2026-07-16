# Prompt for Claude Code

Copy everything below into Claude Code in your project folder (with the 4 CSVs placed alongside).

---

## Task

Build a football match outcome predictor with a Streamlit frontend, supporting **5 selectable algorithms**. Two deliverables:

1. **Model pipeline** (`train_model.py`): trains all 5 algorithms below on match win probability, using the datasets described below.
2. **Streamlit app** (`app.py`): a simple, football-themed UI where the user first **selects which algorithm to use** from a dropdown, then clicks "Start Prediction" to run that specific model and display predicted outcomes for:
   - **Final**: Argentina vs Spain
   - **Third-place match**: England vs France

## Input data (already provided, place in `./data/`)

1. **`top50_matches_2009_2026.csv`** — 2,634 international matches (2009–2026) between current FIFA top-50 teams. Columns include: date, home_team, away_team, tournament, neutral, is_friendly/is_major_tournament/is_world_cup/is_qualifier, home_elo_pre, away_elo_pre, elo_diff, fifa_rank_home_current, fifa_rank_away_current, rolling form (form_pts_5/10, gf_avg, ga_avg for both teams), matches_played_365d, h2h_home_winrate_last5, h2h_matches_count, home_score, away_score, goal_diff, result, home_win. **This is the primary training table** — one row per match, `home_win` (0/1) is usable as the binary label; `result` (home_win/draw/away_win) is available if you want a 3-class model instead.

2. **`team_match_stats_2018_2026.csv`** — 728 rows, one row per team per match, covering major tournaments 2018–2026 (World Cup 2018, Euro 2020, World Cup 2022, Euro 2024, Copa América 2024, **and the full 2026 World Cup through the semifinals**). For 2018–2024 rows: full event-derived metrics (xg, xg_conceded, shots, shots_on_target, big_chances, possession_pct, passes/passes_completed/pass_completion_pct, key_passes, assists, crosses, pressures, counterpressures, tackles, tackles_won, interceptions, blocks, clearances, dribbles, fouls_committed, clean_sheet, result). **For 2026 rows: only goals, goals_conceded, xg, xg_conceded, clean_sheet, result, stage, and an `advanced` flag are populated — shots/possession/passing/defensive-action columns are NaN for 2026**, since that granularity isn't available yet for the current tournament (see file 3 below, which is 2026-only and cleaner to merge on for the recency-weighting step).

3. **`team_match_stats_wc2026.csv`** — 204 rows (102 matches × 2 teams), the 2026 World Cup subset in isolation: goals, goals_conceded, xg, xg_conceded, result, clean_sheet, stage, and an `advanced` flag for knockout matches decided on penalties (the flag distinguishes the shootout winner since the 90-minute scoreline alone reads as a draw). The two semifinals (Spain 2–0 France, Argentina 2–1 England) are included with correct scores but **xg is NaN for those two matches specifically** — not published anywhere free yet. Use this file as the primary source for the recency-weighting step described below, since it isolates 2026 cleanly.

4. **`wc2026_player_stats_espn.csv`** — 104 rows, full 26-man squads for all four semifinalists (Spain, France, Argentina, England) with per-player 2026 World Cup stats: appearances, sub_appearances, goals, assists, shots, shots_on_target, fouls_committed, fouls_suffered, yellow_cards, red_cards, and goalkeeper saves/goals_against. **Critical caveat, must be handled explicitly**: each team's data is a snapshot taken at a different point in the tournament — check the `matches_covered` and `as_of` columns per team before using this file. England is covered through 6 matches (July 12, i.e. through the quarterfinal), Spain and Argentina through 3 matches each (June 28, group stage only), France through 1 match (June 22). **Do not compare raw per-player totals across teams without normalizing by `matches_covered` or `appearances` first** — Kane's 22 shots over 6 games and Messi's 15 over 3 games are not on the same footing, and comparing them raw would systematically make England's players look more productive purely because more of their tournament is captured. Use this file for **within-team relative standing** (who's Spain's most productive attacker relative to other Spain players) and for the star-power feature (max per-90 output on the team), not for cross-team absolute comparisons.

## Feature engineering requirements

- Build a **team quality profile** for each team from `team_match_stats_2018_2026.csv`: per-team averages across all their tournament matches in that file for xg, xg_conceded, possession_pct, pressures, tackles_won, pass_completion_pct, clean_sheet rate, big_chances. Since 2026 rows only have xg/xg_conceded/clean_sheet populated, this will naturally lean on 2018–2024 data for the possession/pressing columns and blend in 2026 for xg/xg_conceded — that's fine, don't try to force-fill the NaNs. Merge as static per-team features onto the main training table (`top50_matches_2009_2026.csv`) by team name, for both home_team and away_team. Teams absent from the tournament file get NaN.

- Build a **squad-quality profile from `wc2026_player_stats_espn.csv`**, respecting the per-team coverage caveat above:
  - First, normalize each player's counting stats (goals, assists, shots, shots_on_target, fouls) to **per-appearance** or **per-90-equivalent** rates using `appearances` (or `matches_covered` as a fallback denominator) — do this before any aggregation, not after.
  - Aggregate to a **squad quality profile per team**: average per-appearance output across the team's players with `appearances > 0`, plus a **star-power feature** (max per-appearance goal contribution — goals + assists — on the team). This captures "does this team have a standout" the same way it was meant to originally, just using rate stats instead of raw totals so the uneven coverage doesn't bias it.
  - Because this file only covers the 4 finalists, this squad-quality profile can only be attached to rows in the training table where Spain, France, Argentina, or England appear (either side). For all other teams in the training data, this feature will be NaN — that's expected and fine, XGBoost and the imputer both handle it.
  - This is a coarser signal than originally planned (rate-based per-appearance rather than full per-90 event data), so treat it as directional/tie-breaking context rather than a heavily-weighted feature. Don't let it dominate the model relative to Elo and form.

- From the main table, engineer the final feature set as **symmetric differences** where sensible (e.g. `elo_diff` already exists; add `form_pts_diff_5`, `form_pts_diff_10`, `xg_profile_diff`, `possession_profile_diff`, `squad_star_power_diff`, etc.) alongside raw home/away values.
- Include `neutral` as a feature (the final and third-place match are both at neutral venues — verify this assumption holds for the training rows so the model isn't systematically biased).
- Do **not** use `home_score`, `away_score`, or `goal_diff` as features (target leakage) — only use pre-match information.
- Some algorithms below need NaN-free, standardized numeric input (Logistic Regression, Random Forest, Bradley-Terry/Poisson) while XGBoost can take raw features with NaNs natively — build **one shared feature table**, then a small preprocessing step (impute + scale) applied only for the algorithms that need it, so all 5 models train from the same underlying data without duplicating feature engineering code.

## Recency weighting — heavy emphasis on the 2026 World Cup

The 2026 World Cup matches are the single most relevant signal for predicting the final and third-place match. Make sure this actually shows up in training, not just in the feature values, and use **`team_match_stats_wc2026.csv` as the source of truth for identifying and tiering these matches** — it's the clean, isolated 2026 file with a `stage` column, so use it rather than re-deriving 2026 status from date filters on the main table:

- **Build the weighting lookup from `team_match_stats_wc2026.csv` first**: extract the set of (date, team, opponent) tuples in that file — these are your 2026 World Cup matches. Join this back onto `top50_matches_2009_2026.csv` (matching on date + home_team/away_team) to flag which rows in the main training table are 2026 World Cup matches, rather than relying on `is_world_cup` + a date cutoff, which could also catch other things incidentally dated in 2026.
- **Add a `sample_weight` column** used during training (all models that support sample weights — Logistic Regression, Random Forest, XGBoost all accept `sample_weight`; weight the likelihood/loss manually for Elo-K-factor and Bradley-Terry instead, see below):
  - Matches flagged as 2026 World Cup get the highest weight tier, and **use the `stage` column from `team_match_stats_wc2026.csv` to grade within that tier** rather than a single flat multiplier: Group stage matches get ~6x baseline, Round of 32/16 ~8x, Quarter-finals ~9x, Semi-finals ~10x — later-stage 2026 matches are more predictive of final-level performance than early group games, and `stage` gives you this for free.
  - Other **major tournament** matches (World Cup 2018/2022, Euros, Copa América — identifiable via `is_major_tournament`/`is_world_cup` in the main table, excluding whatever was already flagged as 2026 above) get a moderate boost (e.g. 2–3x).
  - Everything else (friendlies, qualifiers, Nations League, etc.) gets baseline weight 1x.
  - Additionally apply a smooth **recency decay** on top of the tournament-tier weight (e.g. exponential decay by days-before-cutoff) so within "everything else," 2025–2026 matches still outweigh 2015 matches.
- For the **Elo baseline**, this already exists structurally via the tournament-tier K-factor baked into `home_elo_pre`/`away_elo_pre` in the main table — no extra work needed, just use those columns as-is.
- For **Bradley-Terry**, weight each match's contribution to the likelihood function by the same `sample_weight`, not just fit unweighted.
- Report in the validation output what fraction of total training weight comes from 2026 World Cup matches specifically (and ideally the breakdown by stage), so it's visible that this isn't accidentally negligible given there are relatively few 2026 rows by count.

## Model requirements — all 5 algorithms

Train **all five** on the same time-based split so they're directly comparable, and so the user can pick any one of them at prediction time in the app.

1. **Elo-only baseline** — no training, just the direct formula `P(home_win) = 1 / (1 + 10^((away_elo - (home_elo + home_adv)) / 400))` using `elo_diff` already in the dataset (with a neutral-venue adjustment: no home_adv bonus when `neutral==1`). Pure reference point, not fit to data.

2. **Logistic Regression (`sklearn.linear_model.LogisticRegression`)** — standardized numeric features (`StandardScaler`, NaNs imputed e.g. with median). Stable, interpretable baseline.

3. **Random Forest (`sklearn.ensemble.RandomForestClassifier`)** — same feature set as logistic regression. Moderate depth (e.g. `max_depth=6-10`), `n_estimators=300-500`, to control variance on ~2,634 rows.

4. **Regularized XGBoost (`xgboost.XGBClassifier`)** — tuned specifically to avoid overfitting on ~2,634 rows:
   - Shallow trees: `max_depth=2` or `3`
   - `min_child_weight` on the higher side (try 5–10)
   - Low `learning_rate` (e.g. 0.03–0.05) with a higher `n_estimators` and **early stopping** on the validation set
   - `subsample` and `colsample_bytree` around 0.7–0.8
   - L1/L2 regularization (`reg_alpha`, `reg_lambda`) — nonzero values, don't leave at library defaults

5. **Bradley-Terry model** — a probabilistic pairwise-comparison model fit via logistic regression on `elo_diff`-style strength differences (or implement directly: `P(A beats B) = strength_A / (strength_A + strength_B)` with `strength` fit per team via maximum likelihood, e.g. using the `choix` package if available, or a simple custom MLE loop if not). If `choix` isn't easily installable, implement a straightforward gradient-based MLE for per-team strength ratings instead — keep it simple.

For all five, use the same binary `home_win` target for consistency, and **use the `sample_weight` column described above during training** for every model that supports it (pass `sample_weight=` to `.fit()` for Logistic Regression, Random Forest, and XGBoost; use weighted MLE for Bradley-Terry; keep Elo's existing tournament-based K-factor weighting).

- **Time-based split**: do not shuffle randomly. Train on matches before some cutoff date (e.g. before 2025-01-01) and validate on the most recent matches, since this is a forecasting task and random splits leak future information into training.
- Report **validation accuracy, log loss, and a calibration check (predicted probability vs actual win rate, binned)** for all five models in a single comparison table, printed at the end of `train_model.py`.
- Flag it clearly if the models disagree substantially with each other on the validation set or on the two live predictions — that's a signal worth surfacing, not hiding.
- For **knockout matches with no draw allowed** (both the final and third-place match): after getting P(home_win)/P(away_win)/P(draw) from a model, redistribute the draw probability between the two teams proportionally (or via a simple 50/50 split of the draw mass) to get a final normalized win probability for each side. Apply this same redistribution logic to all five models' outputs so they're comparable.
- Save all trained models (Elo and Bradley-Terry need minimal/no heavy artifacts — just formulas or a small strength table; save the rest as proper model files) plus the shared feature list and preprocessing objects (imputer/scaler), so `app.py` can load everything without retraining.

## Predicting the two specific matches

Since Argentina, Spain, England, and France may not have an existing literal row in the dataset for these exact fixtures at this date, build a small helper function `predict_matchup(team_a, team_b, algorithm, neutral=True)` that:
1. Looks up each team's **most recent pre-match Elo** from `top50_matches_2009_2026.csv` (take the latest known Elo for each team from the last match they appear in — this already reflects their 2026 World Cup run through the semifinals, since the file includes 2026 matches and Elo updates match-by-match).
2. Looks up each team's **latest rolling form** (form_pts_5/10, gf_avg, ga_avg) from their most recent appearance in the dataset.
3. Looks up **head-to-head** history between the two teams from the full dataset (not just post-cutoff).
4. Pulls each team's **tournament quality profile** (xg/possession/pressures) from `team_match_stats_2018_2026.csv`.
5. Pulls each team's **squad quality profile** (star power / per-appearance rates) from `wc2026_player_stats_espn.csv`, using the normalized (per-appearance) version, not raw totals.
6. Assembles a single feature row matching the training schema, applies the correct preprocessing for the requested `algorithm`, and returns a win probability for each side from whichever of the 5 models is selected.

Run this for both matchups across all 5 algorithms (so the comparison table and the app can show any one of them on demand):
- Argentina vs Spain (final, neutral=True)
- England vs France (third-place playoff, neutral=True)

## Streamlit app requirements (`app.py`)

- **Theme**: football/soccer visual identity — pitch-green background accents, ball/trophy emoji or icons, team flags if easy (use emoji flags 🇦🇷🇪🇸🇬🇧🇫🇷 — no need to source actual flag images), clean card-style layout for each match.
- **Algorithm selector**: a dropdown/selectbox at the top of the page listing all 5 options by clear display names, e.g.:
  - "Elo Rating (baseline)"
  - "Logistic Regression"
  - "Random Forest"
  - "XGBoost (Gradient Boosted Trees)"
  - "Bradley-Terry Model"
- **Single "⚽ Start Prediction" button** below the selector — nothing runs until it's clicked.
- On click, using whichever algorithm is currently selected:
  - Show a brief spinner/loading state.
  - Display **two match cards**:
    1. 🏆 **Final: Argentina vs Spain**
    2. 🥉 **Third Place: England vs France**
  - Each card shows the **selected model's** predicted win probability for each team (percentage + progress-bar-style visual) and a one-line takeaway (e.g. "Argentina favored, 54% (XGBoost)").
  - Also show a small caption naming which algorithm produced the result.
  - Add a small "Squad & Form Snapshot" line per team underneath the probability bar (e.g. current Elo, squad star-power score) so it's visible that individual player quality is factoring into the number.
  - Optional nice-to-have (only if trivial): a small expandable "Compare all algorithms" section showing how the other 4 models scored the same matchup.
  - Keep it simple — this is a small side project, not a production dashboard.
- **Cache the loaded models and precomputed matchup features** with `st.cache_resource`/`st.cache_data` at startup so switching the dropdown and clicking "Start" is fast — don't retrain inside the button callback.

## Deliverable structure

```
project/
├── data/
│   ├── top50_matches_2009_2026.csv
│   ├── team_match_stats_2018_2026.csv
│   ├── team_match_stats_wc2026.csv
│   └── wc2026_player_stats_espn.csv
├── train_model.py        # run once to produce all 5 model artifacts + feature list
├── models/
│   ├── elo_config.json          # home advantage constant, etc.
│   ├── logreg.pkl                # includes fitted imputer + scaler
│   ├── random_forest.pkl
│   ├── xgboost_model.json
│   ├── bradley_terry.json        # per-team fitted strengths
│   └── feature_list.json
├── app.py                # streamlit run app.py
└── requirements.txt       # xgboost, pandas, streamlit, scikit-learn, choix (optional)
```

Walk through building `train_model.py` first, show me the **five-way validation comparison table**, and only then build `app.py` once we're confident all five models' predictions are sane (e.g. sanity-check a few known historical results across all five approaches before predicting the two live matches).