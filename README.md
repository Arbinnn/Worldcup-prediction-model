# ⚽ World Cup 2026 — Match Outcome Predictor

Predicts the two remaining fixtures of the 2026 World Cup from 2,634 international matches
(2009–2026) between current FIFA top-50 teams:

- 🏆 **Final** — Argentina vs Spain
- 🥉 **Third place** — England vs France

Six selectable algorithms, a time-based validation split, and a Streamlit front-end.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux

python train_model.py            # tune + train + save artifacts + print the comparison tables
python train_model.py --no-tune  # skip the grid search, reuse models/tuning.json (fast retrain)

streamlit run app.py             # the UI (needs models/ to exist first)
```

`train_model.py` takes a few minutes on the first run — the hyperparameter search fits every
candidate across 4 CV folds. `--no-tune` reuses the saved best parameters and takes seconds.

## Layout

```
├── data/                            # the four input CSVs (provided)
│   ├── top50_matches_2009_2026.csv      # 2,634 matches — the primary training table
│   ├── team_match_stats_2018_2026.csv   # 728 team-match rows, major tournaments 2018–2026
│   ├── team_match_stats_wc2026.csv      # 204 rows — the 2026 WC in isolation, with `stage`
│   └── wc2026_player_stats_espn.csv     # 104 players — the four semifinalists' squads
├── train_model.py                   # everything: features, weighting, tuning, training, prediction
├── app.py                           # Streamlit UI (imports from train_model)
├── models/                          # written by train_model.py
│   ├── elo_config.json                  # home-advantage constant + the formula
│   ├── logreg.pkl                       # pipeline: imputer + scaler + clip + classifier
│   ├── random_forest.pkl                # same pipeline shape
│   ├── xgboost_model.json
│   ├── bradley_terry.json               # per-team fitted strengths + home advantage
│   ├── feature_list.json                # shared feature table + per-model feature subsets
│   └── tuning.json                      # chosen feature set, params, CV log loss per model
└── requirements.txt
```

`train_model.py` is import-safe — training is behind `if __name__ == "__main__"`, so `app.py`
imports `predict_matchup` without retraining.

---

## The models

All six produce `P(home win)` and are scored on the same time-based split, so they are directly
comparable.

| # | Algorithm | Target | Notes |
|---|---|---|---|
| 1 | **Elo** | — | Not fitted. `P = 1/(1+10^((away_elo-(home_elo+adv))/400))`, `adv=0` when neutral. Pure reference point. |
| 2 | **Logistic Regression** | 3-class | Impute → scale → clip pipeline. Best single model. |
| 3 | **Random Forest** | 3-class | 400 trees, tuned depth/leaf size. |
| 4 | **XGBoost** | 3-class | Shallow + heavily regularized. Reads NaN natively, so it keeps the squad features. |
| 5 | **Bradley-Terry** | binary | Weighted MLE for per-team strength + home advantage, via `scipy`. Pairwise by construction. |
| 6 | **Ensemble** | mixed | Unweighted mean of the five. Nothing is fitted, so it can't leak. The app's default. |

Elo and Bradley-Terry stay binary — both are pairwise-strength models with no natural draw term.
The other three predict all three outcomes (`away_win` / `draw` / `home_win`); their `P(home win)`
column is what enters the shared comparison table.

## Features

One shared feature table (73 columns), then per-model subsets — no duplicated feature code.

- **From the main table**: Elo (levels + `elo_diff`), FIFA ranks, rolling form (`form_pts_5/10`,
  `gf_avg`, `ga_avg`), matches played in 365d, head-to-head win rate and meeting count, and the
  match-context flags (`neutral`, `is_friendly`, `is_major_tournament`, `is_world_cup`, `is_qualifier`).
- **Team quality profile** (`team_match_stats_2018_2026.csv`): per-team means of xg, xg_conceded,
  possession, pressures, tackles won, pass completion, clean-sheet rate, big chances. The 2026 rows
  only carry xg/xg_conceded/clean_sheet, so the possession/pressing columns lean on 2018–2024
  automatically — `mean()` skips the NaNs, nothing is force-filled.
- **Squad quality profile** (`wc2026_player_stats_espn.csv`): **rates first, aggregation second.**
  Each team's snapshot covers a different slice of the tournament (England 6 matches, Spain and
  Argentina 3, France 1), so raw totals are not comparable across teams — Kane's 22 shots over 6
  games and Messi's 15 over 3 are not on the same footing. Every counting stat is divided by
  `appearances` (falling back to `matches_covered`) *before* the team average, plus a star-power
  feature: the best goals+assists-per-appearance rate on the squad.
- **Symmetric differences** for everything where a difference is meaningful (`form_pts_diff_5`,
  `prof_xg_diff`, `squad_star_power_diff`, …) alongside the raw home/away levels.
- **No leakage**: `home_score`, `away_score` and `goal_diff` are never features.

## Recency weighting

The 2026 World Cup is the most relevant signal, and this has to show up in *training*, not just in
the feature values. 2026 matches are flagged by joining `(date, {teams})` from the isolated
`team_match_stats_wc2026.csv` — not by `is_world_cup` plus a date cutoff, which would also catch
anything else incidentally dated in 2026.

`sample_weight = tournament tier × exponential recency decay` (4-year half-life), passed to every
model that accepts it and folded into the Bradley-Terry likelihood by hand.

| Tier | Weight |
|---|---|
| 2026 WC — Semi-finals | 10× |
| 2026 WC — Quarter-finals | 9× |
| 2026 WC — Round of 32/16 | 8× |
| 2026 WC — Group | 6× |
| Other major tournaments (WC 2018/2022, Euros, Copa América) | 2.5× |
| Everything else (friendlies, qualifiers, Nations League) | 1× |

Result — **2026 matches are 2.5% of rows by count but 30.2% of total training weight**:

```
Group            39 matches   15.6% of weight
Round of 32      12 matches    6.4% of weight
Round of 16       8 matches    4.3% of weight
Quarter-finals    4 matches    2.4% of weight
Semi-finals       2 matches    1.4% of weight
other majors    485 matches   26.4% of weight
everything else 2084 matches   43.4% of weight
```

Elo needs no extra work here — the tournament-tier K-factor is already baked into `home_elo_pre`.

## Validation

**Time-based split, never shuffled**: train on everything before 2025-01-01 (2,384 matches),
validate on the 250 most recent. A random split would leak the future into the past.

Hyperparameters and feature sets are chosen by **expanding-window CV on the training split only**,
so validation stays clean.

### Results — `P(home win)` on the validation set

| Model | Accuracy | LogLoss | Brier | Features |
|---|---|---|---|---|
| Elo (baseline) | 0.628 | 0.668 | 0.234 | — |
| **Logistic Regression** | **0.696** | **0.604** | **0.208** | 24 |
| Random Forest | 0.664 | 0.619 | 0.215 | 24 |
| XGBoost | 0.660 | 0.606 | 0.209 | 73 |
| Bradley-Terry | 0.632 | 0.641 | 0.226 | — |
| Ensemble | 0.676 | 0.611 | 0.211 | — |
| *(always predict home win)* | *0.432* | | | |

### 3-class metrics (away / draw / home)

| Model | Accuracy | LogLoss |
|---|---|---|
| Logistic Regression | 0.552 | 0.968 |
| Random Forest | 0.536 | 0.983 |
| XGBoost | 0.552 | 0.967 |

All four fitted models beat the Elo baseline. Calibration bins track actual win rates closely for
logreg, XGBoost and the ensemble; **Elo is visibly overconfident at the top end** — it predicts
0.859 where the actual win rate is 0.655.

The tuner pushed everything toward *more* regularization, not less (logreg `C` 0.5 → 0.05, XGBoost
`max_depth` 3 → 2 with 85 trees), and picked the 24-column diff-only feature set over all 73 for
logreg and RF. That's the honest signal that ~2,384 rows cannot support more capacity.

### Sanity check — both semifinals, retrodicted

All six models get both right:

| Matchup | Elo | LogReg | RF | XGB | BT | Ensemble | Actual |
|---|---|---|---|---|---|---|---|
| Spain vs France | 53.2% | 59.4% | 60.8% | 58.1% | 52.1% | 56.7% | Spain won 2–0 ✅ |
| Argentina vs England | 62.6% | 60.6% | 61.5% | 62.7% | 61.0% | 61.7% | Argentina won 2–1 ✅ |

### Predictions

Both are neutral-venue knockout ties, so the draw mass is redistributed 50/50 (see below).

**Final — Argentina vs Spain: a genuine coin flip.** The models split on the favourite and the
script flags this rather than hiding it.

| Model | Argentina | Spain |
|---|---|---|
| Elo | 50.6% | 49.4% |
| Logistic Regression | 49.5% | 50.5% |
| Random Forest | 47.7% | 52.3% |
| XGBoost | 50.5% | 49.5% |
| Bradley-Terry | 45.6% | 54.4% |
| **Ensemble** | **48.8%** | **51.2%** |

**Third place — England vs France: France favoured by all six.**

| Model | England | France |
|---|---|---|
| Elo | 40.9% | 59.1% |
| Logistic Regression | 47.4% | 52.6% |
| Random Forest | 45.0% | 55.0% |
| XGBoost | 45.3% | 54.7% |
| Bradley-Terry | 36.9% | 63.1% |
| **Ensemble** | **43.1%** | **56.9%** |

The 3-class models put the draw at ~30% in regulation for both fixtures.

---

## Two decisions worth knowing about

Both are deviations from the original spec, made deliberately.

### 1. Squad features are XGBoost-only

The spec assumed the squad NaNs were fine because "XGBoost and the imputer both handle it".
XGBoost does. **The imputer does not.**

The player file only covers the four semifinalists, so a squad `*_diff` is non-null in **26 of 2,384
training rows** — but non-null in *every row we actually predict*, since both sides of the final and
the third-place tie are finalists. Median-imputing the other 2,358 rows and standardizing collapses
the column's standard deviation, which makes a near-constant column look informative: the fitted
coefficient comes off 26 rows, lands with an unstable sign, and then dominates the live prediction.
Concretely, it put `squad_shots_p90_diff` **18 standard deviations** out and predicted Argentina at
**16%** against an England side they had just beaten.

Validation never caught this — it holds almost no both-finalist rows. So the squad columns are kept
out of the impute+scale path and given only to XGBoost, which splits on them where present. That is
the directional, tie-breaking role the spec wanted for them anyway. The app still displays star
power per team. Reasoning lives in the `scaled_feature_list` docstring.

The scaled pipeline also clips to ±3σ as a general guard against out-of-distribution rows.

### 2. XGBoost early stopping does not use the validation set

The spec asked for early stopping on the validation set. That tunes a hyperparameter on the data
being reported. `n_estimators` now comes from the CV folds and the final fit never touches
validation — which is why XGBoost's accuracy reads *lower* than an early-stopped run would. The old
number was mildly optimistic; this one is real.

## How the knockout normalization works

Neither the final nor the third-place match can end in a draw, and every prediction is made at a
neutral venue. Both are handled by predicting each matchup **in both orientations** (A home, then B
home) and averaging:

- **3-class models**: average the two orientations, then hand each side half the draw mass.
- **Binary models** (Elo, Bradley-Terry): averaging `P(A wins at home)` with `P(B does not win at
  home)` cancels the home bias *and* splits the draw mass 50/50 in one step — "not home win"
  already lumps draws in with away wins.

## Where the remaining headroom is

Accuracy in the 0.63–0.70 range is near the ceiling for football with this much data — upsets are
common and draws are genuinely hard. The two levers with real room left:

- **Walk-forward validation across multiple cutoffs.** 250 validation rows make these differences
  mostly noise; the 0.696 vs 0.676 gap is well within it.
- **More data.** 2,634 matches is thin for 73 features. This is the binding constraint, and it's
  why every tuning decision came back "regularize harder".

Diminishing returns: tuning the weight tiers and decay half-life against CV, and a Davidson-style
draw term for Bradley-Terry.

## Caveats

- Squad stats are **snapshots at different tournament points** — never compare raw per-player
  totals across teams. The code normalizes by `appearances` before aggregating; anything built on
  top of this file should too.
- xg is NaN for both semifinals specifically (not published yet), so those two matches contribute
  goals/result but not xg to the profiles.
- Teams absent from the tournament file get NaN profiles by design. Nothing is force-filled.
