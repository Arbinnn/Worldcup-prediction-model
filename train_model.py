"""Train 6 win-probability models on international matches through WC2026 semifinals.

Run: python train_model.py            -> tune (time-series CV) + train + save + report
     python train_model.py --no-tune  -> reuse models/tuning.json, skip the search
app.py imports predict_matchup / load_artifacts from here (training is __main__-guarded).
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")

CUTOFF = pd.Timestamp("2025-01-01")  # time-based split, no shuffling
HOME_ADV = 65.0  # Elo points, applied only when neutral == 0
DECAY_HALFLIFE_DAYS = 365 * 4
STAGE_WEIGHT = {
    "Group": 6.0,
    "Round of 32": 8.0,
    "Round of 16": 8.0,
    "Quarter-finals": 9.0,
    "Semi-finals": 10.0,
}
MAJOR_WEIGHT = 2.5
CV_SPLITS = 4

# 3-class target. Alphabetical order matches sklearn's classes_ ordering.
RESULT_CLASSES = ["away_win", "draw", "home_win"]
RESULT_TO_INT = {c: i for i, c in enumerate(RESULT_CLASSES)}
HOME_IDX = RESULT_TO_INT["home_win"]
DRAW_IDX = RESULT_TO_INT["draw"]

BASE_ALGORITHMS = ["elo", "logreg", "random_forest", "xgboost", "bradley_terry"]
ALGORITHMS = BASE_ALGORITHMS + ["ensemble"]
THREE_CLASS = {"logreg", "random_forest", "xgboost"}  # the rest are pairwise/binary by construction
DISPLAY_NAMES = {
    "elo": "Elo Rating (baseline)",
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost (Gradient Boosted Trees)",
    "bradley_terry": "Bradley-Terry Model",
    "ensemble": "Ensemble (blend of all 5)",
}

PROFILE_COLS = [
    "xg", "xg_conceded", "possession_pct", "pressures", "tackles_won",
    "pass_completion_pct", "clean_sheet", "big_chances",
]
SQUAD_COLS = ["squad_goals_p90", "squad_assists_p90", "squad_shots_p90", "squad_sot_p90", "squad_star_power"]
CONTEXT_COLS = ["neutral", "is_friendly", "is_major_tournament", "is_world_cup", "is_qualifier"]
FORM_COLS = ["form_pts_5", "form_pts_10", "gf_avg_5", "gf_avg_10", "ga_avg_5", "ga_avg_10"]

# --- search spaces ---------------------------------------------------------
LOGREG_DEFAULT = {"C": 0.5}
RF_DEFAULT = {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 10}
XGB_DEFAULT = {"max_depth": 3, "min_child_weight": 8, "learning_rate": 0.04,
               "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0}

LOGREG_GRID = [{"C": c} for c in [0.05, 0.1, 0.3, 1.0, 3.0]]
RF_GRID = [{"n_estimators": 400, "max_depth": d, "min_samples_leaf": leaf}
           for d in [6, 8, 12] for leaf in [5, 10, 20]]
XGB_GRID = [{"max_depth": d, "min_child_weight": mcw, "learning_rate": lr,
             "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": lam}
            for d in [2, 3] for mcw in [5, 8] for lr in [0.03, 0.05] for lam in [1.0, 3.0]]


# ------------------------------------------------- shared preprocessing (impute+scale)

def make_preprocessor():
    """Impute + scale for the algorithms that need NaN-free standardized input.

    The clip guards against out-of-distribution rows: median imputation plus scaling can
    push a live feature row many std out when a column is mostly imputed in training.
    Clipping to +/-3 keeps any single feature from swamping Elo and form.

    np.clip by reference, not a local lambda/def -- a module-local function pickles as
    __main__.<name> when train_model.py runs as a script, which app.py cannot unpickle.
    """
    return [("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -3.0, "a_max": 3.0}))]


# ---------------------------------------------------------------- data loading

def load_raw():
    matches = pd.read_csv(os.path.join(DATA, "top50_matches_2009_2026.csv"), parse_dates=["date"])
    tourn = pd.read_csv(os.path.join(DATA, "team_match_stats_2018_2026.csv"), parse_dates=["date"])
    wc26 = pd.read_csv(os.path.join(DATA, "team_match_stats_wc2026.csv"), parse_dates=["date"])
    players = pd.read_csv(os.path.join(DATA, "wc2026_player_stats_espn.csv"))
    return matches, tourn, wc26, players


def team_quality_profile(tourn):
    """Per-team means over tournament matches. 2026 rows only carry xg/xg_conceded/clean_sheet;
    the rest lean on 2018-2024 automatically because NaNs are skipped by mean()."""
    prof = tourn.groupby("team")[PROFILE_COLS].mean()
    return prof.rename(columns={c: f"prof_{c}" for c in PROFILE_COLS})


def squad_quality_profile(players):
    """Rate stats first, aggregate second -- raw totals are NOT comparable across teams
    because each squad snapshot covers a different number of matches (England 6, Spain/Argentina 3, France 1)."""
    p = players.copy()
    denom = p["appearances"].where(p["appearances"] > 0)
    denom = denom.fillna(p["matches_covered"])  # fallback denominator
    for c in ["goals", "assists", "shots", "shots_on_target"]:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
    active = p[p["appearances"].fillna(0) > 0].copy()
    d = denom.loc[active.index]
    active["goals_p90"] = active["goals"] / d
    active["assists_p90"] = active["assists"] / d
    active["shots_p90"] = active["shots"] / d
    active["sot_p90"] = active["shots_on_target"] / d
    active["ga_p90"] = active["goals_p90"] + active["assists_p90"]

    agg = active.groupby("team").agg(
        squad_goals_p90=("goals_p90", "mean"),
        squad_assists_p90=("assists_p90", "mean"),
        squad_shots_p90=("shots_p90", "mean"),
        squad_sot_p90=("sot_p90", "mean"),
        squad_star_power=("ga_p90", "max"),  # best goal-contribution rate on the team
    )
    return agg


# ---------------------------------------------------------- feature engineering

def flag_wc2026(matches, wc26):
    """Flag 2026 WC rows by joining (date, {teams}) from the isolated 2026 file,
    rather than trusting is_world_cup + a date cutoff."""
    stage_by_key = {}
    for _, r in wc26.iterrows():
        stage_by_key[(r["date"], frozenset((r["team"], r["opponent"])))] = r["stage"]
    keys = [(d, frozenset((h, a))) for d, h, a in
            zip(matches["date"], matches["home_team"], matches["away_team"])]
    stages = pd.Series([stage_by_key.get(k) for k in keys], index=matches.index, dtype="object")
    return stages


def sample_weights(matches, wc_stage):
    tier = pd.Series(1.0, index=matches.index)
    tier[matches["is_major_tournament"] == 1] = MAJOR_WEIGHT
    is26 = wc_stage.notna()
    tier[is26] = wc_stage[is26].map(STAGE_WEIGHT).astype(float)
    days_before = (matches["date"].max() - matches["date"]).dt.days.clip(lower=0)
    decay = 0.5 ** (days_before / DECAY_HALFLIFE_DAYS)
    return tier * decay


def build_features(matches, prof, squad):
    df = matches.copy()
    for side in ("home", "away"):
        df = df.merge(prof.add_prefix(f"{side}_"), how="left",
                      left_on=f"{side}_team", right_index=True)
        df = df.merge(squad.add_prefix(f"{side}_"), how="left",
                      left_on=f"{side}_team", right_index=True)

    df["fifa_rank_diff"] = df["fifa_rank_away_current"] - df["fifa_rank_home_current"]  # +ve favours home
    df["matches_played_365d_diff"] = df["home_matches_played_365d"] - df["away_matches_played_365d"]
    for c in FORM_COLS:
        df[f"{c}_diff"] = df[f"home_{c}"] - df[f"away_{c}"]
    for c in PROFILE_COLS:
        df[f"prof_{c}_diff"] = df[f"home_prof_{c}"] - df[f"away_prof_{c}"]
    for c in SQUAD_COLS:
        df[f"{c}_diff"] = df[f"home_{c}"] - df[f"away_{c}"]
    return df


def feature_list(df):
    raw = list(CONTEXT_COLS) + [
        "home_elo_pre", "away_elo_pre", "elo_diff",
        "fifa_rank_home_current", "fifa_rank_away_current", "fifa_rank_diff",
        "home_matches_played_365d", "away_matches_played_365d", "matches_played_365d_diff",
        "h2h_home_winrate_last5", "h2h_matches_count",
    ]
    for side in ("home", "away"):
        raw += [f"{side}_{c}" for c in FORM_COLS]
        raw += [f"{side}_prof_{c}" for c in PROFILE_COLS]
        raw += [f"{side}_{c}" for c in SQUAD_COLS]
    raw += [f"{c}_diff" for c in FORM_COLS]
    raw += [f"prof_{c}_diff" for c in PROFILE_COLS]
    raw += [f"{c}_diff" for c in SQUAD_COLS]
    return [c for c in raw if c in df.columns]


def scaled_feature_list(features):
    """Feature subset for the impute+scale models (logistic regression, random forest).

    Squad columns are dropped here and kept only for XGBoost, which reads NaN natively.
    The player file covers only the four semifinalists, so a squad *_diff is non-null in
    26 of 2384 training rows -- but non-null in *every* row we actually predict, since both
    sides of the final and the third-place tie are finalists. Median-imputing those 2358
    rows and standardizing makes a near-constant column look informative: the fitted
    coefficient comes off 26 rows, lands with an unstable sign, and then dominates the live
    prediction (it put Argentina at 16% against an England side they had just beaten).
    Validation cannot catch this -- it holds almost no both-finalist rows. XGBoost keeps the
    feature and simply splits on it where present, which is the directional, tie-breaking
    role the squad profile is meant to play.
    """
    return [c for c in features if "squad" not in c]


def compact_feature_list(features):
    """Differences + match context only, dropping the raw home/away levels.

    home_x, away_x and x_diff are exactly collinear, so the level columns are pure
    redundancy for the linear model and just extra split candidates for the trees.
    """
    keep = {c for c in features if c.endswith("_diff")}
    keep |= {c for c in CONTEXT_COLS if c in features}
    keep |= {c for c in ["h2h_home_winrate_last5", "h2h_matches_count"] if c in features}
    return [c for c in features if c in keep]


def topk_feature_list(features, df_train, k=24):
    """Top-k by XGBoost gain, fitted on the training split only (no validation leak)."""
    m = build_xgb({**XGB_DEFAULT, "n_estimators": 300})
    m.fit(df_train[features], df_train["result"].map(RESULT_TO_INT),
          sample_weight=df_train["sample_weight"], verbose=False)
    imp = pd.Series(m.feature_importances_, index=features)
    keep = set(imp.sort_values(ascending=False).head(k).index) | set(CONTEXT_COLS)
    return [c for c in features if c in keep]


# ------------------------------------------------------------------- the models

def build_logreg(params):
    return Pipeline(make_preprocessor() + [("clf", LogisticRegression(max_iter=3000, **params))])


def build_rf(params):
    return Pipeline(make_preprocessor() + [("clf", RandomForestClassifier(random_state=0, n_jobs=-1, **params))])


def build_xgb(params):
    return XGBClassifier(objective="multi:softprob", num_class=len(RESULT_CLASSES),
                         eval_metric="mlogloss", random_state=0, **params)


def elo_prob(home_elo, away_elo, neutral):
    adv = 0.0 if neutral else HOME_ADV
    return 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + adv)) / 400.0))


def fit_bradley_terry(df_train, w_train):
    """Weighted MLE for per-team strengths + a home-advantage term.
    P(home win) = sigmoid(r_home - r_away + adv*(1-neutral)). Pairwise by construction, so it
    stays binary (draw counts as not-home-win) while the three ML models go 3-class."""
    teams = sorted(set(df_train["home_team"]) | set(df_train["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    h = df_train["home_team"].map(idx).to_numpy()
    a = df_train["away_team"].map(idx).to_numpy()
    y = df_train["home_win"].to_numpy(float)
    neu = df_train["neutral"].to_numpy(float)
    w = w_train.to_numpy(float)
    n = len(teams)

    def nll(theta):
        r, adv = theta[:n], theta[n]
        z = r[h] - r[a] + adv * (1 - neu)
        p = np.clip(expit(z), 1e-9, 1 - 1e-9)
        loss = -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))
        g = w * (p - y)
        gr = np.zeros(n)
        np.add.at(gr, h, g)
        np.add.at(gr, a, -g)
        gr += 2e-2 * r  # light L2 keeps strengths identified and finite
        return loss + 1e-2 * r @ r, np.append(gr, np.sum(g * (1 - neu)))

    res = minimize(nll, np.zeros(n + 1), jac=True, method="L-BFGS-B")
    r = res.x[:n] - res.x[:n].mean()
    return {"strengths": {t: float(r[i]) for t, i in idx.items()},
            "home_adv": float(res.x[n]), "default_strength": 0.0}


def bt_predict(bt, home_team, away_team, neutral):
    s = bt["strengths"]
    z = (s.get(home_team, bt["default_strength"]) - s.get(away_team, bt["default_strength"])
         + bt["home_adv"] * (0 if neutral else 1))
    return float(expit(z))


# ------------------------------------------------- tuning (time-series CV, train only)

def _fit_predict_fold(kind, params, tr, va, features):
    """Fit on tr, return P(home_win) on va. Every tuning decision uses train-split folds only."""
    if kind in ("logreg", "random_forest"):
        feats = scaled_feature_list(features)
        m = build_logreg(params) if kind == "logreg" else build_rf(params)
        m.fit(tr[feats], tr["result"], clf__sample_weight=tr["sample_weight"])
        return m.predict_proba(va[feats])[:, list(m.classes_).index("home_win")], None
    if kind == "xgboost":
        m = build_xgb({**params, "n_estimators": 800, "early_stopping_rounds": 40})
        m.fit(tr[features], tr["result"].map(RESULT_TO_INT), sample_weight=tr["sample_weight"],
              eval_set=[(va[features], va["result"].map(RESULT_TO_INT))], verbose=False)
        return m.predict_proba(va[features])[:, HOME_IDX], m.best_iteration
    raise ValueError(kind)


def cv_score(kind, params, features, df_train, n_splits=CV_SPLITS):
    """Mean binary log loss over expanding time-ordered folds, plus mean best_iteration for xgb.

    Expanding-window splits, never k-fold: a random fold would train on 2026 and score on 2012,
    which is the leak the whole time-based design exists to avoid.
    """
    dfs = df_train.sort_values("date").reset_index(drop=True)
    scores, iters = [], []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=n_splits).split(dfs):
        tr, va = dfs.iloc[tr_idx], dfs.iloc[va_idx]
        p, best = _fit_predict_fold(kind, params, tr, va, features)
        scores.append(log_loss(va["home_win"], np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
        if best is not None:
            iters.append(best)
    return float(np.mean(scores)), (int(np.mean(iters)) + 1 if iters else None)


def tune(df_train, features):
    """Two sequential passes per model: feature set with default params, then params on the
    winning set. Sequential rather than joint -- a joint grid is 3x the fits for a decision
    the two passes already make, and every fit here is repeated over 4 folds."""
    sets = {
        "all": features,
        "compact": compact_feature_list(features),
        "top24": topk_feature_list(features, df_train, k=24),
    }
    defaults = {"logreg": LOGREG_DEFAULT, "random_forest": RF_DEFAULT, "xgboost": XGB_DEFAULT}
    grids = {"logreg": LOGREG_GRID, "random_forest": RF_GRID, "xgboost": XGB_GRID}
    out = {}
    for kind in ["logreg", "random_forest", "xgboost"]:
        print(f"\n--- tuning {DISPLAY_NAMES[kind]} ({CV_SPLITS}-fold expanding-window CV on train) ---")
        print("  feature sets:")
        best_set, best_score = None, np.inf
        for name, fs in sets.items():
            n = len(scaled_feature_list(fs)) if kind != "xgboost" else len(fs)
            s, _ = cv_score(kind, defaults[kind], fs, df_train)
            print(f"    {name:<8} {n:>3} feats  cv_logloss={s:.4f}")
            if s < best_score:
                best_set, best_score = name, s
        print(f"  -> feature set: {best_set}")

        best_params, best_iters = defaults[kind], None
        for params in grids[kind]:
            s, iters = cv_score(kind, params, sets[best_set], df_train)
            if s < best_score:
                best_score, best_params, best_iters = s, params, iters
        if best_iters is None and kind == "xgboost":
            _, best_iters = cv_score(kind, best_params, sets[best_set], df_train)
        print(f"  -> params: {best_params}  cv_logloss={best_score:.4f}"
              + (f"  n_estimators={best_iters}" if best_iters else ""))
        out[kind] = {"feature_set": best_set, "params": best_params,
                     "cv_logloss": best_score, "n_estimators": best_iters}
    return out, sets


# ------------------------------------------------------------------ evaluation

def calibration_table(y, p, bins=5):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for i in range(bins):
        m = b == i
        if m.sum():
            rows.append((f"{edges[i]:.1f}-{edges[i+1]:.1f}", int(m.sum()),
                         round(float(p[m].mean()), 3), round(float(y[m].mean()), 3)))
    return rows


def predict_all(artifacts, df_valid):
    """P(home_win) on the validation frame for every algorithm."""
    preds = {}
    preds["elo"] = np.array([elo_prob(h, a, n) for h, a, n in
                             zip(df_valid["home_elo_pre"], df_valid["away_elo_pre"], df_valid["neutral"])])
    for kind in ("logreg", "random_forest"):
        feats = artifacts["model_features"][kind]
        m = artifacts[kind]
        preds[kind] = m.predict_proba(df_valid[feats])[:, list(m.classes_).index("home_win")]
    xf = artifacts["model_features"]["xgboost"]
    preds["xgboost"] = artifacts["xgboost"].predict_proba(df_valid[xf])[:, HOME_IDX]
    preds["bradley_terry"] = np.array([bt_predict(artifacts["bradley_terry"], h, a, n) for h, a, n in
                                       zip(df_valid["home_team"], df_valid["away_team"], df_valid["neutral"])])
    preds["ensemble"] = np.mean([preds[a] for a in BASE_ALGORITHMS], axis=0)
    return preds


def three_class_probs(artifacts, kind, X):
    """[P(away_win), P(draw), P(home_win)] for the 3-class models, in RESULT_CLASSES order."""
    feats = artifacts["model_features"][kind]
    if kind == "xgboost":
        return artifacts["xgboost"].predict_proba(X[feats])[0]
    m = artifacts[kind]
    p = m.predict_proba(X[feats])[0]
    order = [list(m.classes_).index(c) for c in RESULT_CLASSES]
    return p[order]


# ------------------------------------------------------------------ prediction

def load_artifacts():
    with open(os.path.join(MODELS, "feature_list.json")) as f:
        meta = json.load(f)
    with open(os.path.join(MODELS, "elo_config.json")) as f:
        elo_cfg = json.load(f)
    with open(os.path.join(MODELS, "bradley_terry.json")) as f:
        bt = json.load(f)
    xgb = XGBClassifier()
    xgb.load_model(os.path.join(MODELS, "xgboost_model.json"))
    return {
        "features": meta["features"],
        "model_features": meta["model_features"],
        "elo_config": elo_cfg,
        "logreg": joblib.load(os.path.join(MODELS, "logreg.pkl")),
        "random_forest": joblib.load(os.path.join(MODELS, "random_forest.pkl")),
        "xgboost": xgb,
        "bradley_terry": bt,
    }


def team_context(matches, prof, squad, team):
    """Latest pre-match Elo + rolling form for a team, from its most recent appearance."""
    rows = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
    last = rows.sort_values("date").iloc[-1]
    side = "home" if last["home_team"] == team else "away"
    ctx = {
        "elo": float(last[f"{side}_elo_pre"]),
        "fifa_rank": float(last[f"fifa_rank_{side}_current"]),
        "matches_played_365d": float(last[f"{side}_matches_played_365d"]),
    }
    for c in FORM_COLS:
        ctx[c] = float(last[f"{side}_{c}"])
    for c in PROFILE_COLS:
        v = prof[f"prof_{c}"].get(team, np.nan)
        ctx[f"prof_{c}"] = float(v) if pd.notna(v) else np.nan
    for c in SQUAD_COLS:
        v = squad[c].get(team, np.nan) if team in squad.index else np.nan
        ctx[c] = float(v) if pd.notna(v) else np.nan
    return ctx


def h2h(matches, team_a, team_b):
    """Win rate of team_a in their last 5 meetings, plus total meetings."""
    m = matches[((matches["home_team"] == team_a) & (matches["away_team"] == team_b)) |
                ((matches["home_team"] == team_b) & (matches["away_team"] == team_a))]
    if m.empty:
        return 0.5, 0
    last5 = m.sort_values("date").tail(5)
    wins = (((last5["home_team"] == team_a) & (last5["result"] == "home_win")) |
            ((last5["away_team"] == team_a) & (last5["result"] == "away_win"))).mean()
    return float(wins), int(len(m))


def _feature_row(ctx_h, ctx_a, h2h_rate, h2h_n, neutral, features):
    row = {
        "neutral": int(neutral), "is_friendly": 0, "is_major_tournament": 1,
        "is_world_cup": 1, "is_qualifier": 0,
        "home_elo_pre": ctx_h["elo"], "away_elo_pre": ctx_a["elo"],
        "elo_diff": ctx_h["elo"] - ctx_a["elo"],
        "fifa_rank_home_current": ctx_h["fifa_rank"], "fifa_rank_away_current": ctx_a["fifa_rank"],
        "fifa_rank_diff": ctx_a["fifa_rank"] - ctx_h["fifa_rank"],
        "home_matches_played_365d": ctx_h["matches_played_365d"],
        "away_matches_played_365d": ctx_a["matches_played_365d"],
        "matches_played_365d_diff": ctx_h["matches_played_365d"] - ctx_a["matches_played_365d"],
        "h2h_home_winrate_last5": h2h_rate, "h2h_matches_count": h2h_n,
    }
    for c in FORM_COLS:
        row[f"home_{c}"], row[f"away_{c}"] = ctx_h[c], ctx_a[c]
        row[f"{c}_diff"] = ctx_h[c] - ctx_a[c]
    for c in PROFILE_COLS:
        row[f"home_prof_{c}"], row[f"away_prof_{c}"] = ctx_h[f"prof_{c}"], ctx_a[f"prof_{c}"]
        row[f"prof_{c}_diff"] = ctx_h[f"prof_{c}"] - ctx_a[f"prof_{c}"]
    for c in SQUAD_COLS:
        row[f"home_{c}"], row[f"away_{c}"] = ctx_h[c], ctx_a[c]
        row[f"{c}_diff"] = ctx_h[c] - ctx_a[c]
    return pd.DataFrame([row]).reindex(columns=features)


def _oriented_probs(artifacts, algorithm, X, ctx_h, ctx_a, home_team, away_team, neutral):
    """One orientation. Returns (P(home_win), P(draw) or None)."""
    if algorithm == "elo":
        return elo_prob(ctx_h["elo"], ctx_a["elo"], neutral), None
    if algorithm == "bradley_terry":
        return bt_predict(artifacts["bradley_terry"], home_team, away_team, neutral), None
    p = three_class_probs(artifacts, algorithm, X)
    return float(p[HOME_IDX]), float(p[DRAW_IDX])


def _matchup_one(artifacts, tables, team_a, team_b, algorithm, neutral):
    matches, prof, squad = tables
    ctx_a = team_context(matches, prof, squad, team_a)
    ctx_b = team_context(matches, prof, squad, team_b)
    rate_ab, n_ab = h2h(matches, team_a, team_b)
    rate_ba, _ = h2h(matches, team_b, team_a)
    f = artifacts["features"]

    X1 = _feature_row(ctx_a, ctx_b, rate_ab, n_ab, neutral, f)
    X2 = _feature_row(ctx_b, ctx_a, rate_ba, n_ab, neutral, f)
    p1, d1 = _oriented_probs(artifacts, algorithm, X1, ctx_a, ctx_b, team_a, team_b, neutral)
    p2, d2 = _oriented_probs(artifacts, algorithm, X2, ctx_b, ctx_a, team_b, team_a, neutral)

    if d1 is None:
        # Binary model: 'not home win' already lumps draws in with away wins, so averaging
        # P(A wins at home) with P(B does not win at home) both cancels home bias and splits
        # the draw mass 50/50 -- the redistribution the knockout format needs.
        p_a, p_draw = (p1 + (1.0 - p2)) / 2.0, None
    else:
        # 3-class: average the two orientations, then hand each side half the draw mass.
        win_a, win_b = (p1 + (1.0 - p2 - d2)) / 2.0, (p2 + (1.0 - p1 - d1)) / 2.0
        p_draw = (d1 + d2) / 2.0
        p_a = win_a + p_draw / 2.0
        p_a /= (win_a + win_b + p_draw)
    return {"p_a": float(p_a), "p_b": float(1.0 - p_a), "p_draw": p_draw,
            "ctx_a": ctx_a, "ctx_b": ctx_b}


def predict_matchup(team_a, team_b, algorithm, neutral=True, artifacts=None, tables=None):
    """P(team_a wins) for a knockout tie with no draw allowed, draw mass split 50/50."""
    artifacts = artifacts or load_artifacts()
    tables = tables or load_tables()
    if algorithm != "ensemble":
        return _matchup_one(artifacts, tables, team_a, team_b, algorithm, neutral)
    outs = [_matchup_one(artifacts, tables, team_a, team_b, a, neutral) for a in BASE_ALGORITHMS]
    p_a = float(np.mean([o["p_a"] for o in outs]))
    draws = [o["p_draw"] for o in outs if o["p_draw"] is not None]
    return {"p_a": p_a, "p_b": 1.0 - p_a, "p_draw": float(np.mean(draws)) if draws else None,
            "ctx_a": outs[0]["ctx_a"], "ctx_b": outs[0]["ctx_b"]}


def load_tables():
    matches, tourn, wc26, players = load_raw()
    return matches, team_quality_profile(tourn), squad_quality_profile(players)


# ---------------------------------------------------------------------- training

def main():
    os.makedirs(MODELS, exist_ok=True)
    matches, tourn, wc26, players = load_raw()
    prof, squad = team_quality_profile(tourn), squad_quality_profile(players)

    wc_stage = flag_wc2026(matches, wc26)
    matches["sample_weight"] = sample_weights(matches, wc_stage)

    df = build_features(matches, prof, squad)
    features = feature_list(df)

    train = df[df["date"] < CUTOFF]
    valid = df[df["date"] >= CUTOFF]
    y_va = valid["home_win"]

    print(f"Train {len(train)} rows (<{CUTOFF.date()})  |  Validate {len(valid)} rows (>={CUTOFF.date()})")
    print(f"Shared feature table: {len(features)} columns")

    # -- weight audit: make it visible that 2026 isn't negligible -------------
    tw = matches["sample_weight"].sum()
    w26 = matches.loc[wc_stage.notna(), "sample_weight"].sum()
    print("\n=== Training-weight composition ===")
    print(f"2026 World Cup rows: {int(wc_stage.notna().sum())} of {len(matches)} matches "
          f"({wc_stage.notna().mean():.1%} by count) but {w26/tw:.1%} of total weight")
    for stage in STAGE_WEIGHT:
        m = wc_stage == stage
        if m.any():
            print(f"  {stage:<15} {int(m.sum()):>3} matches  {matches.loc[m,'sample_weight'].sum()/tw:>6.1%} of weight")
    major = (matches["is_major_tournament"] == 1) & wc_stage.isna()
    print(f"  {'other majors':<15} {int(major.sum()):>3} matches  {matches.loc[major,'sample_weight'].sum()/tw:>6.1%} of weight")
    other = ~major & wc_stage.isna()
    print(f"  {'everything else':<15} {int(other.sum()):>3} matches  {matches.loc[other,'sample_weight'].sum()/tw:>6.1%} of weight")
    print(f"Neutral-venue rate: overall {matches['neutral'].mean():.1%}, "
          f"2026 WC {matches.loc[wc_stage.notna(),'neutral'].mean():.1%} "
          f"(the final and 3rd-place tie are predicted with neutral=1)")

    # -- tune ----------------------------------------------------------------
    tuning_path = os.path.join(MODELS, "tuning.json")
    if "--no-tune" in sys.argv and os.path.exists(tuning_path):
        with open(tuning_path) as f:
            tuning = json.load(f)
        sets = {"all": features, "compact": compact_feature_list(features),
                "top24": topk_feature_list(features, train, k=24)}
        print("\n(reusing models/tuning.json)")
    else:
        tuning, sets = tune(train, features)
        with open(tuning_path, "w") as f:
            json.dump(tuning, f, indent=1)

    model_features = {}
    for kind in ("logreg", "random_forest"):
        model_features[kind] = scaled_feature_list(sets[tuning[kind]["feature_set"]])
    model_features["xgboost"] = sets[tuning["xgboost"]["feature_set"]]

    # -- final fits on the full training split -------------------------------
    models = {}
    logreg = build_logreg(tuning["logreg"]["params"])
    logreg.fit(train[model_features["logreg"]], train["result"], clf__sample_weight=train["sample_weight"])
    models["logreg"] = logreg

    rf = build_rf(tuning["random_forest"]["params"])
    rf.fit(train[model_features["random_forest"]], train["result"], clf__sample_weight=train["sample_weight"])
    models["random_forest"] = rf

    # n_estimators comes from the CV folds, so the final fit never touches validation --
    # early stopping on the validation set would tune a hyperparameter on the data we report.
    xgb = build_xgb({**tuning["xgboost"]["params"], "n_estimators": tuning["xgboost"]["n_estimators"]})
    xgb.fit(train[model_features["xgboost"]], train["result"].map(RESULT_TO_INT),
            sample_weight=train["sample_weight"], verbose=False)
    models["xgboost"] = xgb
    models["bradley_terry"] = fit_bradley_terry(train, train["sample_weight"])

    # -- save ----------------------------------------------------------------
    joblib.dump(logreg, os.path.join(MODELS, "logreg.pkl"))
    joblib.dump(rf, os.path.join(MODELS, "random_forest.pkl"))
    xgb.save_model(os.path.join(MODELS, "xgboost_model.json"))
    with open(os.path.join(MODELS, "bradley_terry.json"), "w") as f:
        json.dump(models["bradley_terry"], f, indent=1)
    with open(os.path.join(MODELS, "elo_config.json"), "w") as f:
        json.dump({"home_adv": HOME_ADV, "formula": "1/(1+10**((away_elo-(home_elo+home_adv))/400))"}, f, indent=1)
    with open(os.path.join(MODELS, "feature_list.json"), "w") as f:
        json.dump({"features": features, "model_features": model_features,
                   "cutoff": str(CUTOFF.date()), "algorithms": ALGORITHMS,
                   "display_names": DISPLAY_NAMES}, f, indent=1)
    print(f"\nSaved artifacts to {MODELS}")

    artifacts = load_artifacts()
    preds = predict_all(artifacts, valid)

    print("\n=== Validation comparison, P(home win) (2025-01-01 onward) ===")
    print(f"{'Model':<34}{'Accuracy':>10}{'LogLoss':>10}{'Brier':>10}{'Features':>10}")
    for a in ALGORITHMS:
        p = np.clip(preds[a], 1e-6, 1 - 1e-6)
        nf = len(model_features[a]) if a in model_features else ""
        print(f"{DISPLAY_NAMES[a]:<34}{accuracy_score(y_va, p > 0.5):>10.3f}"
              f"{log_loss(y_va, p):>10.3f}{np.mean((p - y_va) ** 2):>10.3f}{str(nf):>10}")
    print(f"{'(always predict home win)':<34}{y_va.mean():>10.3f}")

    print("\n=== 3-class metrics (away / draw / home) ===")
    y3 = valid["result"]
    print(f"{'Model':<34}{'Accuracy':>10}{'LogLoss':>10}")
    for kind in ("logreg", "random_forest", "xgboost"):
        feats = model_features[kind]
        if kind == "xgboost":
            p3 = models[kind].predict_proba(valid[feats])
        else:
            m = models[kind]
            p3 = m.predict_proba(valid[feats])[:, [list(m.classes_).index(c) for c in RESULT_CLASSES]]
        pred_lbl = [RESULT_CLASSES[i] for i in p3.argmax(1)]
        print(f"{DISPLAY_NAMES[kind]:<34}{accuracy_score(y3, pred_lbl):>10.3f}"
              f"{log_loss(y3, p3, labels=RESULT_CLASSES):>10.3f}")
    print(f"{'(always predict home win)':<34}{(y3 == 'home_win').mean():>10.3f}")

    print("\n=== Calibration (predicted vs actual home-win rate) ===")
    for a in ALGORITHMS:
        rows = calibration_table(y_va.to_numpy(), preds[a])
        print(f"\n{DISPLAY_NAMES[a]}")
        print(f"  {'bin':<12}{'n':>5}{'pred':>8}{'actual':>8}")
        for b, n, pm, am in rows:
            print(f"  {b:<12}{n:>5}{pm:>8.3f}{am:>8.3f}")

    stack = np.column_stack([preds[a] for a in BASE_ALGORITHMS])
    spread = stack.max(axis=1) - stack.min(axis=1)
    print(f"\nModel disagreement on validation: mean spread {spread.mean():.3f}, max {spread.max():.3f}")
    if spread.mean() > 0.15:
        print("  !! Models disagree substantially -- treat any single model's number with caution.")

    # -- sanity check on known results ---------------------------------------
    tables = (matches, prof, squad)
    print("\n=== Sanity check: known 2026 results (P(first team wins), knockout-normalized) ===")
    known = [("Spain", "France", "SF: Spain won 2-0"),
             ("Argentina", "England", "SF: Argentina won 2-1")]
    print(f"{'Matchup':<24}" + "".join(f"{DISPLAY_NAMES[a][:12]:>14}" for a in ALGORITHMS) + "   actual")
    for a_, b_, note in known:
        line = f"{a_ + ' vs ' + b_:<24}"
        for alg in ALGORITHMS:
            line += f"{predict_matchup(a_, b_, alg, True, artifacts, tables)['p_a']:>14.1%}"
        print(line + f"   {note}")

    print("\n=== Live predictions (neutral venue, knockout: draw mass redistributed) ===")
    for a_, b_, label in [("Argentina", "Spain", "FINAL"), ("England", "France", "THIRD PLACE")]:
        print(f"\n{label}: {a_} vs {b_}")
        ps = []
        for alg in ALGORITHMS:
            r = predict_matchup(a_, b_, alg, True, artifacts, tables)
            ps.append(r["p_a"])
            draw = f"  (pre-split draw {r['p_draw']:.0%})" if r["p_draw"] else ""
            print(f"  {DISPLAY_NAMES[alg]:<34}{a_} {r['p_a']:>6.1%}  |  {b_} {r['p_b']:>6.1%}{draw}")
        base = ps[:len(BASE_ALGORITHMS)]
        if max(base) - min(base) > 0.15:
            print(f"  !! Models disagree: P({a_}) ranges {min(base):.1%}-{max(base):.1%}")
        elif (max(base) > 0.5) != (min(base) > 0.5):
            print(f"  !! Models disagree on the favourite (range {min(base):.1%}-{max(base):.1%})")


if __name__ == "__main__":
    main()
