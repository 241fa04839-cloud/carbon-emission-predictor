"""
model.py  --  Steps 4-6: feature selection, 15 boosting models, model selection,
evaluation, what-if study, and saving the .pickle file.

Run  `python model.py`  once to (re)train everything and write carbon_model.pickle.
The Streamlit app (app.py) then only loads that pickle and calls predict_emission().
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

import preprocessing as prep
from encoding import FeatureEncoder
from scaling import FeatureScaler

PICKLE_FILE = "carbon_model.pickle"
SEED = prep.SEED
TOTAL, RENEW, NONRENEW = (
    "Total_Energy_Consumption_kWh",
    "Renewable_Energy_Consumption_kWh",
    "NonRenewable_Energy_Consumption_kWh",
)


# ---------------------------------------------------------------- feature selection
def select_features(Xs_train: pd.DataFrame, y_train):
    """LASSO (LassoCV, 5-fold), Random Forest (200 trees, importance > mean), hybrid union."""
    lasso = LassoCV(cv=5, random_state=SEED).fit(Xs_train, y_train)
    s_lasso = [c for c, w in zip(Xs_train.columns, lasso.coef_) if w != 0]
    rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xs_train, y_train)
    imp = rf.feature_importances_
    s_rf = [c for c, v in zip(Xs_train.columns, imp) if v > imp.mean()]
    s_hybrid = [c for c in Xs_train.columns if c in set(s_lasso) | set(s_rf)]
    return {"LASSO": s_lasso, "Random Forest": s_rf, "LASSO + Random Forest": s_hybrid}


# ---------------------------------------------------------------- boosting algorithms
def make_boosters():
    return {
        "AdaBoost": AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=200, random_state=SEED),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=SEED),
        "GB (Huber loss)": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, loss="huber", random_state=SEED),
        "Stochastic GB": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, subsample=0.7, random_state=SEED),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=200, random_state=SEED),
    }


def evaluate(model, X, y, threshold):
    p = model.predict(X)
    return {
        "R2": r2_score(y, p) * 100,
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": mean_absolute_error(y, p),
        "AUC": roc_auc_score(y > threshold, p) * 100,
    }


# ---------------------------------------------------------------- prediction (used by the app)
def predict_emission(bundle: dict, query) -> np.ndarray:
    """raw user input -> encode -> scale -> pick selected features -> model.predict"""
    X = prep.prepare_query(query)
    Xe = bundle["encoder"].transform(X)
    Xs = bundle["scaler"].transform(Xe)
    return bundle["model"].predict(Xs[bundle["features"]])


def what_if(bundle: dict, query: dict, fraction: float):
    """Move `fraction` of non-renewable energy to renewable (total unchanged),
    recompute the renewable share, and return (before, after, cut_%)  -- paper Eq. (5)."""
    q = dict(query)
    before = float(predict_emission(bundle, q)[0])
    shift = fraction * q[NONRENEW]
    q[NONRENEW] = q[NONRENEW] - shift
    q[RENEW] = q[RENEW] + shift
    q["Expected_Renewable_Share_Percent"] = 100 * q[RENEW] / q[TOTAL]
    after = float(predict_emission(bundle, q)[0])
    return before, after, 100 * (before - after) / before


# ---------------------------------------------------------------- training
def train(path: str = prep.DATA_FILE, verbose: bool = True) -> dict:
    df = prep.load_data(path)
    X, y = prep.clean_data(df)
    X_tr, X_te, y_tr, y_te = prep.split_data(X, y)

    encoder = FeatureEncoder().fit(X_tr)
    Xe_tr, Xe_te = encoder.transform(X_tr), encoder.transform(X_te)
    scaler = FeatureScaler().fit(Xe_tr)
    Xs_tr, Xs_te = scaler.transform(Xe_tr), scaler.transform(Xe_te)
    y_tr, y_te = y_tr.reset_index(drop=True), y_te.reset_index(drop=True)

    sets = select_features(Xs_tr, y_tr)
    threshold = float(y_te.median())          # High / Low emission split (paper Sec. 3.7)

    rows, fitted = [], {}
    for set_name, feats in sets.items():
        for name, booster in make_boosters().items():
            cv_r2 = cross_val_score(booster, Xs_tr[feats], y_tr, cv=3, scoring="r2").mean() * 100
            booster.fit(Xs_tr[feats], y_tr)
            rows.append({"Feature set": set_name, "Algorithm": name, "CV_R2": cv_r2,
                         **evaluate(booster, Xs_te[feats], y_te, threshold)})
            fitted[(set_name, name)] = booster
            if verbose:
                print(f"{set_name:24s} {name:24s} CV R2={cv_r2:6.2f}  test R2={rows[-1]['R2']:6.2f}")
    results = pd.DataFrame(rows)

    best = results.loc[results["CV_R2"].idxmax()]          # chosen by CV on TRAIN only
    key = (best["Feature set"], best["Algorithm"])

    bundle = {
        "encoder": encoder, "scaler": scaler, "model": fitted[key],
        "features": sets[best["Feature set"]],
        "best_feature_set": best["Feature set"], "best_algorithm": best["Algorithm"],
        "feature_sets": sets, "results": results,
        "high_threshold": threshold,
        "defaults": {**{c: float(X_tr[c].median()) for c in prep.NUMERIC_FEATURES},
                     **{c: X_tr[c].mode()[0] for c in prep.CATEGORICAL_FEATURES}},
        "train_ranges": {c: (float(X_tr[c].min()), float(X_tr[c].max())) for c in prep.NUMERIC_FEATURES},
        "target_stats": {"mean": float(y.mean()), "min": float(y.min()), "max": float(y.max())},
    }
    if verbose:
        print("\nSelected features:", sets)
        print(f"BEST (by CV R2): {key}  -> test metrics:\n{best[['R2','RMSE','MAE','AUC']].to_dict()}")
    return bundle


def save(bundle: dict, path: str = PICKLE_FILE):
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load(path: str = PICKLE_FILE) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    b = train()
    save(b)
    print(f"\nSaved -> {PICKLE_FILE}")
