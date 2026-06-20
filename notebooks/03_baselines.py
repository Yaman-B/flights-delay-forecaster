"""
03_baseline.py
Baseline-evaluation 
A leakage-safe temporal split, locked metrics, and a baseline file.
(majority class -> climatology -> logistic regression). Every later model
number will be measured against this.
"""
# %% --- Setup --------------------------------------------------------------
import sys
from pathlib import Path

# Make `import src` work in the kernel: the kernel runs from notebooks/, so the
# repo root isn't on sys.path by default. Anchor to the pyproject.toml marker.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
from src.paths import DATA_PATH, TARGET_COL
from src.models import temporal_split, evaluate, majority_baseline, climatology_baseline
print("src OK ->", __import__("src").__file__)

# %% --- Setup --------------------------------------------------------------
def find_project_root(markers=("requirements.txt", ".git")):
    here = Path.cwd().resolve()
    for parent in (here, *here.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    raise FileNotFoundError(f"Project root not found above {here}")

PROJECT_ROOT = find_project_root()
DATA_PATH  = PROJECT_ROOT / "data" / "processed" / "flights_weather.parquet"
TARGET_COL = "disrupted"

# %% --- Temporal train / validation / test split --------------------------
def temporal_split(dates):
    """Assign each flight to train/val/test by its scheduled date.
    train: 2023-2024 | val: 2025 | test: 2026+.
    The single source of truth for the split — every baseline and model calls
    this, so comparisons stay apples-to-apples."""
    y = pd.to_datetime(dates).dt.year
    s = np.where(y <= 2024, "train", np.where(y == 2025, "val", "test"))
    return pd.Categorical(s, ["train", "val", "test"], ordered=True)

splits = pd.read_parquet(DATA_PATH, columns=["FlightDate", TARGET_COL])
splits["FlightDate"] = pd.to_datetime(splits["FlightDate"])
splits["split"] = temporal_split(splits["FlightDate"])

report = (splits.groupby("split", observed=True)
          .agg(start=("FlightDate", "min"), end=("FlightDate", "max"),
               flights=("FlightDate", "size"), disrupt_rate=(TARGET_COL, "mean")))
report["pct_of_total"] = report["flights"] / len(splits)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
print(report)
# %% --- Evaluation metrics ----------------------------------------
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, brier_score_loss)

def evaluate(y_true, y_prob, threshold=0.5):
    """The locked metric set. Same function scores every baseline and model."""
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "PR_AUC":   average_precision_score(y_true, y_prob),  # primary (imbalance-honest)
        "ROC_AUC":  roc_auc_score(y_true, y_prob),            # secondary, threshold-free
        "F1@0.5":   f1_score(y_true, y_pred, zero_division=0),# operating point
        "Brier":    brier_score_loss(y_true, y_prob),         # calibration (lower is better)
        "Accuracy": float((y_pred == y_true).mean()),         # shown only as a cautionary foil
    }
# %% --- Baseline 1: majority class (the floor) -----------------------------
# Predict the training-era average disruption probability for every flight.
train_rate = splits.loc[splits["split"] == "train", TARGET_COL].mean()
y_test     = splits.loc[splits["split"] == "test",  TARGET_COL].to_numpy()
maj_prob   = np.full(len(y_test), train_rate)

table = pd.DataFrame({"Majority class": evaluate(y_test, maj_prob)}).T
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
print(f"Constant prediction (train base rate): {train_rate:.4f}\n")
print(table)

# %% --- Baseline ladder, now from the promoted src/ package ----------------
import pandas as pd
from src.paths import DATA_PATH, TARGET_COL
from src.models import temporal_split, evaluate, majority_baseline, climatology_baseline

c = pd.read_parquet(DATA_PATH, columns=["Origin", "Dest", "CRSDepTime", "FlightDate", TARGET_COL])
c["split"] = temporal_split(c["FlightDate"])
c["hour"]  = (c["CRSDepTime"].astype(int) // 100) % 24
tr, te = c[c["split"] == "train"], c[c["split"] == "test"]
yte = te[TARGET_COL].to_numpy()

table = pd.DataFrame({
    "Majority class":             evaluate(yte, majority_baseline(tr, te, TARGET_COL)),
    "Climatology (route x hour)": evaluate(yte, climatology_baseline(tr, te, target=TARGET_COL)),
}).T
print(table)   # must reproduce 0.2445 and 0.3172 exactly


# %% --- Baseline 2: climatology (route x scheduled hour) -------------------
# Historical disruption rate per (origin, dest, local hour), learned on TRAIN.
# Empirical-Bayes shrinkage pulls small/noisy groups toward the global rate;
# unseen route-hours fall back to the global rate.
c = pd.read_parquet(DATA_PATH, columns=["Origin", "Dest", "CRSDepTime", "FlightDate", TARGET_COL])
c["split"] = temporal_split(c["FlightDate"])
c["hour"]  = (c["CRSDepTime"].astype(int) // 100) % 24

ALPHA = 10.0   # a group needs ~ALPHA flights before its own rate outweighs the prior
train = c[c["split"] == "train"]
global_rate = train[TARGET_COL].mean()

agg  = train.groupby(["Origin", "Dest", "hour"])[TARGET_COL].agg(s="sum", n="count")
clim = (agg["s"] + ALPHA * global_rate) / (agg["n"] + ALPHA)   # smoothed rate per group

test   = c[c["split"] == "test"]
keys   = list(zip(test["Origin"], test["Dest"], test["hour"]))
mapped = clim.reindex(keys)                                    # NaN for unseen route-hours
coverage  = float(mapped.notna().mean())
clim_prob = pd.Series(mapped.to_numpy()).fillna(global_rate).to_numpy()
y_test    = test[TARGET_COL].to_numpy()

print(f"Global train rate: {global_rate:.4f} | groups learned: {len(clim):,} | "
      f"test route-hour coverage: {coverage:.1%}")

row   = pd.DataFrame({"Climatology (route x hour)": evaluate(y_test, clim_prob)}).T
table = pd.concat([table, row])
print(table)

# %% --- Baseline 3: logistic regression (first model to COMBINE features) --
import time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

CAT = ["Origin", "Dest", "Reporting_Airline", "hour", "month", "dow"]
NUM = ["snowfall_orig", "snowfall_dest", "wind_gusts_10m_orig", "wind_gusts_10m_dest"]

m = pd.read_parquet(DATA_PATH, columns=["Origin", "Dest", "Reporting_Airline",
        "CRSDepTime", "FlightDate", *NUM, TARGET_COL])
m["split"] = temporal_split(m["FlightDate"])
fd = pd.to_datetime(m["FlightDate"])
m["hour"]  = (m["CRSDepTime"].astype(int) // 100) % 24
m["month"] = fd.dt.month
m["dow"]   = fd.dt.dayofweek
for c in ("Origin", "Dest", "Reporting_Airline"):
    m[c] = m[c].astype("category")

train, test = m[m["split"] == "train"], m[m["split"] == "test"]
Xtr, ytr = train[CAT + NUM], train[TARGET_COL].to_numpy()
Xte, yte = test[CAT + NUM],  test[TARGET_COL].to_numpy()

# Fit ALL preprocessing on train only (scaler stats + category list learned
# from train, applied to test) -- same leakage discipline as the split.
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())]), NUM),
])
clf = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=1000))])

t0 = time.time()
clf.fit(Xtr, ytr)
print(f"fit on {len(Xtr):,} flights in {time.time()-t0:.0f}s")

lr_prob = clf.predict_proba(Xte)[:, 1]
table = pd.concat([table, pd.DataFrame({"Logistic regression": evaluate(yte, lr_prob)}).T])
print(table)

# %% --- What did the linear model lean on? --------
names = clf.named_steps["pre"].get_feature_names_out()
co = pd.Series(clf.named_steps["lr"].coef_[0], index=names).sort_values()
print("Most disruption-INCREASING (log-odds):"); print(co.tail(12)[::-1].round(3))
print("\nMost disruption-DECREASING (log-odds):"); print(co.head(12).round(3))
# %%
