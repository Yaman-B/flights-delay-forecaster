"""Leakage-safe inbound-delay cascade features.

At the prediction cutoff (a flight's SCHEDULED departure) we may not know the
inbound aircraft's final arrival delay. So we use only what's observable then:
the inbound delay right-censored at the scheduled turnaround slack, the slack
itself, the realized spare buffer, and a 'still airborne' flag.
"""
# %% --- Cascade -----------------------------
import numpy as np
import pandas as pd
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

CASCADE_COLS = ["has_inbound", "inbound_gap_h", "inbound_delay_obs",
                "inbound_buffer_min", "inbound_unlanded"]


def add_cascade_features(df, max_gap_h=8.0):
    """Return df with leakage-safe cascade columns added (aligned to df.index).
    Requires Tail_Number, dep_hour_utc, arr_hour_utc, CRSDepTime, ArrDelayMinutes."""
    d = df[["Tail_Number", "dep_hour_utc", "arr_hour_utc",
            "CRSDepTime", "ArrDelayMinutes"]].copy()
    d["dep_hour_utc"] = pd.to_datetime(d["dep_hour_utc"])
    d["arr_hour_utc"] = pd.to_datetime(d["arr_hour_utc"])
    d = d.sort_values(["Tail_Number", "dep_hour_utc", "CRSDepTime"])

    has_tail = d["Tail_Number"].notna() & (d["Tail_Number"].astype(str).str.strip() != "")
    g = d.groupby("Tail_Number", sort=False)
    prev_arr_delay = g["ArrDelayMinutes"].shift(1)
    prev_arr_hour  = g["arr_hour_utc"].shift(1)

    gap_h   = (d["dep_hour_utc"] - prev_arr_hour).dt.total_seconds() / 3600.0
    gap_min = gap_h * 60.0
    linked  = has_tail & prev_arr_delay.notna() & (gap_h > 0) & (gap_h <= max_gap_h)
    obs     = np.minimum(prev_arr_delay, gap_min)   # right-censored at the slack

    out = pd.DataFrame(index=d.index)
    out["has_inbound"]        = linked.astype(int)
    out["inbound_gap_h"]      = gap_h.where(linked)
    out["inbound_delay_obs"]  = obs.where(linked)
    out["inbound_buffer_min"] = (gap_min - obs).where(linked)        # spare turn time
    out["inbound_unlanded"]   = ((prev_arr_delay > gap_min) & linked).astype(int)
    return df.join(out.reindex(df.index))

# %% --- LightGBM with the leakage-safe cascade -----------------------------
import lightgbm as lgb
from src.paths import DATA_PATH, TARGET_COL
from src.models import temporal_split, evaluate
from src.features.cascade import add_cascade_features, CASCADE_COLS

CAT   = ["Origin", "Dest", "Reporting_Airline", "hour", "month", "dow"]
NUM_W = ["snowfall_orig", "snowfall_dest", "wind_gusts_10m_orig",
         "wind_gusts_10m_dest", "temperature_2m_orig"]   # temp: GBM can use its U-shape

cols = list({*CAT[:3], "CRSDepTime", "FlightDate", "Tail_Number", "dep_hour_utc",
             "arr_hour_utc", "ArrDelayMinutes", *NUM_W, TARGET_COL})
m = pd.read_parquet(DATA_PATH, columns=cols)
m["split"] = temporal_split(m["FlightDate"])
fd = pd.to_datetime(m["FlightDate"])
m["hour"], m["month"], m["dow"] = (m["CRSDepTime"].astype(int)//100) % 24, fd.dt.month, fd.dt.dayofweek
m = add_cascade_features(m)
for c in CAT:
    m[c] = m[c].astype("category")          # LightGBM uses category dtype natively

FEATURES = CAT + NUM_W + CASCADE_COLS
tr, va, te = (m[m["split"] == s] for s in ("train", "val", "test"))

dtrain = lgb.Dataset(tr[FEATURES], tr[TARGET_COL])
dval   = lgb.Dataset(va[FEATURES], va[TARGET_COL], reference=dtrain)
params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1)
gbm = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                valid_names=["val"], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

res = evaluate(te[TARGET_COL].to_numpy(), gbm.predict(te[FEATURES]))
ladder = pd.DataFrame({
    "Majority class":      {"PR_AUC":0.2445,"ROC_AUC":0.5000,"F1@0.5":0.0000,"Brier":0.1849,"Accuracy":0.7555},
    "Climatology":         {"PR_AUC":0.3172,"ROC_AUC":0.6066,"F1@0.5":0.0137,"Brier":0.1812,"Accuracy":0.7546},
    "Logistic regression": {"PR_AUC":0.4022,"ROC_AUC":0.6548,"F1@0.5":0.0987,"Brier":0.1727,"Accuracy":0.7628},
}).T
ladder = pd.concat([ladder, pd.DataFrame({"LightGBM (+cascade)": res}).T])
print(ladder.round(4))

imp = pd.Series(gbm.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
print("\nTop features by gain:\n", imp.round(0))
# %% --- Ablation: identical GBM, cascade features REMOVED ------------------
FEATURES_NOCAS = CAT + NUM_W   # everything except CASCADE_COLS

dtrain_nc = lgb.Dataset(tr[FEATURES_NOCAS], tr[TARGET_COL])
dval_nc   = lgb.Dataset(va[FEATURES_NOCAS], va[TARGET_COL], reference=dtrain_nc)
gbm_nc = lgb.train(params, dtrain_nc, num_boost_round=3000, valid_sets=[dval_nc],
                   valid_names=["val"], callbacks=[lgb.early_stopping(100),
                                                   lgb.log_evaluation(200)])

res_nc = evaluate(te[TARGET_COL].to_numpy(), gbm_nc.predict(te[FEATURES_NOCAS]))
ladder = pd.concat([ladder, pd.DataFrame({"LightGBM (no cascade)": res_nc}).T])
order = ["Majority class", "Climatology", "Logistic regression",
         "LightGBM (no cascade)", "LightGBM (+cascade)"]
print(ladder.reindex(order).round(4))

full, nocas = ladder.loc["LightGBM (+cascade)", "PR_AUC"], res_nc["PR_AUC"]
print(f"\nGBM nonlinearity (LR -> no-cascade GBM):  0.4022 -> {nocas:.4f}  (+{nocas-0.4022:.4f})")
print(f"Cascade marginal (no-cascade -> +cascade): {nocas:.4f} -> {full:.4f}  (+{full-nocas:.4f})")
print(f"Cascade = {(full-nocas)/(full-0.4022):.0%} of the LR->full jump; "
      f"+{(full-nocas)/nocas:.0%} over the no-cascade GBM")
# %%
