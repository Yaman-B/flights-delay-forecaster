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
from src.models import temporal_split, evaluate
print("src OK ->", __import__("src").__file__)

# %% --- Cascade validation (ONE-TIME diagnostic; safe to skip on re-runs) ---
# Independently re-derives the cascade to (1) reproduce the EDA's 13%->76% and
# (2) confirm the censoring behaves. The model does NOT use anything from here;
# it builds its features via add_cascade_features in the GBM cell below.
from src.models import wilson_ci
d = pd.read_parquet(DATA_PATH, columns=["Tail_Number", "dep_hour_utc", "arr_hour_utc",
                                        "CRSDepTime", "ArrDelayMinutes", TARGET_COL])
d["dep_hour_utc"] = pd.to_datetime(d["dep_hour_utc"])
d["arr_hour_utc"] = pd.to_datetime(d["arr_hour_utc"])
has_tail = d["Tail_Number"].notna() & (d["Tail_Number"].astype(str).str.strip() != "")
d = d.sort_values(["Tail_Number", "dep_hour_utc", "CRSDepTime"])
g = d.groupby("Tail_Number", sort=False)
prev_arr_delay = g["ArrDelayMinutes"].shift(1)
prev_arr_hour  = g["arr_hour_utc"].shift(1)
gap_h   = (d["dep_hour_utc"] - prev_arr_hour).dt.total_seconds() / 3600.0
gap_min = gap_h * 60.0
linked = has_tail & prev_arr_delay.notna() & (gap_h > 0) & (gap_h <= 8)
inbound_delay_obs = np.minimum(prev_arr_delay, gap_min)
inbound_unlanded  = prev_arr_delay > gap_min
cas = pd.DataFrame(index=d.index)
cas["has_inbound"]       = linked.astype(int)
cas["inbound_delay_obs"] = inbound_delay_obs.where(linked)
cas["inbound_unlanded"]  = (inbound_unlanded & linked).astype(int)
cas["_raw_inbound"]      = prev_arr_delay.where(linked)
cas[TARGET_COL]          = d[TARGET_COL]
cas = cas.sort_index()

base = cas[TARGET_COL].mean()
L = cas[cas["has_inbound"] == 1]
print(f"base {base:.4f} | linked {len(L):,} ({len(L)/len(cas):.1%})\n")
BINS = [-0.001, 15, 30, 60, 120, np.inf]
LAB  = ["on-time 0-15", "15-30", "30-60", "60-120", "120+"]
def buckets(series, title):
    t = L.groupby(pd.cut(series, BINS, labels=LAB), observed=True)[TARGET_COL].agg(rate="mean", n="count")
    t["lift"] = t["rate"] / base
    print(title, "\n", t.round(4), "\n")
buckets(L["_raw_inbound"],      "RAW inbound delay  (EDA reproduction, NOT a model input)")
buckets(L["inbound_delay_obs"], "CENSORED inbound_delay_obs  (leakage-safe, the model input)")
print(f"inbound still airborne at our dep: {L['inbound_unlanded'].mean():.1%} of linked")
print("  disruption | unlanded:", round(L.loc[L.inbound_unlanded == 1, TARGET_COL].mean(), 4))
print("  disruption | landed:  ", round(L.loc[L.inbound_unlanded == 0, TARGET_COL].mean(), 4))

# %% --- LightGBM with the leakage-safe cascade -----------------------------
import lightgbm as lgb
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
                   valid_names=["val"], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

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

# %% --- MLflow: log the two trained runs (no retraining) -------------------
import os, tempfile, mlflow
from src.paths import PROJECT_ROOT

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")   # local SQLite backend
mlflow.set_registry_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("flight-delay-gbm")

def log_run(run_name, model, features, params, test_metrics):
    """Record an already-trained LightGBM run to MLflow: params, test metrics,
    the feature list, gain importances, and the model file itself."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("best_iteration", model.best_iteration)
        mlflow.log_metrics({k.replace("@", "_at_"): float(v) for k, v in test_metrics.items()})
        mlflow.log_dict({"features": list(features)}, "features.json")
        mlflow.log_dict(dict(zip(features, model.feature_importance("gain").tolist())),
                        "feature_importance_gain.json")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.txt")
            model.save_model(path)
            mlflow.log_artifact(path)
    print(f"[{run_name}] logged -> PR-AUC {test_metrics['PR_AUC']:.4f}")

log_run("lightgbm_full_cascade", gbm,    FEATURES,       params, res)
log_run("lightgbm_no_cascade",   gbm_nc, FEATURES_NOCAS, params, res_nc)

# %% --- Calibration: measure -> fit on VAL -> re-measure on test -----------
from sklearn.isotonic import IsotonicRegression

def reliability(y_true, y_prob, n_bins=10):
    """Quantile-binned reliability table + Expected Calibration Error (ECE)."""
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    b = pd.cut(y_prob, edges, include_lowest=True)
    t = (pd.DataFrame({"y": y_true, "p": y_prob, "bin": b})
         .groupby("bin", observed=True)
         .agg(mean_pred=("p", "mean"), obs_freq=("y", "mean"), n=("y", "size")))
    ece = (np.abs(t["mean_pred"] - t["obs_freq"]) * t["n"]).sum() / t["n"].sum()
    return t, ece

y_te = te[TARGET_COL].to_numpy()
p_te = gbm.predict(te[FEATURES])                      # raw probabilities (full model)

tbl_raw, ece_raw = reliability(y_te, p_te)
print(f"RAW: mean predicted {p_te.mean():.4f} vs actual {y_te.mean():.4f} | ECE {ece_raw:.4f}")
print(tbl_raw.round(4), "\n")

# Fit the calibrator on VAL only (2025), then apply to test.
p_va = gbm.predict(va[FEATURES])
iso  = IsotonicRegression(out_of_bounds="clip").fit(p_va, va[TARGET_COL].to_numpy())
p_te_cal = iso.predict(p_te)

tbl_cal, ece_cal = reliability(y_te, p_te_cal)
print(f"CALIBRATED: mean predicted {p_te_cal.mean():.4f} vs actual {y_te.mean():.4f} | ECE {ece_cal:.4f}")
print(tbl_cal.round(4), "\n")

print("raw       :", {k: round(v, 4) for k, v in evaluate(y_te, p_te).items()})
print("calibrated:", {k: round(v, 4) for k, v in evaluate(y_te, p_te_cal).items()})
# %% --- Reliability diagram (before vs after) ------------------------------
import matplotlib.pyplot as plt
from src.paths import FIG_DIR

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect calibration")
ax.plot(tbl_raw["mean_pred"], tbl_raw["obs_freq"], "o-", color="#c0504d", label=f"raw (ECE {ece_raw:.3f})")
ax.plot(tbl_cal["mean_pred"], tbl_cal["obs_freq"], "o-", color="#3b6ea5", label=f"calibrated (ECE {ece_cal:.3f})")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed disruption frequency")
ax.set_title("Reliability: GBM before vs after isotonic calibration")
ax.legend(frameon=False); ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(FIG_DIR / "reliability_calibration.png", dpi=150)
print("saved ->", FIG_DIR / "reliability_calibration.png")
# %% --- Log the calibration run to MLflow ----------------------------------
import os, tempfile, joblib, mlflow
from src.paths import PROJECT_ROOT, FIG_DIR

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")   # same backend as before
mlflow.set_experiment("flight-delay-gbm")

with mlflow.start_run(run_name="isotonic_calibration"):
    mlflow.log_metrics({
        "ece_raw": float(ece_raw), "ece_calibrated": float(ece_cal),
        "brier_raw": float(evaluate(y_te, p_te)["Brier"]),
        "brier_calibrated": float(evaluate(y_te, p_te_cal)["Brier"]),
        "mean_pred_raw": float(p_te.mean()), "mean_pred_cal": float(p_te_cal.mean()),
        "actual_rate": float(y_te.mean()),
    })
    mlflow.log_artifact(str(FIG_DIR / "reliability_calibration.png"))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "calibrator_isotonic.joblib")
        joblib.dump(iso, path)
        mlflow.log_artifact(path)
print("calibration run logged")
# %% --- Threshold tuning: choose the operating point on VAL ----------------
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

# Tune on VALIDATION (2025), raw model (our headline discriminator); report on TEST.
y_va, p_va = va[TARGET_COL].to_numpy(), gbm.predict(va[FEATURES])
y_te, p_te = te[TARGET_COL].to_numpy(), gbm.predict(te[FEATURES])

prec, rec, thr = precision_recall_curve(y_va, p_va)   # thr has one fewer than prec/rec
prec, rec = prec[:-1], rec[:-1]
f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)

t_f1  = float(thr[np.argmax(f1)])                       # threshold maximizing val F1
t_rec = float(thr[np.argmin(np.abs(rec - 0.80))])       # threshold giving ~80% val recall
points = {"default 0.50": 0.50, "max-F1 (val)": t_f1, "high-recall ~80% (val)": t_rec}

def at(t, y, p):
    yhat = (p >= t).astype(int)
    return dict(threshold=t, precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat, zero_division=0),
                f1=f1_score(y, yhat, zero_division=0), flagged_rate=yhat.mean())

tbl = pd.DataFrame({n: at(t, y_te, p_te) for n, t in points.items()}).T
tbl = tbl[["threshold", "precision", "recall", "f1", "flagged_rate"]]
print("Operating points (threshold picked on VAL, metrics on TEST):\n", tbl.round(4))
# %% --- Operating-point curve: precision/recall/F1 vs threshold (val) ------
import matplotlib.pyplot as plt
from src.paths import FIG_DIR

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thr, prec, label="precision", color="#3b6ea5")
ax.plot(thr, rec,  label="recall",    color="#c0504d")
ax.plot(thr, f1,   label="F1",        color="#4f9d4f")
ax.axvline(t_f1, ls="--", color="#666", lw=1)
ax.text(t_f1 + 0.01, 0.04, f"max-F1 @ {t_f1:.2f}", color="#666", fontsize=9)
ax.set_xlabel("Threshold"); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
ax.set_title("Precision / recall / F1 vs threshold (validation)")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG_DIR / "threshold_operating_points.png", dpi=150)
print("saved ->", FIG_DIR / "threshold_operating_points.png")
# %% --- Log threshold tuning to MLflow -------------------------------------
import mlflow
from src.paths import PROJECT_ROOT, FIG_DIR
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("flight-delay-gbm")

best = tbl.loc["max-F1 (val)"]
with mlflow.start_run(run_name="threshold_tuning"):
    mlflow.log_param("tuned_on", "val_2025")
    mlflow.log_param("threshold_max_f1", t_f1)
    mlflow.log_metrics({"test_precision": float(best["precision"]),
                        "test_recall":    float(best["recall"]),
                        "test_f1":        float(best["f1"]),
                        "test_flagged_rate": float(best["flagged_rate"])})
    mlflow.log_artifact(str(FIG_DIR / "threshold_operating_points.png"))
print("threshold tuning logged | max-F1 threshold:", round(t_f1, 4))
# %%
