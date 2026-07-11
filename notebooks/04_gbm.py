# %% --- Setup --------------------------------------------------------------
import sys, os, tempfile
from pathlib import Path

# Put the repo root on sys.path so `import src` works from notebooks/.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap, joblib, mlflow
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

from src.paths import DATA_PATH, TARGET_COL, FIG_DIR, PROJECT_ROOT
from src.models import temporal_split, evaluate
from src.features.cascade import add_cascade_features, CASCADE_COLS

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_registry_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("flight-delay-gbm")
print("src OK ->", __import__("src").__file__)

# %% --- Cascade validation (one-time diagnostic; not used by the model) ----
# Re-derives the cascade independently to reproduce the EDA 13%->76% pattern and
# confirm censoring. The model builds features via add_cascade_features below.
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
buckets(L["_raw_inbound"],      "RAW inbound delay (EDA reproduction)")
buckets(L["inbound_delay_obs"], "CENSORED inbound_delay_obs (model input)")
print(f"inbound still airborne at our dep: {L['inbound_unlanded'].mean():.1%} of linked")
print("  disruption | unlanded:", round(L.loc[L.inbound_unlanded == 1, TARGET_COL].mean(), 4))
print("  disruption | landed:  ", round(L.loc[L.inbound_unlanded == 0, TARGET_COL].mean(), 4))

# %% --- LightGBM with the leakage-safe cascade -----------------------------
CAT   = ["Origin", "Dest", "Reporting_Airline", "hour", "month", "dow"]
NUM_W = ["snowfall_orig", "snowfall_dest", "wind_gusts_10m_orig",
         "wind_gusts_10m_dest", "temperature_2m_orig"]

cols = list({*CAT[:3], "CRSDepTime", "FlightDate", "Tail_Number", "dep_hour_utc",
             "arr_hour_utc", "ArrDelayMinutes", *NUM_W, TARGET_COL})
m = pd.read_parquet(DATA_PATH, columns=cols)
m["split"] = temporal_split(m["FlightDate"])
fd = pd.to_datetime(m["FlightDate"])
m["hour"], m["month"], m["dow"] = (m["CRSDepTime"].astype(int)//100) % 24, fd.dt.month, fd.dt.dayofweek
m = add_cascade_features(m)
for c in CAT:
    m[c] = m[c].astype("category")          # LightGBM handles category dtype natively

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

# %% --- Ablation: identical GBM without the cascade features ---------------
FEATURES_NOCAS = CAT + NUM_W

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
def log_run(run_name, model, features, params, test_metrics):
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
def reliability(y_true, y_prob, n_bins=10):
    """Quantile-binned reliability table + Expected Calibration Error."""
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    b = pd.cut(y_prob, edges, include_lowest=True)
    t = (pd.DataFrame({"y": y_true, "p": y_prob, "bin": b})
         .groupby("bin", observed=True)
         .agg(mean_pred=("p", "mean"), obs_freq=("y", "mean"), n=("y", "size")))
    ece = (np.abs(t["mean_pred"] - t["obs_freq"]) * t["n"]).sum() / t["n"].sum()
    return t, ece

y_te = te[TARGET_COL].to_numpy()
p_te = gbm.predict(te[FEATURES])

tbl_raw, ece_raw = reliability(y_te, p_te)
print(f"RAW: mean predicted {p_te.mean():.4f} vs actual {y_te.mean():.4f} | ECE {ece_raw:.4f}")
print(tbl_raw.round(4), "\n")

p_va = gbm.predict(va[FEATURES])
iso  = IsotonicRegression(out_of_bounds="clip").fit(p_va, va[TARGET_COL].to_numpy())  # fit on VAL only
p_te_cal = iso.predict(p_te)

tbl_cal, ece_cal = reliability(y_te, p_te_cal)
print(f"CALIBRATED: mean predicted {p_te_cal.mean():.4f} vs actual {y_te.mean():.4f} | ECE {ece_cal:.4f}")
print(tbl_cal.round(4), "\n")

print("raw       :", {k: round(v, 4) for k, v in evaluate(y_te, p_te).items()})
print("calibrated:", {k: round(v, 4) for k, v in evaluate(y_te, p_te_cal).items()})

# %% --- Reliability diagram (before vs after) ------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect calibration")
ax.plot(tbl_raw["mean_pred"], tbl_raw["obs_freq"], "o-", color="#c0504d", label=f"raw (ECE {ece_raw:.3f})")
ax.plot(tbl_cal["mean_pred"], tbl_cal["obs_freq"], "o-", color="#3b6ea5", label=f"calibrated (ECE {ece_cal:.3f})")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed disruption frequency")
ax.set_title("Reliability: GBM before vs after isotonic calibration")
ax.legend(frameon=False); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(FIG_DIR / "reliability_calibration.png", dpi=150)
print("saved ->", FIG_DIR / "reliability_calibration.png")

# %% --- MLflow: log the calibration run ------------------------------------
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
# Tune on VAL (2025); report on TEST.
y_va, p_va = va[TARGET_COL].to_numpy(), gbm.predict(va[FEATURES])
y_te, p_te = te[TARGET_COL].to_numpy(), gbm.predict(te[FEATURES])

prec, rec, thr = precision_recall_curve(y_va, p_va)   # thr is one shorter than prec/rec
prec, rec = prec[:-1], rec[:-1]
f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)

t_f1  = float(thr[np.argmax(f1)])
t_rec = float(thr[np.argmin(np.abs(rec - 0.80))])
points = {"default 0.50": 0.50, "max-F1 (val)": t_f1, "high-recall ~80% (val)": t_rec}

def at(t, y, p):
    yhat = (p >= t).astype(int)
    return dict(threshold=t, precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat, zero_division=0),
                f1=f1_score(y, yhat, zero_division=0), flagged_rate=yhat.mean())

tbl = pd.DataFrame({n: at(t, y_te, p_te) for n, t in points.items()}).T
tbl = tbl[["threshold", "precision", "recall", "f1", "flagged_rate"]]
print("Operating points (threshold on VAL, metrics on TEST):\n", tbl.round(4))

# %% --- Operating-point curve: precision/recall/F1 vs threshold (val) ------
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

# %% --- MLflow: log threshold tuning ---------------------------------------
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

# %% --- SHAP: exact per-flight attributions (TreeSHAP) ---------------------
S = te.sample(5000, random_state=42)
X_sample = S[FEATURES]

contribs  = gbm.predict(X_sample, pred_contrib=True)   # exact TreeSHAP, margin (log-odds) space
shap_vals = contribs[:, :-1]
base_val  = contribs[:, -1]

raw_margin = gbm.predict(X_sample, raw_score=True)
err = float(np.abs(shap_vals.sum(1) + base_val - raw_margin).max())
print("additivity check | max reconstruction error:", err)
print("base margin:", round(float(base_val[0]), 4),
      "-> base prob:", round(float(1/(1+np.exp(-base_val[0]))), 4))

X_plot = X_sample.copy()                                # categoricals -> codes for plot coloring
for c in X_plot.select_dtypes("category").columns:
    X_plot[c] = X_plot[c].cat.codes
expl = shap.Explanation(values=shap_vals, base_values=base_val,
                        data=X_plot.to_numpy(), feature_names=list(FEATURES))

mean_abs = pd.Series(np.abs(shap_vals).mean(0), index=FEATURES).sort_values(ascending=False)
print("\nMean |SHAP| (log-odds units):\n", mean_abs.round(4))

# %% --- SHAP global views: beeswarm + bar ----------------------------------
shap.plots.beeswarm(expl, max_display=16, show=False)
plt.title("SHAP summary — impact on disruption log-odds")
plt.tight_layout(); plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

shap.plots.bar(expl, max_display=16, show=False)
plt.title("Mean |SHAP| per feature")
plt.tight_layout(); plt.savefig(FIG_DIR / "shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()
print("saved -> shap_beeswarm.png, shap_bar.png")

# %% --- SHAP local: why THIS flight? (the Phase 4 LLM input) ---------------
p_sample = gbm.predict(X_sample)
i_hi, i_lo = int(np.argmax(p_sample)), int(np.argmin(p_sample))
for tag, i in [("highrisk", i_hi), ("lowrisk", i_lo)]:
    print(f"{tag}: predicted {p_sample[i]:.3f} | actually disrupted = {int(S[TARGET_COL].iloc[i])}")
    shap.plots.waterfall(expl[i], max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_waterfall_{tag}.png", dpi=150, bbox_inches="tight"); plt.close()
print("saved -> shap_waterfall_highrisk.png, shap_waterfall_lowrisk.png")

# %% --- MLflow: log the SHAP analysis --------------------------------------
with mlflow.start_run(run_name="shap_analysis"):
    mlflow.log_param("shap_sample_size", len(X_sample))
    mlflow.log_dict(mean_abs.round(6).to_dict(), "mean_abs_shap.json")
    for fn in ["shap_beeswarm.png", "shap_bar.png",
               "shap_waterfall_highrisk.png", "shap_waterfall_lowrisk.png"]:
        mlflow.log_artifact(str(FIG_DIR / fn))
print("SHAP analysis logged to MLflow")
# %% --- Test the FlightExplainer handoff -----------------------------
import json
from src.llm.explain import FlightExplainer

explainer = FlightExplainer(
    model=gbm, calibrator=iso, features=FEATURES,
    base_rate=float(m[TARGET_COL].mean()),
    cat_categories={c: list(m[c].cat.categories) for c in CAT},
)
for tag, i in [("HIGH", i_hi), ("LOW", i_lo)]:
    print(f"\n===== {tag}-RISK FLIGHT =====")
    print(json.dumps(explainer.explain(S.iloc[i]), indent=2, default=str))

# %%
# %% --- First LLM explanations -----------------------------------
from anthropic import Anthropic
from src.llm.narrate import narrate

import importlib, src.llm.explain, src.llm.narrate
importlib.reload(src.llm.explain); importlib.reload(src.llm.narrate)
from src.llm.explain import FlightExplainer
from src.llm.narrate import narrate
explainer = FlightExplainer(model=gbm, calibrator=iso, features=FEATURES,
                            base_rate=float(m[TARGET_COL].mean()),
                            cat_categories={c: list(m[c].cat.categories) for c in CAT})

client = Anthropic()
for tag, i in [("HIGH", i_hi), ("LOW", i_lo)]:
    obj = explainer.explain(S.iloc[i])
    print(f"\n===== {tag}-RISK: {obj['flight']['origin']}->{obj['flight']['dest']}, "
          f"p={obj['prediction']['probability']:.3f} ({obj['prediction']['risk_level']}) =====")
    print(narrate(obj, client=client))
# %% --- Persist the model bundle (eval + serving load this, no retraining) -
import json
from src.paths import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
gbm.save_model(str(MODELS_DIR / "gbm.txt"))
joblib.dump(iso, MODELS_DIR / "calibrator_isotonic.joblib")
with open(MODELS_DIR / "config.json", "w") as f:
    json.dump({"features": FEATURES, "cat_features": CAT, "num_features": NUM_W,
               "base_rate": float(m[TARGET_COL].mean()), "threshold": 0.25,
               "cat_categories": {c: list(m[c].cat.categories) for c in CAT}}, f, indent=2)
print("saved model bundle ->", MODELS_DIR)

# %% --- Freeze the stratified 250-flight faithfulness eval set -------------
EVAL_DIR = PROJECT_ROOT / "reports" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Calibrated probability + risk tier for ALL test flights (vectorized).
te_cal = iso.predict(gbm.predict(te[FEATURES]))
tier = np.where(te_cal >= 0.50, "high", np.where(te_cal >= 0.25, "elevated", "low"))
strat = pd.DataFrame({"cal_prob": te_cal, "risk_tier": tier,
                      "has_inbound": te["has_inbound"].to_numpy()}, index=te.index)

# Equal coverage across the 6 (tier x has_inbound) cells -> deliberately
# over-samples the rare hard cells (high-risk, no-inbound) that random
# sampling would miss. The frozen parquet is the reproducible artifact.
N_TOTAL, SEED = 250, 7
cells = strat.groupby(["risk_tier", "has_inbound"]).groups
per_cell = N_TOTAL // len(cells)
picked = []
for key, idx in cells.items():
    take = min(per_cell, len(idx))
    picked.extend(pd.Series(list(idx)).sample(take, random_state=SEED).tolist())
if len(picked) < N_TOTAL:                                   # top up from the rest
    rest = strat.drop(index=picked).sample(N_TOTAL - len(picked), random_state=SEED)
    picked.extend(rest.index.tolist())

eval_idx = pd.Index(picked)
eval_set = m.loc[eval_idx].copy()                           # full rows, category dtypes preserved
eval_set["_cal_prob"]  = strat.loc[eval_idx, "cal_prob"].to_numpy()
eval_set["_risk_tier"] = strat.loc[eval_idx, "risk_tier"].to_numpy()
eval_set.to_parquet(EVAL_DIR / "eval_set.parquet")

print(f"frozen eval set: {len(eval_set)} flights -> {EVAL_DIR / 'eval_set.parquet'}")
print(eval_set.groupby(["_risk_tier", "has_inbound"]).size())
# %%
# %% --- Carve the slim serving slice (test period, features prebuilt) ------
SERVE_DIR = PROJECT_ROOT / "serving_data"
SERVE_DIR.mkdir(exist_ok=True)

keep = FEATURES + ["FlightDate", "Tail_Number", TARGET_COL]
slim = m[m["split"] == "test"][keep].copy()
slim["flight_id"] = slim.index
slim.to_parquet(SERVE_DIR / "flights.parquet", index=False)

mb = (SERVE_DIR / "flights.parquet").stat().st_size / 1e6
print(f"serving slice: {len(slim):,} flights, {mb:.1f} MB -> {SERVE_DIR / 'flights.parquet'}")
print("date range:", slim['FlightDate'].min(), "->", slim['FlightDate'].max())
