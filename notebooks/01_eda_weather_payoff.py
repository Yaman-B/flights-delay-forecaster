"""
01_eda_weather_payoff.py
========================
First EDA figure.
Investigates the question: does ORIGIN snowfall predict flight disruption?
We expect the disruption rate to climb as origin snowfall increases.

Run from the repo root:
    python notebooks/01_eda_weather_payoff.py

"""

# %% --- Imports -------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# %% --- Config ------------------------------------------------------------
from pathlib import Path

def find_project_root(markers=("requirements.txt", ".git")):
    """Walk up from the current working directory until a folder containing a
    project marker is found, so paths resolve whether the script runs from the
    repo root or from inside notebooks/."""
    here = Path.cwd().resolve()
    for parent in (here, *here.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    raise FileNotFoundError(f"Project root not found above {here}")

PROJECT_ROOT = find_project_root()
print("Project root:", PROJECT_ROOT)

DATA_PATH  = PROJECT_ROOT / "data" / "processed" / "flights_weather.parquet"
TARGET_COL = "disrupted"
SNOW_COL   = "snowfall_orig"

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TBL_DIR = PROJECT_ROOT / "reports" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)
# %% --- Load the columns from parquet --------------------------------------
df = pd.read_parquet(DATA_PATH, columns=[SNOW_COL, TARGET_COL])
print(f"Loaded {len(df):,} flights, columns = {list(df.columns)}")

# Guardrails in case a column name differs from the assumption above.
for col in (SNOW_COL, TARGET_COL):
    if col not in df.columns:
        raise KeyError(
            f"Column '{col}' not found. Available columns in the parquet: "
            f"{pd.read_parquet(DATA_PATH).columns.tolist()}"
        )

# %% --- Sanity check: overall disruption rate should be ~0.2353 ------------
base_rate = df[TARGET_COL].mean()
print(f"Overall disruption rate (base rate): {base_rate:.4f}")

# %% --- How much snow do flights actually see? (expected heavy zero-inflation)
share_zero = (df[SNOW_COL] == 0).mean()
share_pos = (df[SNOW_COL] > 0).mean()
print(f"Flights with zero origin snowfall: {share_zero:.3%}")
print(f"Flights with some origin snowfall: {share_pos:.3%}")
print("Snowfall summary (cm):")
print(df.loc[df[SNOW_COL] > 0, SNOW_COL].describe())  # describe the >0 tail
print("Max snowfall (cm):", df[SNOW_COL].max())

# %% --- Bin snowfall into interpretable buckets ----------------------------
# Zero gets its own bucket. The positive
# bins are in cm of snow per hour: even ~1 cm/hr is meaningful active snowfall.
bins = [-0.01, 0.0, 0.5, 1.0, 2.0, 4.0, np.inf]
labels = ["0 (none)", "0-0.5", "0.5-1", "1-2", "2-4", "4+"]
# pd.cut with a left edge of -0.01 makes the first interval (-0.01, 0.0] capture
# exactly snowfall == 0; (0.0, 0.5] is the first "light snow" bucket, etc.
df["snow_bucket"] = pd.cut(df[SNOW_COL], bins=bins, labels=labels)


# %% --- Disruption rate + 95% Wilson confidence interval per bucket ---------
def wilson_ci(k, n, z=1.96):
    """95% CI for a proportion (k successes of n). Wilson is well-behaved for
    small n and for rates near 0 or 1, unlike the naive normal interval."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


g = (
    df.groupby("snow_bucket", observed=True)[TARGET_COL]
    .agg(disruption_rate="mean", n_disrupted="sum", n_flights="count")
)
ci = [wilson_ci(k, n) for k, n in zip(g["n_disrupted"], g["n_flights"])]
g["ci_low"] = [c[0] for c in ci]
g["ci_high"] = [c[1] for c in ci]
g["lift_vs_base"] = g["disruption_rate"] / base_rate  # X times the overall rate

pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
print("\nDisruption by origin-snowfall bucket:")
print(g)
g.to_csv(TBL_DIR / "disruption_by_snowfall_orig.csv")

# %% --- Plot ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(g))
rates = g["disruption_rate"].to_numpy() * 100
yerr = np.vstack([
    (g["disruption_rate"] - g["ci_low"]).to_numpy(),
    (g["ci_high"] - g["disruption_rate"]).to_numpy(),
]) * 100

ax.bar(x, rates, yerr=yerr, capsize=4, color="#3b6ea5", edgecolor="white", zorder=3)
ax.axhline(base_rate * 100, ls="--", lw=1, color="#444", zorder=2)
ax.text(len(g) - 0.5, base_rate * 100 + 0.8,
        f"overall {base_rate * 100:.1f}%", ha="right", color="#444")

# Label each bar with its flight count
tops = g["ci_high"].to_numpy() * 100
for xi, (top, n) in enumerate(zip(tops, g["n_flights"])):
    ax.text(xi, top + 0.7, f"n={n:,}", ha="center", va="bottom",
            fontsize=8, color="#666")

ax.set_xticks(x)
ax.set_xticklabels(g.index)
ax.set_xlabel("Origin snowfall in scheduled departure hour (cm)")
ax.set_ylabel("Disruption rate (%)")
ax.set_title("Flight disruption rises sharply with origin snowfall")
ax.set_ylim(0, max(tops) + 6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "disruption_vs_snowfall_orig.png", dpi=150)
print(f"\nSaved figure -> {FIG_DIR / 'disruption_vs_snowfall_orig.png'}")
print(f"Saved table  -> {TBL_DIR / 'disruption_by_snowfall_orig.csv'}")
# %% --- Investigating the 4+ cm/hr bucket --------------------------------------
# The 4+ cm/hr bucket is the most interesting: it has a very low disruption but 
# a very small sample size. Below we investigate deeper.
import pyarrow.parquet as pq

# 1) See what columns actually exist (cheap: reads schema only, no data).
all_cols = pq.read_schema(DATA_PATH).names
print("Available columns:\n", all_cols, "\n")

# 2) Pick the columns useful for the post-mortem, keeping only ones that exist.
def present(cands):
    return [c for c in cands if c in all_cols]

origin_cands = ["Origin", "origin", "origin_airport", "OriginAirport"]
dest_cands   = ["Dest", "dest", "destination", "dest_airport", "DestAirport"]
date_cands   = ["FlightDate", "flight_date", "date", "fl_date"]
canc_cands   = ["Cancelled", "cancelled", "Canceled", "canceled"]
late_cands   = ["ArrDel15", "arr_del15", "arrdel15"]

cols = list(dict.fromkeys(
    [SNOW_COL, TARGET_COL]
    + present(origin_cands) + present(dest_cands) + present(date_cands)
    + present(canc_cands) + present(late_cands)
))
print("Using columns:", cols, "\n")

# 3) Predicate pushdown: only rows with snowfall_orig > 4.0 are read off disk.
#    (The "4+" bucket is (4.0, inf] from pd.cut, i.e. strictly greater than 4.)
heavy = pd.read_parquet(DATA_PATH, columns=cols, filters=[(SNOW_COL, ">", 4.0)])
print(f"Heavy-snow (>4 cm) flights pulled: {len(heavy)}  (expected ~36)\n")

# 4) The key concentration checks.
def col(cands):  # first existing column name from a candidate list
    for c in cands:
        if c in heavy.columns:
            return c
    return None

oc, dc = col(origin_cands), col(date_cands)

if oc:
    print("By ORIGIN airport (is it one airport?):")
    print(heavy[oc].value_counts(), "\n")
if dc:
    print("By DATE (is it one storm?):")
    print(heavy[dc].value_counts().sort_index(), "\n")
if oc and dc:
    print("By (airport, date) — concentration of single events:")
    print(heavy.groupby([oc, dc]).size().sort_values(ascending=False).head(15), "\n")

# 5) Outcome breakdown: of the disrupted ones, cancellations vs lateness?
print(f"Disrupted: {int(heavy[TARGET_COL].sum())} of {len(heavy)} "
      f"({heavy[TARGET_COL].mean():.1%})")
cc, lc = col(canc_cands), col(late_cands)
if cc:
    print(f"  cancelled: {int(heavy[cc].sum())}")
if lc:
    print(f"  arrived 15+ late: {int(heavy[lc].sum())}")

# 6) Full listing 
show = [c for c in [SNOW_COL, oc, dc, dc and "", col(dest_cands), TARGET_COL, cc, lc] if c]
pd.set_option("display.max_rows", 60)
pd.set_option("display.width", 200)
print("\nFull listing (sorted by snowfall):")
print(heavy.sort_values(SNOW_COL, ascending=False)[ [c for c in show if c in heavy.columns] ].to_string(index=False))

# %% ---- Final Figure (with destination snowfall too) --------------------
# # Decision from the above analysis:
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# Reload the 3 columns we need (earlier df held only origin snow + target).
d = pd.read_parquet(DATA_PATH, columns=["snowfall_orig", "snowfall_dest", "disrupted"])
base_rate = d["disrupted"].mean()
print(f"Base rate: {base_rate:.4f}  |  rows: {len(d):,}")

# Shared, cleaned bins (2-4 and 4+ merged into '2+').
BINS   = [-0.01, 0.0, 0.5, 1.0, 2.0, np.inf]
LABELS = ["0 (none)", "0-0.5", "0.5-1", "1-2", "2+"]

def wilson_ci(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; denom = 1 + z**2/n
    c = (p + z**2/(2*n)) / denom
    h = z*np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (c - h, c + h)

def snow_table(snow, target):
    """Disruption rate, count, 95% Wilson CI, lift per snowfall bucket."""
    t = (pd.DataFrame({"target": target, "bucket": pd.cut(snow, BINS, labels=LABELS)})
         .groupby("bucket", observed=True)["target"].agg(rate="mean", k="sum", n="count"))
    ci = [wilson_ci(k, n) for k, n in zip(t["k"], t["n"])]
    t["ci_low"]  = [c[0] for c in ci]
    t["ci_high"] = [c[1] for c in ci]
    t["lift"]    = t["rate"] / base_rate
    return t

orig = snow_table(d["snowfall_orig"], d["disrupted"])
dest = snow_table(d["snowfall_dest"], d["disrupted"])

compare = pd.DataFrame({
    "orig_rate": orig["rate"], "orig_n": orig["n"], "orig_lift": orig["lift"],
    "dest_rate": dest["rate"], "dest_n": dest["n"], "dest_lift": dest["lift"],
})
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
print("\nORIGIN vs DESTINATION:\n", compare)
compare.to_csv(TBL_DIR / "disruption_orig_vs_dest_snowfall.csv")
# 
# %% --- Grouped bar chart: origin vs destination on identical bins ---------
fig, ax = plt.subplots(figsize=(10, 6))
x, w = np.arange(len(LABELS)), 0.38

def draw(x_off, tbl, color, label):
    rates = tbl["rate"].to_numpy() * 100
    yerr = np.vstack([(tbl["rate"]-tbl["ci_low"]).to_numpy(),
                      (tbl["ci_high"]-tbl["rate"]).to_numpy()]) * 100
    ax.bar(x + x_off, rates, w, yerr=yerr, capsize=3, color=color,
           edgecolor="white", label=label, zorder=3)
    for xi, (top, n) in enumerate(zip(tbl["ci_high"]*100, tbl["n"])):
        ax.text(x[xi]+x_off, top+1.0, f"{n:,}", ha="center", va="bottom",
                fontsize=6.5, color="#555")

draw(-w/2, orig, "#3b6ea5", "Origin snowfall")
draw(+w/2, dest, "#d08a3e", "Destination snowfall")
ax.axhline(base_rate*100, ls="--", lw=1, color="#444", zorder=2)
ax.text(len(LABELS)-0.4, base_rate*100+1.2, f"overall {base_rate*100:.1f}%",
        ha="right", color="#444")
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_xlabel("Snowfall in scheduled hour (cm) — departure hour (origin) / arrival hour (destination)")
ax.set_ylabel("Disruption rate (%)")
ax.set_title("Disruption vs snowfall: origin vs destination")
ax.legend(frameon=False); ax.set_ylim(0, 95)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "disruption_vs_snowfall_orig_vs_dest.png", dpi=150)
print("saved ->", FIG_DIR / "disruption_vs_snowfall_orig_vs_dest.png")

# %% --- Wind gusts vs disruption: origin vs destination -------------------

GUST_O, GUST_D = "wind_gusts_10m_orig", "wind_gusts_10m_dest"
d = pd.read_parquet(DATA_PATH, columns=[GUST_O, GUST_D, "disrupted"])
base_rate = d["disrupted"].mean()
print(f"Base rate: {base_rate:.4f}  |  rows: {len(d):,}\n")

# Units + distribution check (gusts should be km/h; median ~tens, tail ~100+).
print("Origin gust quantiles (km/h):")
print(d[GUST_O].describe(percentiles=[.5, .9, .99, .999]).round(1))
print("\nDestination gust quantiles (km/h):")
print(d[GUST_D].describe(percentiles=[.5, .9, .99, .999]).round(1))

# Gust bins (km/h). ~50-65+ is where crosswind limits start to bite.
BINS   = [-0.1, 20, 35, 50, 65, np.inf]
LABELS = ["0-20", "20-35", "35-50", "50-65", "65+"]

def wilson_ci(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k/n; den = 1 + z**2/n
    c = (p + z**2/(2*n))/den
    h = z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))/den
    return (c-h, c+h)

def binned_table(values, target, bins, labels):
    t = (pd.DataFrame({"y": target, "b": pd.cut(values, bins, labels=labels)})
         .groupby("b", observed=True)["y"].agg(rate="mean", k="sum", n="count"))
    ci = [wilson_ci(k, n) for k, n in zip(t["k"], t["n"])]
    t["ci_low"]  = [c[0] for c in ci]
    t["ci_high"] = [c[1] for c in ci]
    t["lift"] = t["rate"]/base_rate
    return t

og = binned_table(d[GUST_O], d["disrupted"], BINS, LABELS)
dg = binned_table(d[GUST_D], d["disrupted"], BINS, LABELS)
compare = pd.DataFrame({
    "orig_rate": og["rate"], "orig_n": og["n"], "orig_lift": og["lift"],
    "dest_rate": dg["rate"], "dest_n": dg["n"], "dest_lift": dg["lift"],
})
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
print("\nGUSTS — ORIGIN vs DESTINATION:\n", compare)
compare.to_csv(TBL_DIR / "disruption_gusts_orig_vs_dest.csv")

# %% --- Grouped bar: gust disruption, origin vs destination ---------------
fig, ax = plt.subplots(figsize=(10, 6))
x, w = np.arange(len(LABELS)), 0.38
def draw(off, tbl, color, label):
    rates = tbl["rate"].to_numpy()*100
    yerr = np.vstack([(tbl["rate"]-tbl["ci_low"]).to_numpy(),
                      (tbl["ci_high"]-tbl["rate"]).to_numpy()])*100
    ax.bar(x+off, rates, w, yerr=yerr, capsize=3, color=color,
           edgecolor="white", label=label, zorder=3)
    for xi, (top, n) in enumerate(zip(tbl["ci_high"]*100, tbl["n"])):
        ax.text(x[xi]+off, top+1.0, f"{n:,}", ha="center", va="bottom",
                fontsize=6.5, color="#555")
draw(-w/2, og, "#3b6ea5", "Origin gusts")
draw(+w/2, dg, "#7a9e54", "Destination gusts")
ax.axhline(base_rate*100, ls="--", lw=1, color="#444", zorder=2)
ax.text(len(LABELS)-0.4, base_rate*100+1.2, f"overall {base_rate*100:.1f}%", ha="right", color="#444")
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_xlabel("Peak wind gust in scheduled hour (km/h) — departure (origin) / arrival (destination)")
ax.set_ylabel("Disruption rate (%)")
ax.set_title("Disruption vs wind gusts: origin vs destination")
ax.legend(frameon=False)
ymax = max(og["ci_high"].max(), dg["ci_high"].max())*100
ax.set_ylim(0, ymax + 8)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "disruption_vs_gusts_orig_vs_dest.png", dpi=150)
print("saved ->", FIG_DIR / "disruption_vs_gusts_orig_vs_dest.png")


