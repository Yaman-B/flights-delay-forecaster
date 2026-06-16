"""
02_eda_nonweather.py
====================
Non-weather EDA for the flight-delay project. Starts with the tail-number
cascade: does an aircraft's inbound delay (the arrival delay of its previous
leg) propagate into its next flight? Later cells (time-of-day, airport
congestion, carrier) will be appended here.

Run from anywhere in the repo:
    python notebooks/02_eda_nonweather.py
Also notebook-friendly via the # %% cell markers.
"""

# %% --- Imports & project paths --------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def find_project_root(markers=("requirements.txt", ".git")):
    """Walk up from the CWD to the repo root so paths resolve whether this runs
    from the repo root or from inside notebooks/."""
    here = Path.cwd().resolve()
    for parent in (here, *here.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    raise FileNotFoundError(f"Project root not found above {here}")

PROJECT_ROOT = find_project_root()
print("Project root:", PROJECT_ROOT)

DATA_PATH  = PROJECT_ROOT / "data" / "processed" / "flights_weather.parquet"
TARGET_COL = "disrupted"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TBL_DIR = PROJECT_ROOT / "reports" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)


# %% --- Shared helper -------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    """95% CI for a proportion (k of n); stable for small n and extreme rates."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / den
    h = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / den
    return (c - h, c + h)


# %% --- Load the columns the cascade needs ---------------------------------
cols = ["Tail_Number", "dep_hour_utc", "arr_hour_utc", "CRSDepTime",
        "ArrDelayMinutes", TARGET_COL]
d = pd.read_parquet(DATA_PATH, columns=cols)
base_rate = d[TARGET_COL].mean()
print(f"Base rate: {base_rate:.4f}  |  rows: {len(d):,}")

# Can't link flights without a tail number.
d = d[d["Tail_Number"].notna() & (d["Tail_Number"].astype(str).str.strip() != "")].copy()

# UTC hour columns -> datetimes for correct chronological ordering.
d["dep_hour_utc"] = pd.to_datetime(d["dep_hour_utc"])
d["arr_hour_utc"] = pd.to_datetime(d["arr_hour_utc"])


# %% --- Build the inbound-delay feature ------------------------------------
# Order each aircraft's flights in true time, then read off the previous leg.
d = d.sort_values(["Tail_Number", "dep_hour_utc", "CRSDepTime"])
g = d.groupby("Tail_Number", sort=False)
d["prev_arr_delay"] = g["ArrDelayMinutes"].shift(1)              # inbound delay (min)
d["prev_arr_hour"]  = g["arr_hour_utc"].shift(1)
d["gap_h"] = (d["dep_hour_utc"] - d["prev_arr_hour"]).dt.total_seconds() / 3600

print("\nTurnaround gap (hours) — sanity check (should peak ~1-3h):")
print(d["gap_h"].describe(percentiles=[.1, .25, .5, .75, .9]).round(2))

# Keep genuine intraday turnarounds: a real inbound leg, gap within a day.
turns = d[d["prev_arr_delay"].notna() & d["gap_h"].between(0, 8)].copy()
print(f"\nLinked quick-turn flights: {len(turns):,}")


# %% --- Disruption rate vs inbound delay -----------------------------------
BINS   = [-np.inf, 0, 15, 30, 60, 120, np.inf]
LABELS = ["on-time (0)", "1-15", "15-30", "30-60", "60-120", "120+"]
turns["inbound_bucket"] = pd.cut(turns["prev_arr_delay"], BINS, labels=LABELS)

t = (turns.groupby("inbound_bucket", observed=True)[TARGET_COL]
     .agg(rate="mean", k="sum", n="count"))
ci = [wilson_ci(k, n) for k, n in zip(t["k"], t["n"])]
t["ci_low"]  = [c[0] for c in ci]
t["ci_high"] = [c[1] for c in ci]
t["lift"]    = t["rate"] / base_rate

pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
print("\nINBOUND DELAY -> THIS FLIGHT'S DISRUPTION:\n", t)
t.to_csv(TBL_DIR / "cascade_inbound_delay.csv")


# %% --- Bar chart: the cascade ---------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(t))
rates = t["rate"].to_numpy() * 100
yerr = np.vstack([(t["rate"] - t["ci_low"]).to_numpy(),
                  (t["ci_high"] - t["rate"]).to_numpy()]) * 100
ax.bar(x, rates, 0.6, yerr=yerr, capsize=4, color="#b5563f", edgecolor="white", zorder=3)
ax.axhline(base_rate * 100, ls="--", lw=1, color="#444", zorder=2)
ax.text(len(t) - 0.5, base_rate * 100 + 1.5, f"overall {base_rate*100:.1f}%",
        ha="right", color="#444")
for xi, (top, n) in enumerate(zip(t["ci_high"] * 100, t["n"])):
    ax.text(xi, top + 1.0, f"{n:,}", ha="center", va="bottom", fontsize=7, color="#555")
ax.set_xticks(x); ax.set_xticklabels(t.index, rotation=15, ha="right")
ax.set_xlabel("Inbound delay: arrival delay of the same aircraft's previous leg (min)")
ax.set_ylabel("This flight's disruption rate (%)")
ax.set_title("The tail-number cascade: inbound delay propagates to the next flight")
ax.set_ylim(0, max(rates) + 12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "cascade_inbound_delay.png", dpi=150)
print("saved ->", FIG_DIR / "cascade_inbound_delay.png")