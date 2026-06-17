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

# %% --- Scheduled hour of day: delays compound through the day ------------
# Local scheduled departure hour at origin (CRSDepTime is local HHMM).
h = pd.read_parquet(DATA_PATH, columns=["CRSDepTime", TARGET_COL])
base_rate = h[TARGET_COL].mean()

h = h[h["CRSDepTime"].notna()].copy()
h["dep_hour_local"] = (h["CRSDepTime"].astype(int) // 100) % 24   # 2400 -> 0

t = (h.groupby("dep_hour_local")[TARGET_COL]
     .agg(rate="mean", k="sum", n="count")
     .reindex(range(24)))
ci = [wilson_ci(k, n) if pd.notna(n) and n > 0 else (np.nan, np.nan)
      for k, n in zip(t["k"], t["n"])]
t["ci_low"]  = [c[0] for c in ci]
t["ci_high"] = [c[1] for c in ci]
t["lift"]    = t["rate"] / base_rate

pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
print("Disruption by local scheduled departure hour:\n", t)
t.to_csv(TBL_DIR / "disruption_by_hour.csv")

# %% --- Plot: the daily disruption curve (operational day starts ~5am) -----
order = list(range(5, 24)) + list(range(0, 5))   # 5,6,...,23,0,1,2,3,4
to = t.reindex(order)
xpos = np.arange(len(order))

MIN_N = 1000
mask = to["n"].to_numpy() >= MIN_N
rate = np.where(mask, to["rate"].to_numpy()*100, np.nan)
lo   = np.where(mask, to["ci_low"].to_numpy()*100, np.nan)
hi   = np.where(mask, to["ci_high"].to_numpy()*100, np.nan)

fig, ax = plt.subplots(figsize=(11, 6))
ax2 = ax.twinx()
ax2.bar(xpos, to["n"].to_numpy(), color="#dcdcdc", width=0.9, zorder=1)
ax2.set_ylabel("Flights scheduled (volume)", color="#999")
ax2.tick_params(axis="y", colors="#999")
ax2.set_ylim(0, np.nanmax(to["n"].to_numpy()) * 3.2)

ax.plot(xpos, rate, "-o", color="#b5563f", lw=2, ms=4, zorder=3)
ax.fill_between(xpos, lo, hi, color="#b5563f", alpha=0.2, zorder=2)
ax.axhline(base_rate*100, ls="--", lw=1, color="#444", zorder=2)
ax.text(xpos[-1], base_rate*100 + 0.8, f"overall {base_rate*100:.1f}%", ha="right", color="#444")

ax.set_zorder(ax2.get_zorder()+1); ax.patch.set_visible(False)
ax.set_xticks(xpos); ax.set_xticklabels(order)
ax.set_xlim(-0.5, len(order)-0.5)
ax.set_ylim(0, np.nanmax(hi) + 5)
ax.set_xlabel("Scheduled departure hour, local time at origin (operational day)")
ax.set_ylabel("Disruption rate (%)")
ax.set_title("Delays compound through the day: disruption by scheduled departure hour")
ax.spines[["top"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "disruption_by_hour.png", dpi=150)
print("saved ->", FIG_DIR / "disruption_by_hour.png")
# %%
# %% --- Airport effect: disruption by origin airport -----------------------
a = pd.read_parquet(DATA_PATH, columns=["Origin", TARGET_COL])
base_rate = a[TARGET_COL].mean()
ap = (a.groupby("Origin")[TARGET_COL]
      .agg(rate="mean", k="sum", n="count").sort_values("rate"))
ci = [wilson_ci(k, n) for k, n in zip(ap["k"], ap["n"])]
ap["ci_low"]=[c[0] for c in ci]; ap["ci_high"]=[c[1] for c in ci]
ap["lift"]=ap["rate"]/base_rate
print(ap)
ap.to_csv(TBL_DIR / "disruption_by_airport.csv")
# %%
# %% --- Plot: airports ranked by disruption --------------------------------
fig, ax = plt.subplots(figsize=(9, 11))
y = np.arange(len(ap)); rates = ap["rate"].to_numpy()*100
xerr = np.vstack([(ap["rate"]-ap["ci_low"]).to_numpy(), (ap["ci_high"]-ap["rate"]).to_numpy()])*100
colors = ["#b5563f" if r > base_rate else "#3b6ea5" for r in ap["rate"]]
ax.barh(y, rates, xerr=xerr, color=colors, edgecolor="white", zorder=3)
ax.axvline(base_rate*100, ls="--", lw=1, color="#444", zorder=2)
ax.text(base_rate*100, len(ap)-0.3, f" overall {base_rate*100:.1f}%", color="#444", va="top", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(ap.index, fontsize=8)
ax.set_ylim(-0.7, len(ap)-0.3)
ax.set_xlabel("Disruption rate (%)")
ax.set_title("Disruption by origin airport")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "disruption_by_airport.png", dpi=150)
print("saved ->", FIG_DIR / "disruption_by_airport.png")
# %%
# %% --- Congestion mechanism: within-airport scheduled-hour load ----------
b = pd.read_parquet(DATA_PATH, columns=["Origin","FlightDate","CRSDepTime",TARGET_COL])
b = b[b["CRSDepTime"].notna()].copy()
b["hour"] = (b["CRSDepTime"].astype(int)//100) % 24

# Scheduled departures from the same airport, same local hour, same day.
b["load"] = b.groupby(["Origin","FlightDate","hour"])["Origin"].transform("size")
# Normalize within airport (percentile rank) so airport size doesn't dominate.
b["load_pct"] = b.groupby("Origin")["load"].rank(pct=True)
b["load_bin"] = pd.cut(b["load_pct"], [0,0.2,0.4,0.6,0.8,1.0],
                       labels=["quietest 20%","20-40%","40-60%","60-80%","busiest 20%"],
                       include_lowest=True)

lb = (b.groupby("load_bin", observed=True)[TARGET_COL].agg(rate="mean", k="sum", n="count"))
ci = [wilson_ci(k, n) for k, n in zip(lb["k"], lb["n"])]
lb["ci_low"]=[c[0] for c in ci]; lb["ci_high"]=[c[1] for c in ci]
lb["lift"]=lb["rate"]/base_rate
print(lb)
lb.to_csv(TBL_DIR / "disruption_by_load.csv")
# %%
# %% --- Plot: within-airport congestion gradient ---------------------------
fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(lb)); rates = lb["rate"].to_numpy()*100
yerr = np.vstack([(lb["rate"]-lb["ci_low"]).to_numpy(), (lb["ci_high"]-lb["rate"]).to_numpy()])*100
ax.bar(x, rates, 0.6, yerr=yerr, capsize=4, color="#4f7a64", edgecolor="white", zorder=3)
ax.axhline(base_rate*100, ls="--", lw=1, color="#444", zorder=2)
ax.text(len(lb)-0.5, base_rate*100+0.8, f"overall {base_rate*100:.1f}%", ha="right", color="#444")
for xi,(top,n) in enumerate(zip(lb["ci_high"]*100, lb["n"])):
    ax.text(xi, top+0.5, f"{n:,}", ha="center", va="bottom", fontsize=7, color="#555")
ax.set_xticks(x); ax.set_xticklabels(lb.index, rotation=15, ha="right")
ax.set_xlabel("Scheduled-hour load, ranked within each airport")
ax.set_ylabel("Disruption rate (%)")
ax.set_title("Congestion: disruption by within-airport scheduled-hour load")
ax.set_ylim(0, max(rates)+6)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "disruption_by_load.png", dpi=150)
print("saved ->", FIG_DIR / "disruption_by_load.png")
# %%
# %% --- Carrier + seasonality: load once, derive fields --------------------
s = pd.read_parquet(DATA_PATH, columns=["Reporting_Airline", "FlightDate", TARGET_COL])
base_rate = s[TARGET_COL].mean()
fd = pd.to_datetime(s["FlightDate"])
s["dow"]   = fd.dt.dayofweek      # 0=Mon ... 6=Sun
s["month"] = fd.dt.month          # 1..12

def rate_table(by):
    t = s.groupby(by, observed=True)[TARGET_COL].agg(rate="mean", k="sum", n="count")
    ci = [wilson_ci(k, n) for k, n in zip(t["k"], t["n"])]
    t["ci_low"]=[c[0] for c in ci]; t["ci_high"]=[c[1] for c in ci]
    t["lift"]=t["rate"]/base_rate
    return t

def bar_rate(t, labels, title, xlabel, fname, rotate=0):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(t)); rates = t["rate"].to_numpy()*100
    yerr = np.vstack([(t["rate"]-t["ci_low"]).to_numpy(), (t["ci_high"]-t["rate"]).to_numpy()])*100
    colors = ["#b5563f" if r > base_rate else "#3b6ea5" for r in t["rate"]]
    ax.bar(x, rates, 0.7, yerr=yerr, capsize=3, color=colors, edgecolor="white", zorder=3)
    ax.axhline(base_rate*100, ls="--", lw=1, color="#444", zorder=2)
    ax.text(len(t)-0.5, base_rate*100+0.6, f"overall {base_rate*100:.1f}%", ha="right", color="#444")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_xlabel(xlabel); ax.set_ylabel("Disruption rate (%)"); ax.set_title(title)
    ax.set_ylim(0, max(rates)+5)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR/fname, dpi=150)
    print("saved ->", FIG_DIR/fname)

# %% --- Carrier ------------------------------------------------------------
car = rate_table("Reporting_Airline").sort_values("rate")
print(car)
car.to_csv(TBL_DIR / "disruption_by_carrier.csv")
bar_rate(car, car.index, "Disruption by carrier", "Reporting airline",
         "disruption_by_carrier.png", rotate=0)

# %% --- Day of week --------------------------------------------------------
dow = rate_table("dow").reindex(range(7))
print(dow)
dow.to_csv(TBL_DIR / "disruption_by_dow.csv")
bar_rate(dow, ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
         "Disruption by day of week", "Scheduled departure day", "disruption_by_dow.png")

# %% --- Month / seasonality ------------------------------------------------
mon = rate_table("month").reindex(range(1, 13))
print(mon)
mon.to_csv(TBL_DIR / "disruption_by_month.csv")
bar_rate(mon, ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
         "Disruption by month (seasonality)", "Scheduled month", "disruption_by_month.png")
# %%
