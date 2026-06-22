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
# %% --- Leakage-safe inbound-delay cascade feature -------------------------
import numpy as np, pandas as pd
from src.paths import DATA_PATH, TARGET_COL

d = pd.read_parquet(DATA_PATH, columns=["Tail_Number", "dep_hour_utc", "arr_hour_utc",
                                        "CRSDepTime", "ArrDelayMinutes", TARGET_COL])
d["dep_hour_utc"] = pd.to_datetime(d["dep_hour_utc"])
d["arr_hour_utc"] = pd.to_datetime(d["arr_hour_utc"])

# Chain each aircraft's legs in true (UTC) chronological order, then read the
# previous leg. (Sorting on the UTC hour is what makes this correct across
# timezones; CRSDepTime breaks the rare same-hour tie.)
has_tail = d["Tail_Number"].notna() & (d["Tail_Number"].astype(str).str.strip() != "")
d = d.sort_values(["Tail_Number", "dep_hour_utc", "CRSDepTime"])
g = d.groupby("Tail_Number", sort=False)
prev_arr_delay = g["ArrDelayMinutes"].shift(1)   # inbound's REALIZED arrival delay (min)
prev_arr_hour  = g["arr_hour_utc"].shift(1)       # inbound's SCHEDULED arrival (UTC)

# Scheduled turnaround slack (fully known at prediction time).
gap_h   = (d["dep_hour_utc"] - prev_arr_hour).dt.total_seconds() / 3600.0
gap_min = gap_h * 60.0

# Genuine intraday turnaround: same aircraft, 0 < slack <= 8h (overnight resets
# absorb any inbound delay, so they carry no cascade signal).
linked = has_tail & prev_arr_delay.notna() & (gap_h > 0) & (gap_h <= 8)

# --- The leakage-safe transform: observe only what's known at scheduled dep --
inbound_delay_obs = np.minimum(prev_arr_delay, gap_min)   # right-censored at the slack
inbound_unlanded  = prev_arr_delay > gap_min              # plane still airborne at our dep

cas = pd.DataFrame(index=d.index)
cas["has_inbound"]       = linked.astype(int)
cas["inbound_gap_h"]     = gap_h.where(linked)              # NaN if no inbound -> GBM handles
cas["inbound_delay_obs"] = inbound_delay_obs.where(linked)  # the leakage-safe cascade feature
cas["inbound_unlanded"]  = (inbound_unlanded & linked).astype(int)
cas["_raw_inbound"]      = prev_arr_delay.where(linked)     # VALIDATION ONLY -- never a model input
cas[TARGET_COL]          = d[TARGET_COL]
cas = cas.sort_index()                                      # restore original row order

print(f"linked flights (usable inbound): {cas['has_inbound'].mean():.1%}")
print(cas[["inbound_gap_h", "inbound_delay_obs"]].describe().round(2))

# %% --- Validate the cascade feature ---------------------------------------
from src.models import wilson_ci
base = cas[TARGET_COL].mean()
L = cas[cas["has_inbound"] == 1]
print(f"base {base:.4f} | linked {len(L):,} ({len(L)/len(cas):.1%})\n")

BINS = [-0.001, 15, 30, 60, 120, np.inf]
LAB  = ["on-time 0-15", "15-30", "30-60", "60-120", "120+"]

def buckets(series, title):
    t = L.groupby(pd.cut(series, BINS, labels=LAB), observed=True)[TARGET_COL].agg(rate="mean", n="count")
    t["lift"] = t["rate"] / base
    print(title, "\n", t.round(4), "\n")

buckets(L["_raw_inbound"],       "RAW inbound delay  (EDA reproduction, NOT a model input)")
buckets(L["inbound_delay_obs"],  "CENSORED inbound_delay_obs  (leakage-safe, the model input)")

print(f"inbound still airborne at our dep: {L['inbound_unlanded'].mean():.1%} of linked")
print("  disruption | unlanded:", round(L.loc[L.inbound_unlanded == 1, TARGET_COL].mean(), 4))
print("  disruption | landed:  ", round(L.loc[L.inbound_unlanded == 0, TARGET_COL].mean(), 4))

