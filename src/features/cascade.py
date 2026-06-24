"""Leakage-safe inbound-delay cascade features.

At the prediction cutoff (a flight's SCHEDULED departure) we may not know the
inbound aircraft's final arrival delay. So we use only what's observable then:
the inbound delay right-censored at the scheduled turnaround slack, the slack
itself, the realized spare buffer, and a 'still airborne' flag.
"""
import numpy as np
import pandas as pd

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