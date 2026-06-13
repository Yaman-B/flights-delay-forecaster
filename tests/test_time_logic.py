"""Synthetic test of the core time-conversion logic used in the weather join.

The goal is to prove that the local-clock-time -> UTC conversion is correct 
on the cases that silently break: no-DST airports, the spring-forward gap, 
the fall-back ambiguous hour, the '2400' midnight code, 
and red-eye arrival via elapsed-minute addition.
"""
import pandas as pd

# Per-airport IANA timezones (subset used in the test; full map lives in the join script)
AIRPORT_TZ = {
    "BOS": "America/New_York",
    "JFK": "America/New_York",
    "PHX": "America/Phoenix",     # Arizona: NO daylight saving
    "HNL": "Pacific/Honolulu",    # Hawaii:  NO daylight saving
}


def hhmm_to_minutes(hhmm: pd.Series) -> pd.Series:
    """HHMM float (e.g. 1430.0, 905.0, 2400.0) -> minutes since local midnight.
    2400 -> 1440 (rolls into next day naturally when added as a timedelta)."""
    h = (hhmm // 100)
    m = (hhmm % 100)
    return h * 60 + m


def to_utc_naive(flight_date: pd.Series, hhmm: pd.Series, airport: pd.Series) -> pd.Series:
    """Build tz-naive UTC timestamps from local FlightDate + HHMM clock time + airport tz.
    Handles DST transitions explicitly. Returns NaT for the (rare) ambiguous fall-back hour."""
    naive_local = pd.to_datetime(flight_date) + pd.to_timedelta(hhmm_to_minutes(hhmm), unit="m")
    tz = airport.map(AIRPORT_TZ)
    out = pd.Series(pd.NaT, index=flight_date.index, dtype="datetime64[ns]")
    for zone, idx in tz.groupby(tz).groups.items():
        local = naive_local.loc[idx]
        utc = (
            local.dt.tz_localize(zone, nonexistent="shift_forward", ambiguous="NaT")
                 .dt.tz_convert("UTC")
                 .dt.tz_localize(None)            # store as naive UTC for clean joining
        )
        out.loc[idx] = utc
    return out


# ---- Synthetic flights covering every edge case --------------------------------
cases = pd.DataFrame({
    "label": [
        "BOS winter 14:30 (EST, UTC-5)",
        "BOS summer 14:30 (EDT, UTC-4)",
        "PHX summer 14:00 (no DST, UTC-7)",
        "PHX winter 14:00 (no DST, UTC-7)",
        "NY spring-forward 02:30 (does not exist)",
        "NY fall-back 01:30 (ambiguous)",
        "BOS code 2400 (midnight -> next day)",
        "JFK red-eye 23:30 + 360min elapsed",
    ],
    "FlightDate": [
        "2024-01-15", "2024-07-15", "2024-07-15", "2024-01-15",
        "2024-03-10", "2024-11-03", "2024-01-15", "2024-01-15",
    ],
    "Origin": ["BOS", "BOS", "PHX", "PHX", "JFK", "JFK", "BOS", "JFK"],
    "CRSDepTime": [1430.0, 1430.0, 1400.0, 1400.0, 230.0, 130.0, 2400.0, 2330.0],
    "CRSElapsedTime": [0, 0, 0, 0, 0, 0, 0, 360.0],
})

dep_utc = to_utc_naive(cases["FlightDate"], cases["CRSDepTime"], cases["Origin"])
arr_utc = dep_utc + pd.to_timedelta(cases["CRSElapsedTime"], unit="m")

expected_dep = [
    "2024-01-15 19:30", "2024-07-15 18:30", "2024-07-15 21:00", "2024-01-15 21:00",
    "2024-03-10 07:00",  # 02:30 shifted forward to 03:00 EDT -> 07:00 UTC
    "NaT",               # ambiguous fall-back hour
    "2024-01-16 05:00",  # 2400 -> 00:00 next day EST -> 05:00 UTC
    "2024-01-16 04:30",  # 23:30 EST -> 04:30 UTC next day
]

print(f"{'case':<42}{'dep_utc (computed)':<22}{'expected':<18}{'ok'}")
print("-" * 90)
all_ok = True
for i, row in cases.iterrows():
    got = "NaT" if pd.isna(dep_utc[i]) else dep_utc[i].strftime("%Y-%m-%d %H:%M")
    exp = expected_dep[i]
    ok = got.startswith(exp) if exp != "NaT" else got == "NaT"
    all_ok &= ok
    print(f"{row['label']:<42}{got:<22}{exp:<18}{'PASS' if ok else 'FAIL'}")

print("-" * 90)
print("red-eye arrival (JFK 23:30 + 360min):", arr_utc[7], "  expected 2024-01-16 10:30")
all_ok &= (not pd.isna(arr_utc[7])) and arr_utc[7].strftime("%Y-%m-%d %H:%M") == "2024-01-16 10:30"
print("\nALL CHECKS:", "PASS" if all_ok else "FAIL")