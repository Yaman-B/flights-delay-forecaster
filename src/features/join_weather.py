"""Join hourly weather onto each flight at BOTH origin and destination.

Usage:
    python src/features/join_weather.py

Inputs:
    data/processed/flights_clean.parquet      (10,889,691 flights, target locked)
    data/raw/weather/all_airports_hourly.parquet   (hourly obs, 40 airports, UTC)

Output:
    data/processed/flights_weather.parquet    (flights + *_orig / *_dest weather)

Design:
  - We join on SCHEDULED times (CRSDepTime / CRSArrTime), never actual times,
    to avoid leaking the lateness we are predicting. Observed weather at the
    scheduled hour stands in for the forecast a live system would use.
  - BTS records clock times in each airport's LOCAL time; weather is in UTC.
    We convert scheduled departure to UTC via the origin airport's IANA
    timezone (which encodes that region's DST rules), then derive scheduled
    arrival in UTC as dep_utc + CRSElapsedTime. This allows for a clean, single
    conversion, no destination timezone lookup and no red-eye midnight-rollover logic.
  - Weather feature columns are auto-detected, so the join does not depend on
    remembering the exact 11 variable names.
"""
from pathlib import Path
import sys
import pandas as pd

FLIGHTS = Path("data/processed/flights_clean.parquet")
WEATHER = Path("data/raw/weather/all_airports_hourly.parquet")
OUT = Path("data/processed/flights_weather.parquet")

# Per-airport IANA timezones for all 40 airports in the dataset.
# America/Phoenix and Pacific/Honolulu do NOT observe daylight saving -- the two
# cases a fixed-offset shortcut would get wrong for part of the year.
AIRPORT_TZ = {
    "ATL": "America/New_York",   "DFW": "America/Chicago",    "DEN": "America/Denver",
    "ORD": "America/Chicago",    "LAX": "America/Los_Angeles","CLT": "America/New_York",
    "MCO": "America/New_York",   "LAS": "America/Los_Angeles","PHX": "America/Phoenix",
    "MIA": "America/New_York",   "SEA": "America/Los_Angeles","IAH": "America/Chicago",
    "JFK": "America/New_York",   "EWR": "America/New_York",   "FLL": "America/New_York",
    "MSP": "America/Chicago",    "SFO": "America/Los_Angeles","DTW": "America/Detroit",
    "BOS": "America/New_York",   "SLC": "America/Denver",     "PHL": "America/New_York",
    "BWI": "America/New_York",   "TPA": "America/New_York",   "SAN": "America/Los_Angeles",
    "LGA": "America/New_York",   "MDW": "America/Chicago",    "BNA": "America/Chicago",
    "IAD": "America/New_York",   "DCA": "America/New_York",   "AUS": "America/Chicago",
    "HNL": "Pacific/Honolulu",   "DAL": "America/Chicago",    "PDX": "America/Los_Angeles",
    "STL": "America/Chicago",    "RDU": "America/New_York",   "HOU": "America/Chicago",
    "SMF": "America/Los_Angeles","MSY": "America/Chicago",    "SJC": "America/Los_Angeles",
    "SNA": "America/Los_Angeles",
}


def hhmm_to_minutes(hhmm: pd.Series) -> pd.Series:
    """HHMM float (1430.0, 905.0, 2400.0) -> minutes since local midnight.
    2400 -> 1440, which rolls into the next day when added as a timedelta."""
    return (hhmm // 100) * 60 + (hhmm % 100)


def to_utc_hour(flight_date: pd.Series, hhmm: pd.Series, airport: pd.Series) -> pd.Series:
    """Local FlightDate + HHMM clock time + airport tz -> tz-naive UTC timestamp.
    DST-aware: spring-forward gap is shifted forward; the ambiguous fall-back
    hour becomes NaT (a few flights/year; they get null weather, reported below)."""
    naive_local = pd.to_datetime(flight_date) + pd.to_timedelta(hhmm_to_minutes(hhmm), unit="m")
    tz = airport.map(AIRPORT_TZ)
    if tz.isna().any():
        bad = sorted(airport[tz.isna()].unique())
        raise KeyError(f"airports missing from AIRPORT_TZ: {bad}")
    out = pd.Series(pd.NaT, index=flight_date.index, dtype="datetime64[ns]")
    for zone, idx in tz.groupby(tz).groups.items():
        local = naive_local.loc[idx]
        out.loc[idx] = (
            local.dt.tz_localize(zone, nonexistent="shift_forward", ambiguous="NaT")
                 .dt.tz_convert("UTC")
                 .dt.tz_localize(None)
        )
    return out


def load_weather() -> tuple[pd.DataFrame, list[str]]:
    wx = pd.read_parquet(WEATHER)
    wx["obs_time_utc"] = pd.to_datetime(wx["obs_time_utc"])
    if wx["obs_time_utc"].dt.tz is not None:                 # normalize to naive UTC
        wx["obs_time_utc"] = wx["obs_time_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    wx["obs_time_utc"] = wx["obs_time_utc"].dt.floor("h")    # weather is already hourly; be safe
    feature_cols = [c for c in wx.columns if c not in ("obs_time_utc", "airport")]
    for c in feature_cols:                                   # halve the memory footprint
        if pd.api.types.is_float_dtype(wx[c]):
            wx[c] = wx[c].astype("float32")
    wx = wx.drop_duplicates(["airport", "obs_time_utc"])
    return wx, feature_cols


def join_side(flights: pd.DataFrame, wx: pd.DataFrame, feats: list[str],
              airport_col: str, hour_col: str, suffix: str) -> pd.DataFrame:
    right = wx.rename(columns={"airport": airport_col, "obs_time_utc": hour_col,
                               **{c: f"{c}{suffix}" for c in feats}})
    return flights.merge(right, on=[airport_col, hour_col], how="left")


def main() -> None:
    for p in (FLIGHTS, WEATHER):
        if not p.exists():
            sys.exit(f"missing input: {p}")

    flights = pd.read_parquet(FLIGHTS)
    n = len(flights)
    print(f"flights in: {n:,}")

    # Scheduled departure -> UTC hour at origin; scheduled arrival -> UTC hour at dest.
    flights["dep_hour_utc"] = to_utc_hour(flights["FlightDate"], flights["CRSDepTime"],
                                          flights["Origin"]).dt.floor("h")
    flights["arr_hour_utc"] = (
        to_utc_hour(flights["FlightDate"], flights["CRSDepTime"], flights["Origin"])
        + pd.to_timedelta(flights["CRSElapsedTime"], unit="m")
    ).dt.floor("h")

    nat_dep = flights["dep_hour_utc"].isna().sum()
    nat_arr = flights["arr_hour_utc"].isna().sum()
    print(f"unresolved departure times (ambiguous DST / missing): {nat_dep:,} ({nat_dep/n:.4%})")
    print(f"unresolved arrival times  (missing CRSElapsedTime):   {nat_arr:,} ({nat_arr/n:.4%})")

    wx, feats = load_weather()
    print(f"weather: {len(wx):,} hourly obs, {wx['airport'].nunique()} airports, "
          f"{len(feats)} features -> {feats}")

    flights = join_side(flights, wx, feats, "Origin", "dep_hour_utc", "_orig")
    flights = join_side(flights, wx, feats, "Dest",  "arr_hour_utc", "_dest")

    # Match-rate report: a probe column from each side (first feature).
    probe = feats[0]
    orig_match = flights[f"{probe}_orig"].notna().mean()
    dest_match = flights[f"{probe}_dest"].notna().mean()
    print(f"origin weather matched: {orig_match:.4%}")
    print(f"dest   weather matched: {dest_match:.4%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    flights.to_parquet(OUT, index=False)
    print(f"rows out: {len(flights):,}  cols: {flights.shape[1]}  -> {OUT}")
    print("EXPECTED: match rates >~99%. If materially lower, STOP and investigate "
          "(timezone gap, weather date coverage, or airport-set mismatch).")


if __name__ == "__main__":
    main()