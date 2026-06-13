"""Validate the weather join before trusting it downstream.

Usage:
    python src/features/validate_weather_join.py

Four checks:
  1. Coverage      -- match rates and unresolved-time counts.
  2. Cross-airport -- Honolulu never snows; a northern airport does. If the join
                      scrambled airports, this is what catches it.
  3. Temporal      -- Boston's snowiest hours must fall in winter months.
  4. Storm spot    -- Boston's snowiest day in Jan 2024 should be the Jan 6-7
                      nor'easter (winter-storm-warning weekend; that season's
                      standout snow event).
Exit code is nonzero if any hard invariant fails.
"""
from pathlib import Path
import sys
import pandas as pd

OUT = Path("data/processed/flights_weather.parquet")


def find_col(cols, needle, suffix):
    hits = [c for c in cols if needle in c.lower() and c.endswith(suffix)]
    if not hits:
        sys.exit(f"no '{needle}{suffix}' column found; got {sorted(cols)[:20]}...")
    return hits[0]


def main() -> None:
    if not OUT.exists():
        sys.exit(f"missing {OUT} -- run join_weather.py first")
    df = pd.read_parquet(OUT)
    n = len(df)
    snow_o = find_col(df.columns, "snow", "_orig")
    print(f"rows: {n:,}   using origin snowfall column: {snow_o}")

    ok = True

    # --- 1. Coverage -----------------------------------------------------------
    for side, col in (("origin", snow_o), ("dest", find_col(df.columns, "snow", "_dest"))):
        rate = df[col].notna().mean()
        flag = "OK" if rate > 0.99 else "LOW"
        ok &= rate > 0.99
        print(f"[coverage] {side:<6} weather matched: {rate:.4%}  [{flag}]")

    # --- 2. Cross-airport: Honolulu vs a snowy northern hub --------------------
    by_airport = df.groupby("Origin")[snow_o].max()
    hnl = by_airport.get("HNL", 0.0)
    north = {a: by_airport.get(a, float("nan")) for a in ("MSP", "DEN", "BOS", "ORD")}
    print(f"[cross-air] HNL max origin snowfall: {hnl:.3f} (expect ~0)")
    print(f"[cross-air] northern hubs max snowfall: "
          + ", ".join(f"{a}={v:.2f}" for a, v in north.items()))
    hnl_ok = hnl < 0.5
    north_ok = max(v for v in north.values() if v == v) > 1.0
    ok &= hnl_ok and north_ok
    print(f"[cross-air] Honolulu-never-snows: {'OK' if hnl_ok else 'FAIL'}; "
          f"northern-hubs-do: {'OK' if north_ok else 'FAIL'}")

    # --- 3. Temporal: snow concentrates in the cold season and NEVER in summer.
    #        The summer-zero check is the real invariant -- a scrambled-timestamp
    #        join would smear snow across all months, so snow on a Jun-Aug Boston
    #        flight would betray it. Dec-Feb alone is too narrow to gate on:
    #        March (and some November) are genuine Boston snow months, so we
    #        report the cold-season fractions for context but only HARD-FAIL on
    #        snow appearing in summer.
    bos_all = df[df["Origin"] == "BOS"].dropna(subset=[snow_o, "dep_hour_utc"]).copy()
    bos_all["dep_dt"] = pd.to_datetime(bos_all["dep_hour_utc"])
    snowing = bos_all[bos_all[snow_o] > 0]    # only hours where it actually snowed
    top = snowing.assign(month=snowing["dep_dt"].dt.month).nlargest(
        min(2000, len(snowing)), snow_o)
    djf_frac = top["month"].isin([12, 1, 2]).mean()
    cold_frac = top["month"].isin([11, 12, 1, 2, 3]).mean()
    summer_frac = top["month"].isin([6, 7, 8]).mean()
    temp_ok = summer_frac < 0.005            # essentially no snowy summer flight-hours
    ok &= temp_ok
    print(f"[temporal] BOS snowiest 2000 hours: {djf_frac:.1%} Dec-Feb, "
          f"{cold_frac:.1%} Nov-Mar, {summer_frac:.2%} Jun-Aug")
    print(f"[temporal] no-snow-in-summer: {'OK' if temp_ok else 'FAIL'} "
          f"(summer fraction must be ~0)")

    # --- 4. Storm spot-check: Jan 2024 Boston nor'easter -----------------------
    # Mean over ALL Boston flight-hours that day (zeros included), so the metric
    # matches what was validated previously and a quiet day can't spike on a
    # single snowy hour.
    jan = bos_all[(bos_all["dep_dt"].dt.year == 2024) &
                  (bos_all["dep_dt"].dt.month == 1)].copy()
    jan["day"] = jan["dep_dt"].dt.date
    daily = jan.groupby("day")[snow_o].mean().sort_values(ascending=False)
    print("[storm] Boston Jan-2024 snowiest days (mean origin snowfall):")
    for day, val in daily.head(3).items():
        print(f"          {day}: {val:.3f}")
    snowiest = str(daily.index[0]) if len(daily) else "n/a"
    storm_ok = snowiest in ("2024-01-06", "2024-01-07")
    print(f"[storm] snowiest day = {snowiest}; expected 2024-01-06/07  "
          f"[{'OK' if storm_ok else 'CHECK'}]")
    # Not a hard failure -- a snow-poor winter can shuffle a quiet day in -- but
    # if this is not the Jan 6-7 weekend, eyeball it before proceeding.

    print("\nHARD INVARIANTS:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()