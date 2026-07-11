"""Consolidate raw BTS monthly zips into one tidy parquet.

Usage:
    python src/data/build_flights.py

Reads every zip in data/raw/bts/, keeps ~29 relevant columns, filters to
flights where BOTH origin and destination are in the top-40 airport set
(so weather features exist at both ends), and writes a single
data/processed/flights.parquet file.

Ends by printing the summary stats needed for the target-variable decision
(delay rate, cancellation rate, diversion rate).
"""

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = Path("data/raw/bts")
OUT_PATH = Path("data/processed/flights.parquet")

AIRPORTS = {
    "ATL", "DFW", "DEN", "ORD", "LAX", "CLT", "MCO", "LAS", "PHX", "MIA",
    "SEA", "IAH", "JFK", "EWR", "FLL", "MSP", "SFO", "DTW", "BOS", "SLC",
    "PHL", "BWI", "TPA", "SAN", "LGA", "MDW", "BNA", "IAD", "DCA", "AUS",
    "HNL", "DAL", "PDX", "STL", "RDU", "HOU", "SMF", "MSY", "SJC", "SNA",
}

# canonical BTS column names, kept as-is on output
WANTED = [
    "FlightDate", "Reporting_Airline", "Tail_Number",
    "Flight_Number_Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "DepTime", "DepDelayMinutes", "DepDel15",
    "TaxiOut", "TaxiIn", "WheelsOff", "WheelsOn",
    "CRSArrTime", "ArrTime", "ArrDelayMinutes", "ArrDel15",
    "Cancelled", "CancellationCode", "Diverted",
    "CRSElapsedTime", "ActualElapsedTime", "Distance",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay",
]

STR_COLS = {"FlightDate", "Reporting_Airline", "Tail_Number", "Origin", "Dest",
            "CancellationCode"}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map actual header names to canonical ones (case/underscore-insensitive)."""
    key = {c.lower().replace("_", ""): c for c in df.columns}
    rename = {}
    for want in WANTED:
        k = want.lower().replace("_", "")
        if k in key:
            rename[key[k]] = want
    df = df.rename(columns=rename)
    missing = [c for c in WANTED if c not in df.columns]
    if missing:
        raise KeyError(f"missing expected columns: {missing}")
    return df[WANTED]


def cast(df: pd.DataFrame) -> pd.DataFrame:
    for c in WANTED:
        if c in STR_COLS:
            df[c] = df[c].astype("string")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["FlightDate"] = df["FlightDate"].astype("string")
    return df


def read_month(zpath: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as f:
            df = pd.read_csv(io.BytesIO(f.read()), low_memory=False)
    df = normalize(df)
    df = df[df["Origin"].isin(AIRPORTS) & df["Dest"].isin(AIRPORTS)]
    return cast(df.reset_index(drop=True))


def main():
    zips = sorted(RAW_DIR.glob("*.zip"))
    if not zips:
        sys.exit(f"no zips found in {RAW_DIR}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    schema = None
    total_rows = 0
    # running tallies for the target-variable decision
    n_cancelled = n_diverted = n_del15 = n_arr_known = 0

    for zpath in zips:
        print(f"processing {zpath.name}...", end=" ", flush=True)
        df = read_month(zpath)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(OUT_PATH, schema, compression="zstd")
        writer.write_table(table.cast(schema))

        total_rows += len(df)
        n_cancelled += int((df["Cancelled"] == 1).sum())
        n_diverted += int((df["Diverted"] == 1).sum())
        known = df["ArrDel15"].notna()
        n_arr_known += int(known.sum())
        n_del15 += int((df.loc[known, "ArrDel15"] == 1).sum())
        print(f"{len(df):,} rows kept")

    if writer is not None:
        writer.close()

    print("\n=== flights.parquet summary ===")
    print(f"total rows (top-40 to top-40):  {total_rows:,}")
    print(f"cancelled:                      {n_cancelled:,} ({n_cancelled/total_rows:.2%})")
    print(f"diverted:                       {n_diverted:,} ({n_diverted/total_rows:.2%})")
    print(f"rows with known arrival status: {n_arr_known:,}")
    print(f"ArrDel15=1 among known:         {n_del15:,} ({n_del15/n_arr_known:.2%})")
    print(f"\nwritten to {OUT_PATH}")


if __name__ == "__main__":
    main()