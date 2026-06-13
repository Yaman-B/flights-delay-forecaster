"""Construct the prediction target and write the cleaned modeling dataset.

Usage:
    python src/data/make_target.py

Reads data/processed/flights.parquet and writes data/processed/flights_clean.parquet.

Target definition (see README):
    disrupted = 1  if the flight was cancelled OR arrived >= 15 minutes late
    disrupted = 0  otherwise
Diverted flights are dropped (0.26% of rows; ambiguous outcomes).
Rows that are neither cancelled nor diverted but have no recorded arrival
status (data errors) are also dropped and counted.

Processes the parquet one row-group at a time so memory stays low.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

IN_PATH = Path("data/processed/flights.parquet")
OUT_PATH = Path("data/processed/flights_clean.parquet")


def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Return (cleaned frame, n_diverted_dropped, n_missing_dropped)."""
    n0 = len(df)
    df = df[df["Diverted"] != 1]
    n_diverted = n0 - len(df)

    cancelled = df["Cancelled"] == 1
    has_status = df["ArrDel15"].notna()
    keep = cancelled | has_status
    n_missing = int((~keep).sum())
    df = df[keep].copy()

    df["disrupted"] = (
        (df["Cancelled"] == 1) | (df["ArrDel15"] == 1)
    ).astype("int8")
    return df, n_diverted, n_missing


def main():
    pf = pq.ParquetFile(IN_PATH)
    writer = None
    schema = None
    rows_in = rows_out = diverted = missing = positives = 0

    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i).to_pandas()
        rows_in += len(df)
        df, n_div, n_miss = transform(df)
        diverted += n_div
        missing += n_miss
        rows_out += len(df)
        positives += int(df["disrupted"].sum())

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(OUT_PATH, schema, compression="zstd")
        writer.write_table(table.cast(schema))
        print(f"row group {i + 1}/{pf.num_row_groups} done", end="\r", flush=True)

    if writer is not None:
        writer.close()

    print("\n=== flights_clean.parquet summary ===")
    print(f"rows in:                    {rows_in:,}")
    print(f"diverted dropped:           {diverted:,} ({diverted/rows_in:.2%})")
    print(f"missing-status dropped:     {missing:,} ({missing/rows_in:.2%})")
    print(f"rows out:                   {rows_out:,}")
    print(f"disrupted=1 (target rate):  {positives:,} ({positives/rows_out:.2%})")
    print(f"\nwritten to {OUT_PATH}")
    print("\nEXPECTED: target rate ~23.5%, diverted ~0.26%. "
          "If these differ materially, stop and investigate before proceeding.")


if __name__ == "__main__":
    main()