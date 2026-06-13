"""Download BTS On-Time Performance data (one zip per month).

Usage:
    python src/data/download_bts.py --start 2023-01 --end 2026-04

Each monthly zip is ~25 MB and contains a single CSV (~600k flights).
Source: Bureau of Transportation Statistics, transtats.bts.gov (public domain).

Notes:
- transtats.bts.gov serves an incomplete SSL certificate chain. We attempt a
  verified connection first and fall back to unverified (with a warning) if
  needed. The data itself is public domain.
- Downloads go to a .part file and are validated as real zips before being
  renamed, so interrupted runs never leave corrupt files that get skipped later.
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

import requests
import urllib3

URL_TMPL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
OUT_DIR = Path("data/raw/bts")
HEADERS = {"User-Agent": "Mozilla/5.0 (research; flight-delay-forecaster)"}
RETRIES = 3


def month_range(start: str, end: str):
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def download(url: str, dest: Path, verify: bool) -> None:
    part = dest.with_suffix(".part")
    with requests.get(url, headers=HEADERS, stream=True, timeout=180, verify=verify) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    if not zipfile.is_zipfile(part):
        part.unlink(missing_ok=True)
        raise ValueError("downloaded file is not a valid zip")
    part.rename(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verify = True
    failures = []

    for year, month in month_range(args.start, args.end):
        url = URL_TMPL.format(year=year, month=month)
        dest = OUT_DIR / f"bts_{year}_{month:02d}.zip"
        if dest.exists() and zipfile.is_zipfile(dest):
            print(f"skip {dest.name} (already downloaded)")
            continue

        ok = False
        for attempt in range(1, RETRIES + 1):
            try:
                print(f"downloading {year}-{month:02d} (attempt {attempt})...", flush=True)
                download(url, dest, verify)
                ok = True
                break
            except requests.exceptions.SSLError:
                if verify:
                    print(
                        "  WARNING: SSL verification failed (known issue with "
                        "transtats.bts.gov cert chain). Continuing without "
                        "verification for this public dataset.",
                        file=sys.stderr,
                    )
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    verify = False
                # retry loop continues with verify=False
            except Exception as e:  # noqa: BLE001
                print(f"  attempt {attempt} failed: {e}", file=sys.stderr)
                time.sleep(2 * attempt)

        if not ok:
            failures.append(f"{year}-{month:02d}")

    if failures:
        print(f"\nFAILED months (re-run to retry): {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)
    print("done.")


if __name__ == "__main__":
    main()