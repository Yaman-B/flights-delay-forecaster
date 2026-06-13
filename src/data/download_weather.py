"""Download hourly weather for top US airports via Open-Meteo's historical archive API.

Free, no API key, queried by lat/lon. Docs: https://open-meteo.com/en/docs/historical-weather-api
Usage:
    python src/data/download_weather.py --start 2023-01-01 --end 2026-04-30
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

# Top ~40 US airports by traffic: IATA -> (lat, lon)
AIRPORTS = {
    "ATL": (33.6407, -84.4277), "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737), "ORD": (41.9742, -87.9073),
    "LAX": (33.9416, -118.4085), "CLT": (35.2140, -80.9431),
    "MCO": (28.4312, -81.3081), "LAS": (36.0840, -115.1537),
    "PHX": (33.4352, -112.0101), "MIA": (25.7959, -80.2870),
    "SEA": (47.4502, -122.3088), "IAH": (29.9902, -95.3368),
    "JFK": (40.6413, -73.7781), "EWR": (40.6895, -74.1745),
    "FLL": (26.0742, -80.1506), "MSP": (44.8848, -93.2223),
    "SFO": (37.6213, -122.3790), "DTW": (42.2162, -83.3554),
    "BOS": (42.3656, -71.0096), "SLC": (40.7899, -111.9791),
    "PHL": (39.8744, -75.2424), "BWI": (39.1774, -76.6684),
    "TPA": (27.9772, -82.5311), "SAN": (32.7338, -117.1933),
    "LGA": (40.7769, -73.8740), "MDW": (41.7868, -87.7522),
    "BNA": (36.1263, -86.6774), "IAD": (38.9531, -77.4565),
    "DCA": (38.8512, -77.0402), "AUS": (30.1945, -97.6699),
    "HNL": (21.3187, -157.9224), "DAL": (32.8471, -96.8518),
    "PDX": (45.5898, -122.5951), "STL": (38.7500, -90.3700),
    "RDU": (35.8801, -78.7880), "HOU": (29.6454, -95.2789),
    "SMF": (38.6951, -121.5908), "MSY": (29.9934, -90.2580),
    "SJC": (37.3639, -121.9289), "SNA": (33.6762, -117.8675),
}
API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m", "dew_point_2m", "relative_humidity_2m",
    "precipitation", "rain", "snowfall", "cloud_cover",
    "pressure_msl", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
]
OUT_DIR = Path("data/raw/weather")


def fetch_airport(iata: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    r = requests.get(API, params=params, timeout=120)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "obs_time_utc"})
    df["obs_time_utc"] = pd.to_datetime(df["obs_time_utc"])
    df["airport"] = iata
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for iata, (lat, lon) in AIRPORTS.items():
        dest = OUT_DIR / f"{iata}.parquet"
        if dest.exists():
            print(f"skip {iata}")
            frames.append(pd.read_parquet(dest))
            continue
        print(f"fetching {iata}...", end=" ", flush=True)
        try:
            df = fetch_airport(iata, lat, lon, args.start, args.end)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}")
            continue
        if df.empty:
            print("WARNING: no data")
            continue
        df.to_parquet(dest, index=False)
        frames.append(df)
        print(f"{len(df):,} rows")
        time.sleep(1)  # stay friendly to the free API

    if not frames:
        raise SystemExit("No data fetched for any airport — check your connection.")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(OUT_DIR / "all_airports_hourly.parquet", index=False)
    print(f"done: {len(combined):,} hourly observations, {combined['airport'].nunique()} airports")


if __name__ == "__main__":
    main()