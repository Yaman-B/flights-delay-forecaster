"""The single temporal split. Everything imports this one, which is what keeps
the baseline ladder and the model strictly comparable."""
import pandas as pd


def temporal_split(dates):
    """Split by calendar period, never at random: train = 2023-2024, val = 2025,
    test = 2026+ (Q1 only in current data).

    Temporal ordering prevents leakage and exposes the year-over-year drift
    (train 23.1% -> test 24.5%) that a random split would hide. Returns an object
    Series aligned to the input index.
    """
    d = pd.to_datetime(dates)
    if not isinstance(d, pd.Series):
        d = pd.Series(d)
    year = d.dt.year
    out = pd.Series(index=d.index, dtype="object")
    out[year <= 2024] = "train"
    out[year == 2025] = "val"
    out[year >= 2026] = "test"
    return out