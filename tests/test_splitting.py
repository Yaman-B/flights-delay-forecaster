import pandas as pd
from src.models import temporal_split


def test_split_assigns_by_year():
    dates = pd.to_datetime(["2023-06-01", "2024-12-31", "2025-01-01",
                            "2025-12-31", "2026-02-15"])
    assert list(temporal_split(dates)) == ["train", "train", "val", "val", "test"]


def test_split_is_exhaustive_and_disjoint():
    dates = pd.Series(pd.date_range("2023-01-01", "2026-03-31", freq="D"))
    s = temporal_split(dates)
    assert s.notna().all()                          # every flight labeled
    assert set(s.unique()) <= {"train", "val", "test"}
    # No calendar year may straddle two splits (contiguity).
    spanned = pd.DataFrame({"y": dates.dt.year, "split": s}).groupby("y")["split"].nunique()
    assert spanned.max() == 1