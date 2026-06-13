# Flight Delay Forecasting w/ LLM Explanation Layer

End-to-end system predicting US flight arrival delays (BTS + NOAA data),
with SHAP-grounded LLM explanations, MLflow/DVC tracking, and automated retraining.

## Quickstart
```bash
pip install -r requirements.txt
python src/data/download_bts.py --start 2023-01 --end 2026-04
python src/data/download_weather.py --start 2023-01-01 --end 2026-04-30
```

## Layout
- `src/data/` — acquisition & cleaning
- `src/features/` — feature engineering (rotation, congestion, weather joins)
- `src/models/` — training, evaluation, baselines
- `src/llm/` — explanation layer + faithfulness evals
- `notebooks/` — EDA and analysis

See PLAN.md for the full project roadmap.

## Target definition

The prediction target is **flight disruption**: a flight is labeled disrupted
(`disrupted = 1`) if it was cancelled or arrived 15 or more minutes late.
The 15-minute threshold is the US Department of Transportation's official
definition of a delayed flight (the `ArrDel15` field in BTS data), which keeps
our results directly comparable to published statistics and prior work.

Cancellations are included as positives because, from both a passenger and an
operational standpoint, a cancellation is the most severe form of disruption —
and because cancellations cluster in exactly the weather-driven events the
model should learn, excluding them would systematically remove the strongest
signal from training. Diverted flights (0.26% of records) are excluded, as
their final outcomes are ambiguous in the data.

Three formulations were considered: (1) delay-only, dropping cancellations;
(2) delay-or-cancellation as a single binary target; (3) multi-class
(on-time / delayed / cancelled). Option 2 was chosen: option 1 discards
operationally meaningful events, and option 3 splits an already imbalanced
positive class across labels, weakening every evaluation metric for little
practical gain.

Measured on 2023-01 through 2026-03 data (flights between the top 40 US
airports): 22.39% of completed flights were delayed ≥15 min, 1.47% of all
flights were cancelled, giving a combined positive rate of ~23.5%.