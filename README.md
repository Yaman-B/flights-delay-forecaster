# Flight Delay Forecasting + LLM Explanation Layer

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
