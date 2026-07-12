#!/usr/bin/env bash
set -e

if [ ! -f models/gbm.txt ] || [ ! -f serving_data/flights.parquet ]; then
  echo "ERROR: missing build artifacts (models/ and/or serving_data/)."
  echo "These are gitignored build outputs. Regenerate them by running"
  echo "notebooks/04_gbm.py before building the image."
  exit 1
fi

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# wait for the API to come up before starting the UI
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:8000/health > /dev/null && break
  sleep 1
done

exec streamlit run src/app/streamlit_app.py \
  --server.port "${PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false