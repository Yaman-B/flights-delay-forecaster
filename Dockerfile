FROM python:3.12-slim

# libgomp is LightGBM's OpenMP runtime; curl is used by the entrypoint healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies: this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code and artifacts
COPY src/ ./src/
COPY models/ ./models/
COPY serving_data/ ./serving_data/
COPY pyproject.toml ./
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# streamlit_app.py reads this; both processes share the container's localhost
ENV API_URL=http://127.0.0.1:8000 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

EXPOSE 8000 8501

CMD ["./docker-entrypoint.sh"]