"""FastAPI serving layer: disruption risk plus a grounded explanation.

Flights come from a prebuilt slice whose features were built by the training
pipeline, so leakage-safety holds by construction. A live deployment would swap
the slice for a real-time feature pipeline.
"""
import json
from contextlib import asynccontextmanager

import lightgbm as lgb
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from fastapi import Request

from src.paths import PROJECT_ROOT
from src.llm.explain import FlightExplainer
from src.llm.narrate import narrate

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # model, calibrator and flight slice load once at startup, not per request.
    models, serving = PROJECT_ROOT / "models", PROJECT_ROOT / "serving_data"
    config = json.loads((models / "config.json").read_text())
    booster = lgb.Booster(model_file=str(models / "gbm.txt"))
    STATE["flights"] = pd.read_parquet(serving / "flights.parquet").set_index("flight_id")
    STATE["explainer"] = FlightExplainer(
        model=booster, calibrator=joblib.load(models / "calibrator_isotonic.joblib"),
        features=config["features"], base_rate=config["base_rate"],
        cat_categories=config["cat_categories"], threshold=config["threshold"])
    print(f"loaded {len(STATE['flights']):,} flights")
    yield
    STATE.clear()


app = FastAPI(title="Flight Disruption Forecaster", version="1.0", lifespan=lifespan)

# rate limit the LLM endpoint; a public /explain is otherwise an open faucet
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class FlightRequest(BaseModel):
    flight_id: int


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "flights_loaded": len(STATE["flights"])}


@app.get("/flights")
def list_flights(origin: str | None = None, dest: str | None = None,
                 date: str | None = None, limit: int = 50):
    """Browse servable flights. Powers the UI's dropdowns."""
    df = STATE["flights"]
    if origin: df = df[df["Origin"] == origin]
    if dest:   df = df[df["Dest"] == dest]
    if date:   df = df[df["FlightDate"].astype(str) == date]
    out = df.head(limit)
    return {
        "count": int(len(df)),
        "flights": [{"flight_id": int(i), "origin": str(r.Origin), "dest": str(r.Dest),
                     "carrier": str(r.Reporting_Airline), "date": str(r.FlightDate)[:10],
                     "dep_hour": int(r.hour)} for i, r in out.iterrows()],
    }


@app.get("/options")
def options():
    """Distinct origins/dests/dates for populating UI selectors."""
    df = STATE["flights"]
    return {"origins": sorted(df["Origin"].astype(str).unique().tolist()),
            "dests":   sorted(df["Dest"].astype(str).unique().tolist()),
            "dates":   sorted(df["FlightDate"].astype(str).str[:10].unique().tolist())}


def _row(flight_id: int):
    try:
        return STATE["flights"].loc[flight_id]
    except KeyError:
        raise HTTPException(404, f"flight_id {flight_id} not found")


@app.post("/predict")
def predict(req: FlightRequest):
    """Structured prediction + SHAP drivers. Fast, no LLM call."""
    return STATE["explainer"].explain(_row(req.flight_id))


@app.post("/explain")
@limiter.limit("10/minute")
def explain(request: Request, req: FlightRequest):
    """Prediction + natural-language explanation (calls the Anthropic API)."""
    obj = STATE["explainer"].explain(_row(req.flight_id))
    try:
        text = narrate(obj)
    except Exception as e:
        raise HTTPException(502, f"explanation service unavailable: {e}")
    return {**obj, "explanation": text}