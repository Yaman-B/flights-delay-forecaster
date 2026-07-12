# ✈️ Flight Disruption Forecaster

Predicts the probability that a US flight arrives 15+ minutes late or is cancelled, and explains
why in plain language, grounded in the model's own SHAP attributions.

**[▶ Live demo](https://huggingface.co/spaces/yamanb/flight-delay-forecaster)**

Built on 10.9M flights (BTS on-time performance, top 40 US airports, 2023 to 2026-Q1) joined to
hourly weather at both origin and destination. Base disruption rate: 23.5%.

---

## Results

Model, evaluated on 2026 Q1, a period it never saw during training:

| Model | PR-AUC | ROC-AUC | Brier | F1 |
|---|---|---|---|---|
| Majority class (no-skill floor) | 0.2445 | 0.500 | 0.1849 | 0.000 |
| Climatology (route × hour, EB-smoothed) | 0.3172 | 0.607 | 0.1812 | 0.014 |
| Logistic regression | 0.4022 | 0.655 | 0.1727 | 0.099 |
| **LightGBM + cascade** | **0.6046** | **0.769** | **0.1429** | **0.538\*** |

\* at a tuned threshold of 0.25, chosen on validation. It's 0.437 at the naive 0.50 default.

That's **+91% PR-AUC over the climatology baseline**. An ablation shows that the gain actually
comes from refitting the identical model with the aircraft-rotation cascade removed, which produces
a score of 0.4370. So the increase from logistic regression splits into **+0.035 from the model class** and
**+0.168 from the cascade feature**, which is 83% of the total.

Explanation layer, scored on a 250-flight stratified eval by an LLM judge:

| Prompt | Faithful |
|---|---|
| Naive (prediction + raw facts, no drivers) | 2.0% |
| **Grounded (SHAP drivers + constrained prompt)** | **98.0%** |

I asked an LLM to explain a prediction without telling it *why* the model made that prediction, and
it just made reasons up: "overnight weather systems", "air traffic congestion". It did this in
96% of explanations. Then, I fed it the model's actual reasons instead, and it stopped.

---

## The critical feature: a leakage-safe cascade

The strongest signal in flight delay is the inbound aircraft. The plane flying your route is
arriving from somewhere else, and if it's running late, so are you. When the inbound leg landed
on time, only 13.2% of flights were disrupted, well below the 23.5% average. When it was two or
more hours late, 75.8% were (a 5.7x difference) 

What makes it the single most valuable feature is that it applies to almost every flight. Snow is
dramatic when it happens, but it barely ever happens. A late inbound aircraft is common.

The problem is that using it naively is cheating. At the moment you'd actually make a prediction,
just before the scheduled departure, you don't yet know how late the inbound plane will *end up*
being. If it's still in the air, all you know is that it's at least as late as the time it has
left. So the feature is capped at that point:

    inbound_delay_observed = min(true_inbound_delay, scheduled_slack)

Censoring makes that feature non-monotonic, because it entangles delay with slack. Three
companions untangle it: the scheduled slack itself, the realized spare buffer
(`slack - observed delay`, which ended up the single most important feature by gain), and a
binary "inbound hasn't landed yet" flag, which on its own predicts 98.4% disruption.

A model trained on the uncensored inbound delay would score higher and be useless in production.
0.6046 is lower than a leaky model's number, and it's real.

---

## Why does the demo score flights that already happened?

It's a backtest, which is how forecasting systems get evaluated.

The model trains on 2023-24 and is tested on 2026 Q1, a period it never saw. Every feature it
uses would be available *before* the flight departs: the schedule, route, carrier, forecast
weather (joined on scheduled times, never actual ones), and the inbound aircraft's status as
known at the scheduled departure. The flights are in the past for you. For the model they're the
future.

Going live is a primarily data problem, not a modeling one. Swapping the historical lookup for published
schedules, a weather forecast, and live aircraft status, and the same model can serve live
predictions.

---

## What the data actually says

Three findings from the EDA that overturned what I expected:

**Delays track weather, not crowding.** Airports range from 16.5% disruption (San Jose) to 29.9%
(Dallas-Fort Worth), and the worst ones sit in thunderstorm country rather than the crowded
Northeast. Atlanta is the busiest airport in the US and it's barely above average.

**Busy hours are the safest.** Within any given airport, the busiest 20% of hours have *fewer*
disruptions than the quietest (20.7% vs 23.4%). My reasoning is that airlines schedule their
heaviest traffic flights in the morning, and delays pile up over the course of a day, so the busiest hours
are also the earliest and cleanest ones. Using flights-per-hour as a feature would have taught the
model the relationship backwards, so airport congestion was left out on purpose.

**The worst month is July, not January.** Summer thunderstorms (33.1% disruption in July) beat
winter snow, and February (20.0%) is actually below average. Snow is far more disruptive per
storm, but it hits only a sliver of flights. Summer storms are less severe and hit almost
everyone, so they move the yearly numbers more.

---

## Architecture

    BTS flights ──┐
                  ├── join on SCHEDULED times ──> features ──> LightGBM ──> probability
    Open-Meteo ───┘   (timezone-aware, IANA)         │                          │
                                                 TreeSHAP                   isotonic
                                                     │                    calibration
                                                     ▼                          │
                                          signed per-flight drivers ────────────┤
                                                     │                          │
                                                     ▼                          ▼
                                        constrained prompt ──> Claude ──> grounded
                                                                         explanation

**Split.** Strictly temporal: train 2023-24, validate on 2025, test on 2026 Q1. This *exposes* a
real upward drift in disruption (23.1% to 24.4%) that a random split would have hidden.

**Calibration.** A model can rank flights well and still emit numbers that don't mean what they
say. I checked: when this model says 70%, roughly 70% of those flights really are disrupted (ECE
0.0092, already good). Isotonic regression made it better but slightly impacted the ranking,
so the raw model does the ranking and the calibrated one produces the number the user sees, where
"72% means 72%" is what matters.

**Explanation.** Exact TreeSHAP (additivity verified to 5e-14), then a structured driver object,
then a constrained prompt to Claude Haiku 4.5. That object is the only sanctioned source of
facts, which is what makes faithfulness checkable in the first place.

**Serving.** FastAPI, with `/predict` fast and free and `/explain` adding the LLM call, so the UI
still renders risk and drivers if the LLM is unavailable. Streamlit front end, containerized,
deployed on HF Spaces.

**Tracking.** MLflow with a SQLite backend. Every run's params, metrics, and artifacts are
versioned.

---

## Some limitations

**No live data feeds.** The demo serves historical flights. Live deployment needs a flight-status
API, which is the one piece that costs money.

**The faithfulness judge is imperfect.** The 98% comes from an LLM judge, so I checked it against
50 explanations I labelled by hand, without seeing its verdicts. It agreed with me 88% of the
time, but that number flatters it: almost everything is faithful, so agreeing by default gets you
most of the way there. Where we disagreed, the judge was usually too harsh, flagging a
reworded driver as if it were invented. So 98% is a floor rather than a ceiling, and the honest
claim is the 2% to 98% gap rather than the 98% on its own.

**The test window is winter-only** (Q1). That affects the absolute numbers but not the relative
comparisons.

**Diverted flights are excluded** (0.26%, ambiguous outcomes). This biases the measured effect of
destination weather downward, since diversions are exactly what severe destination weather causes.

---

## Run it yourself
 
If you just want to try it, use the [live demo](https://huggingface.co/spaces/yamanb/flight-delay-forecaster).
This section is for reproducing the results locally.
 
    git clone <this-repo> && cd flight_delay_forecaster
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements-dev.txt
 
The trained model (`models/`) and the flight data the app serves (`serving_data/`) are build
artifacts, so they aren't in git. Running `notebooks/04_gbm.py` regenerates both, and reproduces
the whole baseline ladder on the way. After that:
 
    docker build -t flight-delay-forecaster .
    docker run --rm -p 7860:7860 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY flight-delay-forecaster
 
The app comes up at `localhost:7860`. You'll need an Anthropic API key for the explanation layer;
predictions and SHAP drivers work without one.


## Stack

Python · LightGBM · SHAP · scikit-learn · MLflow · FastAPI · Streamlit · Docker · Anthropic API