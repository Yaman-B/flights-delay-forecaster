"""Phase 3 -> Phase 4 handoff: turn one flight into a structured explanation.

FlightExplainer bundles the trained model, the isotonic calibrator, and a
feature-to-language mapping. Its explain() emits a JSON-serializable object --
calibrated probability, risk tier, and the top signed SHAP drivers as readable
clauses -- which is the ONLY thing the LLM layer is permitted to draw on.
"""
import numpy as np
import pandas as pd

_MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_CASCADE = {"inbound_gap_h", "inbound_delay_obs", "inbound_buffer_min", "inbound_unlanded"}


def _clean(v):
    """Coerce numpy/pandas scalars to plain JSON-friendly Python values."""
    if v is None or (np.isscalar(v) and pd.isna(v)):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return round(float(v), 3)
    if isinstance(v, (int, float, bool, str)):
        return v
    return str(v)


def _hour_phrase(h):
    h = int(h)
    if h < 6:   return f"a {h:02d}:00 red-eye departure"
    if h < 11:  return f"an early {h:02d}:00 departure"
    if h < 16:  return f"a midday {h:02d}:00 departure"
    if h < 20:  return f"an afternoon {h:02d}:00 departure"
    return f"a late {h:02d}:00 departure"


def _phrase(feature, value, up):
    """Readable clause for one driver; `up` is True when it pushes risk up."""
    f = feature
    if f == "inbound_buffer_min":
        if up:
            v = int(round(value))
            return ("the inbound aircraft has no spare turnaround time" if v <= 0
                    else f"the inbound aircraft has only ~{v} min of spare turnaround time")
        return f"the inbound aircraft has ample spare turnaround time (~{int(round(value))} min)"
    if f == "inbound_delay_obs":
        return (f"the inbound aircraft is running ~{int(round(value))} min behind" if up
                else "the inbound aircraft is on or ahead of schedule")
    if f == "inbound_unlanded":
        return ("the inbound aircraft still hasn't landed by departure time" if up
                else "the inbound aircraft has already landed")
    if f == "inbound_gap_h":
        v = round(float(value), 1)
        return (f"a tight scheduled turnaround (~{v} h)" if up
                else f"a comfortable scheduled turnaround (~{v} h)")
    if f == "Origin":
        return (f"departing {value}, which tends toward higher disruption" if up
                else f"departing {value}, which tends to run smoothly")
    if f == "Dest":
        return (f"arriving into {value}, which tends toward higher disruption" if up
                else f"arriving into {value}, which tends to run smoothly")
    if f == "Reporting_Airline":
        return (f"operated by {value}, which the model links to more disruption" if up
                else f"operated by {value}, a relatively reliable carrier")
    if f == "hour":
        base = _hour_phrase(value)
        return (f"{base}, when delays tend to accumulate" if up
                else f"{base}, before the day's delays build up")
    if f == "month":
        name = _MONTHS[int(value)]
        return (f"{name}, a higher-disruption month" if up else f"{name}, a relatively calm month")
    if f == "dow":
        name = _DOW[int(value)]
        return (f"a {name}, a busier travel day" if up else f"a {name}, a lighter travel day")
    if f in ("snowfall_orig", "snowfall_dest"):
        where = "departure" if f.endswith("orig") else "destination"
        return (f"snowfall at the {where} airport (~{round(float(value), 1)} cm/h)" if up
                else f"no meaningful snow at the {where} airport")
    if f in ("wind_gusts_10m_orig", "wind_gusts_10m_dest"):
        where = "departure" if f.endswith("orig") else "destination"
        return (f"strong wind gusts at the {where} airport (~{int(round(value))} km/h)" if up
                else f"light winds at the {where} airport")
    if f == "temperature_2m_orig":
        v = int(round(value))
        if up:
            return (f"cold conditions at the departure airport (~{v}\u00b0C)" if v <= 5
                    else f"hot conditions at the departure airport (~{v}\u00b0C)")
        return f"mild temperatures at the departure airport (~{v}\u00b0C)"
    return f"{feature} = {value}"


def _magnitude(abs_shap, bands=(0.3, 1.0)):
    lo, hi = bands
    return "strong" if abs_shap >= hi else "moderate" if abs_shap >= lo else "slight"

def _prob_text(p):
    """ human-readable probability phrase; never 0% or 100%."""
    if p >= 0.995: return "over 99%"
    if p < 0.01:   return "under 1%"
    return f"about {int(round(p * 100))}%"

class FlightExplainer:
    def __init__(self, model, calibrator, features, base_rate, cat_dtypes,
                 threshold=0.25, top_up=5, top_down=3, floor=0.1, mag_bands=(0.3, 1.0)):
        self.model = model
        self.calibrator = calibrator
        self.features = list(features)
        self.base_rate = float(base_rate)
        self.cat_dtypes = dict(cat_dtypes)
        self.threshold = threshold
        self.top_up, self.top_down = top_up, top_down
        self.floor, self.mag_bands = floor, mag_bands

    def _row_to_frame(self, s):
        # Rebuild a 1-row frame with the EXACT training dtypes so category codes
        # (and therefore the prediction and its SHAP values) match training.
        X = pd.DataFrame([s[self.features].values], columns=self.features)
        for c in self.features:
            X[c] = X[c].astype(self.cat_dtypes[c]) if c in self.cat_dtypes else pd.to_numeric(X[c])
        return X

    def _risk_level(self, p):
        return "high" if p >= 0.50 else "elevated" if p >= self.threshold else "low"

    def explain(self, flight):
        s = flight if isinstance(flight, pd.Series) else pd.Series(flight)
        X = self._row_to_frame(s)

        raw = float(self.model.predict(X)[0])
        prob = float(self.calibrator.predict([raw])[0]) if self.calibrator is not None else raw

        contribs = self.model.predict(X, pred_contrib=True)[0]   # exact TreeSHAP, log-odds
        shap = dict(zip(self.features, contribs[:-1]))
        has_inbound = int(s.get("has_inbound", 0)) == 1

        rows = []
        for f, sv in shap.items():
            if f == "has_inbound" or (f in _CASCADE and not has_inbound) or abs(sv) < self.floor:
                continue
            up = sv > 0
            rows.append({"feature": f, "value": _clean(s[f]), "impact": round(float(sv), 3),
                         "magnitude": _magnitude(abs(sv), self.mag_bands),
                         "text": _phrase(f, s[f], up), "_up": up})

        inc = sorted([r for r in rows if r["_up"]], key=lambda r: -abs(r["impact"]))[:self.top_up]
        dec = sorted([r for r in rows if not r["_up"]], key=lambda r: -abs(r["impact"]))[:self.top_down]
        for r in inc + dec:
            del r["_up"]

        return {
            "flight": {
                "origin": _clean(s.get("Origin")), "dest": _clean(s.get("Dest")),
                "carrier": _clean(s.get("Reporting_Airline")),
                "dep_hour_local": int(s["hour"]) if "hour" in s.index else None,
                "date": str(pd.to_datetime(s["FlightDate"]).date()) if "FlightDate" in s.index else None,
            },
            "prediction": {
                "probability": max(round(prob, 4), 0.0001), "base_rate": round(self.base_rate, 4),
                "probability_text": _prob_text(prob),
                "risk_level": self._risk_level(prob),
                "flagged": bool(prob >= self.threshold), "threshold": self.threshold,
            },
            "drivers": {"increasing": inc, "decreasing": dec},
            "meta": {"has_inbound": has_inbound, "n_drivers": len(inc) + len(dec)},
        }