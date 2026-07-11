"""Streamlit demo: browse a flight, see its disruption risk, drivers, and a
grounded natural-language explanation. Thin client over the FastAPI service."""
import os
import requests
import pandas as pd
import streamlit as st
import re
import altair as alt

API = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Flight Disruption Forecaster", page_icon="✈️", layout="wide")
st.title("✈️ Flight Disruption Forecaster")
st.caption("Predicts the probability a flight arrives 15+ min late or is cancelled "
           "and explains why, grounded in the model's own SHAP attributions.")


@st.cache_data(ttl=600)
def get_options():
    return requests.get(f"{API}/options", timeout=10).json()


@st.cache_data(ttl=600)
def get_flights(origin, dest, date):
    r = requests.get(f"{API}/flights",
                     params={"origin": origin, "dest": dest, "date": date, "limit": 50}, timeout=10)
    return r.json()


try:
    opts = get_options()
except Exception:
    st.error(f"Can't reach the API at {API}. Start it with: "
             "`uvicorn src.api.main:app --port 8000`")
    st.stop()

c1, c2, c3 = st.columns(3)
origin = c1.selectbox("Origin", opts["origins"], index=opts["origins"].index("DFW")
                      if "DFW" in opts["origins"] else 0)
dest = c2.selectbox("Destination", opts["dests"], index=opts["dests"].index("LGA")
                    if "LGA" in opts["dests"] else 0)
date = c3.selectbox("Date", opts["dates"])

res = get_flights(origin, dest, date)
if not res["flights"]:
    st.warning("No flights on that route and date — try another combination.")
    st.stop()

labels = {f"{f['dep_hour']:02d}:00 · {f['carrier']} · flight {f['flight_id']}": f["flight_id"]
          for f in res["flights"]}
choice = st.selectbox(f"Flight ({res['count']} found)", list(labels))
fid = labels[choice]

obj = requests.post(f"{API}/predict", json={"flight_id": fid}, timeout=15).json()
p, drivers = obj["prediction"], obj["drivers"]

# --- risk headline -----------------------------------------------------------
COLORS = {"low": "#2e7d32", "elevated": "#ef6c00", "high": "#c62828"}

HL = "#ffd166"   # highlight color for grounded terms

KEY_TERMS = {
    "inbound_buffer_min":  ["spare turnaround time", "turnaround time", "buffer time", "buffer"],
    "inbound_delay_obs":   ["behind schedule", "minutes behind", "running late"],
    "inbound_unlanded":    ["hasn't landed", "has not landed", "still in the air", "not yet landed"],
    "inbound_gap_h":       ["scheduled turnaround", "turnaround"],
    "snowfall_orig":       ["snowfall", "snow"],
    "snowfall_dest":       ["snowfall", "snow"],
    "wind_gusts_10m_orig": ["wind gusts", "gusts"],
    "wind_gusts_10m_dest": ["wind gusts", "gusts"],
    "temperature_2m_orig": ["cold temperatures", "cold conditions", "cold", "hot conditions",
                            "mild temperatures", "mild"],
    "hour":                [],   # filled dynamically below
    "month":               [],   # filled dynamically (month name)
    "dow":                 [],   # filled dynamically (day name)
}
_MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _hour_variants(h):
    h = int(h)
    ampm = f"{h-12 if h > 12 else (12 if h == 0 else h)} {'p.m.' if h >= 12 else 'a.m.'}"
    return [f"{h:02d}:00", f"{h}:00", ampm, ampm.replace(".", "")]


def bold_drivers(text, drivers):
    """Highlight the terms the explanation is grounded in — deterministic, UI-side only."""
    terms = set()
    for d in drivers["increasing"] + drivers["decreasing"]:
        f, v = d["feature"], d["value"]
        terms.update(KEY_TERMS.get(f, []))
        if f == "hour":
            terms.update(_hour_variants(v))
        elif f == "month":
            terms.add(_MONTHS[int(v)])
        elif f == "dow":
            terms.add(_DOW[int(v)])
        elif f in ("Origin", "Dest", "Reporting_Airline"):
            terms.add(str(v))
    for t in sorted(terms, key=len, reverse=True):     # longest first
        if not t:
            continue
        text = re.sub(rf"(?<![*\w])({re.escape(t)})(?![\w*])",
                      rf"**:orange[\1]**", text, flags=re.I)
    return text


m1, m2, m3 = st.columns(3)
m1.metric("Disruption risk", p["probability_text"], p["risk_level"].upper())
m2.metric("Typical flight", f"{p['base_rate']:.0%}", "base rate")
m3.metric("Flagged?", "YES" if p["flagged"] else "no", f"threshold {p['threshold']:.0%}")
st.progress(min(p["probability"], 1.0))

# --- explanation (the slow call) --------------------------------------------
st.subheader("Why?")
with st.spinner("Generating explanation…"):
    try:
        text = requests.post(f"{API}/explain", json={"flight_id": fid}, timeout=60).json()["explanation"]
        st.info(bold_drivers(text, drivers))
    except Exception as e:
        st.warning(f"Explanation unavailable ({e}). The prediction and drivers below are unaffected.")

# --- the drivers the explanation is grounded in ------------------------------
st.subheader("Model drivers (SHAP)")
rows = ([{"factor": d["text"], "impact": d["impact"], "dir": "increases risk"}
         for d in drivers["increasing"]] +
        [{"factor": d["text"], "impact": d["impact"], "dir": "reduces risk"}
         for d in drivers["decreasing"]])
if rows:
    df = pd.DataFrame(rows).sort_values("impact")
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("impact:Q", title="SHAP impact (log-odds)"),
            y=alt.Y("factor:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=420, labelOverlap=False, labelFontSize=13)),
            color=alt.Color("dir:N",
                            scale=alt.Scale(domain=["increases risk", "reduces risk"],
                                            range=["#c62828", "#2e7d32"]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["factor", "dir", "impact"],
        )
        .properties(height=46 * max(len(df), 1))       # generous room per bar
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(df[["factor", "dir", "impact"]], hide_index=True, use_container_width=True)
else:
    st.write("No drivers passed the significance floor for this flight.")

st.caption("Impact = SHAP contribution in log-odds. Positive pushes risk up, negative pulls it down. "
           "The explanation above is generated *only* from these drivers.")