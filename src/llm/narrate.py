"""Turn a FlightExplainer object into a grounded, traveler-facing explanation.

The structured object is the ONLY sanctioned source of facts; the system prompt
constrains the model to it so the output stays faithful (and the faithfulness
eval can score it against the same object).
"""
import json
from anthropic import Anthropic

SYSTEM = """You write short, clear flight-disruption-risk explanations for everyday air travelers.

You are given a JSON object describing one flight: a model's predicted disruption probability, a risk level, and the specific factors ("drivers") that pushed the prediction up or down. A flight is "disrupted" if it arrives 15+ minutes late or is cancelled.

Follow these rules exactly:
- Use ONLY the information in the JSON. Do not add any cause, weather condition, air-traffic, holiday, congestion, or airport detail that is not present in the drivers. If a fact is not in the JSON, you do not know it and must not state it.
- State the risk level. For the probability, quote the exact phrase in `prediction.probability_text` (e.g., "about 78%", "over 99%", "under 1%"). Do not compute or state any other number, and never say "0%" or "100%". You may compare qualitatively to a typical flight using `base_rate`, but do not compute exact ratios.
- Each driver has a direction: "increasing" drivers raise the risk, "decreasing" drivers lower it. Never describe a driver in the wrong direction. Lead with increasing drivers for an elevated/high-risk flight and decreasing (protective) drivers for a low-risk flight; emphasize "strong" drivers over "slight" ones.
- This is a probability, never a certainty. Do NOT write that a flight "will" be delayed, "cannot depart on schedule", or that the traveler "can expect" it to arrive on time. Use only "at high/elevated/low risk" framing.
- Write 2-4 sentences of plain text for a traveler — no markdown, asterisks, bold, lists, jargon, or preamble. Output only the explanation. """


def narrate(explanation, client=None, model="claude-haiku-4-5", max_tokens=300, temperature=0):
    """Generate a natural-language explanation from a FlightExplainer object."""
    client = client or Anthropic()
    payload = json.dumps(explanation, indent=2)
    msg = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature, system=SYSTEM,
        messages=[{"role": "user", "content":
                   f"<flight_analysis>\n{payload}\n</flight_analysis>\n\n"
                   "Write the disruption-risk explanation for this flight."}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()

NAIVE_SYSTEM = """You write short flight-disruption-risk explanations for travelers. A flight is "disrupted" if it arrives 15+ minutes late or is cancelled.

You are given a flight's predicted disruption probability and some basic facts about it. Write a 2-4 sentence explanation of why this flight has the risk it does, in plain language for a traveler."""


def narrate_naive(explanation, client=None, model="claude-haiku-4-5", max_tokens=300, temperature=0):
    """BASELINE: prediction + raw flight facts, NO structured drivers, NO faithfulness
    rules. The 'before' arm for the faithfulness ablation."""
    import json
    client = client or Anthropic()
    f, p = explanation["flight"], explanation["prediction"]
    facts = {"origin": f["origin"], "dest": f["dest"], "carrier": f["carrier"],
             "departure_hour": f["dep_hour_local"], "date": f["date"],
             "predicted_disruption_probability": p["probability_text"]}
    msg = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature, system=NAIVE_SYSTEM,
        messages=[{"role": "user", "content":
                   f"<flight>\n{json.dumps(facts, default=str)}\n</flight>\n\n"
                   "Write the disruption-risk explanation."}])
    return "".join(b.text for b in msg.content if b.type == "text").strip()