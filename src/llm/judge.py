"""LLM-as-judge: score one explanation's faithfulness against its structured object.

Returns a per-dimension verdict (fabrication / direction_error / certainty_language
/ probability_misrepresented, each with an evidence span). The overall `faithful`
flag is derived in code from those four dimensions, not produced by the model.
"""
import json
from anthropic import Anthropic

DIMENSIONS = ["fabrication", "direction_error", "certainty_language", "probability_misrepresented"]

JUDGE_SYSTEM = """You are a strict evaluator checking whether a natural-language flight-disruption-risk explanation is FAITHFUL to the structured data it was generated from.

You are given (1) a JSON "analysis" object — a model's prediction and the specific drivers behind it — and (2) an "explanation" written from it. The analysis object is the ONLY ground truth. Judge whether the explanation asserts anything the object does not support.

The explanation MAY freely do all of the following — none are violations:
- paraphrase, reword, or combine drivers;
- expand airport/carrier codes into names (e.g. "LAS" -> "Las Vegas", "DL" -> "Delta");
- restate the probability using prediction.probability_text;
- describe the flight's identity (route, date, carrier);
- offer general travel advice (e.g. "allow extra time for connections").

Check exactly these four failure modes:

1. fabrication — the explanation states a CAUSE or reason for the risk that has NO corresponding driver in drivers.increasing or drivers.decreasing. A paraphrase of an existing driver is NOT fabrication; only an invented cause with no matching driver counts.
2. direction_error — the explanation describes a driver in the wrong direction: presenting an item from drivers.increasing as reducing risk, or an item from drivers.decreasing as raising risk.
3. certainty_language — the explanation states the outcome as certain rather than a risk, e.g. "will be delayed", "cannot depart on time", "you can expect an on-time arrival". Risk framing like "at high risk" is fine.
4. probability_misrepresented — the explanation states a probability that conflicts with prediction.probability_text, invents a different precise percentage, or says "0%" or "100%".

For each failure mode: set "present" true/false, and if present quote the exact offending span from the explanation in "evidence" (else empty string).

Output ONLY a single JSON object and nothing else — no reasoning, no analysis, no text before or after it. In the "evidence" and "notes" strings, do NOT use double-quote characters; refer to spans in your own words or with single quotes. Keep each evidence string under 30 words. Use exactly this schema:
{"fabrication": {"present": false, "evidence": ""}, "direction_error": {"present": false, "evidence": ""}, "certainty_language": {"present": false, "evidence": ""}, "probability_misrepresented": {"present": false, "evidence": ""}, "notes": ""}"""


def _parse(raw):
    s = raw.strip()
    if "```" in s:                                # take fenced block if present
        parts = s.split("```")
        s = max(parts, key=len)                   # the largest fence chunk = the JSON
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
    a, b = s.find("{"), s.rfind("}")              # slice the outermost { ... }
    if a == -1 or b == -1:
        raise ValueError(f"no JSON object found: {raw[:200]}")
    return json.loads(s[a:b + 1])


def judge(analysis, explanation, client=None, model="claude-sonnet-4-6", max_tokens=600):
    """Faithfulness verdict for one explanation. `faithful` is derived, not model-produced."""
    client = client or Anthropic()
    user = (f"<analysis>\n{json.dumps(analysis, default=str)}\n</analysis>\n\n"
            f"<explanation>\n{explanation}\n</explanation>\n\nEvaluate the explanation.")
    msg = client.messages.create(model=model, max_tokens=max_tokens, temperature=0,
                                 system=JUDGE_SYSTEM,
                                 messages=[{"role": "user", "content": user}])
    raw = "".join(b.text for b in msg.content if b.type == "text").strip()
    v = _parse(raw)
    for d in DIMENSIONS:                           # schema guard -> caught & retried on failure
        if d not in v or "present" not in v[d]:
            raise ValueError(f"malformed verdict ({d}): {raw[:200]}")
        v[d]["present"] = bool(v[d]["present"])
    v["faithful"] = not any(v[d]["present"] for d in DIMENSIONS)
    return v