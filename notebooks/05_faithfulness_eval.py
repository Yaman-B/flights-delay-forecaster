# %% --- Setup: load the frozen bundle + eval set  -----------
import sys, json, hashlib
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import lightgbm as lgb, joblib
from anthropic import Anthropic
from src.paths import PROJECT_ROOT
from src.llm.explain import FlightExplainer
from src.llm.narrate import narrate, SYSTEM

MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR   = PROJECT_ROOT / "reports" / "eval"

booster = lgb.Booster(model_file=str(MODELS_DIR / "gbm.txt"))
iso     = joblib.load(MODELS_DIR / "calibrator_isotonic.joblib")
config  = json.loads((MODELS_DIR / "config.json").read_text())
FEATURES, CAT = config["features"], config["cat_features"]

eval_set   = pd.read_parquet(EVAL_DIR / "eval_set.parquet")
CAT_CATEGORIES = config["cat_categories"]
explainer = FlightExplainer(model=booster, calibrator=iso, features=FEATURES,
                            base_rate=config["base_rate"], cat_categories=CAT_CATEGORIES,
                            threshold=config["threshold"])

PROMPT_VERSION = hashlib.md5(SYSTEM.encode()).hexdigest()[:8]   # auto-changes if prompt changes
print(f"loaded bundle + {len(eval_set)} eval flights | prompt {PROMPT_VERSION}")
# %% --- sanity: explainer encodes categoricals to the TRAINING codes -------
maxdiff = 0
for c in CAT:
    want = pd.Categorical(eval_set[c], categories=CAT_CATEGORIES[c]).codes   # training-aligned
    got  = np.array([explainer._codes[c].get(v, -1) for v in eval_set[c]])
    maxdiff = max(maxdiff, int(np.abs(want - got).max()))
print("max code mismatch across CAT cols:", maxdiff)        # expect 0

probs = [explainer.explain(r)["prediction"]["probability"] for _, r in eval_set.head(8).iterrows()]
print("sample calibrated probabilities:", [round(p, 3) for p in probs])
# %% --- Generate 250 explanations -------------
# 250 Haiku calls
GEN_PATH = EVAL_DIR / f"explanations_{PROMPT_VERSION}.jsonl"

done = set()
if GEN_PATH.exists():
    done = {json.loads(l)["flight_id"] for l in GEN_PATH.read_text().splitlines() if l.strip()}
print(f"already done: {len(done)}/{len(eval_set)} -> {GEN_PATH.name}")

client = Anthropic()
with open(GEN_PATH, "a") as f:
    for n, (fid, row) in enumerate(eval_set.iterrows(), 1):
        if int(fid) in done:
            continue
        try:
            obj  = explainer.explain(row)
            text = narrate(obj, client=client)
            f.write(json.dumps({"flight_id": int(fid), "object": obj,
                                "explanation": text}, default=str) + "\n")
            f.flush()
        except Exception as e:
            print(f"  [{fid}] error: {e}")
        if n % 25 == 0:
            print(f"  {n}/{len(eval_set)}")
print("generation complete ->", GEN_PATH.name)
# %% --- Peek at a few generated explanations -------------------------------
recs = [json.loads(l) for l in GEN_PATH.read_text().splitlines() if l.strip()]
print(f"{len(recs)} explanations on file\n")
for r in recs[:3]:
    o = r["object"]
    print(f"--- {o['flight']['origin']}->{o['flight']['dest']} | "
          f"{o['prediction']['risk_level']} ({o['prediction']['probability_text']}) ---")
    print(r["explanation"], "\n")