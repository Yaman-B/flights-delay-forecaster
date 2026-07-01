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

# %% --- Run the judge over all 250 (Sonnet 4.6; resumable, double-keyed) ---
import hashlib
from anthropic import Anthropic
from src.llm.judge import judge, JUDGE_SYSTEM, DIMENSIONS

client = Anthropic()
GEN_PATH      = EVAL_DIR / f"explanations_{PROMPT_VERSION}.jsonl"
JUDGE_VERSION = hashlib.md5(JUDGE_SYSTEM.encode()).hexdigest()[:8]
VERDICT_PATH  = EVAL_DIR / f"verdicts_{PROMPT_VERSION}_{JUDGE_VERSION}.jsonl"

gen = [json.loads(l) for l in GEN_PATH.read_text().splitlines() if l.strip()]
done = ({json.loads(l)["flight_id"] for l in VERDICT_PATH.read_text().splitlines() if l.strip()}
        if VERDICT_PATH.exists() else set())
print(f"judging {len(gen)} | prompt {PROMPT_VERSION} judge {JUDGE_VERSION} | done {len(done)}")

with open(VERDICT_PATH, "a") as f:
    for n, r in enumerate(gen, 1):
        if r["flight_id"] in done:
            continue
        try:
            v = judge(r["object"], r["explanation"], client=client)
            f.write(json.dumps({"flight_id": r["flight_id"], **v}) + "\n"); f.flush()
        except Exception as e:
            print(f"  [{r['flight_id']}] {e}")
        if n % 25 == 0:
            print(f"  {n}/{len(gen)}")
print("judging complete ->", VERDICT_PATH.name)

# %% --- Judge results: PROVISIONAL rates (NOT trusted until validated) -----
V = pd.DataFrame([json.loads(l) for l in VERDICT_PATH.read_text().splitlines() if l.strip()])
print(f"verdicts: {len(V)}   |   faithful: {V['faithful'].mean():.1%}\n")
for d in DIMENSIONS:
    print(f"  {d:28s} {V[d].apply(lambda x: x['present']).mean():.1%}")

flagged = V[~V["faithful"]]
print(f"\n{len(flagged)} flagged — sample evidence:")
for _, r in flagged.head(5).iterrows():
    hit = next(d for d in DIMENSIONS if r[d]["present"])
    print(f"  [{r['flight_id']}] {hit}: {r[hit]['evidence'][:130]}")
# %%
# %% --- Build the blind label set: 50 flights, flagged oversampled ---------
V = pd.DataFrame([json.loads(l) for l in VERDICT_PATH.read_text().splitlines() if l.strip()])
gen_by_id = {r["flight_id"]: r for r in gen}

N_LABEL, SEED = 50, 11
flagged_ids = V.loc[~V["faithful"], "flight_id"].tolist()          # all 5 flagged
faithful_ids = V.loc[V["faithful"], "flight_id"].sample(
    N_LABEL - len(flagged_ids), random_state=SEED).tolist()        # 45 judge-called-faithful
label_ids = flagged_ids + faithful_ids
np.random.RandomState(SEED).shuffle(label_ids)                     # shuffle so order hides which is which

LABEL_PATH = EVAL_DIR / "human_labels.jsonl"
labeled = ({json.loads(l)["flight_id"] for l in LABEL_PATH.read_text().splitlines() if l.strip()}
           if LABEL_PATH.exists() else set())
print(f"{len(label_ids)} to label ({len(flagged_ids)} flagged + {len(faithful_ids)} faithful), "
      f"blind. Already labeled: {len(labeled)}")

# %% --- Label blind (run this cell to do a labeling session) ---------------
RUBRIC = """Choose the verdict for THIS explanation vs its analysis object:
  [f] faithful            — says only what the drivers support
  [1] fabrication         — a CAUSE with no matching driver (paraphrase/among-drivers is OK)
  [2] direction_error     — a driver described with the wrong sign
  [3] certainty_language  — states outcome as certain ("will", "cannot", "expect on-time")
  [4] probability         — a number conflicting with probability_text, or 0%/100%
  (allowed, NOT violations: paraphrasing, expanding codes to names, general travel advice)
  [s] skip   [q] quit and save"""
CODE = {"f": "faithful", "1": "fabrication", "2": "direction_error",
        "3": "certainty_language", "4": "probability_misrepresented"}

todo = [fid for fid in label_ids if fid not in labeled]
print(RUBRIC + f"\n\n{len(todo)} remaining.\n")
with open(LABEL_PATH, "a") as f:
    for k, fid in enumerate(todo, 1):
        r = gen_by_id[fid]
        o = r["object"]
        drv = {"increasing": [d["text"] for d in o["drivers"]["increasing"]],
               "decreasing": [d["text"] for d in o["drivers"]["decreasing"]]}
        print(f"\n───── {k}/{len(todo)} · flight {fid} ─────")
        print(f"risk: {o['prediction']['risk_level']} ({o['prediction']['probability_text']}) | "
              f"has_inbound: {o['meta']['has_inbound']}")
        print("increasing drivers:", drv["increasing"])
        print("decreasing drivers:", drv["decreasing"])
        print(f"\nEXPLANATION:\n{r['explanation']}\n")
        ans = input("verdict [f/1/2/3/4/s/q]: ").strip().lower()
        if ans == "q":
            print("saved, stopped."); break
        if ans == "s" or ans not in CODE:
            print("skipped."); continue
        f.write(json.dumps({"flight_id": fid, "human_label": CODE[ans],
                            "human_faithful": ans == "f"}) + "\n"); f.flush()
        print(f"  recorded: {CODE[ans]}")
# %%
# %% --- Validate the judge against human labels ----------------------------
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, confusion_matrix

H = pd.DataFrame([json.loads(l) for l in (EVAL_DIR / "human_labels.jsonl").read_text().splitlines() if l.strip()])
J = pd.DataFrame([json.loads(l) for l in VERDICT_PATH.read_text().splitlines() if l.strip()])
D = H.merge(J[["flight_id", "faithful"]], on="flight_id").rename(columns={"faithful": "judge_faithful"})

# "unfaithful" = the positive class we care about catching
h_unfaith = ~D["human_faithful"]
j_unfaith = ~D["judge_faithful"]
print(f"labeled: {len(D)} | you-unfaithful: {h_unfaith.sum()} | judge-unfaithful: {j_unfaith.sum()}\n")
print("confusion (rows=you, cols=judge), unfaithful=positive:")
print(confusion_matrix(h_unfaith, j_unfaith), "\n")
print(f"agreement (raw):        {(D['human_faithful'] == D['judge_faithful']).mean():.1%}")
print(f"Cohen's kappa:          {cohen_kappa_score(h_unfaith, j_unfaith):.3f}")
print(f"judge precision (unfaith): {precision_score(h_unfaith, j_unfaith, zero_division=0):.3f}")
print(f"judge recall (unfaith):    {recall_score(h_unfaith, j_unfaith, zero_division=0):.3f}")

print("\nDisagreements:")
for _, r in D[D["human_faithful"] != D["judge_faithful"]].iterrows():
    print(f"  {r['flight_id']}: you={r['human_label']:16s} judge_faithful={r['judge_faithful']}")
# %%
# %% --- Validate the judge + log the faithfulness result to MLflow ---------
import mlflow
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, confusion_matrix
from src.paths import PROJECT_ROOT

J = pd.DataFrame([json.loads(l) for l in VERDICT_PATH.read_text().splitlines() if l.strip()])
H = pd.DataFrame([json.loads(l) for l in (EVAL_DIR / "human_labels.jsonl").read_text().splitlines() if l.strip()])

faithful_rate = J["faithful"].mean()
dim_rates = {d: J[d].apply(lambda x: x["present"]).mean() for d in DIMENSIONS}

D = H.merge(J[["flight_id", "faithful"]], on="flight_id").rename(columns={"faithful": "judge_faithful"})
h_un, j_un = ~D["human_faithful"], ~D["judge_faithful"]
agreement = (D["human_faithful"] == D["judge_faithful"]).mean()
kappa = cohen_kappa_score(h_un, j_un)
prec  = precision_score(h_un, j_un, zero_division=0)
rec   = recall_score(h_un, j_un, zero_division=0)

print(f"faithful rate (judge, n={len(J)}): {faithful_rate:.1%}")
for d, r in dim_rates.items(): print(f"  {d:28s} {r:.1%}")
print(f"\njudge vs human (n={len(D)}): agreement {agreement:.1%} | kappa {kappa:.3f} "
      f"| precision {prec:.3f} | recall {rec:.3f}")
print("confusion (rows=human, cols=judge; unfaithful=positive):\n", confusion_matrix(h_un, j_un))

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("flight-delay-faithfulness")
with mlflow.start_run(run_name=f"faithfulness_{PROMPT_VERSION}"):
    mlflow.log_params({"prompt_version": PROMPT_VERSION, "judge_version": JUDGE_VERSION,
                       "generator": "claude-haiku-4-5", "judge": "claude-sonnet-4-6",
                       "n_eval": len(J), "n_labeled": len(D)})
    mlflow.log_metrics({"faithful_rate": float(faithful_rate),
                        **{f"rate_{d}": float(r) for d, r in dim_rates.items()},
                        "human_agreement": float(agreement), "judge_kappa": float(kappa),
                        "judge_precision_unfaithful": float(prec),
                        "judge_recall_unfaithful": float(rec)})
    mlflow.set_tag("caveat", "judge over-flags paraphrase-as-fabrication; faithful_rate is a "
                             "conservative floor; too few true-unfaithful cases to estimate recall")
    mlflow.log_artifact(str(VERDICT_PATH))
    mlflow.log_artifact(str(EVAL_DIR / "human_labels.jsonl"))
print("\nlogged -> experiment 'flight-delay-faithfulness'")
# %%

# %% --- Baseline arm: naive prompt, same flights, same judge ---------------
import importlib, src.llm.narrate
importlib.reload(src.llm.narrate)
from src.llm.narrate import narrate_naive, NAIVE_SYSTEM

client = Anthropic()
NAIVE_VERSION = hashlib.md5(NAIVE_SYSTEM.encode()).hexdigest()[:8]
NAIVE_GEN  = EVAL_DIR / f"explanations_naive_{NAIVE_VERSION}.jsonl"
NAIVE_VERD = EVAL_DIR / f"verdicts_naive_{NAIVE_VERSION}_{JUDGE_VERSION}.jsonl"

gen = [json.loads(l) for l in (EVAL_DIR / f"explanations_{PROMPT_VERSION}.jsonl").read_text().splitlines() if l.strip()]

# 1) generate naive explanations (resumable)
done = {json.loads(l)["flight_id"] for l in NAIVE_GEN.read_text().splitlines()} if NAIVE_GEN.exists() else set()
with open(NAIVE_GEN, "a") as fh:
    for n, r in enumerate(gen, 1):
        if r["flight_id"] in done: continue
        try:
            txt = narrate_naive(r["object"], client=client)
            fh.write(json.dumps({"flight_id": r["flight_id"], "object": r["object"],
                                 "explanation": txt}, default=str) + "\n"); fh.flush()
        except Exception as e: print(f"  gen [{r['flight_id']}] {e}")
        if n % 50 == 0: print(f"  gen {n}/{len(gen)}")

# 2) judge them (same judge, same rubric)
ndone = {json.loads(l)["flight_id"] for l in NAIVE_VERD.read_text().splitlines()} if NAIVE_VERD.exists() else set()
naive_gen = [json.loads(l) for l in NAIVE_GEN.read_text().splitlines() if l.strip()]
with open(NAIVE_VERD, "a") as fh:
    for n, r in enumerate(naive_gen, 1):
        if r["flight_id"] in ndone: continue
        try:
            v = judge(r["object"], r["explanation"], client=client)
            fh.write(json.dumps({"flight_id": r["flight_id"], **v}) + "\n"); fh.flush()
        except Exception as e: print(f"  judge [{r['flight_id']}] {e}")
        if n % 50 == 0: print(f"  judge {n}/{len(naive_gen)}")

Jn = pd.DataFrame([json.loads(l) for l in NAIVE_VERD.read_text().splitlines() if l.strip()])
naive_rate = Jn["faithful"].mean()
print(f"\nBASELINE (naive) faithful: {naive_rate:.1%}   vs   GROUNDED: {faithful_rate:.1%}")
print(f"lift: +{(faithful_rate - naive_rate)*100:.1f} pts")
for d in DIMENSIONS:
    print(f"  {d:28s} naive {Jn[d].apply(lambda x: x['present']).mean():.1%}")

# %% --- Log baseline arm to MLflow -----------------------------------------
mlflow.set_experiment("flight-delay-faithfulness")
with mlflow.start_run(run_name=f"faithfulness_naive_{NAIVE_VERSION}"):
    mlflow.log_params({"prompt_version": f"naive_{NAIVE_VERSION}", "judge_version": JUDGE_VERSION,
                       "generator": "claude-haiku-4-5", "judge": "claude-sonnet-4-6",
                       "arm": "baseline_no_drivers", "n_eval": len(Jn)})
    mlflow.log_metrics({"faithful_rate": float(naive_rate),
                        **{f"rate_{d}": float(Jn[d].apply(lambda x: x['present']).mean()) for d in DIMENSIONS}})
    mlflow.set_tag("caveat", "same over-strict judge as grounded arm; compare as a DELTA, not absolute")
    mlflow.log_artifact(str(NAIVE_VERD))
print("baseline logged -> compare the two runs in the MLflow UI")
# %%

# %% --- Diagnose any missing naive verdicts --------------------------------
from src.llm.judge import JUDGE_SYSTEM
import importlib, src.llm.judge
importlib.reload(src.llm.judge)

naive_gen = [json.loads(l) for l in NAIVE_GEN.read_text().splitlines() if l.strip()]
ndone = {json.loads(l)["flight_id"] for l in NAIVE_VERD.read_text().splitlines() if l.strip()}
missing = [r for r in naive_gen if r["flight_id"] not in ndone]
print(f"naive verdicts: {len(ndone)}/{len(naive_gen)} | missing: {[r['flight_id'] for r in missing]}\n")

for r in missing:
    user = (f"<analysis>\n{json.dumps(r['object'], default=str)}\n</analysis>\n\n"
            f"<explanation>\n{r['explanation']}\n</explanation>\n\nEvaluate the explanation.")
    msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=600, temperature=0,
                                 system=JUDGE_SYSTEM, messages=[{"role": "user", "content": user}])
    raw = "".join(b.text for b in msg.content if b.type == "text")
    print(f"=== {r['flight_id']} | stop={msg.stop_reason} | out_tokens={msg.usage.output_tokens} ===")
    print(raw)
# %%
