"""Non-learned reference baselines: the bar every real model must clear."""
import numpy as np
import pandas as pd


def majority_baseline(train_df, eval_df, target="disrupted"):
    """Constant prediction = the training-period disruption rate (no-skill ref)."""
    rate = float(train_df[target].mean())
    return np.full(len(eval_df), rate)


def climatology_baseline(train_df, eval_df, group_cols=("Origin", "Dest", "hour"),
                         target="disrupted", alpha=10.0):
    """Empirical-Bayes-smoothed historical disruption rate per group.

    Learns one rate per `group_cols` combination on train_df only, shrunk toward
    the global train rate (a group needs ~alpha flights before its own rate
    outweighs the prior). Unseen groups fall back to the global rate. eval_df
    must already contain every column in group_cols.
    """
    global_rate = float(train_df[target].mean())
    agg = train_df.groupby(list(group_cols))[target].agg(s="sum", n="count")
    rate = (agg["s"] + alpha * global_rate) / (agg["n"] + alpha)
    keys = list(zip(*(eval_df[c] for c in group_cols)))
    mapped = rate.reindex(keys)
    return pd.Series(mapped.to_numpy(), index=eval_df.index).fillna(global_rate).to_numpy()