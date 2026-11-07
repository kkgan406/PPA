"""Fake-positive metrics with explicit pooled / equal-generator macro scopes."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score,
                             balanced_accuracy_score, roc_curve)

SOURCE_GENERATOR = "stable_diffusion_v_1_4"
KEYS = ("ap", "acc", "auc", "ba")


def measures(y, score, threshold: float = 0.0, strict: bool = False):
    y, score = np.asarray(y), np.asarray(score)
    if score.shape != y.shape or score.ndim != 1 or not np.isfinite(score).all():
        raise ValueError("Score/label alignment or finiteness error")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Each metric group needs both real=0 and fake=1")
    pred = score > threshold if strict else score >= threshold
    return dict(ap=float(average_precision_score(y, score)), auc=float(roc_auc_score(y, score)),
                acc=float(accuracy_score(y, pred)), ba=float(balanced_accuracy_score(y, pred)))


def source_ba_threshold(y, score) -> float:
    """Historical Effort: VAL maximum BA; tie closest zero then smaller threshold."""
    fpr, tpr, th = roc_curve(y, score, drop_intermediate=False)
    finite = np.isfinite(th)
    th, ba = th[finite], ((1 + tpr - fpr) / 2)[finite]
    best = ba.max()
    eligible = th[np.isclose(ba, best, atol=1e-12, rtol=0)]
    if abs(measures(y, score)["ba"] - best) <= 1e-12:
        eligible = np.r_[eligible, 0.0]
    return float(sorted(eligible, key=lambda v: (abs(v), v))[0])


def evaluate(frame: pd.DataFrame, score, threshold=0.0, strict=False) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    score = np.asarray(score)
    if score.shape != (len(frame),) or not np.isfinite(score).all():
        raise ValueError("Invalid prediction array")
    rows = []
    for dataset, part in frame.groupby("dataset", sort=False):
        per = []
        for generator, group in part.groupby("generator", sort=False):
            row = dict(dataset=dataset, scope=generator, aggregation="generator", n=len(group),
                       **measures(group.label.to_numpy(), score[group.index], threshold, strict))
            rows.append(row)
            per.append(row)
        macro = f"Macro-{len(per)}"
        scopes = [(macro, per)]
        if dataset == "genimage":
            unseen = [r for r in per if r["scope"] != SOURCE_GENERATOR]
            if unseen:
                scopes.append((f"Macro-{len(unseen)}-unseen", unseen))
        for name, selected in scopes:
            rows.append(dict(dataset=dataset, scope=name, aggregation="equal_generator_macro",
                             n=sum(r["n"] for r in selected),
                             **{k:float(np.mean([r[k] for r in selected])) for k in KEYS}))
        rows.append(dict(dataset=dataset, scope="pooled", aggregation="pooled", n=len(part),
                         **measures(part.label.to_numpy(), score[part.index], threshold, strict)))
    return pd.DataFrame(rows)
