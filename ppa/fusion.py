"""Small source-only linear heads, preserving host-specific score protocols."""
from dataclasses import dataclass, field
import warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from .metrics import measures, source_ba_threshold

GRID = (0.01, 0.1, 1.0, 10.0)
ALPHAS = (0.0, 0.25, 0.5, 1.0)


def fit_linear(x, y, c: float, seed: int = 42):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("Invalid training features")
    for iterations in (3000, 10000, 30000):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", solver="lbfgs", max_iter=iterations, random_state=seed))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x, y)
            return model
        except ConvergenceWarning:
            if iterations == 30000:
                raise


def select_linear(x, y, val, yval, seed=42, selection="ap_acc_auc", validation_dtype="float64"):
    candidates, models = [], {}
    for c in GRID:
        m = fit_linear(x, y, c, seed)
        models[c] = m
        candidates.append(dict(C=c, **measures(yval, m.decision_function(np.asarray(val, dtype=validation_dtype)))))
    keys = ("auc", "ap") if selection == "auc_ap" else ("ap", "acc", "auc")
    best = sorted(candidates, key=lambda r: (*[-r[k] for k in keys], r["C"]))[0]
    return models[best["C"]], [dict(**r, selected=r["C"] == best["C"]) for r in candidates]


@dataclass
class Fusion:
    mode: str
    seed: int = 42
    inference_dtype: str = "float64"
    validation_dtype: str = "float64"
    response: object = None
    head: object = None
    folds: list = field(default_factory=list)
    selection: list = field(default_factory=list)
    threshold: float = 0.0
    alpha: float = 0.0
    score_scaler: object = None
    fold_assignment: object = None
    oof_scores: object = None

    def fit(self, ppr, y, val_ppr, yval, host=None, val_host=None):
        """Call only with source TRAIN/VAL. Dataset guards live at the CLI boundary."""
        if self.mode not in ("ppr", "feature", "oof_lr", "additive", "standardized_additive", "effort"):
            raise ValueError("Unknown fusion protocol")
        if self.mode == "feature":
            self.head, self.selection = select_linear(np.column_stack([host, ppr]), y,
                np.column_stack([val_host, val_ppr]), yval, self.seed,
                validation_dtype=self.validation_dtype)
            return self
        self.response, rows = select_linear(ppr, y, val_ppr, yval, self.seed,
                                            validation_dtype=self.validation_dtype)
        self.selection = [dict(branch="ppr", **r) for r in rows]
        if self.mode == "ppr":
            return self
        if self.mode == "oof_lr":
            x, v = np.asarray(ppr, dtype=np.float64), np.asarray(val_ppr, dtype=np.float64)
            oof = np.full(len(y), np.nan)
            assignment = np.full(len(y), -1, dtype=int)
            val_scores = []
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            for k, (train, hold) in enumerate(splitter.split(x, y)):
                m = fit_linear(x[train], np.asarray(y)[train], self.response[-1].C, self.seed)
                oof[hold] = m.decision_function(x[hold])
                assignment[hold] = k
                self.folds.append(m)
                val_scores.append(m.decision_function(v))
            if not np.isfinite(oof).all() or (assignment < 0).any():
                raise ValueError("Incomplete OOF predictions")
            self.fold_assignment, self.oof_scores = assignment, oof
            self.head, rows = select_linear(np.column_stack([host, oof]), y,
                np.column_stack([val_host, np.mean(val_scores, axis=0)]), yval, self.seed)
            self.selection += [dict(branch="score", **r) for r in rows]
            return self
        response_train = self.response.decision_function(np.asarray(ppr, dtype=self.inference_dtype))
        response_val = self.response.decision_function(np.asarray(val_ppr, dtype=self.inference_dtype))
        h = np.asarray(val_host).reshape(-1)
        if self.mode == "standardized_additive":
            self.score_scaler = StandardScaler().fit(np.column_stack([host, response_train]))
            if (self.score_scaler.var_ <= 0).any():
                raise ValueError("Cannot standardize a constant source score")
            hv, response_val = self.score_scaler.transform(np.column_stack([h, response_val])).T
            h = hv
        candidates = [dict(alpha=a, **measures(yval, h+a*response_val)) for a in ALPHAS]
        best = sorted(candidates, key=lambda r: (-r["ap"], -r["acc"], -r["auc"], r["alpha"]))[0]
        self.alpha = best["alpha"]
        self.selection += [dict(branch="alpha", **r, selected=r["alpha"] == self.alpha) for r in candidates]
        if self.mode == "effort":
            self.threshold = source_ba_threshold(yval, h+self.alpha*response_val)
        return self

    def predict(self, ppr, host=None):
        ppr = np.asarray(ppr, dtype=self.inference_dtype)
        if self.mode == "feature":
            return self.head.decision_function(np.asarray(np.column_stack([host, ppr]), dtype=self.inference_dtype))
        if self.mode == "ppr":
            return self.response.decision_function(ppr)
        if self.mode == "oof_lr":
            response = np.mean([m.decision_function(ppr) for m in self.folds], axis=0)
            return self.head.decision_function(np.column_stack([host, response]))
        response = self.response.decision_function(ppr)
        h = np.asarray(host).reshape(-1)
        if self.score_scaler is not None:
            h, response = self.score_scaler.transform(np.column_stack([h, response])).T
        return h + self.alpha * response
