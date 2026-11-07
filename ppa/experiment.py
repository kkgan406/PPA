"""Shared source fitting and evaluation for the two supervised entry points."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from .data import (read_manifest, validate_splits, load_cache, new_output, write_json,
                   environment, ppr_metadata, load_native_probability)
from .fusion import Fusion
from .metrics import evaluate, KEYS
from .response import file_sha256

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def run(args, host=None):
    f = read_manifest(args.dataset)
    overlap = validate_splits(f)
    spec = None
    if host:
        spec = yaml.safe_load((CONFIG_ROOT / "hosts.yaml").read_text())[host]
    ppr, features, logits, metadata = load_cache(args.cache, f, host, spec["dimension"] if spec else None)
    native_probability = load_native_probability(args.cache, f) if host == "CO-SPY" else None
    tr, va, te = [np.flatnonzero(f.split.to_numpy() == s) for s in ("train", "val", "test")]
    mode = "ppr" if host is None else ("feature" if args.fusion == "feature" else spec["score_protocol"])
    host_values = features if mode == "feature" else logits
    model = Fusion(mode, seed=spec.get("seed", 42) if spec else 42,
                   inference_dtype=spec.get("inference_dtype", "float64") if spec else "float64",
                   validation_dtype=spec.get("validation_dtype", "float64") if spec else "float64")
    model.fit(ppr[tr], f.label.to_numpy()[tr], ppr[va], f.label.to_numpy()[va],
              host_values[tr] if host else None, host_values[va] if host else None)
    out = new_output(args.output_dir)
    joblib.dump(model, out / "model.joblib")
    pd.DataFrame(model.selection).to_csv(out / "source_selection.csv", index=False)
    overlap.to_csv(out / "source_overlap.csv", index=False)
    seal = dict(mode=mode, host=host, source_only=True, OOD_used_for_selection=False,
                source_train_n=len(tr), source_val_n=len(va), threshold=model.threshold,
                threshold_source="source VAL max BA" if mode == "effort" else "fixed zero",
                alpha=model.alpha, seed=model.seed, inference_dtype=model.inference_dtype,
                validation_dtype=model.validation_dtype, score_orientation="higher=fake; real=0/fake=1",
                evaluated_splits=["test"], manifest_sha256=file_sha256(args.dataset),
                source_ids_sha256={s:__import__("hashlib").sha256("\n".join(f.iloc[ix].sample_id).encode()).hexdigest()
                                   for s, ix in [("train", tr), ("val", va)]},
                cache_sha256=file_sha256(args.cache), cache_metadata=metadata,
                model_sha256=file_sha256(out / "model.joblib"), environment=environment())
    if model.fold_assignment is not None:
        ledger = f.iloc[tr][["sample_id", "sha256", "label"]].copy()
        ledger["fold"] = model.fold_assignment
        ledger["ppr_oof_logit"] = model.oof_scores
        ledger.to_csv(out / "oof_ledger.csv", index=False)
        seal["oof_ledger_sha256"] = file_sha256(out / "oof_ledger.csv")
        seal["external_ppr_score"] = "arithmetic mean of five fold logits; full-fit head unused"
    # Persist source decisions before computing any target score/metric.
    write_json(out / "source_seal.json", seal)
    (out / "source_seal.sha256").write_text(file_sha256(out / "source_seal.json") + "\n")
    scores = model.predict(ppr[te], host_values[te] if host else None)
    test = f.iloc[te].reset_index(drop=True)
    predictions = test.drop(columns="path").copy()
    predictions["score"] = scores
    predictions["prediction"] = (scores >= model.threshold).astype(int)
    if host:
        predictions["native_logit"] = logits[te]
        if native_probability is not None:
            predictions["native_probability"] = native_probability[te]
        predictions["native_prediction"] = ((logits[te] > 0) if host == "Effort" else (logits[te] >= 0)).astype(int)
    predictions.to_csv(out / "predictions.csv", index=False, float_format="%.17g")
    results = evaluate(test, scores, model.threshold)
    results["evaluation_range"] = "all supplied test observations; overlap disclosed"
    # A strict content-disjoint descriptive range complements the family exclusion.
    clean = ~test.sha256.isin(f.iloc[np.r_[tr, va]].sha256).to_numpy()
    group_counts = test.loc[clean].groupby(["dataset", "generator"]).label.nunique()
    if clean.any() and len(group_counts) and (group_counts == 2).all():
        strict = evaluate(test.loc[clean], scores[clean], model.threshold)
        strict["evaluation_range"] = "source-content-disjoint test observations"
        results = pd.concat([results, strict], ignore_index=True)
    results["model"] = "PPR-only" if host is None else f"{host}-{args.fusion}"
    results["threshold"] = model.threshold
    results["threshold_source"] = seal["threshold_source"]
    results["positive_label"] = 1
    results["score_orientation"] = "higher=fake"
    results["evaluated_split"] = "test"
    results.to_csv(out / "metrics.csv", index=False, float_format="%.17g")
    comparison_hashes = {}
    if host:
        native, comparison = native_comparison(test, logits[te], results, host,
            f.iloc[np.r_[tr, va]].sha256,
            native_probability[te] if native_probability is not None else None)
        native.to_csv(out / "native_metrics.csv", index=False, float_format="%.17g")
        comparison.to_csv(out / "native_vs_ppa.csv", index=False, float_format="%.17g")
        comparison_hashes = {name: file_sha256(out / name) for name in
                             ["native_metrics.csv", "native_vs_ppa.csv"]}
    write_json(out / "metadata.json", dict(comparison_sha256=comparison_hashes, source_seal_sha256=file_sha256(out / "source_seal.json"),
        metrics_sha256=file_sha256(out / "metrics.csv"), predictions_sha256=file_sha256(out / "predictions.csv"),
        code_sha256={str(p.relative_to(Path(__file__).parent)):file_sha256(p)
                     for p in sorted(Path(__file__).parent.glob("*.py"))},
        warning="Macro including SD1.4 is not strictly all-OOD. Supplied subsets are not full benchmark results."))
    print(results[results.aggregation == "equal_generator_macro"].to_string(index=False))


def native_comparison(test, native_scores, augmented, host, source_hashes, native_probability=None):
    """Report the unchanged native operating point beside existing PPA metrics.

    This does not select a route/head/threshold or alter augmented predictions.
    Effort's native two-logit argmax uses strict margin>0 (ties predict real).
    """
    strict = host == "Effort"
    native = native_evaluate(test, native_scores, host, native_probability)
    native["evaluation_range"] = "all supplied test observations; overlap disclosed"
    clean = ~test.sha256.isin(source_hashes).to_numpy()
    if "source-content-disjoint test observations" in set(augmented.evaluation_range):
        disjoint = native_evaluate(test.loc[clean], np.asarray(native_scores)[clean], host,
            np.asarray(native_probability)[clean] if native_probability is not None else None)
        disjoint["evaluation_range"] = "source-content-disjoint test observations"
        native = pd.concat([native, disjoint], ignore_index=True)
    native["model"] = f"{host}-native"
    native["threshold"] = 0.0
    native["threshold_source"] = "historical native fixed zero"
    native["threshold_comparison"] = ">" if strict else ">="
    native["positive_label"], native["score_orientation"], native["evaluated_split"] = 1, "higher=fake", "test"
    keys = ["dataset", "scope", "aggregation", "n", "evaluation_range"]
    cols = [*keys, *KEYS, "threshold", "threshold_source", "model"]
    comparison = native[cols].merge(augmented[cols], on=keys, how="outer",
        suffixes=("_native", "_ppa"), validate="one_to_one", indicator=True)
    if not comparison._merge.eq("both").all():
        raise ValueError("Native/PPA metric scopes must match")
    comparison = comparison.drop(columns="_merge")
    comparison["threshold_comparison_native"] = ">" if strict else ">="
    comparison["threshold_comparison_ppa"] = ">="
    for key in KEYS:
        comparison[f"delta_{key}"] = comparison[f"{key}_ppa"] - comparison[f"{key}_native"]
    return native, comparison


def native_evaluate(frame, logits, host, native_probability=None):
    """Preserve formal native ranking and operating-point conventions.

    CO-SPY formal unified tables rank its original FP32 sigmoid probabilities
    (saturation ties included), but ACC/BA use canonical raw-logit >=0.
    Effort native argmax ties predict real; other retained native margins use >=0.
    """
    result = evaluate(frame, logits, threshold=0, strict=host == "Effort")
    if host == "CO-SPY":
        if native_probability is None:
            raise ValueError("CO-SPY native ranking requires original saved probabilities; no sigmoid recomputation")
        ranked = evaluate(frame, native_probability, threshold=.5)
        result[["ap", "auc"]] = ranked[["ap", "auc"]]
    result["ranking_score"] = "original saved FP32 sigmoid probability" if host == "CO-SPY" else "native logit/margin"
    return result
