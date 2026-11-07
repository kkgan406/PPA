"""Four-host CLI and native-comparison checks; synthetic data are unit fixtures only."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from test_ppa import manifest
from ppa.data import ppr_metadata, save_cache
from ppa.experiment import native_comparison
from ppa.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
HOSTS = {"CLIP": 768, "DINOv2": 768, "CO-SPY": 1664, "Effort": 1024}


@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("fusion", ["feature", "score"])
def test_four_host_cli_native_comparison(manifest, tmp_path, host, fusion):
    path, f = manifest
    rng = np.random.default_rng(20)
    ppr = rng.normal(size=(len(f), 1536)).astype("float32")
    features = rng.normal(size=(len(f), HOSTS[host])).astype("float32")
    native = rng.normal(size=len(f))
    native[f.split.eq("test").to_numpy()] = 0  # Explicit historical native tie behavior.
    meta = ppr_metadata(host=host, host_frozen=True, host_provenance=dict(
        checkpoint_sha256="b" * 64, code_commit="synthetic-interface-test",
        feature_layer="synthetic", normalization="synthetic", preprocessing="synthetic",
        native_logit="synthetic"))
    cache = tmp_path / "fixture.npz"
    import torch
    probability = torch.sigmoid(torch.from_numpy(native.astype("float32"))).numpy() if host == "CO-SPY" else None
    save_cache(cache, f, ppr, meta, features, native, native_probability=probability)
    out = tmp_path / "output"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/main_results.py"),
        "--dataset", str(path), "--cache", str(cache), "--host", host,
        "--fusion", fusion, "--output-dir", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    seal = json.loads((out / "source_seal.json").read_text())
    assert seal["source_only"] and not seal["OOD_used_for_selection"]
    assert seal["mode"] == ("feature" if fusion == "feature" else "effort" if host == "Effort" else "oof_lr")
    assert seal["threshold_source"] == ("source VAL max BA" if host == "Effort" and fusion == "score" else "fixed zero")
    assert seal["inference_dtype"] == ("float32" if host == "Effort" else "float64")
    pred = pd.read_csv(out / "predictions.csv")
    assert pred.native_prediction.eq(0 if host == "Effort" else 1).all()
    original = pd.read_csv(out / "metrics.csv")
    compare = pd.read_csv(out / "native_vs_ppa.csv")
    for key in ["ap", "acc", "auc", "ba"]:
        np.testing.assert_allclose(compare[f"{key}_ppa"], original[key], atol=1e-14)
        np.testing.assert_allclose(compare[f"delta_{key}"], compare[f"{key}_ppa"] - compare[f"{key}_native"], atol=1e-14)
    assert compare.threshold_ppa.eq(seal["threshold"]).all()
    assert (out / "native_metrics.csv").is_file()


@pytest.mark.parametrize("host", ["CLIMP", "C2P-CLIP", "AIDE"])
def test_excluded_hosts_rejected_by_main_cli(host):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/main_results.py"),
        "--dataset", "unused", "--cache", "unused", "--host", host,
        "--fusion", "feature", "--output-dir", "unused"], capture_output=True, text=True)
    assert result.returncode == 2 and "invalid choice" in result.stderr


def test_scope_and_raw_adapter_guard():
    from ppa.hosts import FrozenHost, preprocess
    assert set(yaml.safe_load((ROOT / "configs/hosts.yaml").read_text())) == set(HOSTS)
    for host in ["CLIMP", "C2P-CLIP", "AIDE", "CO-SPY", "Effort"]:
        with pytest.raises(ValueError):
            FrozenHost(host, {})
        with pytest.raises(ValueError):
            preprocess(host)


def test_native_comparison_retains_negative_gains_and_scopes():
    frame = pd.DataFrame(dict(dataset=["genimage"] * 8, generator=["ADM"] * 4 + ["stable_diffusion_v_1_4"] * 4,
        label=[0, 0, 1, 1] * 2, sha256=list("abcdefgh")))
    native_score = np.array([-2, -1, 1, 2] * 2)
    augmented = evaluate(frame, -native_score)
    augmented["evaluation_range"] = "all supplied test observations; overlap disclosed"
    clean = frame.sha256.ne("e")
    disjoint = evaluate(frame.loc[clean], -native_score[clean])
    disjoint["evaluation_range"] = "source-content-disjoint test observations"
    augmented = pd.concat([augmented, disjoint], ignore_index=True)
    augmented["threshold"], augmented["threshold_source"], augmented["model"] = 0, "fixed zero", "CLIP-feature"
    n, comparison = native_comparison(frame, native_score, augmented, "CLIP", ["e"])
    assert comparison.delta_ap.lt(0).all() and comparison.delta_acc.eq(-1).all()
    assert len(comparison) == len(augmented)
    assert "Macro-1-unseen" in set(comparison.scope)


def test_cospy_native_probability_ranking_preserves_saturation_ties():
    import torch
    from ppa.experiment import native_evaluate
    frame = pd.DataFrame(dict(dataset=["genimage"] * 4, generator=["ADM"] * 4, label=[0, 1, 0, 1]))
    logits = np.array([-2, 1, 90, 100], dtype=np.float32)
    result = native_evaluate(frame, logits, "CO-SPY", torch.sigmoid(torch.from_numpy(logits)).numpy())
    ranked = evaluate(frame, torch.sigmoid(torch.from_numpy(logits)).numpy(), .5)
    raw = evaluate(frame, logits)
    np.testing.assert_array_equal(result.ap, ranked.ap)
    np.testing.assert_array_equal(result.auc, ranked.auc)
    np.testing.assert_array_equal(result.acc, raw.acc)
    assert not result.ap.equals(raw.ap)


def test_missing_cospy_native_rank_is_explicitly_rejected(manifest, tmp_path):
    from ppa.data import load_native_probability
    path, frame = manifest
    cache = tmp_path / "missing.npz"
    save_cache(cache, frame, np.zeros((len(frame),1536), dtype="float32"), ppr_metadata())
    with pytest.raises(ValueError, match="original saved"):
        load_native_probability(cache, frame)
