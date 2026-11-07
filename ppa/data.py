"""Content-bound manifests and portable, pickle-free feature caches."""
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .preprocessing import PROTOCOL
from .response import ORDER, ALEXNET_SHA256, LPIPS_SHA256, file_sha256

FIELDS = ("sample_id", "sha256", "label", "generator", "split", "dataset")
PPR_DEFINITION = "alex-v0.1/raw-abs/L4L5/stat-major-MSM1536/std-correction0/jpeg90-50/v1"


def read_manifest(path):
    path = Path(path)
    f = pd.read_csv(path, dtype={"sample_id": str, "sha256": str})
    needed = {*FIELDS, "path"}
    if not needed <= set(f):
        raise ValueError(f"Missing manifest fields: {sorted(needed-set(f))}")
    if f[list(needed)].isna().any().any() or not len(f) or not f.sample_id.is_unique:
        raise ValueError("Empty manifest, missing values or duplicate image IDs")
    if not f.label.isin([0, 1]).all():
        raise ValueError("Labels must be real=0/fake=1")
    if not f.sha256.map(lambda s: bool(re.fullmatch(r"[0-9a-f]{64}", s))).all():
        raise ValueError("Expected full lowercase content SHA256")
    f["path"] = [str((path.parent / p).resolve()) for p in f.path]
    return f.reset_index(drop=True)


def validate_splits(f):
    if not f.split.isin(["train", "val", "test"]).all():
        raise ValueError("Only train/val/test split names are supported")
    tr, va, te = (f[f.split == s] for s in ("train", "val", "test"))
    if not all(len(p) for p in [tr, va, te]):
        raise ValueError("Need source TRAIN, source VAL, and TEST")
    for part in [tr, va]:
        if set(part.dataset) != {"source"} or not part.generator.isin(["SD1.4", "stable_diffusion_v_1_4"]).all():
            raise ValueError("TRAIN and VAL must use SD1.4 source only")
        if set(part.label) != {0, 1} or not part.sha256.is_unique:
            raise ValueError("Source split needs both classes and unique contents")
    if set(tr.sha256) & set(va.sha256) or set(tr.path) & set(va.path):
        raise ValueError("Source TRAIN/VAL leakage")
    if not te.dataset.isin(["genimage", "commgen15"]).all():
        raise ValueError("TEST supports GenImage and CommGen15 only")
    if (f.groupby("sha256").label.nunique() > 1).any():
        raise ValueError("Identical content has conflicting labels")
    # Target overlap is disclosed, never used for source model selection.
    source = set(tr.sha256) | set(va.sha256)
    rows = []
    for (dataset, generator), group in te.groupby(["dataset", "generator"]):
        rows.append(dict(dataset=dataset, generator=generator, n=len(group),
            train_overlap=int(group.sha256.isin(tr.sha256).sum()),
            val_overlap=int(group.sha256.isin(va.sha256).sum()),
            source_disjoint_n=int((~group.sha256.isin(source)).sum())))
    return pd.DataFrame(rows)


def ppr_metadata(**extra):
    return dict(schema=1, ppr_definition=PPR_DEFINITION, descriptor_order=list(ORDER),
                preprocessing=PROTOCOL, seed=42, dtype="float32", frozen=True,
                checkpoints={"alexnet": ALEXNET_SHA256, "lpips": LPIPS_SHA256}, **extra)


def save_cache(path, frame, ppr, metadata, features=None, logits=None, native_probability=None):
    arrays = {k: frame[k].to_numpy(dtype=np.int64 if k == "label" else str) for k in FIELDS}
    arrays.update(ppr=np.asarray(ppr, dtype=np.float32), metadata=json.dumps(metadata, sort_keys=True))
    for name, a in [("features", features), ("logits", logits), ("native_probability", native_probability)]:
        if a is not None:
            arrays[name] = np.asarray(a)
    np.savez_compressed(path, **arrays)


def load_cache(path, frame, host=None, dimension=None):
    with np.load(path, allow_pickle=False) as a:
        meta = json.loads(a["metadata"].item())
        for key, value in ppr_metadata().items():
            if meta.get(key) != value:
                raise ValueError(f"PPR cache protocol mismatch: {key}")
        ids = pd.Index(a["sample_id"].astype(str))
        take = ids.get_indexer(frame.sample_id)
        if not ids.is_unique or len(ids) != len(frame) or (take < 0).any():
            raise ValueError("Cache image IDs do not match manifest")
        for key in FIELDS:
            if not np.array_equal(a[key][take], frame[key].to_numpy()):
                raise ValueError(f"Cache alignment mismatch: {key}")
        ppr = a["ppr"][take]
        if ppr.shape != (len(frame), 1536) or ppr.dtype != np.float32 or not np.isfinite(ppr).all():
            raise ValueError("Expected finite float32 MSM1536 cache")
        features = logits = None
        if host:
            if meta.get("host") != host or meta.get("host_frozen") is not True:
                raise ValueError("Host identity/freeze metadata missing")
            provenance = meta.get("host_provenance", {})
            for key in ("checkpoint_sha256", "code_commit", "feature_layer", "normalization", "preprocessing", "native_logit"):
                if not provenance.get(key):
                    raise ValueError(f"Missing host provenance: {key}")
            if not re.fullmatch(r"[0-9a-f]{64}", provenance["checkpoint_sha256"]):
                raise ValueError("Expected full host checkpoint SHA256")
            features, logits = a["features"][take], a["logits"][take]
            if features.shape != (len(frame), dimension) or logits.shape != (len(frame),):
                raise ValueError("Host feature dimension or native logit shape mismatch")
            if not np.isfinite(features).all() or not np.isfinite(logits).all():
                raise ValueError("Non-finite host cache")
    return ppr, features, logits, meta


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment():
    import os
    import platform
    from importlib.metadata import version
    return dict(python=platform.python_version(), packages={p:version(p) for p in
        ["torch", "torchvision", "lpips", "Pillow", "numpy", "pandas", "scikit-learn", "joblib"]},
        thread_environment={k:os.environ.get(k) for k in
                            ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]})


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Output directory must be new or empty; preserve previous source seals")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_native_probability(path, frame):
    """Load original CO-SPY probability ranks after the ordinary cache validation.

    Raw logits remain the fusion input. Recomputing sigmoid can change historical
    FP32 saturation/tie behavior across devices, so reporting requires saved ranks.
    """
    with np.load(path, allow_pickle=False) as a:
        if "native_probability" not in a:
            raise ValueError("CO-SPY native comparison requires original saved native_probability; do not recompute sigmoid")
        ids = pd.Index(a["sample_id"].astype(str))
        take = ids.get_indexer(frame.sample_id)
        if not ids.is_unique or len(ids) != len(frame) or (take < 0).any():
            raise ValueError("Native probability image IDs do not match manifest")
        for key in FIELDS:
            if not np.array_equal(a[key][take], frame[key].to_numpy()):
                raise ValueError(f"Native probability alignment mismatch: {key}")
        probability = a["native_probability"][take]
        if (probability.shape != (len(frame),) or probability.dtype != np.float32
                or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any()):
            raise ValueError("Expected original finite FP32 native probabilities in [0,1]")
    return probability
