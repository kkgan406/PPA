import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from ppa.preprocessing import canonicalize, perturb, tensor01, ar_resize, min_size, codec_roundtrip, PROTOCOL
from ppa.response import (FrozenAlex, response_maps, global_response, structured_descriptor,
                          extract_pair, LAYER_CHANNELS)
from ppa.fusion import Fusion, fit_linear
from ppa.metrics import measures, evaluate, source_ba_threshold
from ppa.data import (read_manifest, validate_splits, ppr_metadata, save_cache, load_cache)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def image():
    return Image.fromarray(np.random.default_rng(10).integers(0,256,(97,193,3),dtype=np.uint8))


def test_canonicalization(image):
    expected = codec_roundtrip(min_size(ar_resize(image)), **PROTOCOL["base"])
    actual = canonicalize(image)
    assert actual.size == (512,257) and actual.mode == "RGB"
    np.testing.assert_array_equal(actual, expected)
    extreme = canonicalize(Image.new("RGB", (2000,20)))
    assert min(extreme.size) == 64 and max(extreme.size) > 512


def test_exif():
    image = Image.new("RGB",(97,193))
    image.getexif()[274] = 6
    assert canonicalize(image).size == (512,257)


def test_jpeg_deterministic_sequential(image):
    base = canonicalize(image)
    a = perturb(base,"jpeg","a")
    b = perturb(base,"jpeg","b")
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    expected = tensor01(codec_roundtrip(base,format="JPEG",quality=50,subsampling=2))
    torch.testing.assert_close(a,expected,rtol=0,atol=0)


def test_noise_keyed_and_blur(image):
    base = canonicalize(image)
    torch.testing.assert_close(perturb(base,"noise","a"),perturb(base,"noise","a"),rtol=0,atol=0)
    assert not torch.equal(perturb(base,"noise","a"),perturb(base,"noise","b"))
    assert perturb(base,"blur","a").shape == tensor01(base).shape


def test_response_shape_definition_order():
    generator = torch.Generator().manual_seed(11)
    a = [torch.rand((2,c,3,4),generator=generator) for c in LAYER_CHANNELS]
    b = [torch.rand((2,c,3,4),generator=generator) for c in LAYER_CHANNELS]
    maps = response_maps(a,b)
    assert [m.shape[1] for m in maps] == list(LAYER_CHANNELS)
    scalar = global_response(maps)
    expected_scalar = torch.stack([r.mean((-2,-1)).mean(1) for r in maps],1).mean(1)
    torch.testing.assert_close(scalar,expected_scalar,atol=0,rtol=0)
    desc = structured_descriptor(maps)
    assert scalar.shape == (2,) and desc.shape == (2,1536)
    expected = np.concatenate([op(r.numpy()) for op in
        [lambda r:r.mean((-2,-1)),lambda r:r.std((-2,-1),ddof=0),lambda r:r.max((-2,-1))]
        for r in maps[-2:]],axis=1)
    np.testing.assert_allclose(desc,expected,atol=1e-7,rtol=1e-6)
    with pytest.raises(ValueError):
        response_maps(a[:-1],b)
    b[-1] = b[-1][:,:,:2]
    with pytest.raises(ValueError):
        response_maps(a,b)


def test_official_checkpoint_frozen_real_forward(image):
    # Intentionally a real checkpoint test; offline missing assets are reported as skips.
    try:
        model = FrozenAlex("cpu")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    model.train(True)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    # Smaller valid canvas speeds the backbone interface test; canonical protocol tested separately.
    x = tensor01(image)
    s,d,m = extract_pair(model,x,x)
    assert s.shape == (1,) and d.shape == (1,1536)
    assert s.eq(0).all() and d.eq(0).all()
    assert not any(p.grad is not None for p in model.parameters())


@pytest.fixture
def source_arrays():
    rng = np.random.default_rng(42)
    y = np.tile([0,1],15)
    yv = np.tile([0,1],6)
    x = rng.normal(size=(30,8)) + y[:,None]*.4
    v = rng.normal(size=(12,8)) + yv[:,None]*.4
    h = rng.normal(size=(30,3)) + y[:,None]*.2
    hv = rng.normal(size=(12,3)) + yv[:,None]*.2
    return x,y,v,yv,h,hv


def test_feature_fusion_train_scaler(source_arrays):
    x,y,v,yv,h,hv = source_arrays
    m = Fusion("feature").fit(x,y,v,yv,h,hv)
    np.testing.assert_allclose(m.head[0].mean_,np.column_stack([h,x]).mean(0))
    assert m.head[-1].class_weight == "balanced"
    assert np.isfinite(m.predict(v,hv)).all()


def test_score_oof_matches_independent_fold_replay(source_arrays):
    from sklearn.model_selection import StratifiedKFold
    x,y,v,yv,h,hv = source_arrays
    s,sv = h[:,0],hv[:,0]
    m = Fusion("oof_lr").fit(x,y,v,yv,s,sv)
    for k,(train,hold) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y)):
        assert (m.fold_assignment[hold] == k).all()
        np.testing.assert_allclose(m.folds[k][0].mean_,x[train].mean(0))
        np.testing.assert_allclose(m.oof_scores[hold],m.folds[k].decision_function(x[hold]))
    mean = np.mean([f.decision_function(v) for f in m.folds],0)
    np.testing.assert_allclose(m.predict(v,sv),m.head.decision_function(np.column_stack([sv,mean])))
    np.testing.assert_allclose(m.head[0].mean_,np.column_stack([s,m.oof_scores]).mean(0))


@pytest.mark.parametrize("mode",["ppr","additive","standardized_additive","effort"])
def test_special_protocols(source_arrays,mode):
    x,y,v,yv,h,hv = source_arrays
    s,sv = h[:,0],hv[:,0]
    m = Fusion(mode).fit(x,y,v,yv,s,sv)
    score = m.predict(v,sv)
    assert np.isfinite(score).all()
    r = m.response.decision_function(v)
    if mode in ["additive","effort"]:
        np.testing.assert_allclose(score,sv + m.alpha*r)
    if mode == "standardized_additive":
        tr = m.response.decision_function(x)
        np.testing.assert_allclose(m.score_scaler.mean_,np.column_stack([s,tr]).mean(0))
        hs,rs = m.score_scaler.transform(np.column_stack([sv,r])).T
        np.testing.assert_allclose(score,hs+m.alpha*rs)
    if mode == "effort":
        assert m.threshold == source_ba_threshold(yv,score)


def test_metric_macro_vs_pooled_and_source_exclusion():
    f = pd.DataFrame(dict(dataset=["genimage"]*8, generator=["a"]*4+["stable_diffusion_v_1_4"]*4,
                          label=[0,0,1,1]*2))
    score = np.array([-2,-1,1,2,2,1,-1,-2])
    rows = evaluate(f,score).set_index("scope")
    assert rows.loc["Macro-2","auc"] == .5
    assert rows.loc["Macro-1-unseen","auc"] == 1
    assert rows.loc["pooled","auc"] == measures(f.label,score)["auc"]
    assert measures([0,1],[0,0])["acc"] == .5
    with pytest.raises(ValueError):
        measures([0,1],[0,np.nan])


@pytest.mark.parametrize("mode",["feature","effort","standardized_additive"])
def test_host_specific_float32_inference(source_arrays,mode):
    x,y,v,yv,h,hv = source_arrays
    host = h if mode == "feature" else h[:,0]
    val_host = hv if mode == "feature" else hv[:,0]
    m = Fusion(mode,inference_dtype="float32").fit(x,y,v,yv,host,val_host)
    if mode == "feature":
        expected = m.head.decision_function(np.column_stack([val_host,v]).astype("float32"))
    else:
        r = m.response.decision_function(v.astype("float32"))
        if mode == "standardized_additive":
            a,b = m.score_scaler.transform(np.column_stack([val_host,r])).T
            expected = a+m.alpha*b
        else:
            expected = val_host+m.alpha*r
    np.testing.assert_array_equal(m.predict(v,val_host),expected)


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for split,n in [("train",20),("val",10),("test",10)]:
        for i in range(n):
            sid = f"{split}-{i}"
            rows.append(dict(sample_id=sid,path=f"images/{sid}.png",label=i%2,
                sha256=hashlib.sha256(sid.encode()).hexdigest(),split=split,
                dataset="source" if split != "test" else "genimage",
                generator="SD1.4" if split != "test" else "ADM"))
    f = pd.DataFrame(rows)
    path = tmp_path/"manifest.csv"
    f.to_csv(path,index=False)
    return path,read_manifest(path)


def test_cache_alignment_and_protocol(manifest,tmp_path):
    path,f = manifest
    x = np.random.default_rng(3).normal(size=(len(f),1536)).astype("float32")
    dest = tmp_path/"cache.npz"
    reverse = f.iloc[::-1]
    save_cache(dest,reverse,x[::-1],ppr_metadata())
    np.testing.assert_array_equal(load_cache(dest,f)[0],x)
    bad = f.copy()
    bad.loc[0,"label"] = 1-bad.loc[0,"label"]
    with pytest.raises(ValueError,match="alignment"):
        load_cache(dest,bad)
    metadata = ppr_metadata()
    metadata["descriptor_order"] = list(reversed(metadata["descriptor_order"]))
    save_cache(dest,f,x,metadata)
    with pytest.raises(ValueError,match="protocol"):
        load_cache(dest,f)


def test_split_leakage_and_overlap(manifest):
    path,f = manifest
    assert validate_splits(f).train_overlap.sum() == 0
    bad = f.copy()
    bad.loc[20,"sha256"] = bad.loc[0,"sha256"]
    with pytest.raises(ValueError,match="leakage"):
        validate_splits(bad)
    bad = f.copy()
    bad.loc[30,"sha256"] = bad.loc[0,"sha256"]
    assert validate_splits(bad).train_overlap.sum() == 1


@pytest.mark.parametrize("script",["motivation.py","ppr_only.py","main_results.py","extract.py"])
def test_cli_help(script):
    result = subprocess.run([sys.executable,str(ROOT/"scripts"/script),"--help"],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    assert "--dataset" in result.stdout


@pytest.mark.parametrize("fusion",[None,"feature","score"])
def test_synthetic_cli_end_to_end(manifest,tmp_path,fusion):
    path,f = manifest
    rng = np.random.default_rng(4)
    x = rng.normal(size=(len(f),1536)).astype("float32")
    h = rng.normal(size=(len(f),768)).astype("float32")
    s = rng.normal(size=len(f))
    meta = ppr_metadata(host="CLIP",host_frozen=True,host_provenance=dict(
        checkpoint_sha256="a"*64,code_commit="synthetic-interface-test",feature_layer="synthetic",
        normalization="synthetic",preprocessing="synthetic",native_logit="synthetic"))
    cache = tmp_path/"synthetic.npz"
    save_cache(cache,f,x,meta,h,s)
    out = tmp_path/"run"
    script = "ppr_only.py" if fusion is None else "main_results.py"
    cmd = [sys.executable,str(ROOT/"scripts"/script),"--dataset",str(path),"--cache",str(cache),"--output-dir",str(out)]
    if fusion:
        cmd += ["--host","CLIP","--fusion",fusion]
    result = subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    seal = json.loads((out/"source_seal.json").read_text())
    assert seal["source_train_n"] == 20 and not seal["OOD_used_for_selection"]
    assert len(pd.read_csv(out/"predictions.csv")) == 10
    assert np.isfinite(pd.read_csv(out/"metrics.csv")[["ap","acc","auc","ba"]]).all().all()
