> Scope update: CLIMP, C2P-CLIP and AIDE are excluded from current public interfaces. This document is retained as historical paper/audit evidence; see [current four-host scope](four-host-release-scope.md). Historical interface commands are not current reproduction instructions.

# AIDE A3 Release Decision — Phase 3

**Decision:** preserve Native=A0, Score=A2, Feature=A3. A3 is the historical
nonlinear exception. The phase3 instruction permits evaluating a faithful minimal
interface; the previous generic LR-only release limitation is not a reason to
replace or relabel A3. A minimal **external-code inference contract** is now
provided. The original MLP implementation, host implementation, original scaler
file and checkpoints are not redistributed. Their publication permissions and
acquisition endpoints remain unresolved.

## What is provided

- `ppa/aide.py:ExternalAIDEA3`: freezes and calls a user-supplied original head.
  It defines no MLP and has no fit/train path, optimizer or LR alternative.
- `scripts/aide_feature.py`: target-only, inference-only cache CLI; does not
  extract raw A0 features or refit any source classifier.
- `configs/aide_a3.yaml`: external source factories/hashes and required local
  asset-path placeholders. It contains no model URL or license grant.

This preserves the actual archived eight-tensor state contract:
`response_proj.0.weight[256,1536]`, bias256, LayerNorm weight/bias256,
`fc.fc1.weight[1024,2560]`, bias1024, `fc.fc2.weight[2,1024]`, bias2.
LayerNorm normalized_shape256/eps1e-5 and exact GELU (approximate=none) are checked.
The external original code supplies `FeatureFusion` and `Mlp`; only their known
factory interface is called. Any temporary constructor initialization is fully
replaced by strict loading before inference can occur.

A3 checkpoint SHA256 is
`c9f695262b8d8d5617c792be1246e6e1b2ad0b53f3ae82461173afcd3bdce519`.
Original A1 scaler-pipeline SHA256 is
`81a31654fb655cc4eb301ef1889b902c222197dcbaa6b677387817c703f5ff7d`.
The original StandardScaler must report1536 features and12398 TRAIN fit rows;
no refitting is possible in the interface. The A0 cache must bind the original
checkpoint `8c3bcae2bffb60311bbe15c6b2f64a8fdc8a7fd155a5b4e263cc0b8d8d78307f`.

Inference preserves FP32 raw PPR input -> original scaler transform -> FP32 MLP
input, raw host2304, original concat/order, eval/frozen mode and two-logit
fake-minus-real output. There is no AMP, extra normalization, projection change,
scaler replacement, parameter selection or threshold recalibration.

## Dependency boundary and conditional usage

The config names the **original project audit checkout** (`audit_models.py`,
`audit_common.py`, `models/AIDE.py`, `models/srm_filter_kernel.py`) and pins its
source hashes. These audit helpers are not promised to exist in the upstream
AIDE repository. A user must obtain a legally usable copy from the owner; cloning
upstream AIDE alone does not supply the project's A3 helper/checkpoint. External
imports also need that environment's torch/torchvision/numpy/kornia/open_clip/clip
dependencies. The local validated environment is the archived AIDE environment,
not a newly tested clean public installation.

Create a local copy of the template, set the authorized original code root,
A3 checkpoint and A1 scaler paths, then supply a bound target-only cache:

```bash
python scripts/aide_feature.py --dataset data/aide-test.csv --cache cache/aide-test.npz --config configs/aide_a3.local.yaml --output-dir outputs/aide-a3
```

Paths in the local YAML are resolved by Python relative to the current working
directory; no endpoint is inferred. The manifest/cache contract is unchanged.
Outputs are `predictions.csv` (ID/hash/label/native A0 logit/A3 margin/prediction),
`metrics.csv` (generator/macro/pooled metrics, fixed zero threshold), and
`metadata.json` (exact source/asset/cache/code hashes and no-training flag).
The generic `main_results.py --host AIDE --fusion feature` stays explicitly
blocked because it fits a linear head and cannot reproduce this trained MLP.

## Verification and actual release scope

The existing25 tests pass; three new contract/CLI rejection checks pass. The
new API and CLI pass two actual real-image-bound cached observations. A separate
check recomputes PPR and uses the archive's raw A0 on the same two real images,
then calls the public external A3 wrapper and compares the original frozen A3
predictions. This verifies compatibility, not a publicly supplied raw A0 adapter.
Exact measured errors and scopes are in `docs/evidence/phase3-*`.

Support is **B, conditional external-assets cache inference**, not A public raw.
Availability for a fresh public user remains **blocked** until the original
helper/scaler/checkpoint can be obtained lawfully and dependency instructions
can be exercised in a clean environment. A2 remains its existing separate
TRAIN-standardized additive score route. No A0/A1/A2/A3 formal file or metric is
changed, and the MLP is never described as a linear classifier.

Resolved: architecture identity, minimal interface feasibility, exact tensor and
scaler compatibility, inference behavior on measured observations.
Requires author decision: final manuscript exception wording, which original
helper/asset package to release, project license.
Requires external permission: upstream/project checkpoint/model dependency terms
where not already established.
Technically unresolved: public asset endpoints, raw A0 adapter and clean-install
reproducibility. This is not a formally reproducible published AIDE release.
