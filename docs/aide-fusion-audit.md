> Scope update: CLIMP, C2P-CLIP and AIDE are excluded from current public interfaces. This document is retained as historical paper/audit evidence; see [current four-host scope](four-host-release-scope.md). Historical interface commands are not current reproduction instructions.

> Historical Phase 1/2 record. Current Phase 3 scope and blockers: [release readiness](release-readiness.md), [AIDE decision](aide-release-decision.md), [public support](public-host-support.md). Earlier A3 scope restrictions are superseded only for the conditional external inference contract; generic AIDE LR feature fusion remains blocked.

# AIDE Fusion Protocol Audit (phase 2)

**Finding: formal feature fusion F is A3, a learned projection plus MLP. It is
not StandardScaler + LogisticRegression.** The latter is A1 (PPR-only); A2 is
an additive score rule using A1. No formal model/result was changed.

## Training -> checkpoint -> prediction -> table chain

Paths below are relative to the separate AIDE research checkout.

| Evidence | Audited role |
| --- | --- |
| `audit_models.py:FeatureFusion` | PPR projection and two-layer nonlinear fusion classifier |
| `audit_pipeline.py:train_a3`, `source_scores`, `seal_source`, `evaluate` | TRAIN fitting, source VAL selection, immutable source seal, A0/A1/A2/A3 predictions |
| `outputs/a3_source_config.json`, `outputs/aide_msm_frozen_source_config.json` | Selected epoch5, dimensions/hyperparameters; source seal SHA256 `d1f55baa8aebaf091791611e254fe8c9767b4eae12aab821f638a3307cde2dbc` |
| `checkpoints/aide_msm_feature.pth` | A3 SHA256 `c9f695262b8d8d5617c792be1246e6e1b2ad0b53f3ae82461173afcd3bdce519` |
| `cache/aide/{genimage,commgen15}_all_scores.npz`, `cache/genimage8_extension/extra_all_scores.npz` | Bound IDs/labels and actual A3/A2 score arrays |
| `results/genimage8_extension/genimage8_metrics.csv`, `results/commgen15_metrics.csv` | Independently recomputed AP/ACC/AUC/BA match within 1e-12 |
| `audit_genimage8.py:NAMES`, PPR `results/unified_cross_host/table_*.md` | F=A3, S=A2, Native=A0; 216 printed numeric cells verified |

A0 is the source-trained official-code AIDE detector, selected epoch20/20;
checkpoint SHA256 `8c3bcae2bffb60311bbe15c6b2f64a8fdc8a7fd155a5b4e263cc0b8d8d78307f`.
A1 is the TRAIN-fitted scaler + balanced LBFGS LR (seed0, selected C=0.01),
SHA256 `81a31654fb655cc4eb301ef1889b902c222197dcbaa6b677387817c703f5ff7d`.
Every listed checkpoint was rehashed this round.

## Actual A3 structure and training

Frozen host h = semantic256 + artifact2048 = 2304D. Reuse A1's source-TRAIN
StandardScaler on the raw PPR1536 vector. Then:

```text
scaled PPR1536 -> Linear(1536,256) -> LayerNorm(256) -> GELU
concat(host2304, projectedPPR256) -> Linear(2560,1024) -> GELU -> Linear(1024,2)
fake-positive margin = output[:,1] - output[:,0]
```

Strict checkpoint loading confirms `response_proj.0.weight=[256,1536]`,
`fc.fc1.weight=[1024,2560]`, `fc.fc2.weight=[2,1024]`, plus the recorded biases
and LayerNorm tensors. This cannot be interpreted as a single LR head.

Training: 3,018,498 trainable projection/classifier parameters; seed0; five epochs
completed and source VAL selected epoch5; batch256, shuffle=True, drop_last=True;
AdamW lr1e-4, eps1e-8, weight_decay0 (default betas .9/.999); cosine per-step
schedule to min lr1e-6; official label-smoothed cross entropy (smoothing0.1).
Host backbones, semantic projection and BN buffers stay frozen through cached
eval features. The old classifier's first2304 columns/bias and final layer
initialize A3; new projected-PPR classifier columns initialize to zero, initially
preserving A0 predictions. No A3 re-training was performed in this audit.

A2 uses `TRAIN-z(native AIDE margin) + alpha * TRAIN-z(full-fit A1 logit)`.
Alpha grid0/.25/.5/1 selects alpha0.25 on source VAL AP -> ACC -> AUC -> smaller
alpha; statistics are source TRAIN only. No OOF response is used for A2. All
AIDE formal thresholds are fixed margin>=0. Source VAL is never merged into fit.

## Corresponding archived main-table results

Units below are percent. These are **archived formal-table observations**, not
new paper results. No PPA manuscript was supplied; manuscript-to-table agreement
cannot be certified beyond this code/checkpoint/table chain.

| Dataset / scope | Model | AP (%) | ACC (%) |
| --- | --- | ---: | ---: |
| genimage / Macro-8 | A0 | 97.116863 | 85.659720 |
| genimage / Macro-8 | A2 | 97.771838 | 82.714921 |
| genimage / Macro-8 | A3 | 97.160119 | 79.850759 |
| genimage / Macro-7-unseen | A0 | 96.705173 | 83.667060 |
| genimage / Macro-7-unseen | A2 | 97.453589 | 80.257529 |
| genimage / Macro-7-unseen | A3 | 96.754422 | 76.973487 |
| commgen15 / Macro-15 | A0 | 98.766742 | 88.363666 |
| commgen15 / Macro-15 | A2 | 99.182104 | 82.787233 |
| commgen15 / Macro-15 | A3 | 97.623906 | 75.423195 |

A3 does not consistently improve A0; all negative outcomes are retained. The
machine-readable evidence contains all per-generator metrics, source mapping
and comparison errors, not just favorable rows.

## Can the release candidate reproduce it faithfully?

- A2 score route: **B, frozen-cache reproduction verified** in the candidate;
  local archive raw-image inference also passes on two real images this round.
- A3 feature route: **D in the candidate**: its CLI deliberately rejects the
  historical MLP. In the archive, its actual checkpoint strict-loads and runs;
  two real-image A3 scores reproduce saved outputs within 9.54e-7. This does not
  make A3 publicly supported or authorize its replacement by LR.

Recommended minimal resolution: keep F=A3 in the provenance and paper table;
retain the candidate's explicit rejection until the owner resolves the previous
LR-only/no-new-MLP release scope. If faithful A3 release is authorized later,
provide its exact small inference head, original scaler/checkpoint and source
seal, with the nonlinear exception stated explicitly. Do not relabel A1 as A3,
retrain an LR substitute, remove an unfavorable row or change a formal score.
