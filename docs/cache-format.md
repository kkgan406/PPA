# Data and cache contract

The manifest requires `sample_id,path,sha256,label,generator,split,dataset`.
Use source-only SD1.4 TRAIN/VAL, real=0/fake=1. `path` is relative to the manifest
or an explicit local absolute path; it is not copied into prediction outputs.
Keep the original frozen image IDs, particularly for noise. Compute full file
SHA256 before extraction; changed image bytes are rejected.

A cache is a pickle-free NPZ with:

| Key | Shape/type | Meaning |
| --- | --- | --- |
| sample_id, sha256, generator, split, dataset | N Unicode strings | Observation binding |
| label | N integers | Real=0, fake=1 |
| ppr | N x 1536 float32 | JPEG90 -> JPEG50 raw AlexNet descriptor |
| features | N x host dimension | Original host representation (for main results) |
| logits | N finite floats | Original fake-positive native logit/margin; always the score-fusion input |
| native_probability | N float32, CO-SPY Main Results only | Original saved official sigmoid probabilities for native AP/AUC; do not recompute from logits |
| metadata | scalar JSON string | Protocol and checkpoint provenance |

`ppa.data.save_cache` and `ppr_metadata` are the producer interface. The consumer
joins by ID and verifies every binding field, shape, precision, finite value,
statistic order, canonicalization, frozen status, and checkpoint identity. An ID
permutation is supported; missing, extra or duplicate IDs are rejected. The
metadata is a producer declaration, not independent proof of checkpoint fidelity.
Publish a producer audit before treating a newly supplied host cache as official.

`ppr_metadata()` supplies the fixed protocol identifier, order, pixel transforms,
seed, dtype, and these original checkpoints:

| Asset | SHA256 |
| --- | --- |
| torchvision alexnet-owt-7be5be79.pth | 7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02 |
| LPIPS package weights/v0.1/alex.pth | df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0 |

For main results, add `host`, `host_frozen=true` and `host_provenance` containing
`checkpoint_sha256`, `code_commit`, `feature_layer`, `normalization`,
`preprocessing`, `native_logit`. Record checkpoint/file hashes, pinned code
revision, source-head selection, and numerical equivalence in the producer audit.
Never supply probabilities where a native logit is required.

Optional raw pretrained-host extraction takes a local YAML (paths are relative
to the current working directory):

```yaml
model_dir: checkpoints/clip
weights: checkpoints/clip/pytorch_model.bin
checkpoint_sha256: TODO_FULL_SHA256
code_commit: 32bd64288804d66eefd0ccbe215aa642df71cc41
native_head: models/original-native-head.joblib
native_head_sha256: TODO_FULL_SHA256
feature_layer: projected image_embeds
normalization: L2
preprocessing: canonical Q90 -> bicubic shortest224 -> center224 -> official CLIP normalization
native_logit: original source-only frozen LR decision_function; fake-positive
```

DINOv2 also needs `code_root` pointing to the official pinned checkout and
the original `weights`. For DINOv2, `code_commit` is checked against the local
Git HEAD and tracked modifications are rejected. No alternative backbone is used.
Local head serialization is loaded only when the user supplies a matching hash.

Source selection writes `source_seal.json` before predicting TEST. Five-fold
TRAIN PPR scores are held out once; VAL/TEST use the arithmetic mean of the five
fold logits. TRAIN contents are required to be unique to prevent cross-fold
content leakage. Full-fit standalone PPR is a separate model.

Each generator group must contain both labels. Macro is an equal-weight average
of unrounded per-generator metrics; pooled is computed on all observations.
For GenImage, the source-family exclusion and the content-disjoint range are
different scopes and are labeled separately. Partial manifests produce partial
macros, never an asserted full benchmark score.

## Portable manifests

The [portable manifest plan](manifest-release-plan.md) retains original IDs and
adds `source_split`, `dataset_version` and source-manifest hash provenance. These
extra fields do not authorize regenerated IDs or changed row membership. The
readiness catalog binds locally prepared drafts; full manifests are not bundled.

## Native versus PPA reporting

Main Results adds `native_metrics.csv` and `native_vs_ppa.csv`, joined by dataset,
scope, aggregation, observation count and evaluation range. Signed metric
changes are not clipped. Original PPA `metrics.csv` remains separate.

CO-SPY formal unified native AP/AUC retain its **saved FP32 probabilities**,
including saturation/device ties. Supply `native_probability` through the optional
`save_cache(..., native_probability=...)` argument using the original bound
`cospy_probability.npy` rows. Its ID/label/domain/split/content bindings and
finite FP32 [0,1] values are verified. If absent, Main Results explicitly rejects
native comparison; it never silently re-sigmoids/re-ranks an old logit-only cache.
The augmented fusion itself still uses original raw logits. Native ACC/BA use
canonical raw logit>=0 as in the unified formal table; the older detector-only
C0 report's probability>.5 convention remains historical and distinct.

Effort native uses strict margin>0; feature PPA uses >=0; score PPA uses its
source-VAL-selected threshold. Neither rank arrays nor target metrics choose
fusion hyperparameters or operating points. An enriched CO-SPY cache remains
cache reproduction, not raw extraction support.
