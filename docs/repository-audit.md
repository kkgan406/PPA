> Scope update: CLIMP, C2P-CLIP and AIDE are excluded from current public interfaces. This document is retained as historical paper/audit evidence; see [current four-host scope](four-host-release-scope.md). Historical interface commands are not current reproduction instructions.

> Historical Phase 1/2 record. Current Phase 3 scope and blockers: [release readiness](release-readiness.md), [AIDE decision](aide-release-decision.md), [public support](public-host-support.md). Earlier A3 scope restrictions are superseded only for the conditional external inference contract; generic AIDE LR feature fusion remains blocked.

# Repository audit and migration record

Phase2 follow-up: [release audit](phase2-release-audit.md), [host status matrix](host-reproducibility-matrix.md), and [AIDE checkpoint/table audit](aide-fusion-audit.md) supersede the initial support/license limitations below where explicitly resolved.

Audit date: 2026-11-07 (Asia/Shanghai). Research HEAD:
`890786c` (`research/clip-l14-lora-performance`). Before any source changes, Git
status, branch and last five commits were inspected. The research worktree had
modified `.gitignore` and two unified result tables plus many untracked scripts,
models, figures, environments and results. They were not overwritten, stashed,
deleted or committed. A separate candidate repository avoids importing that
history. The exact initial status and 342 file-content hashes are retained in
the local audit bundle, outside the release candidate.

The audit inventoried 258 Python files through their ASTs (module documentation,
imports, functions and classes). The core chains below were read in implementation
and checked against callers, configurations, manifests, cache metadata and source
seals. The inventory is broader than the code review; unrelated historical
experiments are not certified by this cleanup.

## Implementation-to-result mapping

Paths in the first column are relative to the research archive, unless prefixed
`AIDE/` for its separately checked-out official-code reproduction.

| Original file(s) | Actual implementation / call chain | Formal-result role | Decision / candidate destination |
| --- | --- | --- | --- |
| `src/p0_1/canonical.py` | `load_verified` checks bytes; EXIF/RGB; `ar_resize` Python rounding; min64 isotropic bicubic; explicit Pillow codecs | Canonical Motivation and downstream PPR | Consolidate exact helpers in `ppa/preprocessing.py` |
| `src/perturbations/transforms.py` | Q50 recoding of supplied base, tensor Gaussian blur5/sigma1, CPU SHA256-ID-seeded noise0.03; callers supply decoded Q90 | All three Motivation perturbations; JPEG main descriptor | Retain exact parameters/RNG; omit unrelated resize ablation |
| `src/backbones/lpips_alex.py` | Frozen official LPIPS v0.1 `scaling_layer` then five raw post-ReLU AlexNet outputs | Raw PPR definition | `ppa/response.py`; verify full weights hashes; no trainable backbone |
| `src/responses/aggregate_response.py` | Absolute per-channel spatial mean, then mean channels for each layer; optional resize-only pooling | Historical raw global scalar | Preserve no-alignment route; global = equal mean of five layer means |
| `src/p0_1/extract.py`, `collect.py` | Paired same-canvas forwards and content-bound cache reuse; Q90 response scalar; Native inherited separately | Source of cited Motivation AUCs | Merge relevant extraction into `scripts/motivation.py`; retain image IDs |
| `src/p0_1/source.py`, `metrics.py`, `configs/direction_v1.json` | Source sign freeze; inherited signs across canonical views; condition-specific signs only diagnostic | Blur -1 / JPEG -1 / Noise +1 for primary results | Keep primary signs; no target reorientation |
| `src/p0_1/evaluate.py` | `collect -> evaluations -> per-generator equal macro`; verifies inherited Native lineage | Canonical six-domain Motivation results | Recompute global-response metrics; residual/matching diagnostics outside release scope |
| `scripts/l14_lpips_audit.py` | `prepare` binds/rebuilds formal MSM from E3 statistic caches; `compute` returns a separate learned LPIPS scalar control and raw layer outputs | Descriptor and older CLIP controls | Merge MSM only; do not rename learned distance as paper global response |
| `scripts/e3_spatial.py`, `cache/e3` provenance via formal metadata | `map_stats` computes channelwise mean/std(correction=0)/amax; content-deduplicated extraction and sealed row mapping | Upstream lineage of cached MSM1536 | Preserve provenance; do not copy caches or historical spatial ablations |
| `scripts/unified_cross_host.py` | Explicit IDs/hash/label/domain joins; TRAIN scaler/LR; VAL AP/ACC/AUC selection; five-fold PPR OOF, fold-mean external logits | Formal CLIP-L14/DINOv2/CLIMP feature/score tables and unified PPR head | Consolidate linear protocol in `ppa/fusion.py`; canonical native heads remain external |
| `scripts/unified_cross_host_extract.py` | Official frozen adapters; extra-domain cache extension and content reuse; original h/logit hooks | GenImage8 extension and dedicated host transfer | Keep cache contract; do not copy third-party trees or silently reuse disputed extension bytes |
| `src/stage_b/backbones.py` | Frozen projected CLIP; DINO normalized CLS without additional L2; CLIMP original projection; distinct canonical geometry adapters | Three pretrained hosts (main CLIP is L14, not old B16) | Port relevant adapters to `ppa/hosts.py`; require local hash-verified assets/native heads |
| `src/cospy_msm/cospy_adapter.py`, `scripts/cospy_extract.py` | Official semantic/artifact/fusion model; pre-hooks capture 1152+512 actual fc inputs, not two branch logits | CO-SPY h1664 and native fusion logit | Cache contract preserved; raw public adapter not certified/ported |
| `scripts/cospy_train.py`, `cospy_evaluate.py` | C1 full-fit PPR, C2 native+five-fold response logits, C3 h1664+MSM; class-balanced LR | CO-SPY main routes | Preserve OOF protocol, dimension3200; native official probability threshold pertains to baseline only |
| `scripts/c2p_msm.py` | Official source-trained LoRA model forward_eval and final fc pre-hook; C2 joint LR; C3 raw native+alpha*full-fit PPR | C2P-CLIP main routes | Preserve raw additive alpha grid and full-fit response, not OOF-LR |
| `scripts/effort_msm_audit.py`, `effort_inference_transfer.py` | Official demo/AST fidelity gates, feat1024 and fake-minus-real margin; source-selected additive alpha then VAL BA threshold | Effort main routes | Preserve fusion and inference precision; raw official fidelity loader stays external |
| `AIDE/audit_models.py`, `audit_pipeline.py` | h2304 and native margin; TRAIN-standardized additive A2; A3 trains projection+MLP | AIDE score route; historical feature row conflicts with requested scope | Preserve A2; explicitly block A3; do not re-train or substitute LR |
| `scripts/publish_ppr_only_tables.py` | Reuses sealed full-TRAIN MSM head, predicts original target caches, averages per-generator metrics | Standalone PPR-only GenImage8/CommGen15 | Keep full-fit standalone distinction; no host baseline training |
| `scripts/publish_embedding_host_fs_tables.py`, `split_unified_fs_tables.py` | Verifies and formats existing per-generator records into separate F/S tables; no learning | Table provenance | Replace duplicated presentation code with table-ready `metrics.csv` |
| `src/ppr_visualization/run_layerwise_visualization.py` | Saves raw absolute C,H,W; channel mean bilinear only for display; no feature normalization | Layer-wise response visualization | Minimal deterministic maps/PNG option in Motivation CLI |
| `scripts/ppr_dinov2_macro8.py`, its report and cache seals | Detects old extra-domain descriptor mismatch; independently recomputes consistent_v2, retaining old arrays | Later descriptor lineage audit, not authority to silently replace main tables | Record blocker and preserve both lineages in archive |

## Result and data provenance

| Archived evidence | Meaning / selection | Release handling |
| --- | --- | --- |
| `results/p0_1/scores.csv` + `table_canonicalization_macro.csv` | 75,991 independent GenImage6 observations per perturbation; raw five-layer scalar; fixed source signs | All three canonical AUCs replayed; sanitized measured replay evidence only |
| `cache/l14_lpips_audit/formal/{split}/metadata.json` + `manifest.csv` | Q90->Q50, FP32 MSM stat order; original source12398/VAL800 and target content binding | Mandatory portable cache contract; bytes/absolute paths not bundled |
| `models/unified_cross_host/source_seal.json` | TRAIN-only heads; source-VAL C decisions; five response fold models and OOF assignment hashes | Candidate source seal mirrors protocol; original source head/checkpoint rights remain unresolved |
| `results/unified_cross_host/ppr_only_metrics.csv` and its verification JSON | Full-fit standalone MSM head, not five-fold average | Full target replay against saved predictions/metrics; source hash recorded |
| `results/unified_cross_host/per_generator_long.csv`, F/S verification | Existing unified main results, including historically observed OOD domains | Do not invent new runs or promote static tables to live code output |
| `results/unified_cross_host/data_audit.json` | Source counts, extra two GenImage domains, 11,999 SD1.4 train-content overlaps | Separate source-family and source-content-disjoint metric ranges |
| Dedicated host source seals and cache metadata | Exact checkpoint/code/feature and score protocol; external pretraining overlap may be unknown | Required before formal raw-host reproduction claim |

The full GenImage8 manifest is 99,991 observations; CommGen15 is 66,785. Original
six GenImage domains exclude both glide and SD1.4. Real images are paired within
generator groups according to frozen manifests; duplicate content may appear in
multiple target observations. Macro averages generator metrics, not unique-image
metrics. The main target sets were historically observed and are retrospective
development benchmarks, not fresh confirmation.

## Excluded assets (retained in research archive)

Residualization, covariate-matching, historical Stage B dimensionality studies,
LoRA/rank sweeps, all-layer/PCA/spatial ablations, robustness sweeps, and unrelated
ForgeLens/PGC/LATTE/LaRE/AEROBLADE hosts are outside the requested two-part scope.
Their scripts, results, checkpoints, figures and environments remain intact.
No unrelated host is silently relabeled as one of the seven formal hosts.

No `.venv*`, raw images, `third_party` source, weights, joblib heads, model arrays,
feature caches, logs, private absolute paths, or temporary run outputs are copied
into the release candidate. The example manifest intentionally has TODO hashes;
the loader rejects them until replaced. The framework overview uses a code-native
Mermaid diagram, without inventing a paper figure or redistributing image data.

## Git history and publication audit

All reachable research refs were scanned: **186 commits, 2,183 unique blob
versions, 1,732 textual blobs**. Heuristic patterns found no token-like strings,
private keys or credential assignments. **223 text blobs contain local absolute
paths**, and **219 blob versions** match manifest/binary/large-artifact risk
categories. These counts are blob versions, not necessarily distinct filenames.
They justify keeping the research history out of the candidate history; they do
not prove exhaustive secret absence or legal clearance. No history rewrite,
deletion, commit, push, or GitHub Release was performed.

The original ignore rules did not prevent already-tracked model/cache artifacts
and large manifests from existing in history. The candidate has broader ignore
rules and a separately inspected publishable file list; ignore rules alone are
not treated as an audit. Its Git HEAD is `release/ppa-open-source` with no commit.
