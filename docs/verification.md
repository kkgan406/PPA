# Phase 5 — candidate freeze verification

Current frozen scope: Motivation, PPR-only and CLIP/DINOv2/CO-SPY/Effort
Feature/Score. **No algorithm, runtime code/configuration, formal model, formal
result or data split changed in this phase.** Historical phase records below
are retained; they do not expand current public support.

| Check | Phase5 result / evidence scope |
| --- | --- |
| Complete existing tests | **40 passed**,3 upstream warnings,no skips;41.45s; all current tests retained |
| Four-host frozen feature/score |16 host/route/dataset pairs on actual bound cached observations;24 GenImage6 or60 CommGen15 per pair; maximum score error 0, existing atol/rtol1e-10 |
| PPR-only frozen model | Both target datasets replayed on same real bound subsets; maximum error 0, atol/rtol1e-12; original TRAIN scaler n12398 |
| Source-only fit provenance | Original feature/score/PPR scalers rechecked at12398 TRAIN rows; fold scalers9918/9919; heads/coefficients strict hash checked, no fitting |
| Threshold provenance | Effort original alpha=.25 and source-VAL BA threshold2.008856108429543; remaining PPA operating points0. Native Effort strict>0; CO-SPY original saved probability ranking |
| Prior eight real subset CLIs | Source seals,model/output hashes and all current candidate module hashes rechecked; no refit/re-extraction claimed in this phase |
| Motivation | Fresh JPEG/blur/noise CLI on4 actual original observations each; training-free; no full-benchmark claim |
| Data alignment / metrics | Tests cover ID/hash/label/domain/split alignment, leakage, source-only selection, OOF, macro/pooled/disjoint metrics, native tie/ranking and signed negative gains |
| README / freeze safety | CLI --help flags, shell syntax, local links/imports, exact code/config/test preservation and final file scan recorded in final checks |

Frozen route replay is cache B. The earlier limited candidate CLIP raw check
remains the measured raw scope; no new DINO/CO-SPY/Effort public raw claim.
Only unit fixtures fit disposable models while running the mandated tests.
Phase5 inference checks never fit/train a formal model. No full formal training
or full target benchmark was rerun because no code/algorithm error required it.

Evidence: [Phase5 route checks](evidence/phase5-route-checks.json),
[final checks](evidence/phase5-final-checks.json),
[file freeze](evidence/phase5-freeze.json), [author review](phase5-author-review.md).
Local full logs/manifests/assets remain outside the candidate. Fresh online
installation/downloads, public exact assets and final manuscript sign-off are
not verified. Three existing warnings come from upstream torchvision/LPIPS.

---

# Final four-host scope verification — 2026-11-07

**Current public scope: CLIP-L/14, DINOv2, CO-SPY, Effort.** Earlier phase
records below retain their historical seven-host scope and do not re-enable
excluded interfaces. No formal Host/fusion model, result or split was changed.

| Check | Actual result / scope |
| --- | --- |
| Baseline before exclusion | 28 passed, 3 upstream warnings, no skips, including all3 AIDE-specific cases |
| Current complete relevant suite | **40 passed, 3 upstream warnings, no skips**; original25 shared cases unchanged plus15 scope/reporting checks |
| Motivation | JPEG/blur/noise real CLI, four original real GenImage observations each, actual AlexNet/PPR extraction, maps/PNG enabled; partial smoke scope, not full paper result |
| PPR-only | Real-cache CLI subset:20 source TRAIN,4 source VAL,4 GenImage and4 CommGen15 observations; disposable source-only LR fit, not formal refit |
| Four hosts x feature/score frozen replay | 16 host/route/dataset pairs,24 GenImage6 or60 CommGen15 observations per pair; actual original bound features/native logits and hash-checked frozen heads; maximum score error 0 under pre-existing atol/rtol1e-10 |
| All eight Main Results CLIs | Actual bound cache subsets with20 TRAIN/4 VAL/8 TEST, both target datasets; source-only disposable fusion fitting, seals/threshold/source/metrics/predictions/native comparison checked; no frozen Host training |
| Candidate CLIP raw after adapter reduction | Four real images, fresh CPU PPR+frozen CLIP/native head; maximum errors: PPR 3.0279159545898438e-05, feature 5.5693089962005615e-07, native logit 2.6211630909322992e-05; original tolerances PPR2e-5/2e-4, feature3e-4/3e-4, score.003/.003 |
| Native comparison and aggregation | All generator/macro/pooled/disjoint PPA rows match standalone metrics when joined by dataset/scope/range; signed negative gains retained; Effort strict native ties, selected score threshold and CO-SPY saturation ranks tested |
| CO-SPY original ranks | Native AP/AUC use original saved FP32 `native_probability`; logit-only cache rejected explicitly. No device-dependent sigmoid recomputation used. Raw logits remain fusion inputs |
| README CLI | Four experiment --help commands/flags, eight main routes, PPR-only, three Motivation forms and optional CLIP extraction checked; shell block syntax/import/config/link checks recorded in final evidence |

Effort frozen replay retains **alpha=0.25** and
**source-VAL threshold=2.0088561084295429**. Tiny subset fits
may select different disposable parameters and are not formal paper estimates.
No OOD metrics select a fusion route, threshold, source scaler or coefficient.
Unit fixtures are synthetic interface tests; all real/cache records are labeled
separately. CLIP retains limited public raw A. Other retained public raw routes
remain D; cache replay remains B, not a raw/fresh-public-checkout claim.

Two local audit-harness assumptions were corrected during validation: CSV rows
must be paired by their scope keys rather than incidental order, and original
archive loaders need their original working directory/dataset metadata. These
were checker fixes, not relaxed tolerances or altered algorithms/results. The
initial logs remain outside candidate. No failing shared test was removed.
The AIDE-only3 passing tests moved with their excluded interface; their exact
bytes and all original25 shared cases are preserved.

Sanitized [route evidence](evidence/four-host-route-verification.json),
[original fusion identities](evidence/four-host-fusion-identities.json),
[checkpoint inventory](evidence/four-host-checkpoint-inventory.csv),
[final preservation/safety/CLI checks](evidence/four-host-final-checks.json).
Local images/caches/configs/serialized heads/logs stay outside candidate.
No fresh online dependency installation, missing-asset hosting, DINO original
source-pin recovery, CO-SPY/Effort public raw wrapper or complete benchmark
extraction is claimed. Final manuscript/license/permissions remain blockers.

---

# Phase 3 verification — 2026-11-07 (Asia/Shanghai)

The previous phase records below are historical. Current public support is in
[the public matrix](public-host-support.md); archive checks never upgrade public
raw status. **No formal head/model was trained or refitted in Phase 3.**

| Check | Actual result / scope |
| --- | --- |
| Existing required suite | 25 passed, 3 upstream warnings, no skips; original `tests/test_ppa.py` rerun |
| New A3 contract tests | 3 passed: wrong asset hash rejected before deserialization, input alignment rejected before inference, CLI help |
| Candidate A3 cached API | Two real-image-bound observations; frozen/eval FP32; original scaler fitted on 12,398 TRAIN rows; maximum archived-score error 4.76837158203125e-7 |
| Candidate A3 CLI | Real target-only manifest/cache and authorized local config; predictions exactly match API; predictions/metrics/metadata written; no fitting/raw host extraction |
| Two-real-image alignment | Archive raw A0 and fresh CPU PPR feed candidate external A3. Host feature error 2.384185791015625e-7; native margin error 4.76837158203125e-7; PPR error 2.4199485778808594e-5 under original atol2e-5/rtol2e-4 |
| Frozen A3 raw comparison | Candidate external call equals original A3 on the same raw inputs exactly; saved historical-score maximum error 9.5367431640625e-7; original score tolerance atol/rtol .003 |
| Exact portable manifest preparation | Eight local drafts preserve ordered ID/source-split/domain/label/content bindings and record CSV checksums; no images/weights/features copied; publication/version/permissions pending |

A3 real-image host extraction used the **archive A0**, not a public raw adapter.
This validates only conditional external A3 cache B; AIDE public raw remains D.
The two-image scope is not a benchmark estimate. Existing 25 tests include
synthetic disposable unit fixtures; those fixtures are not paper reproduction.
No thresholds or tolerances were tuned against target outcomes.

Sanitized evidence: [cached A3](evidence/phase3-a3-cache.json),
[raw-input comparison](evidence/phase3-a3-raw-comparison.json),
[manifest catalog](evidence/phase3-manifest-readiness.json), and
[final checks](evidence/phase3-final-checks.json).
Local paths/configs/data/cache/logs remain outside the candidate.
Fresh online installation, exact public asset acquisition, final-manuscript
agreement and full-target raw extraction remain unverified/blocked. The original
AIDE environment used its installed packages and the existing LPIPS package path
for the local raw check, without installation/substitution.

---

# Verification report

## Phase 2: release blocker audit (2026-11-07, Asia/Shanghai)

The same existing suite was rerun: **25 passed, 3 upstream warnings, no skips**.
No algorithm, training parameter or formal result was changed in this round.
The CPU sandbox cannot see CUDA; an explicit outside-sandbox driver check found
an idle V100S32GB, and the read-only raw audit used that GPU.

- All seven hosts passed two real GenImage6 observations (one real/one fake),
  image-byte SHA256 verification, fresh CPU FP32 PPR, actual frozen GPU host,
  original native score and frozen feature/score inference. Maximum actual raw
  fusion-score error was below4.5e-5. Fixed comparison tolerances: PPR2e-5/2e-4,
  host feature3e-4/3e-4, native/fusion score3e-3/3e-3; no target metric tuning.
- CLIP and CLIMP used the existing **candidate FrozenHost**; DINOv2 and four
  dedicated detectors used **archive adapters**, so their public cache routes
  remain B. AIDE A3 used its actual strict-loaded MLP in the archive; it remains
  D in the candidate. See the [status matrix](host-reproducibility-matrix.md).
- AIDE's initial environment lacked LPIPS. The retry appended the existing
  original LPIPS package path, preserving verified weights and the detector's own
  environment. No package/model was installed or substituted for that retry.
- AIDE checkpoint tensor structure, all112 per-generator/macro metric records
  and216 numeric table cells were independently checked. The original F=A3
  (MLP), S=A2 mapping is confirmed; unfavorable outcomes remain untouched.
- 24 original checkpoint/head/config assets were rehashed;156 DINOv2 and16 CLIMP
  source-file bindings match their original sealed config. DINOv2's upstream
  commit remains unresolved even though the snapshot contents are known.
- Frozen dataset counts, original ID formulas for every manifest row, source
  TRAIN/VAL leakage, all target source intersections and generator mappings were
  checked. Macro-7-unseen has87991 observations and zero current source-content
  overlap. This is distinct from unknown host pretraining overlap.
- README Motivation/extraction/PPR-only/main-feature/main-score command forms
  ran again on actual local data/caches with new output directories. CLI help is
  covered by the rerun suite. Venv creation/activation and Python version passed;
  an isolated offline editable build/install passed; locked requirements were
  satisfied in an offline resolver dry-run. Fresh online installation/downloads
  were **not tested**, and the README's literal placeholder paths require assets.

The raw result records and checkpoint/data/table audit CSV/JSON are under
`docs/evidence/phase2-*`. Public-vs-archive support and remaining release
conditions are in the [phase2 report](phase2-release-audit.md). Historical phase1
measurements below are retained with their original CPU-only scope.

## Phase 1 record (CPU-only execution)

Date: 2026-11-07 (Asia/Shanghai). Tests ran on CPU with Python 3.10.20,
PyTorch 2.4.1+cu121, torchvision 0.19.1, LPIPS 0.1.4, Pillow 12.3.0,
NumPy 2.2.6, scikit-learn 1.7.2 and pytest 9.1.1. CUDA was unavailable
inside this execution environment; no GPU or host-training run is claimed.

## Passed

- **25 pytest cases**, including an actual official-weight AlexNet forward
  (no checkpoint skip in this run). Canonical geometry/EXIF, deterministic JPEG,
  seeded noise, layer shapes/order, raw global scalar, 1536D descriptor,
  frozen parameters, feature fusion, held-out score OOF, special additive
  routes, FP32 sklearn inference, source-only scaler scope, macro/pooled metrics,
  leakage and cache-misalignment rejection, CLI help and synthetic CLI runs.
- Independent imports of all `ppa` modules from a different working directory.
  No import depends on the research archive's `src` package.
- **32 real images** from source TRAIN/VAL, GenImage and CommGen15: canonical
  pixels, perturbations and raw response/statistic math agree with original
  implementations. Historical JPEG descriptors agree at the archive's disclosed
  `atol=2e-5, rtol=2e-4`; maximum absolute difference was `4.9591064453125e-05`.
  This is numerical agreement, not a bitwise historical-cache claim.
- Real-image Motivation CLIs for JPEG/blur/noise (four images per perturbation),
  including layer-map NPZ/PNG generation. One generated visualization was visually
  inspected. The small manifest is a smoke test, not a full paper result.
- Raw frozen CLIP-L14 plus PPR extraction from four real images. Maximum host
  feature error versus original content-bound cache: `5.569308996200562e-07`,
  within the pre-existing adapter tolerance `3e-4/3e-4`.
- Real cached-feature CLIs: PPR-only, CLIP feature fusion, CLIP score fusion
  (20 TRAIN, 4 VAL, 4 GenImage, 4 CommGen15). All wrote source seals, selection
  records, predictions and finite metrics. These source subsets do not estimate
  benchmark performance.
- Historical Motivation replay: **75,991 images per perturbation**. Recomputed
  canonical equal-generator AUC matches the original table exactly:

  | Perturbation | Replayed Macro-6 AUC | Difference from original |
  | --- | ---: | ---: |
  | Blur | 0.76656499874845 | 0 |
  | JPEG | 0.7665392221783837 | 0 |
  | Noise | 0.6927067529072722 | 0 |

- Historical full-fit PPR-only head replay: **99,991 GenImage-8** and **66,785
  CommGen15** predictions reproduced; AP/ACC/AUC/BA maximum table difference
  `1.1102230246251565e-16`. Includes Macro-7-unseen and pooled scopes. The old
  extra-domain descriptor-lineage limitation is retained, not resolved by replay.
- Full original source refit through the candidate CLI: **12,398 TRAIN + 800
  VAL**, four C candidates, original two-thread BLAS execution. Selected C=0.01;
  source scaler means, LR coefficients and all 800 VAL scores exactly match the
  archived standalone head in this environment. Source VAL AP=0.9912140348561189,
  AUC=0.99069375, ACC=0.9625, BA=0.9624999999999999. Only the lightweight LR was
  trained; no host was trained. A different thread setting initially produced
  coefficient differences around 1e-11 with unchanged metrics; thread environment
  is now recorded in run metadata. See `full-source-ppr-verification.json`.
- **26 host-route/split sample replays** through the candidate `Fusion.predict`:
  six hosts x two routes x two datasets, plus AIDE score x two datasets. Each
  generator/class contributes two observations (24 GenImage6 / 60 CommGen15
  observations per route). All match archived scores at `1e-10/1e-10`.
  AIDE feature fusion was deliberately excluded because it uses an incompatible
  historical MLP. These are fixed-head replay checks, not raw detector inference.
- Research preservation: branch, HEAD, complete Git status and the 342 recorded
  source/config/test/document/table hashes remained unchanged.

Sanitized machine-readable reports are in [evidence](evidence/). Historical replay
CSV files are explicitly labeled; they are not new full raw-image experiments.
The local-only audit bundle retains full paths, original state, file hashes,
replay scripts, intermediate caches and run artifacts outside the release tree.

## Failed / deliberately rejected

No required unit/interface check remains failing. AIDE feature fusion is rejected
by the CLI with the actual projection/MLP conflict, not falsely reported as
reproduced. The archive's old extra-domain descriptor lineage remains unresolved.

An initial fixed-head replay caught the Effort/AIDE FP32 input behavior; preserving
that behavior fixed the discrepancy without relaxing the `1e-10` comparison.
Exact bitwise equality of separately refitted LR coefficients is not a portable
claim across BLAS thread settings or hardware.

## Not tested / unavailable

- Fresh full target-image extraction and complete paper reproduction for all
  seven hosts; GPU execution; fresh-environment online dependency installation.
- Raw DINOv2/CLIMP adapters on this CPU environment, and raw detector extraction
  for CO-SPY/C2P-CLIP/AIDE/Effort in this candidate.
- Historical AIDE feature projection/MLP; host training; external detector
  pretraining content overlap; final dataset/model redistribution permissions.
- A fresh public checkout reproducing the exact original split, because approved
  portable frozen manifests and source-trained checkpoints are not yet released.

## Missing dependencies / assets

Core CPU imports/tests had no missing dependency. Optional host extraction still
requires the exact official code checkout, source checkpoint/native head and
its environment. CLIMP's Mamba/CUDA dependencies and external detector stacks
are not replaced with approximations. No weights, caches, dataset images or
unapproved third-party code are included in the candidate.

## Commands actually exercised

The public README's three experiment command forms and all four `--help`
interfaces were executed with local manifest/cache paths. The local-only scripts
`verify_local.py` and `replay_hosts.py` read archived provenance and compare
against the candidate without editing the archive. Their output artifacts are
not mixed into the new-run tables produced by the CLIs.
