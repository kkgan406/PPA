> Scope update: CLIMP, C2P-CLIP and AIDE are excluded from current public interfaces. This document is retained as historical paper/audit evidence; see [current four-host scope](four-host-release-scope.md). Historical interface commands are not current reproduction instructions.

> Historical Phase 1/2 record. Current Phase 3 scope and blockers: [release readiness](release-readiness.md), [AIDE decision](aide-release-decision.md), [public support](public-host-support.md). Earlier A3 scope restrictions are superseded only for the conditional external inference contract; generic AIDE LR feature fusion remains blocked.

# Release Blocker Resolution & Reproducibility Audit — Phase 2

2026-11-07, Asia/Shanghai. Work used the existing candidate structure. **No
algorithm, formal training hyperparameter, checkpoint, split or paper-result
file was changed.** All raw probes and temporary install/run artifacts stay in
the local-only audit bundle. The project LICENSE was not selected or modified.

## A. AIDE Audit

[Detailed training/checkpoint/result audit](aide-fusion-audit.md).

F is actual A3: source-TRAIN-scaled PPR1536 -> Linear256/LayerNorm/GELU,
concatenated with frozen host2304 -> MLP2560/1024/2. AdamW lr1e-4, batch256,
cosine min1e-6, label smoothing0.1, seed0, five epochs, source VAL selected5.
Strict checkpoint tensor shapes confirm the nonlinear architecture. A1 is the
separate scaler/balanced LR PPR model; A2 is TRAIN-z(native)+.25*TRAIN-z(A1).

The archived F=A3/S=A2/Native=A0 mapping is verified against112 metric records and
216 printed numeric cells. GenImage Macro-8 A3 AP/ACC=97.160119/79.850759%;
A2=97.771838/82.714921%; A0=97.116863/85.659720%. Negative effects are preserved.
The A3 actual checkpoint also passes two-real-image archive inference; the
candidate feature CLI remains deliberately blocked. No LR substitute was made.
No manuscript was supplied, so these are audited archived main-table values,
not independently confirmed manuscript rows.

## B. Host Reproducibility Matrix

[Full matrix with official code/commit, acquisition, hash, preprocessing,
feature location, native score, fusion and verified scope](host-reproducibility-matrix.md).

| Host | Public candidate route | Newly executed raw verification |
| --- | --- | --- |
| CLIP | A limited inference + B cache | candidate raw adapter + fresh PPR + both frozen fusion heads, n=2 |
| DINOv2 | B cache; raw blocked by missing snapshot Git pin | original hash-bound snapshot + fresh PPR + both fusion heads, n=2 |
| CLIMP | A limited inference + B cache, CUDA/assets required | candidate raw adapter + fresh PPR + both fusion heads, n=2 |
| CO-SPY | B cache; raw adapter not shipped | archive official adapter + fresh PPR + both fusion heads, n=2 |
| C2P-CLIP | B cache; raw adapter not shipped | archive official adapter + fresh PPR + both fusion heads, n=2 |
| AIDE | B score cache; D feature/public raw | actual A0/A3 archive checkpoints + fresh PPR + A2, n=2 |
| Effort | B cache; raw fidelity adapter not shipped | exact checkpoint with auxiliary-only load proof + fresh PPR + both heads, n=2 |

A=raw-image inference verified for the stated subset/environment; B=frozen-cache
reproduction; C=interface/smoke only; D=unsupported/blocked. A local archive A
result never upgrades its absent public adapter or public B route. No full
benchmark training/raw extraction/fresh-download reproduction is claimed.

All24 inventoried assets were rehashed.156 DINOv2 and16 CLIMP source files match
the original sealed source hashes. CLIMP's binding to the current clean commit
is now locally established; DINOv2's upstream commit is still unknown.

## C. Dataset Protocol Audit

[Generation rules, manifests/IDs/hashes, counts, label/domain mapping,
intersections, macro definition and thresholds](dataset-protocol-audit.md).

Source TRAIN12398 (6199/class), VAL800 (400/class) have zero shared content.
GenImage8=99991; CommGen15=66785. Macro-8 includes SD1.4 and11999 TRAIN-content
overlaps. Macro-7 excludes the entire12000-observation SD1.4 domain, leaves87991
observations and has zero overlap with current source TRAIN/VAL. Unknown host
pretraining overlap is a separate limitation. Macro AP averages unrounded
per-generator AP; pooled is separate. Thresholds are fixed zero except Effort's
source-VAL-selected score-fusion operating point and native baseline strict ties.

The exact original ID formulas were checked for every manifest row. Extension
IDs depended on original absolute paths; do not regenerate them after relocation.
Target duplicates are retained as frozen observations. The exact permission-cleared
portable manifests still need release. The content-disjoint scope is not
identical to Macro-7 and can be unavailable when a retained group is single-class.

## D. Updated Release Blockers and safety

[Current blockers and resolved subquestions](release-blockers.md);
[actual code/license/model/data terms](licenses.md).

Resolved audit questions: AIDE MLP identity/result mapping, seven-host local
raw inference, CLIMP source binding, exact data counts/overlap, and previously
missed README license declarations (CLIMP Apache-2.0, Effort CC BY-NC4.0).

Still blocked: authorized A3 nonlinear release scope; source-trained
checkpoint/native-head hosting and redistribution; portable exact manifests;
DINOv2 source pin; dedicated detector public raw adapters; final manuscript
feature-cache lineage; project license/citation; unresolved model/data terms.
The old extra-domain MSM/consistent_v2 discrepancy is recorded rather than
silently repaired. Official checkpoint availability is not redistribution permission.

The safety scan rechecked all186 reachable research commits (2183 unique blobs,
1732 textual blobs): no heuristic credential-like/key/assignment matches;
223 textual blobs contain local absolute paths. Original history and experiment
assets are not imported into the candidate. The candidate's explicitly enumerated
publishable files have no detected private absolute paths, secret-like matches,
large files, dataset images, weights or feature-cache/model artifacts. This is
heuristic evidence, not exhaustive secret detection/legal clearance. Six archive
working trees/HEADs and the342 original file hashes are unchanged.

## E. Verification Results and README command audit

[Verification record](verification.md) and [sanitized raw evidence](evidence/phase2-raw-host-verification.json).

- Existing pytest suite: **25 passed**, three upstream warnings, no skips.
- Seven hosts x two actual real images: fresh raw branch inference passes. GPU
  was unavailable inside the sandbox but verified/used outside it with approval.
- AIDE first attempt lacked LPIPS in its dedicated environment; the recorded
  successful retry used the existing verified original package path, no install
  or substituted model. Its actual MLP head strict-loads; no training occurred.
- PPR/features/native/fusion all have finite values and fixed-tolerance reference
  checks; maximum actual raw fusion-score error is below4.5e-5.
- Frozen-cache AIDE metrics and table mapping pass; all112 metric records and216
  printed numeric cells agree with archived formal tables.
- README Motivation, extract, PPR-only, main-feature and main-score command forms
  were rerun on genuine local assets. All help entry points pass. Literal
  placeholder paths require real prepared manifests/caches; they are not presented
  as ready-to-run paper assets.
- Venv creation/activation passed. An isolated offline editable package build and
  install passed, and an offline dependency resolver dry-run found the existing
  lock satisfied. Fresh online dependency installation/model downloads are not
  tested or asserted. README now states this distinction and output schemas.

Full target raw extraction, host retraining and a new formal benchmark run were
not performed. No C-level synthetic interface result is called paper reproduction.

## F. Recommended Minimal Fixes

1. Resolve the AIDE release-scope exception while preserving F=A3 and its exact
   checkpoint, scaler and unfavorable results; do not replace it with LR.
2. Publish approved portable manifests with original IDs and content bindings,
   plus exact source-trained checkpoints/native heads and the24-asset inventory.
3. Recover DINOv2's exact source pin (or explicitly authorize a hash-bound snapshot
   loader later); retain CLIMP's verified source pin and specify its original
   weight revision/environment. Do not substitute moving upstream main.
4. Ship only necessary, previously verified detector inference adapters after
   dependency/license review, with the native view/feature/logit contracts in the
   matrix. Do not copy complete third-party repositories or create a new model.
5. Have the authors identify the final manuscript's extension-cache lineage and
   approve project/model/data release terms. Preserve both archived lineages and
   source-overlap disclosures. Document missing content-disjoint metrics as NA.

These are provenance/availability fixes. They do not authorize retraining,
OOD selection, altered thresholds or changed formal table values.

## G. Git Status

Candidate branch: `release/ppa-open-source`, still **no commits**. `git status`
shows the original candidate plus audit documentation/evidence as untracked;
there were no staged changes. `git log -5 --oneline` returns the expected
"branch has no commits" error, rather than a fabricated commit list.

This phase changes documentation and adds small sanitized textual audit evidence;
no `ppa/`, `scripts/`, `configs/`, `tests/`, requirements, project layout or
LICENSE changes. The original research branch/HEAD, AIDE and inspected upstream
working-tree states remain intact. No commit, push, tag, history rewrite,
archive deletion or release publication was performed.
