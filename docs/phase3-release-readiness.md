> Historical seven-host audit. The current public scope is four hosts; see [scope report](four-host-release-scope.md). Commands/interfaces described below may have been excluded.

# Phase 3 — Paper–Code Consistency & Release Finalization

**Decision: not ready for formal reproducible public release.** This is an
unpublished audited candidate. No final manuscript was provided, so paper/code
agreement awaits concrete author checks rather than an asserted sign-off.
No new algorithm, formal training, formal result change or split change occurred.

## 1. Paper–Code Consistency Matrix

[Ten-row matrix and author checklist](paper-code-consistency.md) covers global
scalar, structured1536, Q50/canonicalization, PPR-only, feature/score fusion,
AIDE A3/A2, Effort source-VAL threshold and dataset evaluation. Standard routes
and host exceptions are explicit. AIDE A3 is nonlinear; score fusion is not
uniformly LR/OOF. Final equation/table references require the authors' manuscript.

## 2. AIDE A3 Release Decision

[Decision and dependency contract](aide-release-decision.md): preserve Native=A0,
Score=A2, Feature=A3. Provide a minimal **external original-code inference-only**
caller; do not copy the MLP or substitute LR. Exact2304+1536 dimensions,
1536->256 projection/LayerNorm/GELU,2560->1024->2 GELU classifier, eight strict
checkpoint tensors, original TRAIN-only scaler and FP32 behavior are retained.
The new route is conditional external-assets cache B; generic feature LR/raw A0
remain D. Authorized acquisition of original helper and A0/A1/A3 is unresolved.

## 3. Seven-Host Public Support Matrix

[Full matrix](public-host-support.md) separates public raw support, cache,
license, exact checkpoints and measured end-to-end scope. CLIP and CLIMP retain
limited candidate raw A; DINOv2/CO-SPY/C2P/AIDE/Effort remain public raw D.
All seven retain measured frozen routes, with AIDE A3 conditional external B.
Archive raw checks do not upgrade these public statuses. No third-party tree
is copied and no release permission is inferred from local availability.

## 4. Exact Manifest Readiness

[Portable plan](manifest-release-plan.md) and [catalog](evidence/phase3-manifest-readiness.json):
eight complete local CSV drafts preserve original IDs/order/source splits,
labels/domains/content hashes and record binding/manifest checksums. Full drafts
remain outside candidate. Dataset version binds archived manifest bytes;
unknown upstream official versions are null and require confirmation.
Historical Macro-8=99,991 includes SD1.4 and11,999 TRAIN-overlap observations;
Macro-7=87,991 excludes SD1.4 entirely. Content-disjoint is a third scope.
No historical metric or split was changed.

## 5. License / Checkpoint Blockers

[Typed blocker register](release-blockers.md) marks every item Resolved,
Requires author decision, Requires external permission or Technically unresolved.
Project LICENSE remains unchanged/undecided. Manuscript/final MSM lineage/version
need author decisions; data/helper/source/weight permissions and approved asset
acquisition remain unresolved. DINO pin and dedicated public raw wrappers also
remain open. An inventory/hash is not a license grant or public download.

## 6. Verification Report

[Verification](verification.md): existing25 tests and new3 contract tests pass.
New A3 CLI runs on actual two-image-bound cached data. Cached score maximum
error4.77e-7; real-input comparison error9.54e-7 against saved archive, exact
against original A3 on the same inputs. PPR/native features/logits checked.
This is a limited inference alignment check, not a full public raw benchmark.
README command options match real CLI help; formal-fitting examples were not
executed during Phase3. Fresh online installation is unverified. Final preservation
and safety results are in [machine-readable evidence](evidence/phase3-final-checks.json).

## 7. Final Git Status

Candidate branch: `release/ppa-open-source`, unborn/no commits. Candidate files
remain untracked/unstaged, so `git log -5 --oneline` reports no commits (exit128).
No git add/commit/push/tag, GitHub Release, history rewrite or research-file deletion
was performed. Original research source/result hashes and six archive repository
states are checked against the audit snapshots; details in final-check evidence.

Publication must wait for the listed author decisions, permissions and technical
reproducibility gaps. The present candidate must not be labeled a formally
reproducible release.
