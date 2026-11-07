# Phase 5 — Release Candidate Freeze & Author Review

**Decision: four-host candidate frozen for author review; not a public/formally
reproducible release.** No staging/commit/tag/push/Release or publication is
performed. Freeze is a SHA256 file inventory, not an immutable Git tag or a
claim that author/legal approval has been obtained.

## 1. Frozen Release Scope

Only Motivation (JPEG/blur/noise, training-free Global Scalar ROC-AUC), PPR-only
(L4/L5 Mean/Std/Max1536D, source-only scaler/balanced LR), and CLIP-L/14,
DINOv2, CO-SPY, Effort Feature/Score fusion are in scope. CLIMP/C2P-CLIP/AIDE
are not re-enabled. Shared already-tested code remains. All runtime/config/test
bytes are unchanged from the start of Phase5; this phase adds review documentation
and evidence only. No formal model/result/split is changed.

The [freeze inventory](evidence/phase5-freeze.json) binds every publishable
candidate file except itself by path/SHA256. Its inventory digest covers the
canonical sorted path->file-hash map. Verify before any future approved local
commit; any changed/additional/missing file invalidates this snapshot until
reviewed and re-frozen. Git-ignored environments/private assets are outside the
inventory and are not approved publication inputs.

## 2. Paper–Code Consistency Checklist

[Current eleven-item checklist](phase5-paper-checklist.md): no final manuscript
was located in the bounded source/attachment search. Research tables/figures
and task descriptions do not substitute for final paper source. Every final
paper location/revision/sign-off remains pending; no invented page/equation or
completed manuscript audit is asserted. Code facts include exact response math,
geometry, scalar,1536D order, classifiers/fusion, exceptions, metrics/thresholds,
GenImage overlap and GenImage/CommGen15 result lineage. Resolve wording/results
mapping with the author, never rewrite verified code to fit an assumed paper.

## 3. Four-Host Reproduction Matrix

| Host | Public raw inference | Frozen feature / score | Checkpoint / external requirement | License / known blocker |
| --- | --- | --- | --- | --- |
| CLIP-L/14 | Limited real-sample verified; no complete benchmark claim | B / B | Exact official base, original source native LR and frozen heads | PPA grant undecided; asset terms/acquisition need approval |
| DINOv2 | Blocked for original snapshot; interface only | B / B | Original156-file bound source snapshot, weights/native/fusion heads | Upstream commit unresolved; no moving-main substitution |
| CO-SPY | Blocked: public raw adapter absent | B / B | Exact components/frozen heads and original saved FP32 native_probability | MIT source observed; component/model terms and raw producer availability unresolved |
| Effort | Blocked: public fidelity adapter absent | B / B | Exact official detector, PPR/feature heads and recorded inference behavior | CC BY-NC4.0 noncommercial restriction; bundled/model terms, raw wrapper/acquisition unresolved |

B is actual frozen-cache replay, not public raw A or a fresh-user asset claim.
[Detailed provenance](host-reproducibility-matrix.md) retains preprocessing,
features/native logits, commits/hashes and host exceptions. Effort score remains
source-VAL selected, never forced into the OOF/LR route or zero threshold.

## 4. Asset / License Readiness

[Typed asset inventory](phase5-asset-readiness.md) uses exactly READY,
AUTHOR DECISION REQUIRED, EXTERNAL PERMISSION REQUIRED, TECHNICALLY BLOCKED.
READY checks include local portable preservation, known hashes/pins and command
interfaces, not whole public acquisition. Full manifest publication/version,
unknown source pin, original heads and asset-specific grants remain open.
Project LICENSE is unchanged/undecided; no permission or link is invented.

## 5. Test Results

[Verification](verification.md):40 existing tests pass,3 upstream warnings,no
skips.16 original frozen host/route/dataset combinations pass; standalone PPR
replay passes; fresh real JPEG/blur/noise CLI passes. Eight earlier real-subset
CLI seals/output/model/current-code hashes are rechecked without refitting.
Original TRAIN scalers12398 and fold scalers9918/9919 checked; original Effort
alpha=.25/threshold2.008856108429543 preserved. No full formal training,
full-target extraction or new public raw host validation is implied.

## 6. Author Decision Checklist

All items are **pending**; empty boxes are not approval.

- [ ] Supply final manuscript source/revision/SHA256 and complete every equation/table reference and signature in the paper checklist.
- [ ] Confirm final paper terminology, seven-host full-result scope and four-host public scope; preserve all negative gains and historical exclusions.
- [ ] Identify final extra-domain MSM cache lineage and map each final result table to source seals/checkpoints/features; do not replace historical arrays/numbers.
- [ ] Approve exact portable manifests, preserved IDs/splits/labels/domains/content hashes, upstream dataset versions and overlap disclosures (Macro8/7/content-disjoint).
- [ ] Select project LICENSE with contributor rights confirmed; approve notices and Effort noncommercial restrictions without relicensing third-party assets.
- [ ] Provide documented dataset/manifest, model/component/native/fusion-head permissions and approved exact acquisition routes; delegate rights confirmation to the appropriate rights holders where necessary.
- [ ] Supply approved final paper metadata/citation; do not fill DOI/venue/BibTeX by inference.
- [ ] Resolve or explicitly keep blocked DINO original source pin and CO-SPY/Effort public raw adapters; approve truthful cache-only limitations without expanding claims.
- [ ] Review file inventory/tests/safety evidence and explicitly authorize a **private local candidate snapshot commit**, if desired. No authority to tag/push/publish follows from a local commit approval.
- [ ] Separately authorize any future public GitHub publication only after its required rights/assets/readiness gates are satisfied.

For each decision record reviewer, date, exact artifact/hash, decision and
supporting grant/revision. No owner signature/decision is manufactured here.

## 7. Remaining Release Blockers

[Register](release-blockers.md): final paper/final feature lineage/dataset version/
project license; data/model/head rights; missing approved exact assets/manifests;
DINO source pin; absent CO-SPY/Effort raw wrappers; unverified clean public/full
paper reproduction. Excluded-host packaging is out of scope; its absence is
explicit, not an invitation to substitute a new algorithm.

## 8. Git Status and three separate decisions

Branch `release/ppa-open-source` is unborn/no commits. Files remain untracked,
nothing staged. `git log -5 --oneline` exits128 for the empty history. Six original
archive statuses/HEADs/branches and342 protected file hashes are compared with
start/prior audit records. [Final safety evidence](evidence/phase5-final-checks.json).

| Question | Decision |
| --- | --- |
| Can a local release commit be made? | **Technically suitable only for a private candidate snapshot; AUTHOR DECISION REQUIRED.** After explicit owner authorization, a local commit can record this reviewed candidate with all blockers disclosed. It is not a certified final/formally reproducible release; no commit is executed now |
| Ready for public GitHub publication? | **No.** Project/asset/data rights, exact public acquisition, source/protocol decisions and author approval remain open |
| Ready for complete formal paper reproduction? | **No.** Final manuscript unavailable, exact data/head/cache-lineage/public dependencies unresolved, only four of seven paper hosts exposed, full benchmark/fresh installation not verified |

The frozen scope is reviewable locally. Freeze does not convert a TODO or a
local audit into author permission, public availability or complete reproduction.
