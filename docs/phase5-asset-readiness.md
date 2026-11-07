# Phase 5 Asset / License Readiness

These are concrete states, not placeholders counted as completed work. READY
applies only to the stated artifact/check. It does not clear other gates. No
image, weight, full manifest, cache or third-party code tree is bundled.

| Item | Status | Evidence / exact action required |
| --- | --- | --- |
| Frozen four-host code/config scope | READY | Only CLIP/DINOv2/CO-SPY/Effort main choices; Motivation/PPR-only and shared code retained; hash freeze inventory |
| Portable-manifest preservation mechanics | READY | Eight local drafts preserve original IDs/order/source-split/domain/label/content bindings and CSV checksums; [catalog](evidence/phase3-manifest-readiness.json) |
| Exact manifest package/version selection | AUTHOR DECISION REQUIRED | Confirm final source snapshots, original split mapping, unknown upstream versions, extension feature lineage and approved package |
| Dataset / manifest publication terms | EXTERNAL PERMISSION REQUIRED | Confirm image-list/metadata and dataset terms; images/full drafts stay external |
| Portable manifests obtainable by public users | TECHNICALLY BLOCKED | No approved exact public package/acquisition endpoint; schema/catalog alone do not reproduce a split |
| Exact CLIP code/base binding | READY | HF revision32bd64288804d66eefd0ccbe215aa642df71cc41, Transformers4.44.2 and original base hash inventoried; local measured scope only |
| DINOv2 original source version | TECHNICALLY BLOCKED |156 source byte bindings known but upstream commit unresolved; original snapshot cannot pass candidate Git-pin guard |
| CO-SPY / Effort official code versions | READY | Original commits pinned in [protocol matrix](host-reproducibility-matrix.md); no source copied |
| Checkpoint / native / fusion hash identities | READY | [Inventory](evidence/four-host-checkpoint-inventory.csv), [fusion identities](evidence/four-host-fusion-identities.json), LPIPS/AlexNet hashes in cache contract; exact local frozen models checked |
| Exact checkpoint/head public acquisition | TECHNICALLY BLOCKED | Approved native/fusion-head endpoints and complete exact acquisition instructions absent; official generic model is not a substitute |
| Model/component/head release rights | EXTERNAL PERMISSION REQUIRED | Verify actual CLIP/DINO assets, CO-SPY SigLIP/VAE/components and original native/fusion heads separately from source licenses |
| Effort and bundled component clearance | EXTERNAL PERMISSION REQUIRED | Observed CC BY-NC4.0 noncommercial restriction must be preserved; assess exact weight/DeepfakeBench terms, not an unrestricted grant |
| Project LICENSE | AUTHOR DECISION REQUIRED | Unchanged/undecided; owner and contributors approve a grant. No license selected by agent |
| Final paper / metadata / citation | AUTHOR DECISION REQUIRED | Final manuscript not located; approved authors/title/version/paper URL/BibTeX needed; no DOI/venue invented |
| Example commands / CLI syntax | READY | Concrete options, shell syntax and tests checked; commands require authorized exact local assets/new outputs; placeholders are explicitly prerequisites, not runnable public data |
| CO-SPY native ranking contract | READY | Requires original saved FP32 native_probability; missing original ranks explicitly rejected, never re-sigmoided |
| Public raw coverage | TECHNICALLY BLOCKED | CLIP limited verified; DINO original pin blocked; CO-SPY/Effort public raw wrappers absent. Frozen-cache B is not raw A |
| Fresh clean online/full-paper reproduction | TECHNICALLY BLOCKED | Not verified; absent manifests/heads and excluded full-paper hosts prevent a complete claim |
| Local candidate snapshot approval | AUTHOR DECISION REQUIRED | Technical checks permit a private candidate snapshot commit; owner must explicitly authorize any Git mutation; no release label or publication implied |

## Acquisition boundaries

Use exact pinned official repository instructions referenced in the host matrix,
subject to those terms. No new dataset/model URLs are invented. Project native
heads, frozen fusion heads and source-specific artifacts are separate assets.
Neither possession of a local checkpoint nor a source MIT declaration grants
weight/data publication rights. Missing grants/endpoints are open work, not READY.

See [author decisions](phase5-author-review.md), [licenses](licenses.md) and
[remaining blockers](release-blockers.md). The final scope freeze is a candidate
snapshot, not public availability, a legal grant or a formal paper sign-off.
