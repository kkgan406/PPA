> Phase5: scope/runtime/config/tests frozen, no formal retraining or result change. Final author decisions and exact READY/blocked asset states are in [author review](phase5-author-review.md) and [asset readiness](phase5-asset-readiness.md). No blocker below is implicitly cleared by freeze.

# Release blockers — final four-host scope

**Not ready for a formally reproducible public release.** Reducing scope does
not clear remaining rights, exact provenance or missing public dependencies.
CLIMP/C2P/AIDE public packaging is excluded, not repaired or replaced. Historical
paper results/limitations remain in prior audits and are not rewritten.

| Item | Status | Evidence / minimal remaining action |
| --- | --- | --- |
| Four-host interface/configuration scope | Resolved | Main CLI accepts only CLIP/DINOv2/CO-SPY/Effort; AIDE-only runtime/config/tests preserved outside candidate; CLIMP adapter branch removed; shared code/tests retained |
| Retained fusion fidelity | Resolved | Original feature/score algorithms retained; no fusion implementation edits. Effort original alpha=.25 and source-VAL threshold2.008856108429543 retained in frozen replay; no OOD route selection |
| Native versus augmented reporting | Resolved | Separate native metrics/paired signed gains; PPA metrics remain untouched; CO-SPY FP32 probability ranking and Effort native strict tie convention disclosed |
| Final paper/code correspondence | Requires author decision | No final manuscript supplied. Authors must approve scope wording and exact final equations/table correspondence |
| Final extension MSM lineage | Requires author decision | Old extension descriptors differ from consistent_v2; choose final intended lineage without replacing formal arrays or numbers |
| Project LICENSE and citation | Requires author decision | LICENSE unchanged/undecided; contributors' rights and approved citation needed |
| Dataset version / frozen manifest publication | Requires author decision | Validate actual upstream versions and approve exact frozen ID/content binding; current local portable drafts are not released |
| Dataset / manifest rights | Requires external permission | No established publication grant; full images/manifests stay external |
| Retained model / native / fusion asset terms | Requires external permission | Code/model grants separate; CO-SPY components and Effort CC BY-NC4.0/bundled dependencies require exact asset review; native/fusion heads need approved release terms |
| Exact public asset acquisition | Technically unresolved | Exact native/fusion-head and approved manifest endpoints/instructions missing; hashes alone cannot supply assets |
| DINOv2 original source pin | Technically unresolved | Original156-file byte binding known; upstream commit absent. Do not replace original snapshot with moving main |
| CO-SPY / Effort public raw adapters | Technically unresolved | Not shipped; cache B/raw D. Archive raw verification is not public raw availability |
| Fresh install / complete target extraction | Technically unresolved | No clean online installation/full-target raw benchmark claimed; limited real checks do not complete paper reproduction |
| Data scope disclosure | Resolved | Historical Macro-8 includes11,999 TRAIN-overlap observations; Macro-7 excludes whole SD1.4 domain; content-disjoint is separate |
| Research/result preservation | Resolved | No original research edits or asset deletion; preservation hashes/status and candidate safety in final scope evidence |

See [scope delivery](four-host-release-scope.md), [public matrix](public-host-support.md),
[verification](verification.md), [licenses](licenses.md), and [manifest plan](manifest-release-plan.md).
