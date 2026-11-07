# Final four-host public release scope

**Scope work complete; formal public release remains blocked.** This is an
open-source packaging decision, not a change to the paper's seven-host results.
CLIP-L/14 and DINOv2 represent general-purpose pretrained hosts; CO-SPY and Effort
represent dedicated AIGI detectors. No replacement host or algorithm is added.

## A. Candidate directory

```text
PPA/
  README.md / LICENSE / pyproject.toml / requirements*.txt
  configs/                  # ppr.yaml, hosts.yaml (four only), manifest_release.yaml
  ppa/                      # shared preprocessing/response/data/metrics/fusion/experiment; CLIP/DINO raw loader
  scripts/                  # motivation.py, extract.py, ppr_only.py, main_results.py
  tests/                    # original25 shared cases; four-host scope/reporting checks
  examples/                 # schema template only
  docs/                     # current four-host protocols/support/blockers; labeled historical audits
    evidence/               # sanitized evidence/hashes only; no model/data/cache
```

No original research tree is moved or modified. A raw DINO interface is not a
claim that the original source snapshot can satisfy its existing Git pin.

## B. Four-host support

[Full matrix](public-host-support.md): CLIP limited Public raw verified A;
DINOv2 original raw Blocked D; CO-SPY/Effort missing raw adapters D. All four
feature/score frozen routes are Cache verified B. External dependencies and exact
assets are required. No row means all-target reproduction or license clearance.

[Protocol/provenance matrix](host-reproducibility-matrix.md) preserves embedding,
native logit, preprocessing, code/version/checkpoint hashes, source-only
normalization/selection and thresholds. [Current inventory](evidence/four-host-checkpoint-inventory.csv)
filters original rows without changing asset identities. Effort's noncommercial
CC BY-NC4.0 terms remain explicit. DINO source pin is still unresolved.

## C. Removed or excluded files/code

| Candidate path / surface | Action | Reason |
| --- | --- | --- |
| `ppa/aide.py` | Moved outside candidate to local audit preservation bundle | AIDE-only external-head inference wrapper |
| `scripts/aide_feature.py` | Moved outside candidate, not published | AIDE-only CLI |
| `configs/aide_a3.yaml` | Moved outside candidate | AIDE-only source/asset factory template |
| `tests/test_aide_contract.py` | Moved with its excluded interface, baseline3 cases passed | Tests solely for removed AIDE API; no failing test hidden |
| `configs/hosts.yaml` | Excluded CLIMP/C2P/AIDE keys | Four-host default config only; retained entries semantically unchanged |
| `scripts/main_results.py` | Excluded three CLI choices | Exactly four supported hosts, both fusion routes |
| `scripts/extract.py`, `ppa/hosts.py` | Removed CLIMP choice/preprocessing/model/projection branch and exclusive imports | No CLIMP raw entry/Mamba loader; retained CLIP/DINO branches unchanged |
| `requirements-hosts.txt` | Removed CLIMP/Mamba installation instruction | No exclusive CLIMP package existed in this file; Transformers/safetensors retained for CLIP |
| README/current protocols/support/license/blockers | Removed excluded hosts' active commands/requirements/promises | Paper seven-host history remains explicitly distinct |

No standalone C2P adapter/dependency file existed in the candidate. Its default
config/CLI choice was the active surface. No third-party source/weight was copied.
The tested shared `Fusion` class and its existing additive/precision unit checks
are retained, including low-level legacy helper branches; those branches do not
expose additional hosts in public entry points or configuration. Existing shared
25 tests are preserved byte-for-byte. Historical seven-host audit documents,
checkpoint inventories and numerical evidence are retained and labeled as past
scope. [Historical Phase2 matrix](phase2-host-reproducibility-matrix.md) and
[Phase3 license record](phase3-licenses.md) preserve prior conclusions.

## D. Formal paper results

Unchanged. No formal checkpoint/cache/result/split/manifest was edited and no
formal Host or fusion head was refitted. Only disposable source fusion heads
are fitted for limited interface checks. `ppa/fusion.py`, Motivation/PPR modules
and the retained host configuration values remain unchanged. Effort frozen replay
uses alpha=.25, threshold2.008856108429543 from source VAL, never a forced zero.
CO-SPY native AP/AUC retain original saved FP32 sigmoid ranking (including saturation
ties); its unified-table native ACC/BA use raw logit>=0, while Effort native
argmax ties use margin>0. These reporting rules do not alter fusion predictions.

Main results keep the original `metrics.csv` PPA values and add native metrics
and `native_vs_ppa.csv` with signed differences, including negative gains.
GenImage and CommGen15 remain the targets. Macro-8/source overlap, Macro-7 and
content-disjoint disclosures are unchanged. [Data audit](dataset-protocol-audit.md).

## E. Verification

See [verification](verification.md) and [route evidence](evidence/four-host-route-verification.json)
for exact case counts and measured errors. Current suite: **40 passed**,3 upstream warnings/no skips; pre-exclusion28 passed. All16 frozen host/route/dataset pairs and8 real-cache subset CLIs pass, as do real JPEG/blur/noise, PPR-only and4-image candidate CLIP extraction. Checks distinguish unit fixtures,
real-cache frozen original-head replay, tiny real-data CLI fusion fits and
limited candidate CLIP raw inference. Real target subsets are not benchmark
results. No DINOv2/CO-SPY/Effort public raw upgrade is asserted.

## F. Remaining blockers

[Typed register](release-blockers.md): final paper/extension lineage/license/data
version require author decisions; dataset/model/native/fusion release rights
require permission; public exact assets, DINO pin, CO-SPY/Effort raw wrappers and
fresh full reproduction remain technically unresolved. Excluded-host packaging
is outside scope, not a blocker that demands a substitute algorithm.

## G. Git status and preservation

Branch `release/ppa-open-source` remains unborn/no commits; candidate files are
untracked/unstaged. `git log -5 --oneline` exits128 because there is no commit.
No add/commit/push/tag/release/history rewrite or research archive deletion.
Final original342-file hashes, six archive statuses/HEADs/branches, unchanged
LICENSE/core algorithms, excluded-file byte preservation and candidate file
safety/link checks are in [final evidence](evidence/four-host-final-checks.json).
