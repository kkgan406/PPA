# Dataset Protocol Audit (phase 2)

All counts and intersections below were recomputed from the frozen manifests.
Original IDs, ordered labels, generator names and full content SHA256 remain
unchanged. Manifest hashes and ID-generation checks are in
[summary evidence](evidence/phase2-dataset-summary.json); per-generator counts and
source intersections are in [counts](evidence/phase2-dataset-counts.csv) and
[overlap](evidence/phase2-dataset-overlap.csv). This round rehashed every image
used in the raw probes, not every source/target image file again. Manifest hashes and
checkpoint hashes were independently recomputed.

## Frozen observations

| Split / scope | Observations | Real=0 | Fake=1 | Unique content SHA256 |
| --- | ---: | ---: | ---: | ---: |
| Source TRAIN | 12,398 | 6,199 | 6,199 | 12,398 |
| Source VAL | 800 | 400 | 400 | 800 |
| GenImage6 formal / Motivation | 75,991 | 37,993 | 37,998 | 75,967 |
| GenImage added glide + SD1.4 | 24,000 | 12,000 | 12,000 | 24,000 |
| GenImage8 | 99,991 | 49,993 | 49,998 | 99,959 |
| GenImage7-unseen | 87,991 | 43,993 | 43,998 | 87959 |
| CommGen15 | 66,785 | 33,392 | 33,393 | 64,814 |

Source is SD1.4 only. Source TRAIN/VAL share no path or content. The source
scanner (`src/data/scan_dataset.py`, `src/source.py`) enumerates sorted class
directories, hashes actual file bytes and quarantines TRAIN contents appearing
in VAL (one real image). `scripts/stage_b2.py:prepare` preserves the inherited
balanced1024 anchor, sorts each class's remaining paths, uses independent
`default_rng(42)` permutations, takes nested class prefixes, and balances the full
source to6199/class; one extra fake is excluded for balance. This is a particular
frozen source split, not a generic official dataset training split.

Source VAL is the frozen800 manifest. GenImage6 comes from the prior target
manifest after removing contents present in effective source TRAIN/VAL; the
canonicalization study froze that exact independent subset. GenImage8
`scripts/unified_cross_host.py:prepare` retains every old-six observation and
adds all12,000 glide and12,000 SD1.4 observations from their paired class folders.
No target response is used to select those samples.

CommGen15 `scripts/stage_c.py:discover` scans paired `0_real`/`1_fake` generator
folders, hashes bytes, rejects contents already present in the frozen source,
VAL, quarantine, balance-excluded source, and old GenImage6 development pool.
Content duplicates within/between target generator observations are retained
and share feature rows. The archive records1,971 repeated CommGen15 content
observations and32 repeated GenImage8 observations; they are not extra independent
images or silently deduplicated metric units. CommGen15 was fresh for the older
Stage C branch, but is a historically observed development target by the later
PPA main-results work; do not call it fresh confirmation now.

## IDs, mapping and portable manifests

All historical ID rules below were checked against every frozen row:

- TRAIN/VAL/old GenImage: first24 hex of SHA256 of Python
  `json.dumps([split, relative_path], sort_keys=True, default=str)`.
- CommGen15: same JSON digest with `['stage-c', relative_path]`.
- Added GenImage domains: first24 hex of SHA256 of the original string
  `'unified-genimage8:' + original_absolute_image_path`.

The extension's ID rule is path-dependent. Preserve its frozen IDs when moving
data; do not regenerate them from a new absolute root. Image SHA256 is the full
64-hex digest of file bytes, distinct from the observation ID. Preserve the IDs
also for keyed noise. All manifest IDs are unique; repeated bytes can have
multiple observation IDs. A manifest's CSV byte hash changes when paths are
made portable: publish the original manifest hash plus a migration binding of
IDs/hashes/labels/domain/split, rather than pretending the two CSVs are identical.

Real=0 and fake=1 come from explicit class directories. Canonical public split
names are `train`, `val`, `test`; the archive also used `genimage` and `fresh_ood`.
Changing only that manifest metadata for the public loader does not change an
image assignment. Use `dataset=source`, `genimage`, `commgen15` namespaces.

GenImage names: `ADM`, `BigGAN`, `Midjourney`, `VQDM`, `glide`,
`stable_diffusion_v_1_4`, `stable_diffusion_v_1_5`, `wukong`. Original six omit
`glide` and `stable_diffusion_v_1_4`.

CommGen15 names: `Doubao_videos`, `akool`, `chatgpt-image`, `flux`, `google-imagen`,
`hailuo`, `hunyuan`, `ideogram`, `kling`, `lexica`, `midjourney`, `nano_banana`,
`sora`, `stable-diffusion`, `veo`. Preserve spelling/case and dataset namespace;
these names are frozen groups, not a license to merge similar model families.

## Macro-8 versus Macro-7-unseen

**Macro-8** is the arithmetic mean of the eight per-generator metrics on99,991
observations. It includes the SD1.4 source family: **11,999 of its12,000
observations have content in current source TRAIN**; none overlaps VAL. It is
not strictly all-OOD.

**Macro-7-unseen** removes the entire12,000-observation SD1.4 group before equally
averaging the remaining seven generator metrics, on87,991 observations. Those
seven groups have zero content overlap with current source TRAIN and VAL.
"Unseen" refers to the current source protocol, not a guarantee about unknown
host pretraining/detector-training corpora. Neither macro is computed by
subtracting one number from the other after rounding.

A **source-content-disjoint** range is another scope: remove all observations
whose content appears in TRAIN/VAL and recompute estimable generator groups.
It can leave the single unmatched SD1.4 observation, for which binary ROC-AUC
cannot be estimated. It must not be silently labeled Macro-7. The current
candidate emits a content-disjoint aggregate only when every retained group has
both classes; for the full GenImage8 case this can be unavailable. Its
`source_overlap.csv` still exposes the excluded observation counts. This is a
reporting limitation, not a reason to manufacture missing metrics or tune a new
threshold. Macro-7 remains directly available.

## Metrics and operating points

AP uses fake-positive raw ranking scores; `ap` is average precision on[0,1].
Macro AP = mean of unrounded generator AP; pooled AP is a separate calculation
on all supplied observations. The same equal-generator rule applies to AUC,
ACC and balanced accuracy. Dataset cardinality alone never labels a partial
manifest as a complete benchmark.

| Route | ACC decision / threshold source |
| --- | --- |
| PPR-only and feature LR | decision_function>=0, fixed; TRAIN scaler, source VAL C |
| CLIP/DINOv2/CLIMP/CO-SPY OOF score LR | fused logit>=0, fixed; fold scalers on TRAIN subsets only |
| C2P score | native+alpha*PPR>=0; source VAL alpha at fixed zero |
| AIDE Native/A2/A3 archived evaluation | respective score/margin>=0; A2 TRAIN-only z statistics and source VAL alpha |
| Effort score | source VAL selects alpha at threshold0, then maximum-BA threshold; ties nearest0 then smaller |
| Native CO-SPY baseline | historical official probability>0.5; fusion inputs are raw logits |
| Native Effort baseline | two-class argmax, fake-minus-real>0; exact tie is real |

No target OOD labels choose C, alpha, checkpoint epoch, orientation or ACC
threshold. Full raw benchmark extraction, final portable split release and
external pretraining overlap are still outside the completed verification scope.
