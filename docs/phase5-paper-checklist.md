# Phase 5 Paper–Code Consistency Checklist

**AUTHOR DECISION REQUIRED — no final manuscript located.** Search covered the
candidate and research repository, research `docs` LaTeX/table files, shallow
project/report paper paths, and supplied attachments. Sixteen research LaTeX
files are table fragments, without `documentclass`/`begin{document}`; attached
PPA-related texts are task descriptions, not author-approved manuscript sources.
No attached tex/bib/pdf was found. This bounded search is not a claim about
unmounted author machines or unknown external locations.

For every row below: **paper location/revision = NOT PROVIDED; author sign-off =
PENDING**. A code fact verified by tests/archive replay is not paper verification.
No paper/code conflict can be asserted at an invented page/equation. Suggestions
below are concrete author checks, not confirmed manuscript defects. No code was
changed to fit an assumed paper narrative.

| Item | Candidate code location | Actual frozen implementation | Required author check / recommended manuscript action |
| --- | --- | --- | --- |
| PPR mathematical definition | [`ppa/response.py:53`](../ppa/response.py#L53) | Raw post-ReLU AlexNet absolute activation difference after official v0.1 input scaling; no learned perceptual-distance weight/channel normalization | Confirm equation defines raw response rather than LPIPS distance; amend Methods if different |
| Canonicalization / perturbations | [`ppa/preprocessing.py:51`](../ppa/preprocessing.py#L51); [`ppa/preprocessing.py:61`](../ppa/preprocessing.py#L61) | EXIF/RGB, AR bicubic long512, isotropic min64, decoded JPEG90 -> JPEG50/subsampling2; blur kernel5 sigma1; keyed CPU Gaussian sigma.03 seed42 | Confirm Q50 input is decoded Q90; state min64 safeguard and fixed blur/noise parameters |
| Global Scalar | [`ppa/response.py:69`](../ppa/response.py#L69) | Equal mean of five layer means, each averaged over C/H/W; JPEG/blur orientation -1, noise +1 inherited from source VAL | Confirm equal layer weighting and source-fixed signs; do not call a learned distance training-free scalar |
| Structured1536D | [`ppa/response.py:74`](../ppa/response.py#L74) | L4/L5 each256 channels; stat-major Mean4/Mean5/Std4/Std5/Max4/Max5; population std | Confirm dimension/order/population std; no map resizing/padding in descriptor |
| PPR-only | [`ppa/fusion.py:33`](../ppa/fusion.py#L33); [`ppa/experiment.py:19`](../ppa/experiment.py#L19) | TRAIN scaler+balanced LR; VAL C grid.01/.1/1/10 via AP/ACC/AUC/smaller C; full-fit inference, >=0; no TRAIN+VAL refit | Confirm source-only selection and distinguish standalone head from OOF fold ensemble |
| Feature fusion | [`ppa/fusion.py:64`](../ppa/fusion.py#L64) | Retained four hosts: original embedding+1536 -> TRAIN scaler/balanced LR; no Host training | Confirm actual embedding definitions/dimensions; full paper AIDE nonlinear exception stays separate historical scope, never labeled LR |
| Score fusion | [`ppa/fusion.py:74`](../ppa/fusion.py#L74) | CLIP/DINO/CO-SPY: five-fold source PPR OOF + native logit -> LR; external PPR fold mean | Confirm folds/scalers/selection; avoid a blanket LR/OOF description that includes additive hosts |
| Host-specific exceptions | [`ppa/fusion.py:106`](../ppa/fusion.py#L106); [`ppa/experiment.py:137`](../ppa/experiment.py#L137) | Effort alpha .25 / source-VAL BA threshold2.008856108429543 in frozen formal head; FP32 inference; native >0. CO-SPY native AP/AUC use saved FP32 probabilities, canonical ACC/BA raw>=0 | Name threshold provenance, tie/precision/ranking conventions. Full seven-host AIDE/C2P exceptions must be confirmed from historical audit, without re-enabling them |
| Evaluation metrics | [`ppa/metrics.py:34`](../ppa/metrics.py#L34); [`ppa/experiment.py:103`](../ppa/experiment.py#L103) | Real0/fake1, generator AP/ACC/AUC/BA; equal unrounded generator macro vs pooled; signed gains retained | Confirm units, aggregation, baseline ranking/threshold and all negative gains; no OOD parameter selection |
| GenImage source overlap | [`ppa/data.py:30`](../ppa/data.py#L30); [`ppa/metrics.py:7`](../ppa/metrics.py#L7) | Macro8 N99991 includes SD1.4 and11999 TRAIN-overlap observations; Macro7 N87991 removes whole SD1.4 domain; content-disjoint separate | Confirm overlap disclosure; remove any strict-all-unseen claim for Macro8; retain historical numbers |
| GenImage / CommGen15 result rows | [`ppa/experiment.py:70`](../ppa/experiment.py#L70) | Source TRAIN12398/VAL800; CommGen15 N66785; exact ID/content manifests and archive replay; extension MSM lineage unresolved | Provide exact final table/equation revision and source-seal/feature-lineage mapping, including seven-host rows; do not substitute cache arrays to match prose |

## Author evidence required to close each row

Record final source filename + immutable revision/SHA256, section/page/equation,
result table/row/units, matching experiment/source-seal/checkpoint/cache hashes,
applicable exception, conflict resolution, approver and approval date. When a
conflict is discovered, log the actual paper location and code location together
with the observed protocol; propose a manuscript correction or author decision.
Keep the verified implementation/formal results unchanged pending that decision.

Historical full-paper exceptions remain in [Phase3 checklist](paper-code-consistency.md),
[AIDE audit](aide-fusion-audit.md) and [Phase2 matrix](phase2-host-reproducibility-matrix.md).
They are author review evidence, not additional current public interfaces.
The unresolved extension descriptor lineage must be resolved by identifying the
intended formal artifact, not silently regenerating IDs/features or retraining.
