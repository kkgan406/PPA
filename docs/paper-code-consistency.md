> Scope update: CLIMP, C2P-CLIP and AIDE are excluded from current public interfaces. This document is retained as historical paper/audit evidence; see [current four-host scope](four-host-release-scope.md). Historical interface commands are not current reproduction instructions.

# Paper–Code Consistency Matrix — Phase 3

**No final paper/manuscript was supplied.** This is an implementation-backed
checklist for the authors, not a completed manuscript audit. Equation numbers,
section/page references, final table row identifiers and author sign-off remain
blank until the final paper is available. The ten code facts below are grounded
in the phase1/phase2 audits and the actual current code, not reconstructed from
an assumed paper description. No formal experiment was changed to fit a narrative.

A/B/C/D always describe an implementation route: A=limited public raw-image
inference verified, B=frozen-cache inference verified, C=interface/smoke only,
D=blocked. Earlier archive raw verification is not public raw verification.

| # / item | Historical implementation to preserve | Standard / exception | Verified implementation and remaining gap | Exact author check (paper reference/sign-off pending) |
| --- | --- | --- | --- | --- |
| 1. PPR Global Scalar Response | Raw post-ReLU absolute differences; spatial mean per channel, channel mean per layer, then **equal mean of five layer means**. Frozen LPIPS v0.1 input scaling; no learned distance weights or feature channel normalization | Standard Motivation protocol | `ppa/response.py:global_response`, original aggregate_response/p0_1 chain; actual frozen Alex forward/tests and historical raw-score AUC replay verified | Does the equation specify equal layer weighting? Does any “Global Scalar” table accidentally use the separate learned LPIPS distance control? Are blur=-1/JPEG=-1/noise=+1 inherited source-VAL orientations stated, not test-selected signs? |
| 2. Structured PPR1536 | L4 and L5 each256 channels; absolute response; population std(correction0); concatenate Mean_L4,Mean_L5,Std_L4,Std_L5,Max_L4,Max_L5 | Standard main PPR descriptor | `structured_descriptor`; 1536D/order/precision tests and32 real-image parity checks verified; no raw feature-map resizing/padding | Does the descriptor formula/order match the archived1536D vector? Does “std” mean population std? Are raw F and LPIPS-normalized distance clearly distinguished? |
| 3. Canonicalization / JPEG Q50 | EXIF transpose/RGB -> bicubic aspect-preserving long512 -> isotropic min64 -> Pillow JPEG90/subsampling2 -> decode -> JPEG50/subsampling2 recode. Python round; Q90 optimize/progressiveFalse; no square stretch | Standard PPR view; host views can differ | `ppa/preprocessing.py`; pixel/JPEG/RNG parity verified | Is Q50 applied to decoded Q90, not original bytes? Does the resize description preserve the min64 safeguard (extreme long side can exceed512)? Are Pillow/torchvision versions and fixed blur/noise parameters recorded? |
| 4. PPR-only classification | Source SD1.4 TRAIN12398 only fits scaler + balanced LBFGS LR; VAL800 selects C=.01/.1/1/10 by AP/ACC/AUC/smaller C. No TRAIN+VAL refit; full-fit head at inference, fixed margin>=0 | Standard standalone linear protocol | `ppa/fusion.py`, experiment/data guards; original source refit matched archived head in phase1; no formal refit in phase3 | Is standalone independent of any Host? Is its full-fit logit distinguished from the five-fold response ensemble inside some score routes? Are source-only selection and label1=fake explicit? |
| 5. Feature-Level Fusion | Six hosts: frozen host feature + PPR1536 -> TRAIN scaler + balanced LR. Host dimensions768/768/768/1664/768/1024 for CLIP/DINOv2/CLIMP/CO-SPY/C2P/Effort | Standard for six; **AIDE is the nonlinear exception** | Six routes B; CLIP/CLIMP limited A with local exact assets. AIDE's generic LR entry remains D; separate external A3 interface B with local original assets | Does any blanket “all fusion is linear” or “only LR is trained” statement incorrectly include AIDE A3? Are CO-SPY actual branch features, native host views and host-specific normalization retained? |
| 6. Score-Level Fusion | CLIP/DINOv2/CLIMP/CO-SPY use native logit + five-fold PPR OOF -> scaler/LR; VAL/TEST PPR logit is fold mean. C2P raw additive, AIDE TRAIN-z additive, Effort raw additive plus VAL threshold | Four-host OOF standard plus three exceptions | `Fusion` modes and26 fixed-head cache replays verified; detector public raw paths are D even if archive raw passed | Does the paper avoid describing every score route as LR or OOF? Are in-sample full-fit response scores for additive hosts, source-only alpha grids, and C2P historical alpha0 disclosed? |
| 7. AIDE A3 nonlinear feature | Original TRAIN A1 scaler; PPR1536 -> Linear256/LayerNorm/GELU; concat host2304 -> Linear2560/1024/GELU/2; fake-minus-real logit; exact A3 checkpoint, seed0/epoch5 historical source selection | Host-specific nonlinear exception, Native=A0/Feature=A3 | Original code/checkpoint/table mapping verified. Phase3 `ppa/aide.py` wraps the **external original head**; strict FP32 state/scaler and real cached/raw-archive comparisons verified. No MLP implementation/weight copied; fresh public assets unresolved | Does the main F row identify A3, not A1 LR? Are the nonlinear head's capacity/training and mixed ACC outcomes acknowledged? Author must confirm this exception in final Methods/Table text |
| 8. AIDE A2 score | TRAIN-z(A0 native margin) + alpha * TRAIN-z(full-fit A1 logit), selected alpha=.25; source-only mean/population std; margin>=0; no OOF and no fusion LR | Host-specific score exception, Score=A2 | Candidate score B plus archive raw check; original A1 FP32 scaler-input precision preserved | Is TRAIN centering distinguished from unchanged raw A0 threshold? Are no OOD statistics used? Does the S row map to A2 and not A3? |
| 9. Effort operating point | Source VAL first selects additive alpha using AP/ACC at0/AUC/smaller alpha, then maximum-BA threshold; tie closest0 then smaller. Native baseline uses strict margin>0 | Host-specific threshold/precision/load exception | `Fusion(mode='effort')`, source_ba_threshold and fixed-head replay verified. Official serialization strict-load still fails on288 auxiliary-only tensors; inference gate passed in archive; public raw remains D | Does ACC name the actual VAL-derived threshold? Is the baseline tie rule distinct? Do not claim serialization strict-load PASS, a universal zero threshold, or OOD threshold selection |
| 10. GenImage / CommGen15 evaluation | Fake-positive AP/AUC; equal unrounded generator macro distinct from pooled. GenImage8 N99991 includes SD1.4 source family and11999 current TRAIN overlaps; Macro7 N87991 excludes SD1.4; CommGen15 N66785 | Standard metric aggregation; source-family and content-disjoint scopes are distinct | Manifest/ID/hash/count/overlap checks and cached metrics verified; exact portable drafts validated locally, not released. Extension MSM lineage and unknown host pretraining overlap remain unresolved | Are Macro8/Macro7/content-disjoint clearly labeled and table units correct? No strict all-OOD claim for Macro8; no fresh-confirmation claim for historically observed targets. Authors must select/identify final extension feature lineage without editing archived results |

## Required author sign-off package

Supply the final manuscript revision (file SHA256 or immutable revision), then
for **each row** record: Methods/appendix equation and page, table/figure row,
exact experiment/source seal, any host exception, author decision and date.
Until then all ten manuscript-comparison statuses are **Requires author decision**,
even where code correctness is independently verified. A title/request alone is
not a final paper, and no DOI/BibTeX/venue/author list is inferred here.

Specific statements requiring correction if present in the final paper:

- “PPR scalar is LPIPS perceptual distance” conflicts with the raw five-layer definition.
- “All feature fusion classifiers are linear” conflicts with AIDE A3.
- “All score fusion is LR / OOF” conflicts with C2P, AIDE and Effort.
- “All ACC uses zero threshold” conflicts with Effort's score route.
- “GenImage Macro8 is strictly unseen OOD” conflicts with the11999 overlaps.
- “AIDE A3 improves every metric” conflicts with archived ACC and CommGen15 outcomes.
- “Archive raw inference verifies every public adapter” confuses implementation scope.

Evidence: [AIDE audit](aide-fusion-audit.md), [phase2 host matrix](host-reproducibility-matrix.md),
[dataset audit](dataset-protocol-audit.md), [phase3 support](public-host-support.md),
and [verification](verification.md).
