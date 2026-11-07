> Historical seven-host audit. The current public scope is four hosts; see [scope report](four-host-release-scope.md). Commands/interfaces described below may have been excluded.

> Historical Phase 1/2 record. Current Phase 3 scope and blockers: [release readiness](release-readiness.md), [AIDE decision](aide-release-decision.md), [public support](public-host-support.md). Earlier A3 scope restrictions are superseded only for the conditional external inference contract; generic AIDE LR feature fusion remains blocked.

# Host Reproducibility Matrix (phase 2)

Status meanings: **A** raw-image end-to-end verified, **B** frozen-cache
reproduction verified, **C** interface/smoke only, **D** unsupported/blocked.
A/B/C/D describe a stated route and environment, not an entire benchmark claim.
A public cache route is never promoted to A because an archive adapter works.

This round used two fixed, byte-hash-verified GenImage6 images (one real/one
fake), freshly recomputed CPU FP32 PPR, frozen GPU host and original frozen fusion
heads. No synthetic image or target cache was used as the prediction input.
Target caches were comparison references only. V100S GPU access required running
outside the CPU-only sandbox. A only certifies this small inference scope, not
full training/benchmark reproduction or fresh-install availability.

| Host | Candidate raw path | Candidate feature route | Candidate score route | Local archive raw audit this round |
| --- | --- | --- | --- | --- |
| CLIP | A (existing FrozenHost) | A limited raw inference; B archived cache replay | A limited raw inference; B archived cache replay | A, both routes, n=2 |
| DINOv2 | D: old torch-hub snapshot has no Git metadata; candidate requires a pinned Git checkout | B | B | A, original snapshot + candidate fusion, n=2 |
| CLIMP | A (existing FrozenHost, CUDA and exact assets) | A limited raw inference; B cache replay | A limited raw inference; B cache replay | A, both routes, n=2 |
| CO-SPY | D: detector raw adapter not shipped | B | B | A, original adapter + candidate fusion, n=2 |
| C2P-CLIP | D: detector raw adapter not shipped | B | B | A, original adapter + candidate fusion, n=2 |
| AIDE | D: detector raw adapter not shipped | D: historical MLP exception unresolved | B | A, original A0/A3 + candidate A2, n=2; A3 archive-only |
| Effort | D: official fidelity adapter not shipped | B | B | A with explicit auxiliary-only load caveat, n=2 |

Historical B checks comprise 26 host-route/dataset pairs on real frozen
GenImage6/CommGen15 features, not interface-only synthetic tests. The automated
unit suite also includes C-level synthetic interface checks, labeled separately.
No A entry above certifies a public fresh checkout can obtain unreleased assets.

## Official code version and checkpoint identity

| Host | Official source and code version | Actual checkpoint lineage |
| --- | --- | --- |
| CLIP | [OpenAI CLIP HF model](https://huggingface.co/openai/clip-vit-large-patch14); Transformers4.44.2 implementation; model revision `32bd64288804d66eefd0ccbe215aa642df71cc41` | Official ViT-L/14 FP32 model; original source-only native LR must also be supplied |
| DINOv2 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2); original hub snapshot, upstream Git commit unresolved | Official dinov2_vitb14_pretrain.pth; 156 source files match the original sealed config hashes |
| CLIMP | [NimrodShabtay/CLIMP](https://github.com/NimrodShabtay/CLIMP) @ `3ff1279bb87f905596e6187f0e4a14f2cd28e818` | Official Mamba1 vision/projection; 16 original source hashes match the clean checkout; original HF weights revision unresolved |
| CO-SPY | [Megum1/Co-Spy](https://github.com/Megum1/Co-Spy) @ `4be0ba51fc6c96a06d5d3b12e36439d5d1507e16` | Official SD-v1_4 semantic/artifact/fusion plus official SigLIP and VAE config |
| C2P-CLIP | [Official repo](https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection) @ `5c1a66cdd70f21e7130a28b9f5ff53e94dea5bc2` | Project source-trained SD1.4 checkpoint, epoch3; not a substitute upstream release checkpoint |
| AIDE | [shilinyan99/AIDE](https://github.com/shilinyan99/AIDE) @ `6725b710d5c437ab2f59792908ce0377dfc907de` | Project source-trained A0 epoch20; A3 epoch5; exact pretrained ConvNeXt/ResNet also recorded |
| Effort | [YZY-stack/Effort-AIGI-Detection](https://github.com/YZY-stack/Effort-AIGI-Detection) @ `96f5dea2b534d400cfd7003f053c7e93c8e16461` | Official SD-v1.4 checkpoint; all forward tensors present; strict serialization misses 288 auxiliary-only tensors |

Full SHA256, actual rehashed sizes and acquisition lineage for **24 assets/heads**
are in [checkpoint inventory](evidence/phase2-checkpoint-inventory.csv). Hashes
identify multiple CO-SPY files separately, rather than inventing one model hash.
DINOv2 and CLIMP's preserved source-file bindings are in
[source snapshot evidence](evidence/phase2-source-binding.json). HF revision,
implementation-package version, Git commit, and byte hash are different fields.

## Exact original view, feature, native logit and fusion

All PPR branches use the fixed Q90->Q50 AlexNet1536 protocol. Native dedicated
host inputs do not receive the PPR canonicalization. No extra normalization,
new projection or classifier was introduced in this round.

| Host | Original host input preprocessing | Feature location / dimension | Native score | Feature fusion | Score fusion |
| --- | --- | --- | --- | --- | --- |
| CLIP | EXIF/RGB canonical Q90; bicubic shortest224/center224; CLIP mean/std | projected image_embeds768, L2 | original frozen source LR decision_function; historical AUC/AP/smaller-C selection | concat768+1536 -> TRAIN scaler + balanced LR | native + PPR 5-fold OOF -> TRAIN scaler/balanced LR; external PPR fold-mean |
| DINOv2 | canonical Q90; bicubic shortest256/center224; ImageNet mean/std | final x_norm_clstoken768; no extra L2 | original sealed source LR | concat768+1536 -> LR | same 5-fold OOF LR protocol |
| CLIMP | canonical Q90; bilinear square224; ImageNet mean/std | pretrained vision1024 -> projection768, L2 | original sealed source LR | concat768+1536 -> LR | same 5-fold OOF LR protocol |
| CO-SPY | PIL RGB native; bilinear shortest384/center384/ToTensor; semantic mean/std .5/.5; artifact bilinear224 antialiasFalse, mean0/std1 | actual semantic fc input1152 + artifact fc input512; not branch logits | official final fusion Linear(2,1) raw logit | concat1664+1536 -> LR | official logit + PPR 5-fold OOF -> LR |
| C2P-CLIP | PIL RGB native; official small-image tiling safeguard, CenterCrop224, CLIP mean/std | final model.fc pre-hook, normalized projected embedding768 | official forward_eval -> final fc output | concat768+1536 -> LR | native + alpha*full-fit PPR; source grid0/.25/.5/1; historical alpha=0 |
| AIDE | PIL RGB -> tensor; deterministic official DCT patch selection + full image; five square256 views, bilinear resize and ImageNet mean/std | final fc pre-hook: semantic256 + artifact2048 | original two logits fake-minus-real | original TRAIN-scaled PPR -> 1536-to-256 projection, 2560-to-1024-to-2 MLP (candidate D) | TRAIN-z(native)+alpha*TRAIN-z(full-fit PPR), alpha=.25, no OOF |
| Effort | OpenCV BGR read -> RGB, cv2 INTER_LINEAR square224, CLIP mean/std | official output feat1024 | cls[:,1]-cls[:,0] | concat1024+1536 -> LR | native+alpha*full-fit PPR; source VAL alpha then maximum-BA threshold |

LR is StandardScaler + class-balanced LBFGS LogisticRegression. C grid
.01/.1/1/10 uses source VAL AP -> ACC -> AUC -> smaller C for formal fusion;
no TRAIN+VAL refit. OOF is StratifiedKFold(5, shuffle=True, random_state=42),
each TRAIN image held out once, each fold scaler fit only on its fit subset.
AIDE seed0 and FP32 scaler inputs, and Effort FP32 inference/FP64 C-selection,
remain host-specific; other formal linear routes use explicit FP64 sklearn inputs.

## Remaining gaps and how to acquire weights

Use the pinned official README/download script: CO-SPY `pretrained/download.sh`
for SD-v1_4; C2P-CLIP's official Google Drive folder and CLIP HF instructions;
AIDE's official pretrained instructions; Effort's official checkpoint folder.
CLIMP's README links [CLIMP-Mamba1](https://huggingface.co/NimrodShabtay1986/CLIMP-Mamba1).
The supplied project's **source-trained C2P/AIDE detector heads and all native
source LR heads are separate assets**. Their download endpoints/redistribution
permission are still TODO; upstream off-the-shelf weights cannot replace them.

Every local raw audit passed at fixed, predeclared feature/PPR/score tolerances;
maximum actual raw fusion-score error across the seven hosts was below 4.5e-5.
These tolerances were not tuned on target performance. The AIDE environment
initially lacked LPIPS; the retry used the already-installed original LPIPS
package path, with the same verified weights, without installing or changing a
model. Raw-script evidence is [here](evidence/phase2-raw-host-verification.json).
