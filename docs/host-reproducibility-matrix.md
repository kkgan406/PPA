# Four-host reproducibility and protocol matrix

Current support is [here](public-host-support.md); the unchanged seven-host
Phase2 audit is [historical evidence](phase2-host-reproducibility-matrix.md).
All original checkpoint hashes for the retained hosts are in the
[filtered asset inventory](evidence/four-host-checkpoint-inventory.csv).
No cache/local-archive check certifies a public raw chain or complete benchmark.

| Host | Official source / exact code binding | Checkpoint lineage | Input view | Original embedding / native logit |
| --- | --- | --- | --- | --- |
| CLIP | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14), revision `32bd64288804d66eefd0ccbe215aa642df71cc41`, Transformers4.44.2 | Official ViT-L/14 + original source-only native LR | EXIF/RGB canonical decoded Q90, bicubic shortest224/center224, official CLIP mean/std | projected image_embeds768, L2; original frozen LR decision_function (historical native selection AUC/AP/smaller C) |
| DINOv2 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), original hub snapshot, upstream commit **unresolved**;156 sealed source hashes | Official dinov2_vitb14_pretrain.pth + original native LR | canonical Q90, bicubic shortest256/center224, ImageNet mean/std | x_norm_clstoken768, no additional L2; original frozen source LR decision_function |
| CO-SPY | [Megum1/Co-Spy](https://github.com/Megum1/Co-Spy) @ `4be0ba51fc6c96a06d5d3b12e36439d5d1507e16` | Official SD-v1_4 semantic/artifact/fusion plus SigLIP/VAE; hashes separate per component | Native PIL RGB, bilinear shortest384/center384/ToTensor; semantic .5/.5; artifact bilinear224 antialiasFalse mean0/std1 | actual semantic fc input1152 + artifact fc input512, no extra normalization; official final Linear(2,1) raw fusion logit |
| Effort | [YZY-stack/Effort-AIGI-Detection](https://github.com/YZY-stack/Effort-AIGI-Detection) @ `96f5dea2b534d400cfd7003f053c7e93c8e16461` | Official SD-v1.4 detector; strict serialization missing288 auxiliary-only tensors, historical forward completeness/demo gates passed | Native OpenCV BGR->RGB, INTER_LINEAR square224, CLIP mean/std | official feat1024, no extra L2; cls[:,1]-cls[:,0], native ties use strict margin>0 |

Native dedicated-detector views are separate from PPR canonical Q90->Q50.
Do not resize or normalize cached embeddings or replace native margins with
probabilities. No raw CO-SPY/Effort adapter is supplied. CLIP raw checks remain
limited real samples; DINOv2 original public raw remains blocked by code binding.

| Host | Feature fusion | Score fusion | Source-only selection / fixed target operating point |
| --- | --- | --- | --- |
| CLIP | concat768+1536 -> TRAIN StandardScaler + balanced LR | native + five-fold held-out TRAIN PPR logits -> TRAIN scaler/LR; external PPR mean of five fold models | Source VAL AP/ACC/AUC/smaller C; fixed >=0 |
| DINOv2 | concat768+1536 -> same LR | same five-fold OOF LR | Same; fixed >=0 |
| CO-SPY | concat1664+1536 -> same LR | official native logit + five-fold PPR OOF -> same LR | Same; fixed >=0 |
| Effort | concat1024+1536 -> same LR | native + alpha*full-fit PPR; **not a score-fusion LR** | Source VAL alpha first at zero; source-VAL maximum-BA threshold then frozen; feature >=0, score >=selected threshold |

Fusion C grid .01/.1/1/10; LBFGS/class_weight=balanced; no TRAIN+VAL refit.
Five-fold StratifiedKFold shuffles with seed42; each scaler sees only its fit
subset. Effort alpha grid0/.25/.5/1 uses VAL AP/ACC/AUC/smaller alpha; historical
alpha=.25 and threshold=2.008856108429543 are retained in frozen replay. BA threshold ties closest zero then smaller
threshold. Original sealed coefficients/scalers are not refitted for replay.
Target metrics, including negative gains, never select the head, alpha or route.

Effort retains FP64 fitting/C-selection but FP32 inference scaler input. Other
retained routes use original FP64 sklearn inputs. Backbone precision and
preprocessing remain unchanged. See [host detail](hosts.md), [cache contract](cache-format.md),
[verification](verification.md) and [licenses](licenses.md).

Use pinned official README acquisition instructions: CO-SPY `pretrained/download.sh`
for SD-v1_4 and Effort's official checkpoint folder; never invent an endpoint or
substitute a different checkpoint. Exact project native/fusion heads still need
approved acquisition instructions and release rights. No source tree or weights
are bundled. DINO code byte evidence remains in the historical
[source bindings](evidence/phase2-source-binding.json).

CO-SPY formal unified native AP/AUC use original saved FP32 sigmoid probabilities
(including saturation ties), while canonical native ACC/BA use raw logit>=0.
The older detector-only C0 report used probability>.5; these are disclosed as
separate conventions, not silently interchangeable. Native-versus-PPA reporting
preserves the unified table convention. Effort native remains strict margin>0.
