# Frozen four-host protocols

Current public scope is CLIP-L/14, DINOv2, CO-SPY and Effort. See the
[protocol/provenance matrix](host-reproducibility-matrix.md) for input geometry,
feature/native-logit definitions, exact code versions, hashes and operating points;
[public support](public-host-support.md) separately states what is available.

All four feature routes concatenate the original frozen embedding with PPR1536
and fit a source TRAIN scaler/balanced LR selected on source VAL. CLIP/DINOv2/
CO-SPY score routes use source-only five-fold PPR OOF plus native logit -> LR.
Effort preserves raw additive full-fit PPR and source-VAL-selected alpha and BA
threshold; it is not unified into the OOF/LR score route. Native Effort uses
strict >0; augmented score uses >= its selected threshold. No OOD selection.

Optional raw extraction supports only CLIP/DINOv2 interfaces. CLIP is limited
public raw verified; DINO's original source pin is unresolved. CO-SPY/Effort use
bound caches until faithful public adapters are available. Their upstream MIT
and CC BY-NC4.0 declarations respectively do not clear all asset redistribution.
