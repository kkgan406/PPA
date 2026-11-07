# Licensing and attribution — four-host scope

The project [LICENSE](../LICENSE) remains **undecided and unchanged**. The owner
must choose/approve a grant with contributors' rights confirmed. This audit does
not confer code, checkpoint, dataset or manifest redistribution permission.

| Source | Observed terms / provenance | Remaining limitation |
| --- | --- | --- |
| LPIPS0.1.4 | Installed package BSD-style terms, copyright2018 Richard Zhang et al. | Dependency only; model terms separate |
| PyTorch / torchvision | Official packages/AlexNet dependency | No checkpoint bundled |
| CLIP / Transformers | Official model revision and Transformers4.44.2 | Code/runtime terms do not automatically clear model/native-head redistribution |
| DINOv2 | Original snapshot has LICENSE and separate code/model license documents | Review exact ViT-B/14 asset terms; original upstream code pin unresolved |
| CO-SPY | Actual MIT LICENSE, copyright2025 Siyuan Cheng | SigLIP, VAE, semantic/artifact/fusion assets have separate terms/provenance |
| Effort | README explicitly declares **CC BY-NC 4.0** | Preserve noncommercial restriction; clarify bundled DeepfakeBench components and exact model terms; do not present as unrestricted commercial use |
| GenImage / CommGen15 | No established image/manifest redistribution authorization | Images and full manifest drafts remain external; confirm official acquisition/version/terms |

No third-party source tree, checkpoint, dataset image, cache or serialized
experiment model is included. Installation follows pinned official sources
under their own terms. Hashes identify assets; they are not permission grants.

CLIMP/C2P/AIDE implementations/configurations are excluded from current runtime.
Their past audits remain [historical license evidence](phase3-licenses.md),
not dependencies or support promises of this four-host release. No original
research file/asset is removed or relicensed. See [blockers](release-blockers.md).

Phase5 freeze does not alter these terms or grant rights. See the [typed asset/permission readiness](phase5-asset-readiness.md) and [owner decision checklist](phase5-author-review.md).
