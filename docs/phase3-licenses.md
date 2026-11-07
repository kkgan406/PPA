> Historical seven-host audit. The current public scope is four hosts; see [scope report](four-host-release-scope.md). Commands/interfaces described below may have been excluded.

# Licensing and attribution — phase 2

The original PPA research repository had no top-level project LICENSE. The
candidate [LICENSE](../LICENSE) remains **undecided and unchanged**. The owner
must approve the grant and confirm the original contributors' rights. Nothing
in this audit chooses a project license or changes third-party terms.

This round read actual LICENSE files **and README declarations**, correcting the
phase1 inference that an absent root LICENSE meant no declared license.

| Source | Observed code / declaration evidence | Weight/data redistribution status |
| --- | --- | --- |
| LPIPS0.1.4 | Installed package LICENSE contains the original BSD-style redistribution terms, copyright2018 Richard Zhang et al. | Package is a dependency; verify model terms separately |
| PyTorch / torchvision | Official packages/AlexNet asset; dependencies only | No pretrained weights bundled; user obtains official assets |
| CLIP / Transformers | Official HF model and Transformers4.44.2 runtime | Code/runtime terms do not by themselves clear model redistribution |
| DINOv2 | Official snapshot contains LICENSE plus distinct code/model license documents | Review the actual ViT-B/14 asset terms; do not apply cell/X-ray model terms indiscriminately |
| CLIMP | README `License` explicitly says **Apache-2.0**; no standalone LICENSE found in inspected checkout | Original HF model-card/revision and redistribution terms remain to be documented separately |
| CO-SPY | Actual MIT LICENSE, copyright2025 Siyuan Cheng | SigLIP, VAE/artifact, semantic and fusion weights are separate dependencies with their own provenance/terms |
| C2P-CLIP | No LICENSE found in the inspected official tree; README does not establish a general source redistribution grant | Code/weights permission unresolved; Drive availability is not authorization |
| AIDE | Actual MIT LICENSE, copyright2024 Shilin Yan | Project source-trained A0/A3 and pretrained ConvNeXt/ResNet require their own release review |
| Effort | README badge explicitly declares **CC BY-NC4.0**, linking the license; no standalone LICENSE found | Noncommercial restriction must be preserved/clarified; review original detector and bundled DeepfakeBench components/weights separately |
| GenImage / CommGen15 | No image redistribution authorization established by this audit | Images remain outside release; official acquisition/terms and permission for portable manifests still pending |
| AIDE Chameleon dataset | README separately restricts Chameleon to academic research and prohibits commercial use | Dataset-specific statement, not a replacement for AIDE's MIT code LICENSE; Chameleon is not part of the audited evaluation |

No third-party source tree, checkpoint, dataset image, feature cache or serialized
experiment model is included in the candidate. The source/checkpoint inventory
lists names and hashes, not redistributable model files. Loading an official
model locally for this audit is not permission to publish its weights.

Recommended minimal action: preserve the observed notices; document model/data
terms at the exact revision; obtain permission or keep third-party assets
external. Do not relicense an upstream tree under the eventual PPA license.
These observations are release evidence, not legal clearance.

## Phase 3 external interface boundary

`ppa/aide.py` calls an owner-supplied original A3 head after checking declared
external source hashes and the exact A3/A1 assets. The candidate includes no
original MLP implementation, upstream source tree, serialized scaler or weights.
It does not create a license grant for externally supplied code or assets.
The original audit helper is separate from the official AIDE code and requires
rights/acquisition review. The project LICENSE is unchanged and undecided.
The typed [blockers](release-blockers.md) govern publication readiness.
