# Four-host public support

The paper evaluated seven hosts. The current candidate focuses on four
representative hosts; excluded paper experiments and their historical audit
records are not changed. Public means candidate code, not a published or
license-cleared repository. A=limited Public raw verified, B=Cache verified,
C=Smoke only, D=Blocked. External dependencies/assets remain prerequisites.

| Group / Host | Public raw | Feature / score cache | External dependency required | Checkpoint available? | License cleared? |
| --- | --- | --- | --- | --- | --- |
| General-purpose / CLIP-L/14 (`CLIP`) | A: existing candidate loader, limited real-image checks | B / B | Exact Transformers/base model and original native LR; authorized caches/heads for fusion | Official base identity/hash known, exact heads local; approved native/fusion-head public endpoint missing | No blanket clearance; PPA license undecided and model terms separate |
| General-purpose / DINOv2 | D for original snapshot; loader interface only | B / B | Exact original code snapshot, official weights, original native LR | Original weights local/rehashed;156 source hashes known; upstream commit unresolved | Exact code/model grant review remains required |
| Dedicated / CO-SPY | D: public detector raw adapter absent | B / B | Original semantic/artifact/fusion and SigLIP/VAE assets for cache production | Local exact components rehashed; follow pinned official instructions, terms pending | MIT source observed; component/weight terms separate |
| Dedicated / Effort | D: public fidelity raw adapter absent | B / B | Official detector + original inference environment/cache producer | Official checkpoint local/rehashed; forward-completeness caveat retained | README declares **CC BY-NC 4.0**, including noncommercial restriction; bundled components/model terms need review |

Only CLIP retains measured candidate raw support. DINOv2's original snapshot
has no recovered upstream commit; moving upstream-main is not an equivalent.
CO-SPY/Effort archive raw checks do not supply public raw adapters. Cache B
verifies frozen-head prediction replay on real bound observations, not a fresh
user's ability to obtain unpublished assets or reproduce an entire benchmark.

CO-SPY Main Results requires its original saved `native_probability` cache field
for faithful native AP/AUC comparison; old logit-only caches need enrichment
from the bound original probability arrays, not sigmoid recomputation.

All eight feature/score routes remain exposed through `scripts/main_results.py`.
Original host view/embedding/native score, source normalization, classifier
selection and thresholds are described in [the protocol matrix](host-reproducibility-matrix.md).
Source/version/checkpoint hashes and license evidence are retained there and in
[the current inventory](evidence/four-host-checkpoint-inventory.csv).
CLIMP, C2P-CLIP and AIDE are outside this release scope; no substitute algorithms
are introduced. Historical seven-host evidence remains explicitly historical.
