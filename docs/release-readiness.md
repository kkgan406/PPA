> Current freeze and three separate release judgments: [Phase5 author review](phase5-author-review.md). Asset states: [Phase5 readiness](phase5-asset-readiness.md).

# Current release readiness — four-host scope

**Unpublished candidate; not ready for formal reproducible public release.**
The current entry points cover Motivation, PPR-only and four representative
Host Fusion routes: CLIP-L/14, DINOv2, CO-SPY, Effort. The paper's seven-host
experiments are unchanged. The previous [Phase3 report](phase3-release-readiness.md)
is historical; its AIDE interface/scope statements are no longer current.

The current [delivery report](four-host-release-scope.md) lists the directory,
excluded files, support matrix, protocol checks, tests, blockers and Git status.
[Public support](public-host-support.md) keeps CLIP limited raw A and the other
three raw D/cache B. Both feature/score cache paths remain available for all four.
CO-SPY/Effort have no public raw wrapper; DINOv2's original source pin is unresolved.

Motivation and PPR-only code are unchanged. Host score fusion is not uniform:
CLIP/DINOv2/CO-SPY keep source-only five-fold OOF LR; Effort keeps additive
full-fit PPR plus source-VAL alpha and maximum-BA threshold. Native comparison
is separate reporting, retaining signed negative gains and original operating
points. No formal model refit, formal result edit or data split modification.

Final paper/MSM-lineage decisions, project license, data/model rights, exact
public assets and missing raw implementations remain [typed blockers](release-blockers.md).
No final manuscript was supplied, so the existing [author checklist](paper-code-consistency.md)
is not a completed paper sign-off. No publication/staging/commit/tag/history rewrite
is authorized or performed. See [verification](verification.md) and
[final safety evidence](evidence/four-host-final-checks.json).
