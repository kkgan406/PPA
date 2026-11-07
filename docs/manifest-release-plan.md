# Exact Manifest Release Plan — Phase 3

**Readiness: local portable drafts validated; not approved or released.** The
full CSV drafts and local conversion/checking script remain outside the candidate
in the private audit bundle. No images, features, model files or full dataset
manifests are added to the candidate. Public artifacts here are a schema/policy
and a checksum/readiness catalog only. This is a concrete reviewed draft rather
than a proposal to regenerate an experimental split.

## Draft package and invariants

Eight local draft manifests were generated from the exact archived manifests:
source_train12398, source_val800, genimage6/motivation75991, genimage_extra24000,
genimage8=99991, commgen15=66785 and combined experiment179974. GenImage8 is an
ordered concatenation of old six plus the two added domains. Motivation is the
unchanged old-six membership; combined experiment does not repartition any data.
All frozen IDs, ordered observations, original split strings, label assignments,
generator names and image content hashes match their originals exactly.

Each release CSV has:

| Field | Preservation / meaning |
| --- | --- |
| sample_id | Original frozen24-hex observation ID, immutable; never regenerated |
| path | Relative path only, e.g. `../data/genimage/ADM/0_real/image.jpg`; sole relocated input field |
| sha256 | Original full64-hex image-file byte hash |
| label | Original0=real,1=fake |
| generator | Original domain spelling/case within dataset namespace |
| source_split | Exact original split string (`train`, `val`, `genimage`, `fresh_ood`, etc.) |
| split | Loader-compatible metadata mapping only: source train/val unchanged; historical target names map to test; no observation reassignment |
| dataset | source / genimage / commgen15 namespace |
| dataset_version | `archived-manifest-sha256:<original CSV byte hash>`; identifies a real frozen local snapshot, not an invented upstream release tag |
| source_manifest_sha256 | Full original CSV byte hash for the row's original fragment |

The package `manifest-index.json` records each relative CSV filename, exact
UTF-8 CSV-byte SHA256, count, source manifest checksum(s), ordered observation
binding SHA256, original-ID/split preservation and publication status. The
binding is SHA256 of canonical JSON records over
`sample_id,source_split,generator,label,sha256`, preserving row order. It excludes
the relocated path but not label/domain/split. Every binding is equal before and
after relocation. A portable CSV's file hash differs from an absolute-path CSV's
hash and is never represented as identical.

All upstream dataset versions in the draft index are explicitly **null / requires
author confirmation**. `dataset_version` supplies the known archive snapshot
identity without inventing an official GenImage/CommGen15 release revision.
The owner must add the actual upstream version/source/terms before publication.

## Relative image layout (no files copied by this task)

```text
bundle/
  manifests/source_train.csv, source_val.csv, motivation.csv,
            genimage6.csv, genimage_extra.csv, genimage8.csv,
            commgen15.csv, experiment.csv
  manifest-index.json
  data/source/train/{0_real,1_fake}/...   # user obtains licensed images
  data/source/val/{0_real,1_fake}/...
  data/genimage/<original-domain>/{0_real,1_fake}/...
  data/commgen15/<original-domain>/{0_real,1_fake}/...
```

The runtime accepts the extra provenance columns without changing image
processing. Original extension IDs were SHA256 of the original absolute-path
string. **Copy those frozen IDs verbatim**; never recompute them from the new
relative path. Keyed noise uses the preserved sample_id, not the new filename.
The source-label/group mapping and repeated target observations are retained.

Old caches used historical split strings; migrate their binding metadata only
through the same checked mapping, without changing any feature/logit byte or row
association. Record old/new cache checksums separately. The candidate's cache
consumer will reject an unconverted old `genimage` split against a new `test`
manifest rather than silently assuming alignment. No existing cache is overwritten
or converted in place by this task.

## Three evaluation scopes — never interchangeable

| Scope | Definition | Interpretation |
| --- | --- | --- |
| Historical Macro-8 | Equal mean of eight original GenImage generator metrics, N99991 | Includes SD1.4 source family and11999 TRAIN-overlap observations; not strict all-OOD |
| Macro-7-unseen | Exclude the entire12000-observation SD1.4 domain; equal mean over seven domains, N87991 | Zero overlap with current source TRAIN/VAL; unknown external host pretraining overlap is separate |
| Content-disjoint | Remove each observation whose image hash occurs in source TRAIN/VAL, then report estimable groups/pooled ranges explicitly | Not equivalent to removing a whole domain; can leave a single-class SD1.4 group, whose AUC is unavailable |

No formal metric, threshold, OOD selection or original split changed. CommGen15
retains66785 observations, including original repeated content/group associations.
Existing per-generator macro/pooled semantics remain as audited.

## Publication gates and owner review

1. **Resolved:** local draft membership/row-order/ID/hash/label/domain/source-split
   preservation and exact draft checksums; no path-derived ID regeneration.
2. **Requires author decision:** approve this layout, source-to-public name mapping,
   final manuscript scope, actual upstream versions and final manifest package.
3. **Requires external permission:** confirm dataset/image-list/metadata publication
   terms as applicable. Data access and permission to publish lists are distinct.
4. **Technically unresolved:** publicly obtainable approved exact manifests, upstream
   version identifiers, and final cache-lineage binding for the added GenImage
   domains. Local CSV validation does not resolve these release questions.

The [draft checksum catalog](evidence/phase3-manifest-readiness.json) identifies
what was actually generated and checked. No full manifest is publicly bundled
pending these gates. [Schema/policy](../configs/manifest_release.yaml) is not a
substitute for the original frozen data or an official dataset download link.
