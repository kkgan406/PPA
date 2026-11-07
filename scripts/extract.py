"""Extract JPEG PPR and optionally an original frozen pretrained host."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Manifest CSV; paths relative to manifest")
    p.add_argument("--output", required=True, help="New NPZ cache path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--host", choices=["CLIP", "DINOv2"])
    p.add_argument("--host-config", help="Local YAML with checkpoint paths/hashes and frozen native head")
    args = p.parse_args()
    import numpy as np
    import torch
    import yaml
    from ppa.data import read_manifest, save_cache, ppr_metadata, environment
    from ppa.preprocessing import load_rgb, canonicalize, perturb, tensor01
    from ppa.response import FrozenAlex, extract_pair, file_sha256
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Preserve the existing cache; use a new path")
    if bool(args.host) != bool(args.host_config):
        p.error("--host and --host-config are required together")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    f = read_manifest(args.dataset)
    model = FrozenAlex(args.device)
    host = None
    if args.host:
        from ppa.hosts import FrozenHost
        spec = yaml.safe_load(Path(args.host_config).read_text())
        host = FrozenHost(args.host, spec, args.device)
    values, features, logits = [], [], []
    for i, row in enumerate(f.itertuples()):
        base = canonicalize(load_rgb(row.path, row.sha256))
        _, descriptor, _ = extract_pair(model, tensor01(base), perturb(base, "jpeg", row.sample_id))
        values.append(descriptor.cpu().numpy()[0])
        if host is not None:
            h, s = host.extract(base)
            features.append(h)
            logits.append(s)
        if (i+1) % 100 == 0 or i+1 == len(f):
            print(f"Extracted {i+1}/{len(f)}", flush=True)
    extra = dict(manifest_sha256=file_sha256(args.dataset), environment=environment(),
                 device=args.device, batch="one same-canvas pair; no padding or resize of features")
    if host is not None:
        extra.update(host=args.host, host_frozen=True, host_provenance={k:spec[k] for k in
            ["checkpoint_sha256", "code_commit", "feature_layer", "normalization", "preprocessing", "native_logit",
             "native_head_sha256"]})
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_cache(destination, f, np.stack(values), ppr_metadata(**extra),
               np.stack(features) if host is not None else None,
               np.asarray(logits) if host is not None else None)


if __name__ == "__main__":
    main()
