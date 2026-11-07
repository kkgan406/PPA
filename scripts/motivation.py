"""Training-free raw PPR ROC-AUC and layer-wise response maps."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Single-split manifest CSV; frozen independent GenImage6 subset for paper")
    p.add_argument("--perturbation", choices=["jpeg", "blur", "noise"], required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--visualize", type=int, default=0, help="First N image IDs per domain/class; save raw maps and PNG")
    args = p.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from ppa.data import read_manifest, write_json, environment, new_output
    from ppa.preprocessing import canonicalize, load_rgb, tensor01, perturb, PROTOCOL
    from ppa.response import FrozenAlex, extract_pair, file_sha256
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    f = read_manifest(args.dataset)
    if f.split.nunique() != 1 or f.dataset.nunique() != 1:
        raise ValueError("Motivation requires one explicitly identified dataset/split")
    if args.visualize < 0:
        p.error("--visualize must be nonnegative")
    out = new_output(args.output_dir)
    model = FrozenAlex(args.device)
    config = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/ppr.yaml").read_text())
    direction = config["motivation_orientation"][args.perturbation]
    chosen = set(f.sort_values("sample_id").groupby(["generator", "label"], sort=False).head(args.visualize).sample_id)
    rows = []
    for i, row in enumerate(f.itertuples()):
        base = canonicalize(load_rgb(row.path, row.sha256))
        scalar, _, maps = extract_pair(model, tensor01(base), perturb(base, args.perturbation, row.sample_id))
        result = {k:getattr(row,k) for k in ["sample_id", "sha256", "label", "generator", "dataset", "split"]}
        result.update(global_response=float(scalar[0]), score=direction*float(scalar[0]), orientation=direction)
        result.update({f"layer_{j+1}":float(m.mean((-2,-1)).mean(1)[0]) for j,m in enumerate(maps)})
        rows.append(result)
        if row.sample_id in chosen:
            # Filenames never interpolate user-provided IDs.
            stem = __import__("hashlib").sha256(row.sample_id.encode()).hexdigest()[:24]
            np.savez_compressed(out/f"maps_{stem}.npz", **{f"layer_{j+1}":m.cpu().numpy()[0] for j,m in enumerate(maps)})
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 6, figsize=(15, 3))
            axes[0].imshow(base)
            axes[0].set_title(f"label={row.label}")
            for j,m in enumerate(maps):
                spatial = m.mean(1, keepdim=True)
                display = torch.nn.functional.interpolate(spatial, size=(base.height,base.width), mode="bilinear", align_corners=False)
                axes[j+1].imshow(display.cpu().numpy()[0,0], cmap="magma", vmin=0)
                axes[j+1].set_title(f"L{j+1} (own scale)")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out/f"maps_{stem}.png", dpi=150)
            plt.close(fig)
        if (i+1) % 100 == 0 or i+1 == len(f):
            print(f"Responses {i+1}/{len(f)}", flush=True)
    responses = pd.DataFrame(rows)
    responses.to_csv(out/"responses.csv", index=False, float_format="%.17g")
    metrics = []
    for generator, group in responses.groupby("generator", sort=False):
        metrics.append(dict(scope=generator, n=len(group),
            auc=float(roc_auc_score(group.label,group.score)),
            raw_auc=float(roc_auc_score(group.label,group.global_response))))
    metrics.append(dict(scope=f"Macro-{len(metrics)}", n=len(f),
        auc=float(np.mean([r["auc"] for r in metrics])), raw_auc=float(np.mean([r["raw_auc"] for r in metrics]))))
    metrics.append(dict(scope="pooled", n=len(f), auc=float(roc_auc_score(responses.label,responses.score)),
                        raw_auc=float(roc_auc_score(responses.label,responses.global_response))))
    pd.DataFrame(metrics).to_csv(out/"metrics.csv", index=False, float_format="%.17g")
    write_json(out/"metadata.json", dict(dataset=str(f.dataset.iloc[0]), split=str(f.split.iloc[0]), n=len(f),
        perturbation=args.perturbation, preprocessing=PROTOCOL, manifest_sha256=file_sha256(args.dataset),
        checkpoints=model.checkpoints, seed=42, orientation=direction,
        orientation_source=config["orientation_source"], positive_label=1, score="orientation * raw global response",
        scalar="equal mean of 5 raw post-ReLU absolute response layer means; no learned LPIPS distance",
        aggregation="equal-generator macro plus pooled; no per-target sign selection",
        environment=environment(), device=args.device,
        visualization="raw C,H,W NPZ; channel mean bilinear align_corners=False for display; each layer own color scale",
        responses_sha256=file_sha256(out/"responses.csv"), code_sha256=file_sha256(__file__)))
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
