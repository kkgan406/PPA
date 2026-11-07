"""Raw post-ReLU LPIPS AlexNet activation response, without learned distance."""
from pathlib import Path
import hashlib

import torch

LAYER_CHANNELS = (64, 192, 384, 256, 256)
ORDER = ("Mean_L4", "Mean_L5", "Std_L4", "Std_L5", "Max_L4", "Max_L5")
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
LPIPS_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class FrozenAlex(torch.nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        import lpips
        backbone = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
        learned = Path(lpips.__file__).parent / "weights/v0.1/alex.pth"
        for path, expected in [(backbone, ALEXNET_SHA256), (learned, LPIPS_SHA256)]:
            if not path.is_file():
                raise FileNotFoundError(f"Install the documented official checkpoint first: {path.name}")
            if file_sha256(path) != expected:
                raise ValueError(f"Unexpected checkpoint: {path.name}")
        self.model = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval()
        self.checkpoints = {backbone.name: ALEXNET_SHA256, "lpips-v0.1-alex.pth": LPIPS_SHA256}
        self.requires_grad_(False)
        self.eval().to(device)

    def train(self, mode: bool = True):
        # Keep the feature extractor frozen even if a caller toggles its parent.
        return super().train(False)

    @torch.inference_mode()
    def forward(self, image01: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if image01.dtype != torch.float32 or image01.ndim != 4 or image01.shape[1] != 3:
            raise ValueError("Expected NCHW RGB float32")
        if not torch.isfinite(image01).all() or image01.min() < 0 or image01.max() > 1:
            raise ValueError("Expected finite pixels in [0,1]")
        if min(image01.shape[-2:]) < 63:
            raise ValueError("AlexNet requires minimum side >=63; no silent resizing")
        x = image01.to(next(self.parameters()).device) * 2 - 1
        return tuple(self.model.net(self.model.scaling_layer(x)))


def response_maps(original, transformed) -> tuple[torch.Tensor, ...]:
    if len(original) != 5 or len(transformed) != 5:
        raise ValueError("Expected five ordered AlexNet ReLU outputs")
    maps = []
    for a, b, channels in zip(original, transformed, LAYER_CHANNELS):
        if a.shape != b.shape or a.ndim != 4 or a.shape[1] != channels:
            raise ValueError("Layer shape/order mismatch; no pooling, padding or resizing")
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError("Feature extraction must use float32")
        r = (a - b).abs()
        if not torch.isfinite(r).all():
            raise ValueError("Non-finite response")
        maps.append(r)
    return tuple(maps)


def global_response(maps) -> torch.Tensor:
    """Equal mean over five layer means, each over C,H,W. Shape [N]."""
    return torch.stack([r.mean((-2, -1)).mean(1) for r in maps], dim=1).mean(1)


def structured_descriptor(maps) -> torch.Tensor:
    """L4/L5 raw absolute response; population std; stat-major concatenation."""
    if len(maps) != 5 or maps[3].shape[1] != 256 or maps[4].shape[1] != 256:
        raise ValueError("Expected AlexNet L4/L5 with 256 channels each")
    late = maps[3:5]
    values = torch.cat([torch.cat([op(r) for r in late], dim=1) for op in (
        lambda r: r.mean((-2, -1)), lambda r: r.std((-2, -1), correction=0),
        lambda r: r.amax((-2, -1)))], dim=1)
    if values.shape[1] != 1536 or not torch.isfinite(values).all():
        raise ValueError("Invalid structured PPR")
    return values


@torch.inference_mode()
def extract_pair(model, base01: torch.Tensor, transformed01: torch.Tensor):
    if base01.shape != transformed01.shape:
        raise ValueError("Pair geometry differs")
    n = len(base01)
    levels = model(torch.cat([base01, transformed01], dim=0))
    maps = response_maps([x[:n] for x in levels], [x[n:] for x in levels])
    return global_response(maps), structured_descriptor(maps), maps
