"""Image protocol retained from the research canonical/perturbation helpers."""
from io import BytesIO
import hashlib
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torchvision.transforms.functional import gaussian_blur, pil_to_tensor

SEED = 42
PROTOCOL = {
    "color": "EXIF transpose then RGB", "resize": "bicubic long512 then isotropic min64",
    "rounding": "Python round, each dimension >=1 before min64",
    "base": {"format": "JPEG", "quality": 90, "subsampling": 2,
             "optimize": False, "progressive": False},
    "jpeg": {"quality": 50, "subsampling": 2},
    "blur": {"kernel_size": 5, "sigma": 1.0},
    "noise": {"sigma": 0.03, "seed": SEED,
              "rng": "CPU torch.Generator; sha256(seed:sample_id)[:8] little endian mod(2**63-1)"},
}


def load_rgb(path: str | Path, expected_sha256: str) -> Image.Image:
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Image content SHA256 mismatch")
    with Image.open(BytesIO(content)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def ar_resize(image: Image.Image, long_side: int = 512) -> Image.Image:
    scale = long_side / max(image.size)
    return image.resize(tuple(max(1, round(d * scale)) for d in image.size), Image.Resampling.BICUBIC)


def min_size(image: Image.Image, minimum: int = 64) -> Image.Image:
    if min(image.size) < minimum:
        scale = minimum / min(image.size)
        return image.resize(tuple(round(d * scale) for d in image.size), Image.Resampling.BICUBIC)
    return image


def codec_roundtrip(image: Image.Image, **parameters) -> Image.Image:
    stream = BytesIO()
    image.save(stream, **parameters)
    stream.seek(0)
    with Image.open(stream) as decoded:
        return decoded.convert("RGB")


def canonicalize(image: Image.Image) -> Image.Image:
    """Long side can exceed 512 after the inherited min-short-side safeguard."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    return codec_roundtrip(min_size(ar_resize(rgb)), **PROTOCOL["base"])


def tensor01(image: Image.Image) -> torch.Tensor:
    return pil_to_tensor(image).to(torch.float32).div(255).unsqueeze(0)


def perturb(image: Image.Image, name: str, sample_id: str, seed: int = SEED) -> torch.Tensor:
    """Perturb the decoded canonical base, not the original file."""
    x = tensor01(image)
    if name == "jpeg":
        return tensor01(codec_roundtrip(image, format="JPEG", **PROTOCOL["jpeg"]))
    if name == "blur":
        p = PROTOCOL["blur"]
        return gaussian_blur(x, [p["kernel_size"]] * 2, [p["sigma"]] * 2).clamp(0, 1)
    if name == "noise":
        key = f"{seed}:{sample_id}".encode()
        key_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(key_seed)
        return (x + torch.randn(x.shape, generator=generator) * PROTOCOL["noise"]["sigma"]).clamp(0, 1)
    if name == "identity":
        return x
    raise ValueError(f"Unknown perturbation: {name}")
