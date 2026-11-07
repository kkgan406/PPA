"""Original frozen pretrained-host adapters; detector outputs use the cache interface.

Adapted from src/stage_b/backbones.py. No host training or checkpoint substitution.
"""
import subprocess
import joblib
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from .response import file_sha256


def preprocess(name):
    if name == "CLIP":
        geometry = [T.Resize(224, interpolation=InterpolationMode.BICUBIC), T.CenterCrop(224)]
        mean, std = [.48145466,.4578275,.40821073], [.26862954,.26130258,.27577711]
    elif name == "DINOv2":
        geometry = [T.Resize(256, interpolation=InterpolationMode.BICUBIC), T.CenterCrop(224)]
        mean, std = [.485,.456,.406], [.229,.224,.225]
    else:
        raise ValueError("Raw adapter supports CLIP/DINOv2; use audited detector caches otherwise")
    return T.Compose([*geometry, T.ToTensor(), T.Normalize(mean, std)])


class FrozenHost(torch.nn.Module):
    def __init__(self, name, spec, device="cpu"):
        super().__init__()
        if name not in ("CLIP", "DINOv2"):
            raise ValueError("Public raw adapter supports CLIP/DINOv2 only")
        self.name = name
        if file_sha256(spec["weights"]) != spec["checkpoint_sha256"]:
            raise ValueError("Host checkpoint SHA256 mismatch")
        if name == "CLIP":
            from transformers import CLIPVisionModelWithProjection
            self.encoder = CLIPVisionModelWithProjection.from_pretrained(spec["model_dir"], local_files_only=True).float()
        elif name == "DINOv2":
            commit = subprocess.check_output(["git", "-C", spec["code_root"], "rev-parse", "HEAD"], text=True).strip()
            if commit != spec["code_commit"]:
                raise ValueError("Host code commit mismatch")
            if subprocess.check_output(["git", "-C", spec["code_root"], "diff", "HEAD", "--name-only"], text=True).strip():
                raise ValueError("Official host tracked source has modifications")
            self.encoder = torch.hub.load(spec["code_root"], "dinov2_vitb14", source="local", pretrained=False)
            self.encoder.load_state_dict(torch.load(spec["weights"], map_location="cpu", weights_only=True), strict=True)
        else:
            raise ValueError("No raw adapter for this host; see docs/hosts.md")
        self.requires_grad_(False)
        self.eval().to(device)
        if file_sha256(spec["native_head"]) != spec["native_head_sha256"]:
            raise ValueError("Frozen native head SHA256 mismatch")
        self.native_head = joblib.load(spec["native_head"])
        if self.native_head.n_features_in_ != 768:
            raise ValueError("Expected original 768D native host head")

    def train(self, mode=True):
        return super().train(False)

    @torch.inference_mode()
    def forward(self, x):
        if self.name == "CLIP":
            value = self.encoder(pixel_values=x).image_embeds
        elif self.name == "DINOv2":
            return self.encoder.forward_features(x)["x_norm_clstoken"]
        return torch.nn.functional.normalize(value, dim=-1)

    @torch.inference_mode()
    def extract(self, canonical):
        x = preprocess(self.name)(canonical).unsqueeze(0).to(next(self.parameters()).device)
        features = self(x).cpu().numpy()
        logits = self.native_head.decision_function(np.asarray(features, dtype=np.float64))
        return features[0], float(logits[0])
