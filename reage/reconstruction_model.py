"""Atlas-conditioned volume reconstruction and gestational-week atlas selection."""

from pathlib import Path
import math

import numpy as np
import torch
from torch import nn

from .data import read_volume


IMAGE_SHAPE = (128, 160, 128)


def _block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.SiLU(),
    )


class DenoisingAE3D(nn.Module):
    """Whole-volume encoder–decoder with two pooling stages and a nine-class tissue head."""

    def __init__(self):
        super().__init__()
        self.enc1 = _block(2, 8)
        self.enc2 = _block(8, 16)
        self.bottom = _block(16, 32)
        self.pool = nn.AvgPool3d(2)
        self.up2 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.up1 = nn.ConvTranspose3d(16, 8, 2, stride=2)
        self.dec2 = _block(32, 16)
        self.dec1 = _block(16, 8)
        self.output = nn.Conv3d(8, 1, 1)
        self.segmentation_head = nn.Conv3d(8, 9, 1)

    def forward(self, image, atlas, return_segmentation=False):
        if image.ndim != 5 or image.shape != atlas.shape or image.shape[1] != 1:
            raise ValueError("Expected image and atlas tensors with matching B×1×D×H×W dimensions.")
        x = torch.cat((image, atlas), dim=1).contiguous(memory_format=torch.channels_last_3d)
        shallow = self.enc1(x)
        deep = self.enc2(self.pool(shallow))
        x = self.bottom(self.pool(deep))
        x = self.dec2(torch.cat((self.up2(x), deep), dim=1))
        x = self.dec1(torch.cat((self.up1(x), shallow), dim=1))
        reconstruction = self.output(x)
        if return_segmentation:
            return reconstruction, self.segmentation_head(x)
        return reconstruction


def load_reconstruction(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = DenoisingAE3D()
    model.load_state_dict(payload.get("model", payload), strict=True)
    return model.to(device=device, memory_format=torch.channels_last_3d).eval().requires_grad_(False)


def read_prepared_image(path, *, training=False):
    """Load an MRI or atlas after positive-foreground P1/P99 scaling to [0, 1]."""
    image = np.asarray(read_volume(path), dtype=np.float32)
    if image.shape != IMAGE_SHAPE or not np.isfinite(image).all():
        raise ValueError(f"{path} must contain finite values on the {IMAGE_SHAPE} grid.")
    if image.min() < -1e-6 or image.max() > 1 + 1e-6 or not np.any(image > 0):
        raise ValueError(f"Expected a nonempty, normalized image in {path}.")
    if training:
        image = image.astype(np.float16).astype(np.float32)
    return np.ascontiguousarray(image)


def atlas_week(age_weeks):
    age = float(age_weeks)
    if not math.isfinite(age) or age <= 0:
        raise ValueError("Gestational age must be a positive number of weeks.")
    return min(38, max(21, math.floor(age + 0.5)))


def load_atlas(atlas_dir, age_weeks):
    """Read the nearest prepared weekly atlas, propagating the endpoint atlas beyond its range."""
    week = atlas_week(age_weeks)
    paths = [Path(atlas_dir) / f"week_{week}{suffix}" for suffix in (".nii.gz", ".nii", ".npy")]
    available = [path for path in paths if path.is_file()]
    if len(available) != 1:
        raise ValueError(f"Expected one atlas named week_{week} (.nii.gz, .nii, or .npy) in {atlas_dir}.")
    return read_prepared_image(available[0], training=True)
