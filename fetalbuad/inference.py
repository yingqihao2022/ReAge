"""Compute the brain-age, reconstruction and fused maps for one examination."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .age_model import load_age_model
from .data import read_volume
from .reconstruction_model import load_atlas, load_reconstruction, read_prepared_image


def fuse_maps(original_age, reconstructed_age, residual, foreground, brain):
    """Center the paired age change within the MRI foreground and fold its two signs."""
    change = reconstructed_age - original_age
    centered = (change - change[foreground].mean()) * foreground
    magnitude = centered.abs()
    scale = magnitude.sum() / foreground.sum()
    upper = 0.5 * (1 + centered / (magnitude + scale).clamp_min(1e-8))
    age_map = torch.maximum(upper, 1 - upper) * brain
    # Take the larger of the two complementary, residual-scaled age responses.
    positive = (residual * upper).double()
    fused = torch.maximum(positive, residual.double() - positive)
    return {"H_age": age_map, "H_rec": residual, "H_final": fused}


@torch.inference_mode()
def predict(image, atlas, brain, foreground, recon_model, age_model):
    """Predict a single prepared examination. Tensors have shape (1, 1, X, Y, Z)."""
    if image.ndim != 5 or image.shape[:2] != (1, 1):
        raise ValueError("Supply one three-dimensional, single-channel examination.")
    if any(t.shape != image.shape for t in (atlas, brain, foreground)):
        raise ValueError("The image, atlas and masks must share a voxel grid.")
    if brain.dtype != torch.bool or foreground.dtype != torch.bool:
        raise ValueError("Brain and foreground masks must be Boolean tensors.")
    if not foreground.any() or (foreground & ~brain).any():
        raise ValueError("The foreground must be a nonempty subset of the brain mask.")
    values = image[foreground]
    mean = values.mean()
    std = values.std(correction=0).clamp_min(1e-8)
    original = torch.where(foreground, (image - mean) / std, 0)
    original_age = age_model(original)[2]

    with torch.autocast(device_type=image.device.type, dtype=torch.bfloat16,
                        enabled=image.is_cuda and torch.cuda.is_bf16_supported()):
        reconstructed = recon_model(image, atlas).float().clamp(0, 1)
    residual = (image - reconstructed).abs() * brain
    normalized = torch.where(foreground, (reconstructed - mean) / std, 0)
    reconstructed_age = age_model(normalized)[2]
    maps = fuse_maps(original_age, reconstructed_age, residual, foreground, brain)
    score = maps["H_final"][brain].mean().item()
    if not np.isfinite(score):
        raise ValueError("The models produced a non-finite score.")
    return maps, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Prepared MRI on the atlas grid")
    parser.add_argument("--mask", type=Path, required=True, help="Whole-brain scoring mask")
    parser.add_argument("--foreground", type=Path, help="Aligned MRI foreground mask for voxel-age prediction")
    parser.add_argument("--age-weeks", type=float, required=True)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--recon-checkpoint", type=Path, required=True)
    parser.add_argument("--age-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    image = read_prepared_image(args.image)
    atlas = load_atlas(args.atlas_dir, args.age_weeks)
    brain = read_volume(args.mask) > 0.5
    foreground = (read_volume(args.foreground) > 0.5) if args.foreground else brain & (image > 0)
    # Keep native voxel order. Use the physical frame, when supplied, for export.
    nifti_paths = [p for p in (args.image, args.mask, args.foreground) if p is not None and p.suffix != ".npy"]
    affine = None
    for path in nifti_paths:
        transform = nib.load(path).affine
        if affine is not None and not np.allclose(affine, transform, rtol=1e-5, atol=1e-4):
            raise ValueError("Image and masks must be prepared on the same physical grid.")
        affine = transform
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True
    recon = load_reconstruction(args.recon_checkpoint, device)
    age = load_age_model(args.age_checkpoint, device)
    inputs = [torch.from_numpy(np.ascontiguousarray(a))[None, None].to(device)
              for a in (image, atlas, brain, foreground)]
    maps, score = predict(*inputs, recon, age)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in maps.items():
        array = value[0, 0].cpu().numpy()
        if affine is None:
            np.save(args.output / f"{name}.npy", array, allow_pickle=False)
        else:
            nib.save(nib.Nifti1Image(array, affine), args.output / f"{name}.nii.gz")
    result = json.dumps({"score": score}, allow_nan=False)
    (args.output / "score.json").write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
