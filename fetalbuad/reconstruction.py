"""Train the atlas-conditioned 3D denoising autoencoder on normal examinations."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import read_volume
from .reconstruction_model import DenoisingAE3D, IMAGE_SHAPE, atlas_week, load_atlas, read_prepared_image


class ReconstructionDataset(Dataset):
    def __init__(self, manifest):
        manifest = Path(manifest)
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"image", "segmentation", "age_weeks"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{manifest} requires columns image, segmentation, age_weeks.")
            self.records = [
                (manifest.parent / row["image"], manifest.parent / row["segmentation"],
                 atlas_week(row["age_weeks"]))
                for row in reader
            ]
        if not self.records:
            raise ValueError(f"No examinations in {manifest}.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, segmentation_path, week = self.records[index]
        image = read_prepared_image(image_path, training=True)
        labels = read_volume(segmentation_path)
        if (labels.shape != IMAGE_SHAPE or not np.isfinite(labels).all()
                or labels.min() < 0 or labels.max() > 8 or not np.any(labels > 0)
                or not np.array_equal(labels, labels.astype(np.uint8))):
            raise ValueError(f"Expected aligned tissue labels from 0 to 8 in {segmentation_path}.")
        return image[None], week, np.ascontiguousarray(labels, dtype=np.uint8)


def corrupt_image(image, warp_rng, noise_rng):
    """Warp the MRI and add spatially correlated noise inside its positive foreground."""
    shape = image.shape[-3:]
    batch = len(image)
    offsets = torch.randn((batch, 3, 4, 4, 4), generator=warp_rng).to(image.device)
    offsets = F.interpolate(offsets, size=shape, mode="trilinear", align_corners=False)
    axes = [torch.linspace(-1 + 1/n, 1 - 1/n, n, device=image.device) for n in shape]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij")[::-1], dim=-1)
    scale = image.new_tensor([2 * 1.5 / n for n in reversed(shape)])
    warped = F.grid_sample(image, grid + offsets.movedim(1, -1) * scale, align_corners=False)
    noise = torch.randn((batch, 1, 16, 16, 16), generator=noise_rng).to(image.device)
    noise = F.interpolate(noise, size=shape, mode="trilinear", align_corners=False)
    shifts = [int(torch.randint(max(1, n // 16), (), generator=noise_rng)) for n in shape]
    noise = torch.roll(noise, shifts, dims=(-3, -2, -1))
    return warped + 0.2 * noise * (warped > 0)


def reconstruction_losses(prediction, target, logits, labels):
    foreground = target > 0
    error = ((prediction.float() - target).square() * foreground).flatten(1).sum(1)
    mse = error / foreground.flatten(1).sum(1).clamp_min(1)
    probs = logits.float().softmax(dim=1).flatten(2)
    tissues = torch.zeros_like(probs).scatter_(1, labels.flatten(1)[:, None].long(), 1)
    intersection = (probs * tissues).sum(-1)
    dice = (2 * intersection + 1e-6) / (probs.sum(-1) + tissues.sum(-1) + 1e-6)
    return mse, 1 - dice[:, 1:].mean(1)


def run_epoch(model, loader, atlases, device, warp_rng, noise_rng, optimizer=None):
    model.train(optimizer is not None)
    errors = []
    dice_losses = []
    with torch.set_grad_enabled(optimizer is not None):
        for image, weeks, labels in loader:
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            atlas = torch.stack([atlases[week] for week in weeks.tolist()])
            noisy_image = corrupt_image(image, warp_rng, noise_rng)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction, logits = model(noisy_image, atlas, return_segmentation=True)
                mse, dice = reconstruction_losses(prediction, image, logits, labels)
                loss = mse.mean() + 0.01 * dice.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite reconstruction loss.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            errors.extend(mse.detach().tolist())
            dice_losses.extend(dice.detach().tolist())
    return float(np.mean(errors)), float(np.mean(dice_losses))


def train(args):
    device = torch.device(args.device)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA training uses bfloat16. Choose a compatible GPU or use --device cpu.")
    torch.backends.cudnn.allow_tf32 = True
    train_data = ReconstructionDataset(args.train)
    val_data = ReconstructionDataset(args.val)
    train_images = {image.resolve() for image, _, _ in train_data.records}
    if any(image.resolve() in train_images for image, _, _ in val_data.records):
        raise ValueError("Training and validation manifests share an image.")
    weeks = {week for dataset in (train_data, val_data) for _, _, week in dataset.records}
    atlases = {week: torch.from_numpy(load_atlas(args.atlas_dir, week)).to(device)[None] for week in weeks}
    loader_args = dict(batch_size=args.batch_size, pin_memory=device.type == "cuda")
    shuffle_rng = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=shuffle_rng, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    torch.manual_seed(args.seed)
    model = DenoisingAE3D().to(device=device, memory_format=torch.channels_last_3d)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    noise_rng = torch.Generator().manual_seed(args.seed)
    warp_rng = torch.Generator().manual_seed(args.seed + 2)
    best_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        mse, dice = run_epoch(model, train_loader, atlases, device, warp_rng, noise_rng, optimizer)
        print(f"Epoch {epoch:3d}  training MSE {mse:.6f}  tissue Dice loss {dice:.6f}", flush=True)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            val_warp_rng = torch.Generator().manual_seed(args.seed + 3)
            val_noise_rng = torch.Generator().manual_seed(args.seed + 1)
            mse, _ = run_epoch(model, val_loader, atlases, device, val_warp_rng, val_noise_rng)
            print(f"           validation MSE {mse:.6f}", flush=True)
            if mse < best_mse:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temp = args.output.with_suffix(".tmp.pt")
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_mse": mse}, temp)
                temp.replace(args.output)
                best_mse = mse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="CSV of normal training examinations.")
    parser.add_argument("--val", type=Path, required=True, help="CSV of normal validation examinations.")
    parser.add_argument("--atlas-dir", type=Path, required=True, help="Directory of normalized weekly atlases.")
    parser.add_argument("--output", type=Path, required=True, help="Path for the best validation checkpoint.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
