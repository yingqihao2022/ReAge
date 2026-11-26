"""Train a voxel-wise brain-age predictor on healthy fetal MRI."""

import argparse
import csv
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .age_model import VoxelAgeNet
from .data import read_volume


class AgeVolumes(Dataset):
    """Prepared MRI, nine-class tissue labels and gestational age in weeks."""

    def __init__(self, manifest):
        manifest = Path(manifest)
        self.cases = []
        self.images = set()
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            fields = {"image", "segmentation", "age_weeks"}
            if not fields.issubset(rows.fieldnames or ()):
                raise ValueError(f"{manifest} requires columns {sorted(fields)}")
            for row in rows:
                image = (manifest.parent / row["image"]).resolve()
                tissue = manifest.parent / row["segmentation"]
                days = 7 * float(row["age_weeks"])
                if not np.isfinite(days) or days <= 0:
                    raise ValueError("Gestational age must be finite and positive.")
                if image in self.images:
                    raise ValueError(f"Repeated training image: {image}")
                self.cases.append((image, tissue, days))
                self.images.add(image)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        image_path, tissue_path, days = self.cases[index]
        image = np.asarray(read_volume(image_path), dtype=np.float64)
        tissue = read_volume(tissue_path)
        if image.shape != (128, 160, 128) or tissue.shape != image.shape:
            raise ValueError("MRI and tissue labels must share the 128 × 160 × 128 grid.")
        if not np.array_equal(tissue, np.rint(tissue)) or tissue.min() < 0 or tissue.max() > 8:
            raise ValueError("Tissue labels must be integers in the range 0–8.")
        if random.random() > 0.5:
            axis = random.choice((0, 1))
            image, tissue = np.flip(image, axis), np.flip(tissue, axis)
        foreground = image > 0
        values = image[foreground]
        if not values.size:
            raise ValueError("MRI has no positive foreground.")
        std = values.std()
        if std == 0:
            raise ValueError("MRI foreground must contain more than one intensity.")
        normalized = np.zeros(image.shape, dtype=np.float32)
        normalized[foreground] = (values - values.mean()) / std
        return (
            torch.from_numpy(normalized[None]),
            torch.tensor([days], dtype=torch.float32),
            torch.from_numpy(np.ascontiguousarray(tissue, dtype=np.int64)),
            torch.from_numpy(np.ascontiguousarray(foreground[None])),
        )


def tissue_dice_loss(logits, labels):
    probability = logits.sigmoid()
    target = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
    axes = (0, 2, 3, 4)
    overlap = (probability * target).sum(dim=axes)
    denominator = probability.square().sum(dim=axes) + target.sum(dim=axes)
    return 1 - (2 * overlap / denominator.clamp_min(1e-6)).mean()


def age_objective(predictions, age_days, tissue, foreground):
    """Global age MAE, nine-class tissue Dice and full-volume voxel-age MAE."""
    global_age, tissue_logits, voxel_age = predictions
    jitter_days = torch.rand_like(voxel_age) * 4 - 2
    voxel_target = (age_days.view(-1, 1, 1, 1, 1) + jitter_days) * foreground
    return (
        F.l1_loss(global_age, age_days)
        + 1000 * tissue_dice_loss(tissue_logits, tissue)
        + F.l1_loss(voxel_age, voxel_target)
    )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            image, age, tissue, foreground = (part.to(device, non_blocking=True) for part in batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = age_objective(model(image), age, tissue, foreground)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite brain-age objective.")
            if training:
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Training CSV")
    parser.add_argument("--val", type=Path, required=True, help="Validation CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint file")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Epochs and batch size must be positive; workers must be nonnegative.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train, val = AgeVolumes(args.train), AgeVolumes(args.val)
    if train.images & val.images:
        raise ValueError("Training and validation contain overlapping images.")
    if min(len(train), len(val)) < args.batch_size:
        raise ValueError("Each split must contain at least one complete batch.")
    options = dict(batch_size=args.batch_size, num_workers=args.workers,
                   pin_memory=device.type == "cuda", drop_last=True)
    train_loader = DataLoader(train, shuffle=True, **options)
    val_loader = DataLoader(val, shuffle=False, **options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = VoxelAgeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        training_loss = run_epoch(model, train_loader, device, optimizer)
        schedule.step()
        validation_loss = run_epoch(model, val_loader, device)
        print(f"Epoch {epoch:3d}  train {training_loss:.5f}  validation {validation_loss:.5f}", flush=True)
        if epoch > args.epochs // 2 and validation_loss < best:
            best = validation_loss
            checkpoint = {"net": model.state_dict(), "epoch": epoch, "validation_loss": best}
            temporary = args.output.with_suffix(".tmp.pt")
            torch.save(checkpoint, temporary)
            temporary.replace(args.output)


if __name__ == "__main__":
    main()
