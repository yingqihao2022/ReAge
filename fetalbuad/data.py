"""Read prepared MRI, segmentation and mask volumes in native array order."""

from pathlib import Path

import nibabel as nib
import numpy as np


def read_volume(path):
    """Load a three-dimensional NIfTI or NumPy array without reorienting it."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        array = np.asanyarray(nib.load(path).dataobj)
    if array.ndim != 3 or array.dtype.kind not in "biuf":
        raise ValueError(f"Expected a numeric 3D volume at {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite voxel values in {path}")
    return array
