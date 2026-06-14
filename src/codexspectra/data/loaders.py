from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def load_cube(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a hyperspectral cube and optional wavelength vector.

    Returns cube as (height, width, bands). For unknown formats the function fails loudly
    so dataset-specific readers can be added without silently corrupting spectral axes.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    wavelengths = None
    if suffix == ".npy":
        cube = np.load(path)
    elif suffix == ".npz":
        obj = np.load(path)
        keys = list(obj.keys())
        cube_key = "cube" if "cube" in keys else keys[0]
        cube = obj[cube_key]
        if "wavelengths" in obj:
            wavelengths = obj["wavelengths"]
    elif suffix == ".mat":
        mat: dict[str, Any] = loadmat(path)
        candidates = [v for k, v in mat.items() if not k.startswith("__") and isinstance(v, np.ndarray)]
        cubes = [v for v in candidates if v.ndim == 3]
        if not cubes:
            raise ValueError(f"No 3D cube found in {path}")
        cube = max(cubes, key=lambda a: a.size)
        for key in ["wavelengths", "lambda", "wl", "bands"]:
            if key in mat:
                wavelengths = np.ravel(mat[key])
                break
    elif suffix == ".bil":
        raise NotImplementedError(
            "ENVI .bil reading needs dataset-specific header parsing. Convert with the "
            "HYPERDOC tools or add a reader using spectral.io.envi."
        )
    else:
        raise ValueError(f"Unsupported cube format: {path}")

    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube with 3 dims, got {cube.shape} from {path}")
    if cube.shape[0] < cube.shape[-1] and cube.shape[1] < cube.shape[-1]:
        # likely already H,W,B
        pass
    elif cube.shape[0] < 16 and cube.shape[-1] > 16:
        cube = np.moveaxis(cube, 0, -1)
    return cube, wavelengths
