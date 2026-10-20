"""
Data for image regression.

Input x in Omega = [-1,1]^2 (pixel coordinates), output y in [0,1]^3 (RGB).

The target image I has fN x fN pixels, where f (a power of two) is the
subsampling factor: f = 2 for the 2N x 2N images and f = 4 for the
4N x 4N images (Cathedral and SST on the nested grids).  Pixel (i, j), 0-based, sits at the centre
of its fine cell,

    x_ij = (s_i, s_j),   s_i = -1 + (i + 0.5) h/f,   i = 0, ..., fN-1,

with h = 2/N.  The f x f blocks of pixels partition Omega into N x N
cells of side h.

  train   one pixel per cell, the one with indices (fi, fj), at its true
          coordinates (N^2 pixels).  It lies inside its cell, offset from
          the cell centre by h/2 - h/(2f) per axis.
  val     the diagonally opposite pixel of each cell, indices
          (fi + f/2, fj + f/2), offset (h/2, h/2) from the training pixel
          (N^2 pixels).  Used only to select kappa.
  unseen  all other pixels, used neither for training nor for selection.
  test    all fN x fN pixels.
  nested  the test grids m = N, 2N, ..., fN, i.e. every (fN/m)-th pixel
          per axis.  m = N is the training grid and m = fN the
          full test grid.

On every grid of m x m pixels, (|Omega|/m^2) sum ||F(x) - I(x)||^2 is the
grid error of the piecewise-constant approximation at resolution m.
"""

import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Image and pixel coordinates
# ---------------------------------------------------------------------

def load_image(path, N, factor=2):
    """Read the image at `path` as a float tensor (fN, fN, 3) in [0,1].
    The image is used as it is: its side must be exactly f * N."""
    from skimage import io

    img = np.asarray(io.imread(str(path)))
    if img.ndim == 2:                       # grayscale -> RGB
        img = np.stack([img] * 3, axis=-1)
    img = img[:, :, :3].astype(np.float64)  # drop an alpha channel
    if img.max() > 1.0:                     # 8-bit values -> [0,1]
        img = img / 255.0

    side = int(factor) * int(N)
    if img.shape[:2] != (side, side):
        raise ValueError(f"{path}: image is {img.shape[0]} x {img.shape[1]}, "
                         f"but f * N = {side}; choose N = side / f")
    return torch.from_numpy(np.clip(img, 0.0, 1.0)).float()


def fine_axis(N, device=None, dtype=torch.float32, factor=2):
    """Pixel centres s_i = -1 + (i + 0.5) h/f, i = 0, ..., fN-1."""
    h = 2.0 / int(N)
    f = int(factor)
    return -1.0 + (torch.arange(f * int(N), device=device, dtype=dtype)
                   + 0.5) * (h / f)


def _lattice(ax1, ax2):
    X, Y = torch.meshgrid(ax1, ax2, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


# ---------------------------------------------------------------------
# Train / val / unseen / test sets
# ---------------------------------------------------------------------

def make_train_test(image, N, device, dtype=torch.float32, factor=2):
    """Build the data sets described in the module docstring.

    Returns a dict with
      x_train, y_train    (N^2, 2), (N^2, 3)
      x_val, y_val        (N^2, 2), (N^2, 3)
      x_unseen, y_unseen  ((f^2 - 2) N^2, 2), ((f^2 - 2) N^2, 3)
      x_test, y_test      (f^2 N^2, 2), (f^2 N^2, 3)
      test_nested         list of {"m", "x", "y"}, m = N, 2N, ..., fN
    """
    f = int(factor)
    if f < 2 or (f & (f - 1)) != 0:
        raise ValueError("factor must be a power of two >= 2")
    side = f * int(N)
    I = load_image(image, N, factor=f).to(device=device, dtype=dtype)
    ax = fine_axis(N, device=device, dtype=dtype, factor=f)

    # Test: all pixels, in row-major order (pixel (i, j) -> i * side + j).
    x_test = _lattice(ax, ax)
    y_test = I.reshape(-1, 3)

    # Train and val: one pixel per cell each.
    i_tr = torch.arange(0, side, f, device=device)
    i_va = torch.arange(f // 2, side, f, device=device)
    x_train = _lattice(ax[i_tr], ax[i_tr])
    y_train = I[i_tr][:, i_tr].reshape(-1, 3)
    x_val = _lattice(ax[i_va], ax[i_va])
    y_val = I[i_va][:, i_va].reshape(-1, 3)

    # Unseen: every pixel that is in neither train nor val.
    used = torch.zeros(side, side, dtype=torch.bool, device=device)
    used[i_tr[:, None], i_tr] = True
    used[i_va[:, None], i_va] = True
    keep = ~used.reshape(-1)
    x_unseen, y_unseen = x_test[keep], y_test[keep]

    # Nested test grids m = N, 2N, ..., fN (every s-th pixel, s = f, ..., 1).
    test_nested = []
    s = f
    while s >= 1:
        idx = torch.arange(0, side, s, device=device)
        test_nested.append({"m": side // s,
                            "x": _lattice(ax[idx], ax[idx]),
                            "y": I[idx][:, idx].reshape(-1, 3)})
        s //= 2

    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_unseen": x_unseen, "y_unseen": y_unseen,
        "x_test": x_test, "y_test": y_test,
        "test_nested": test_nested,
    }
