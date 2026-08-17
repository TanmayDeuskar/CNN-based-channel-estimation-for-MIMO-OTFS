"""Shared utilities used by training, evaluation and benchmarking."""
import random
import numpy as np
import scipy.io as sio

try:
    import h5py
    HAS_H5PY = True
except Exception:
    h5py = None
    HAS_H5PY = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def to_numpy_complex(arr):
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64)
    if arr.dtype.names is not None:
        names = tuple(n.lower() for n in arr.dtype.names)
        if "real" in names and "imag" in names:
            real = arr["real"].astype(np.float32)
            imag = arr["imag"].astype(np.float32)
            return (real + 1j * imag).astype(np.complex64)
    return arr.astype(np.complex64)


def load_mat_var(path, varname):
    """Load a MATLAB variable, supporting v7.2 and v7.3 files."""
    try:
        mat = sio.loadmat(path, variable_names=[varname])
        if varname in mat:
            return mat[varname]
    except NotImplementedError:
        pass

    if not HAS_H5PY:
        raise RuntimeError("MAT v7.3 file encountered but h5py is not installed.")

    with h5py.File(path, "r") as f:
        if varname in f:
            return f[varname][:]
        for key in f.keys():
            if varname in key:
                return f[key][:]

    raise KeyError(f"Could not locate {varname} in {path}")


def load_phi(path):
    return to_numpy_complex(load_mat_var(path, "Phi"))


def orient_phi(phi, num_dd):
    """Return Phi with shape (numDD, Ptot)."""
    if phi.shape[0] == num_dd:
        return phi
    if phi.shape[1] == num_dd:
        return phi.T.copy()
    raise RuntimeError(
        f"Phi shape {phi.shape} incompatible with numDD={num_dd}"
    )


def load_state_dict(checkpoint_path, device):
    """Load common checkpoint formats and strip DataParallel prefixes."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    try:
        return state
    except Exception:
        return {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }


def strip_module_prefix(state):
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }


def complex_to_str_grid(mat):
    """Format a complex matrix for CSV output."""
    m, n = mat.shape
    grid = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            re = float(mat[i, j].real)
            im = float(mat[i, j].imag)
            sign = "+" if im >= 0 else "-"
            grid[i, j] = f"{re:.6e}{sign}{abs(im):.6e}j"
    return grid
