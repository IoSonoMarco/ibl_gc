from typing import Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class BasisInfo:
    basis: np.ndarray          # shape: (n_lags, n_basis)
    lag_times: np.ndarray      # lag values in seconds, shape: (n_lags,)
    basis_type: str
    dt: float
    window: Tuple[float, float]


def make_raised_cosine_basis(
    n_basis: int,
    dt: float,
    window: Tuple[float, float],
    nonlinear: bool = False,
    eps: float = 1e-6
) -> BasisInfo:
    """
    Create a raised-cosine basis over a lag window.

    Parameters
    ----------
    n_basis : int
        Number of basis functions.
    dt : float
        Bin size in seconds.
    window : tuple (t_min, t_max)
        Lag window in seconds. Can include negative lags for event kernels.
    nonlinear : bool
        If True, place basis centers in log-time (useful for spike history).
        Only valid for nonnegative windows in practice.
    eps : float
        Small constant for log transform.

    Returns
    -------
    BasisInfo
    """
    t_min, t_max = window
    if t_max <= t_min:
        raise ValueError("window must satisfy t_max > t_min")

    lag_times = np.arange(t_min, t_max + dt/2, dt)

    if nonlinear:
        if t_min < 0:
            raise ValueError("nonlinear/log basis currently requires nonnegative lag window")
        warped = np.log(lag_times + eps)
        centers = np.linspace(warped[0], warped[-1], n_basis)
        spacing = centers[1] - centers[0] if n_basis > 1 else 1.0
        X = warped[:, None] - centers[None, :]
    else:
        centers = np.linspace(lag_times[0], lag_times[-1], n_basis)
        spacing = centers[1] - centers[0] if n_basis > 1 else (lag_times[-1] - lag_times[0] + dt)
        X = lag_times[:, None] - centers[None, :]

    # raised cosine bumps
    arg = np.clip(np.pi * X / (2 * spacing), -np.pi, np.pi)
    basis = (np.cos(arg) + 1.0) / 2.0
    basis[np.abs(X) > spacing] = 0.0

    # normalize columns to unit L2 norm for numerical stability
    norms = np.linalg.norm(basis, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    basis = basis / norms

    return BasisInfo(
        basis=basis,
        lag_times=lag_times,
        basis_type="raised_cosine_log" if nonlinear else "raised_cosine",
        dt=dt,
        window=window,
    )