import numpy as np
from dataclasses import dataclass
from scipy.signal import fftconvolve
from typing import List, Dict
from lib.analysis.kernel_regression.basis import BasisInfo


@dataclass
class EventKernelOperator:
    row_idx: np.ndarray       # shape: (n_events, n_lags), absolute bin row for each event/lag
    valid: np.ndarray         # shape: (n_events, n_lags), whether row_idx is in bounds
    basis: np.ndarray         # shape: (n_lags, n_basis)
    lag_times: np.ndarray     # shape: (n_lags,)
    lag_bins: np.ndarray      # shape: (n_lags,)
    n_bins: int
    dt: float

def make_event_kernel_operator(
    event_times: np.ndarray,
    bin_edges: np.ndarray,
    basis_info: BasisInfo,
) -> EventKernelOperator:
    """
    Precompute the mapping from trial events to time-bin rows for a given basis.
    """
    event_times = np.asarray(event_times, dtype=float)
    n_bins = len(bin_edges) - 1

    # map each event to the bin containing it
    event_bin_idx = np.searchsorted(bin_edges, event_times, side="right") - 1
    valid_events = np.isfinite(event_times) & (event_bin_idx >= 0) & (event_bin_idx < n_bins)

    # convert lag times to integer bin offsets
    lag_bins = np.rint(basis_info.lag_times / basis_info.dt).astype(int)

    # absolute row index for every (event, lag)
    row_idx = event_bin_idx[:, None] + lag_bins[None, :]
    valid = valid_events[:, None] & (row_idx >= 0) & (row_idx < n_bins)

    return EventKernelOperator(
        row_idx=row_idx,
        valid=valid,
        basis=basis_info.basis,
        lag_times=basis_info.lag_times,
        lag_bins=lag_bins,
        n_bins=n_bins,
        dt=basis_info.dt
    )
    
###

@dataclass
class DesignMatrixBlock:
    X: np.ndarray
    column_names: List[str]
    metadata: Dict


def apply_event_kernel_operator(
    event_values: np.ndarray,
    op: EventKernelOperator,
    center: bool = False,
    scale: bool = False,
    dtype=np.float32,
    covariate_name: str = ""
) -> DesignMatrixBlock:
    """
    Apply a precomputed event kernel operator to trial values.
    """
    vals = np.asarray(event_values, dtype=dtype).copy()
    if vals.ndim != 1:
        raise ValueError("event_values must be 1D")
    if len(vals) != op.row_idx.shape[0]:
        raise ValueError("event_values length must match number of events")

    finite = np.isfinite(vals)
    vals[~finite] = 0.0

    m = None
    s = None

    if center:
        m = vals[finite].mean() if np.any(finite) else 0.0
        vals = vals - m

    if scale:
        s = vals[finite].std() if np.any(finite) else 1.0
        if s == 0:
            s = 1.0
        vals = vals / s

    n_basis = op.basis.shape[1]
    X = np.zeros((op.n_bins, n_basis), dtype=dtype)

    # event-additive exact construction
    for i, v in enumerate(vals):
        if v == 0:
            continue
        keep = op.valid[i]
        if not np.any(keep):
            continue
        rows = op.row_idx[i, keep]                # absolute bin rows for this event
        X[rows, :] += v * op.basis[keep, :]       # add scaled basis footprint

    column_names = [f"{covariate_name}_basis_{k}" for k in range(n_basis)]
    metadata = {
        "covariate_name": covariate_name,
        "lag_times": op.lag_times,
        "lag_bins": op.lag_bins,
        "center": center,
        "scale": scale,
        "mean_before_centering": m,
        "std_before_scaling": s,
    }

    return DesignMatrixBlock(X=X, column_names=column_names, metadata=metadata)

###

def build_spike_history_design(
    binned_spikes: np.ndarray,
    basis_info: BasisInfo,
    covariate_name: str = "spk_hist"
) -> DesignMatrixBlock:
    """
    Build a causal spike-history design block.

    Parameters
    ----------
    binned_spikes : array, shape (n_bins,)
        Spike counts for one neuron.
    basis_info : BasisInfo
        Should typically use a nonnegative lag window, e.g. (0.001, 0.2).
    covariate_name : str
        Prefix for column names.

    Returns
    -------
    DesignMatrixBlock

    Notes
    -----
    To ensure strict causality, the convolved signal is shifted by one bin:
    history at time t depends only on spikes before t.
    """
    y = np.asarray(binned_spikes, dtype=float)
    n_bins = len(y)

    if basis_info.window[0] < 0:
        raise ValueError("Spike-history basis should use nonnegative lags")

    basis = basis_info.basis
    n_basis = basis.shape[1]

    X = np.zeros((n_bins, n_basis), dtype=float)

    for k in range(n_basis):
        kernel = basis[:, k]
        full = fftconvolve(y, kernel, mode="full")
        conv_same = full[:n_bins]

        # strict causality: shift forward by 1 bin
        xk = np.zeros(n_bins, dtype=float)
        xk[1:] = conv_same[:-1]
        X[:, k] = xk

    column_names = [f"{covariate_name}_basis_{k}" for k in range(n_basis)]

    metadata = {
        "covariate_name": covariate_name,
        "basis_type": basis_info.basis_type,
        "basis_window": basis_info.window,
        "lag_times": basis_info.lag_times,
        "causal": True,
    }

    return DesignMatrixBlock(X=X, column_names=column_names, metadata=metadata)

###

@dataclass
class ContinuousDesignBlock:
    X: np.ndarray
    column_names: list
    metadata: dict


def bin_continuous_to_session_bins(
    t_cont: np.ndarray,
    x_cont: np.ndarray,
    bin_edges: np.ndarray,
    agg: str = "mean",
    fill_value: float = 0.0,
    dtype=np.float32
) -> np.ndarray:
    """
    Bin a continuous time series into session bins.

    Parameters
    ----------
    t_cont : array, shape (n_samples,)
        Timestamps of continuous signal.
    x_cont : array, shape (n_samples,)
        Signal values.
    bin_edges : array, shape (n_bins + 1,)
        Session bin edges.
    agg : {"mean", "last", "sum"}
        Aggregation rule within each bin.
    fill_value : float
        Value for bins with no samples.
    """
    t_cont = np.asarray(t_cont, dtype=float)
    x_cont = np.asarray(x_cont, dtype=float)

    valid = np.isfinite(t_cont) & np.isfinite(x_cont)
    t_cont = t_cont[valid]
    x_cont = x_cont[valid]

    n_bins = len(bin_edges) - 1
    out = np.full(n_bins, fill_value, dtype=dtype)

    if len(t_cont) == 0:
        return out

    bin_idx = np.searchsorted(bin_edges, t_cont, side="right") - 1
    keep = (bin_idx >= 0) & (bin_idx < n_bins)

    t_cont = t_cont[keep]
    x_cont = x_cont[keep]
    bin_idx = bin_idx[keep]

    if agg == "mean":
        counts = np.bincount(bin_idx, minlength=n_bins)
        sums = np.bincount(bin_idx, weights=x_cont, minlength=n_bins)
        nz = counts > 0
        out[nz] = (sums[nz] / counts[nz]).astype(dtype)

    elif agg == "sum":
        sums = np.bincount(bin_idx, weights=x_cont, minlength=n_bins)
        touched = np.bincount(bin_idx, minlength=n_bins) > 0
        out[touched] = sums[touched].astype(dtype)

    elif agg == "last":
        order = np.argsort(t_cont)
        t_cont = t_cont[order]
        x_cont = x_cont[order]
        bin_idx = bin_idx[order]
        out[:] = fill_value
        out[bin_idx] = x_cont.astype(dtype)

    else:
        raise ValueError("agg must be one of {'mean', 'last', 'sum'}")

    return out


def build_continuous_design_block(
    x_binned: np.ndarray,
    covariate_name: str,
    center: bool = False,
    scale: bool = False,
    dtype=np.float32
) -> ContinuousDesignBlock:
    """
    Wrap a 1D binned continuous regressor as a design block.
    """
    x = np.asarray(x_binned, dtype=dtype).copy()
    finite = np.isfinite(x)
    x[~finite] = 0.0

    m = None
    s = None

    if center:
        m = x[finite].mean() if np.any(finite) else 0.0
        x = x - m

    if scale:
        s = x[finite].std() if np.any(finite) else 1.0
        if s == 0:
            s = 1.0
        x = x / s

    return ContinuousDesignBlock(
        X=x[:, None],
        column_names=[covariate_name],
        metadata={
            "covariate_name": covariate_name,
            "center": center,
            "scale": scale,
            "mean_before_centering": m,
            "std_before_scaling": s,
        }
    )

###

def build_slow_trial_drift_design(
    n_trials,
    n_bins_per_trial,
    n_basis=6,
    dtype=np.float32,
    covariate_name="drift"
):
    u = np.linspace(0.0, 1.0, n_trials)

    centers = np.linspace(0.0, 1.0, n_basis)
    spacing = centers[1] - centers[0] if n_basis > 1 else 1.0

    Xtrial = np.zeros((n_trials, n_basis), dtype=dtype)
    for k, c in enumerate(centers):
        d = u - c
        arg = np.clip(np.pi * d / (2 * spacing), -np.pi, np.pi)
        x = (np.cos(arg) + 1.0) / 2.0
        x[np.abs(d) > spacing] = 0.0
        Xtrial[:, k] = x

    X = np.repeat(Xtrial[:, None, :], n_bins_per_trial, axis=1).reshape(-1, n_basis)

    column_names = [f"{covariate_name}_basis_{k}" for k in range(n_basis)]
    metadata = {
        "covariate_name": covariate_name,
        "n_trials": n_trials,
        "n_bins_per_trial": n_bins_per_trial,
        "basis_type": "raised_cosine_trial_index"
    }

    return DesignMatrixBlock(X=X, column_names=column_names, metadata=metadata)