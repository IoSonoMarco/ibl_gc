import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class TimeBinning:
    bin_edges: np.ndarray      # shape: (n_bins + 1,)
    bin_centers: np.ndarray    # shape: (n_bins,)
    dt: float


def make_time_binning(
    t_start: float,
    t_stop: float,
    dt: float
) -> TimeBinning:
    """
    Create global time bins for the session.
    """
    if t_stop <= t_start:
        raise ValueError("t_stop must be greater than t_start")
    bin_edges = np.arange(t_start, t_stop + dt, dt)
    if bin_edges[-1] < t_stop:
        bin_edges = np.append(bin_edges, t_stop)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return TimeBinning(bin_edges=bin_edges, bin_centers=bin_centers, dt=dt)

###

def bin_spike_times(
    spike_times: np.ndarray,
    bin_edges: np.ndarray
) -> np.ndarray:
    """
    Bin spike times into counts per bin.
    """
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    return counts.astype(float)

###

def build_event_train(
    event_times: np.ndarray,
    event_values: np.ndarray,
    bin_edges: np.ndarray
) -> np.ndarray:
    """
    Create a weighted event train over time bins.

    Each event contributes its scalar value to the bin containing its time.

    Parameters
    ----------
    event_times : array, shape (n_events,)
    event_values : array, shape (n_events,)
    bin_edges : array, shape (n_bins + 1,)

    Returns
    -------
    train : array, shape (n_bins,)
    """
    event_times = np.asarray(event_times)
    event_values = np.asarray(event_values)

    if event_times.shape != event_values.shape:
        raise ValueError("event_times and event_values must have the same shape")

    n_bins = len(bin_edges) - 1
    train = np.zeros(n_bins, dtype=float)

    # assign each event to a bin
    idx = np.searchsorted(bin_edges, event_times, side="right") - 1
    valid = (idx >= 0) & (idx < n_bins) & np.isfinite(event_values) & np.isfinite(event_times)

    # multiple events falling in the same bin are summed
    np.add.at(train, idx[valid], event_values[valid])

    return train

###

def make_event_mask(
    bin_centers: np.ndarray,
    event_times: np.ndarray,
    event_window: List[np.ndarray]
) -> np.ndarray:
    """
    Convert time intervals into a boolean mask over bins.

    Parameters
    ----------
    bin_centers : array, shape (n_bins,)
    event_times : array, shape (n_events,)
    event_window: list, shape (2,)

    Returns
    -------
    mask : bool array, shape (n_bins,)
    """
    mask = np.zeros(len(bin_centers), dtype=bool)
    
    intervals = np.column_stack([
        event_times + event_window[0],
        event_times + event_window[1]
    ])

    if intervals.size == 0:
        return mask

    for start, stop in intervals:
        mask |= (bin_centers >= start) & (bin_centers < stop)

    return mask