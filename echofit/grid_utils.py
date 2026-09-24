"""
grid_utils.py
=============

Shared logic for picking the driver Fourier series' finest resolvable
timescale from a set of (possibly irregularly sampled) observation times.

Used by both ``EchoFit.build_grid`` (fitting) and ``synthetic.py``
(ground-truth generation) so the two always agree on what frequency grid
a given set of observation times implies -- see CLAUDE.md's "Known rough
edges" note on why this needs to be robust rather than the raw minimum gap.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def estimate_dt_min(
    t_arrays: Iterable[np.ndarray],
    percentile: float = 5.0,
    floor_frac: float = 1e-3,
    t_span: Optional[float] = None,
) -> float:
    """Robust estimate of the shortest timescale the driver should resolve.

    Sets the driver frequency grid's upper bound (``w_max = pi / dt_min``).
    Uses a low percentile of consecutive observation gaps rather than the
    strict minimum: for irregularly/randomly sampled data the single
    tightest gap between any two points can be pathologically small (its
    expectation shrinks much faster than the typical spacing as more points
    are added), which would otherwise blow up ``w_max`` and force the model
    to try to resolve spurious high-frequency structure.

    Parameters
    ----------
    t_arrays : iterable of array_like
        One array of observation times per band (need not be pre-sorted).
    percentile : float
        Percentile (0-100) of consecutive gaps to use, e.g. 5 = 5th
        percentile. Lower is closer to (and more sensitive like) the raw
        minimum; higher is more conservative (coarser resolution).
    floor_frac : float
        Floor on the returned ``dt_min`` as a fraction of ``t_span``, to
        avoid a degenerate (near-zero) estimate when very few points are
        available.
    t_span : float, optional
        Total observed time baseline (days), used for ``floor_frac`` and as
        the fallback when fewer than two total points are available.

    Returns
    -------
    dt_min : float
    """
    # A plain np.concatenate([]) raises rather than returning a size-0
    # array, so the "no bands have 2+ points" fallback below has to be
    # checked on this list directly, before concatenating -- found via a
    # coverage report showing that fallback as unreachable dead code (it
    # crashed here first, for every t_arrays that should have hit it).
    diffs = [np.diff(np.sort(np.asarray(t, float))) for t in t_arrays if len(t) > 1]
    if not diffs:
        dt_min = t_span if t_span is not None else 1.0
    else:
        dt_min = float(np.percentile(np.concatenate(diffs), percentile))
    if t_span is not None:
        dt_min = max(dt_min, floor_frac * t_span)
    return dt_min
