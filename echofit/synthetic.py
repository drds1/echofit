"""
synthetic.py
============

Minimal synthetic-data generator used for the demo notebook and tests: draws
a DRW driver on a dense Fourier grid, echoes it into several bands with the
true forward model, and adds Gaussian noise -- so the resulting light curves
have the correct lag ordering with wavelength (longer wavelength => longer
lag) by construction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .forward_model import response_function, transfer_coeffs, compute_echo
from .grid_utils import estimate_dt_min


def make_frequency_grid(n_freq: int, t_span: float, dt_min: float) -> np.ndarray:
    """Log-spaced angular-frequency grid from the light-curve baseline up to
    (roughly) the Nyquist frequency set by the finest sampling."""
    w_min = 2.0 * np.pi / t_span
    w_max = np.pi / dt_min
    return np.geomspace(w_min, w_max, n_freq)


def _sample_uniform_excluding_gaps(
    rng: np.random.Generator,
    t_span: float,
    gaps: Sequence[Tuple[float, float]],
    size: int,
) -> np.ndarray:
    """Uniformly sample ``size`` times in ``[0, t_span)``, skipping the given
    ``(gap_start, gap_duration)`` windows -- e.g. a seasonal/weather/downtime
    gap in an observing campaign. Density stays uniform over the remaining
    (observable) time, it's just zero inside the gaps.
    """
    allowed = []  # list of (start, end) intervals actually observable
    cursor = 0.0
    for start, duration in sorted(gaps, key=lambda g: g[0]):
        start = float(np.clip(start, 0.0, t_span))
        end = float(np.clip(start + duration, start, t_span))
        if start > cursor:
            allowed.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < t_span:
        allowed.append((cursor, t_span))
    if not allowed:
        raise ValueError("gaps cover the entire time span; no observable time remains")

    lengths = np.array([b - a for a, b in allowed])
    edges = np.concatenate([[0.0], np.cumsum(lengths)])
    u = rng.uniform(0.0, edges[-1], size=size)
    interval_idx = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, len(allowed) - 1)
    starts = np.array([a for a, _ in allowed])[interval_idx]
    return starts + (u - edges[interval_idx])


def generate_synthetic_dataset(
    bands: Optional[Dict[str, float]] = None,
    M_BH: float = 1.0e8,
    log_mdot_true: float = 0.2,
    inclination_true: float = 35.0,
    sigma_drw_true: float = 0.3,
    tau_drw_true: float = 30.0,
    t_span: float = 200.0,
    n_obs_per_band: int = 60,
    n_freq: int = 60,
    n_tau: int = 400,
    tau_max: float = 60.0,
    noise_level: float = 0.02,
    gaps: Optional[Sequence[Tuple[float, float]]] = None,
    seed: int = 0,
) -> dict:
    """Generate a synthetic multi-band reverberation-mapping dataset.

    Parameters
    ----------
    bands : dict, optional
        Mapping ``band_name -> wavelength_angstrom``. Defaults to five SDSS-
        like bands spanning u through z, which guarantees a range of lags.
    M_BH, log_mdot_true, inclination_true, sigma_drw_true, tau_drw_true :
        Ground-truth physical parameters used to generate the data.
    t_span : float
        Total light-curve baseline, days.
    n_obs_per_band : int
        Number of (irregular) observation epochs per band, before removing
        any that fall in ``gaps``.
    n_freq, n_tau, tau_max : int, int, float
        Resolution of the driver Fourier grid and the lag grid used to
        build the ground-truth response functions.
    noise_level : float
        Fractional Gaussian noise added to each band (relative to that
        band's echo amplitude).
    gaps : sequence of (start, duration), optional
        Observing gaps (days) applied to every band identically -- e.g.
        ``[(50, 14), (150, 21)]`` for a 2-week gap starting day 50 and a
        3-week gap starting day 150 (weather/downtime/seasonal-style
        campaign structure). ``n_obs_per_band`` observations are still drawn
        uniformly at random, just excluding these windows, so the same
        total point count is spread more densely over the remaining time.
    seed : int
        RNG seed.

    Returns
    -------
    data : dict
        ``{"bands": {name: {"t", "y", "yerr", "wavelength"}}, "truth": {...},
        "freqs": array, "tau_grid": array}``.
    """
    rng = np.random.default_rng(seed)

    if bands is None:
        bands = {"u": 3543.0, "g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}

    sorted_bands = sorted(bands.items(), key=lambda kv: kv[1])
    # sample every band's observation times up front so the frequency grid
    # below is derived from the *actual* (irregular) cadence -- via the same
    # estimator EchoFit.build_grid() uses -- rather than a prior guess. This
    # keeps the ground-truth driver and the fitting basis on the same grid.
    if gaps:
        t_by_band = {
            name: np.sort(_sample_uniform_excluding_gaps(rng, t_span, gaps, n_obs_per_band))
            for name, _ in sorted_bands
        }
    else:
        t_by_band = {
            name: np.sort(rng.uniform(0.0, t_span, size=n_obs_per_band))
            for name, _ in sorted_bands
        }

    dt_min = estimate_dt_min(t_by_band.values(), t_span=t_span)
    freqs = make_frequency_grid(n_freq, t_span, dt_min)
    tau_grid = np.linspace(0.0, tau_max, n_tau)

    # -- draw a DRW driver realization on the fixed Fourier grid ---------
    dw = np.gradient(freqs)
    power = sigma_drw_true ** 2 * tau_drw_true / (1.0 + (freqs * tau_drw_true) ** 2)
    amp_scale = np.sqrt(power * np.clip(dw, 1e-8, None))
    S_true = rng.normal(0.0, amp_scale)
    C_true = rng.normal(0.0, amp_scale)

    out_bands = {}
    per_band_truth = {}
    for name, wavelength in sorted_bands:
        t = t_by_band[name]

        psi = np.asarray(
            response_function(
                tau_grid,
                log_mdot=log_mdot_true,
                wavelength=wavelength,
                inclination=inclination_true,
                M_BH=M_BH,
            )
        )
        A, B = transfer_coeffs(tau_grid, psi, freqs)
        echo = np.asarray(compute_echo(S_true, C_true, freqs, np.asarray(A), np.asarray(B), t))

        S_band_true = rng.uniform(0.8, 1.5)
        C_band_true = rng.uniform(-0.5, 0.5)
        y_clean = S_band_true * echo + C_band_true

        yerr = np.full_like(y_clean, noise_level * (np.std(y_clean) + 1e-3))
        y = y_clean + rng.normal(0.0, yerr)

        out_bands[name] = {
            "t": t,
            "y": y,
            "yerr": yerr,
            "wavelength": wavelength,
        }
        per_band_truth[name] = {
            "S_band": S_band_true,
            "C_band": C_band_true,
            "tau_mean": float(np.trapz(tau_grid * psi, tau_grid)),
        }

    truth = {
        "M_BH": M_BH,
        "log_mdot": log_mdot_true,
        "inclination": inclination_true,
        "sigma_drw": sigma_drw_true,
        "tau_drw": tau_drw_true,
        "S": S_true,
        "C": C_true,
        "bands": per_band_truth,
    }

    return {"bands": out_bands, "truth": truth, "freqs": freqs, "tau_grid": tau_grid, "gaps": gaps}
