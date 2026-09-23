"""
plotting.py
===========

Three plotting entry points, matching the EchoFit API:

* ``plot_raw_lightcurves``   -- one panel per band, ordered by wavelength.
* ``plot_lightcurve_fits``   -- posterior predictive light curves (68%/95%
                                 credible bands) with the corresponding
                                 response function psi(tau) alongside each
                                 band.
* ``plot_mcmc_diagnostics``  -- trace plots per (scalar) parameter.

All functions take plain numpy-able arrays / dicts so they have no
dependency on JAX or NumPyro themselves.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def _band_colors(bands: Dict[str, dict]):
    """Consistent color per band, ordered and colored by wavelength."""
    ordered = sorted(bands.items(), key=lambda kv: kv[1]["wavelength"])
    wavelengths = np.array([d["wavelength"] for _, d in ordered])
    norm = plt.Normalize(wavelengths.min(), wavelengths.max())
    cmap = plt.get_cmap("plasma_r")
    colors = {name: cmap(norm(d["wavelength"])) for name, d in ordered}
    return ordered, colors


def plot_raw_lightcurves(bands: Dict[str, dict], figsize_per_panel=(7, 1.8)):
    """One panel per band (no overlay), ordered shortest -> longest wavelength.

    Parameters
    ----------
    bands : dict
        ``{band_name: {"t", "y", "yerr", "wavelength"}}``.
    """
    ordered, colors = _band_colors(bands)
    n = len(ordered)
    fig, axes = plt.subplots(
        n, 1, figsize=(figsize_per_panel[0], figsize_per_panel[1] * n), sharex=True
    )
    if n == 1:
        axes = [axes]

    for ax, (name, d) in zip(axes, ordered):
        ax.errorbar(
            d["t"], d["y"], yerr=d["yerr"], fmt="o", ms=4, color=colors[name],
            ecolor=colors[name], alpha=0.85, capsize=3, elinewidth=1.2, capthick=1.2,
        )
        ax.set_ylabel(f"{name}\n({d['wavelength']:.0f} \u00c5)")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("time [days]")
    fig.suptitle("Raw multi-band light curves")
    fig.tight_layout()
    return fig, axes


def plot_lightcurve_fits(
    bands: Dict[str, dict],
    t_fine: np.ndarray,
    y_pred_samples: Dict[str, np.ndarray],
    tau_grid: np.ndarray,
    psi_samples: Dict[str, np.ndarray],
    driver_samples: Optional[np.ndarray] = None,
    figsize_per_row=(10, 2.2),
    driver_row_height=1.8,
):
    """Posterior-predictive light curves with 68%/95% credible bands, plus
    the corresponding response function psi(tau) next to each band.

    Parameters
    ----------
    bands : dict
        Observed data, ``{band_name: {"t", "y", "yerr", "wavelength"}}``.
    t_fine : (n_fine,) array
        Dense time grid the posterior predictive was evaluated on.
    y_pred_samples : dict
        ``{band_name: array of shape (n_samples, n_fine)}`` posterior
        predictive draws of the light curve, evaluated at ``t_fine``.
    tau_grid : (n_tau,) array
        Lag grid the response functions live on.
    psi_samples : dict
        ``{band_name: array of shape (n_samples, n_tau)}`` posterior draws
        of psi(tau) for that band.
    driver_samples : (n_samples, n_fine) array, optional
        Posterior draws of the raw driver X(t) (see
        ``forward_model.driver_at``), evaluated at the same ``t_fine`` as
        ``y_pred_samples``. If given, plotted as an extra panel above the
        per-band rows, sharing the same time axis so lags between the
        driver and each band's echo are visually alignable.
    """
    ordered, colors = _band_colors(bands)
    n = len(ordered)
    has_driver = driver_samples is not None
    n_rows = n + (1 if has_driver else 0)
    height_ratios = ([driver_row_height] if has_driver else []) + [figsize_per_row[1]] * n
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(figsize_per_row[0], sum(height_ratios)),
        gridspec_kw={"width_ratios": [3, 1], "height_ratios": height_ratios},
    )
    if n_rows == 1:
        axes = axes[None, :]

    if has_driver:
        ax_drv, ax_drv_unused = axes[0, 0], axes[0, 1]
        lo95, lo68, med, hi68, hi95 = np.percentile(driver_samples, [2.5, 16, 50, 84, 97.5], axis=0)
        ax_drv.fill_between(t_fine, lo95, hi95, color="0.5", alpha=0.15)
        ax_drv.fill_between(t_fine, lo68, hi68, color="0.5", alpha=0.35)
        ax_drv.plot(t_fine, med, color="0.2", lw=1.5)
        ax_drv.set_ylabel("driver\nX(t)")
        ax_drv.set_title("Inferred driving light curve", fontsize=10)
        ax_drv.grid(alpha=0.25)
        ax_drv.sharex(axes[1, 0])
        ax_drv_unused.axis("off")

    for row, (name, d) in enumerate(ordered):
        ax_lc, ax_psi = axes[row + has_driver, 0], axes[row + has_driver, 1]
        color = colors[name]

        preds = y_pred_samples[name]
        lo95, lo68, med, hi68, hi95 = np.percentile(preds, [2.5, 16, 50, 84, 97.5], axis=0)

        ax_lc.fill_between(t_fine, lo95, hi95, color=color, alpha=0.15, label="95% CI")
        ax_lc.fill_between(t_fine, lo68, hi68, color=color, alpha=0.35, label="68% CI")
        ax_lc.plot(t_fine, med, color=color, lw=1.5, label="posterior median")
        ax_lc.errorbar(
            d["t"], d["y"], yerr=d["yerr"], fmt="o", ms=4, color="k",
            ecolor="k", alpha=0.7, capsize=3, elinewidth=1.2, capthick=1.2, label="data",
        )
        ax_lc.set_ylabel(f"{name}\n({d['wavelength']:.0f} \u00c5)")
        ax_lc.grid(alpha=0.25)
        if row == 0:
            ax_lc.legend(fontsize=8, ncol=2, loc="upper right")

        psis = psi_samples[name]
        plo95, plo68, pmed, phi68, phi95 = np.percentile(psis, [2.5, 16, 50, 84, 97.5], axis=0)
        ax_psi.fill_between(tau_grid, plo95, phi95, color=color, alpha=0.15)
        ax_psi.fill_between(tau_grid, plo68, phi68, color=color, alpha=0.35)
        ax_psi.plot(tau_grid, pmed, color=color, lw=1.5)
        ax_psi.set_ylabel(r"$\psi(\tau)$")
        ax_psi.grid(alpha=0.25)

    axes[-1, 0].set_xlabel("time [days]")
    axes[-1, 1].set_xlabel(r"$\tau$ [days]")
    fig.suptitle("Posterior-predictive light curves and response functions")
    fig.tight_layout()
    return fig, axes


def plot_mcmc_diagnostics(
    samples: Dict[str, np.ndarray],
    param_names: Optional[Sequence[str]] = None,
    max_params: int = 12,
):
    """Trace plots per scalar parameter.

    Parameters
    ----------
    samples : dict
        ``mcmc.get_samples(group_by_chain=True)`` output, i.e.
        ``{name: array of shape (n_chains, n_samples, ...)}``. Non-scalar
        (vector) sites -- e.g. the per-frequency driver coefficients ``S``,
        ``C`` -- are summarized by their mean across the vector dimension
        so the figure stays readable.
    param_names : sequence of str, optional
        Restrict to these parameter names; defaults to all scalar-friendly
        sites in ``samples``, capped at ``max_params``.
    """
    if param_names is None:
        param_names = list(samples.keys())[:max_params]

    n = len(param_names)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, param_names):
        arr = np.asarray(samples[name])
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1).mean(axis=-1)
        n_chains = arr.shape[0]
        for c in range(n_chains):
            ax.plot(arr[c], lw=0.7, alpha=0.8, label=f"chain {c}" if len(param_names) == 1 else None)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("sample")
    fig.suptitle("MCMC trace diagnostics")
    fig.tight_layout()
    return fig, axes
