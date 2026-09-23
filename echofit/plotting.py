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


def _band_colours(bands: Dict[str, dict]):
    """Consistent colour per band, ordered and coloured by wavelength."""
    ordered = sorted(bands.items(), key=lambda kv: kv[1]["wavelength"])
    wavelengths = np.array([d["wavelength"] for _, d in ordered])
    norm = plt.Normalize(wavelengths.min(), wavelengths.max())
    cmap = plt.get_cmap("plasma_r")
    colours = {name: cmap(norm(d["wavelength"])) for name, d in ordered}
    return ordered, colours


def plot_raw_lightcurves(bands: Dict[str, dict], driver: Optional[dict] = None, figsize_per_panel=(7, 1.8)):
    """One panel per band (no overlay), ordered shortest -> longest wavelength,
    plus a driver panel first if given.

    Parameters
    ----------
    bands : dict
        ``{band_name: {"t", "y", "yerr", "wavelength"}}``.
    driver : dict, optional
        ``{"t", "y", "yerr"}`` for a light curve directly observing the
        driver (see ``EchoFit.add_driver_lightcurve``), plotted in its own
        panel first, ahead of the wavelength-ordered bands.
    """
    ordered, colours = _band_colours(bands)
    n = len(ordered) + (1 if driver is not None else 0)
    fig, axes = plt.subplots(
        n, 1, figsize=(figsize_per_panel[0], figsize_per_panel[1] * n), sharex=True
    )
    if n == 1:
        axes = [axes]

    if driver is not None:
        ax = axes[0]
        ax.errorbar(
            driver["t"], driver["y"], yerr=driver["yerr"], fmt="o", ms=4, color="0.2",
            ecolor="0.2", alpha=0.85, capsize=3, elinewidth=1.2, capthick=1.2,
        )
        ax.set_ylabel("driver")
        ax.grid(alpha=0.25)
        axes = axes[1:]

    for ax, (name, d) in zip(axes, ordered):
        ax.errorbar(
            d["t"], d["y"], yerr=d["yerr"], fmt="o", ms=4, color=colours[name],
            ecolor=colours[name], alpha=0.85, capsize=3, elinewidth=1.2, capthick=1.2,
        )
        ax.set_ylabel(f"{name}\n({d['wavelength']:.0f} \u00c5)")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("time [days]")
    fig.suptitle("Raw multi-band light curves")
    fig.tight_layout()
    return fig, fig.axes


def plot_lightcurve_fits(
    bands: Dict[str, dict],
    t_fine: np.ndarray,
    y_pred_samples: Dict[str, np.ndarray],
    tau_grid: np.ndarray,
    psi_samples: Dict[str, np.ndarray],
    driver_samples: Optional[np.ndarray] = None,
    driver_points: Optional[tuple] = None,
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
    driver_points : (t, X_implied, yerr_implied) tuple, optional
        A registered driver light curve's own data, back-transformed
        through the posterior-mean ``S_driver``/``C_driver`` into the same
        units as ``driver_samples`` (i.e. ``(y - C_driver) / S_driver``) so
        it can be overlaid on the driver panel as a direct cross-check that
        the inferred X(t) actually tracks what was observed.
    """
    ordered, colours = _band_colours(bands)
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
        if driver_points is not None:
            t_d, X_d, yerr_d = driver_points
            ax_drv.errorbar(
                t_d, X_d, yerr=yerr_d, fmt="o", ms=4, color="C3",
                ecolor="C3", alpha=0.7, capsize=3, elinewidth=1.2, capthick=1.2,
                label="driver data (back-transformed)",
            )
            ax_drv.legend(fontsize=7, loc="upper right")
        ax_drv.set_ylabel("driver\nX(t)")
        ax_drv.set_title("Inferred driving light curve", fontsize=10)
        ax_drv.grid(alpha=0.25)
        ax_drv.sharex(axes[1, 0])
        ax_drv_unused.axis("off")

    for row, (name, d) in enumerate(ordered):
        ax_lc, ax_psi = axes[row + has_driver, 0], axes[row + has_driver, 1]
        colour = colours[name]

        preds = y_pred_samples[name]
        lo95, lo68, med, hi68, hi95 = np.percentile(preds, [2.5, 16, 50, 84, 97.5], axis=0)

        ax_lc.fill_between(t_fine, lo95, hi95, color=colour, alpha=0.15, label="95% CI")
        ax_lc.fill_between(t_fine, lo68, hi68, color=colour, alpha=0.35, label="68% CI")
        ax_lc.plot(t_fine, med, color=colour, lw=1.5, label="posterior median")
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
        ax_psi.fill_between(tau_grid, plo95, phi95, color=colour, alpha=0.15)
        ax_psi.fill_between(tau_grid, plo68, phi68, color=colour, alpha=0.35)
        ax_psi.plot(tau_grid, pmed, color=colour, lw=1.5)
        ax_psi.set_ylabel(r"$\psi(\tau)$")
        ax_psi.grid(alpha=0.25)

    axes[-1, 0].set_xlabel("time [days]")
    axes[-1, 1].set_xlabel(r"$\tau$ [days]")
    fig.suptitle("Posterior-predictive light curves and response functions")
    fig.tight_layout()
    return fig, axes


def plot_power_spectrum(
    freqs: np.ndarray,
    S_samples: np.ndarray,
    C_samples: np.ndarray,
    sigma_drw_samples: np.ndarray,
    tau_drw_samples: np.ndarray,
    figsize=(7, 5),
):
    """Posterior driver power spectrum vs. the DRW prior it was drawn under.

    The empirical periodogram-style estimate per posterior draw is
    ``P(w_k) = (S_k**2 + C_k**2) / (2 * dw_k)``, where ``dw_k`` is the local
    frequency-grid spacing -- matching how ``model.drw_prior_scale`` sets
    ``Var(S_k) = Var(C_k) = power(w_k) * dw_k`` in the first place, so this
    is directly comparable to the Lorentzian ``power(w)`` curve overlaid
    from the same posterior draws' ``sigma_drw``/``tau_drw``. Since ``freqs``
    is log-spaced (geomspace), skipping the ``dw_k`` normalisation would
    flatten the apparent log-log slope purely from the growing bin width at
    high frequency -- not a real physical effect.

    A DRW's power spectrum is Lorentzian, ``power(w) = sigma_drw**2 *
    tau_drw / (1 + (w*tau_drw)**2)``: flat for ``w << 1/tau_drw``, and
    falling off as ``w**-2`` (pure random walk / Brownian motion) for
    ``w >> 1/tau_drw``. A dashed reference line at that ``-2`` slope is
    overlaid so the high-frequency asymptote is easy to eyeball.

    Parameters
    ----------
    freqs : (n_freq,) array
        Driver angular frequency grid (rad/day).
    S_samples, C_samples : (n_samples, n_freq) array
        Posterior draws of the driver's sine/cosine Fourier coefficients.
    sigma_drw_samples, tau_drw_samples : (n_samples,) array
        Posterior draws of the DRW hyperparameters.
    """
    dw = np.clip(np.gradient(freqs), 1e-8, None)
    P_samples = (S_samples ** 2 + C_samples ** 2) / (2.0 * dw[None, :])
    plo95, plo68, pmed, phi68, phi95 = np.percentile(P_samples, [2.5, 16, 50, 84, 97.5], axis=0)

    fit_power = (
        sigma_drw_samples[:, None] ** 2 * tau_drw_samples[:, None]
        / (1.0 + (freqs[None, :] * tau_drw_samples[:, None]) ** 2)
    )
    flo95, flo68, fmed, fhi68, fhi95 = np.percentile(fit_power, [2.5, 16, 50, 84, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(freqs, plo95, phi95, color="0.6", alpha=0.15, label="posterior P(w) 95% CI")
    ax.fill_between(freqs, plo68, phi68, color="0.6", alpha=0.3, label="posterior P(w) 68% CI")
    ax.plot(freqs, pmed, color="k", lw=1.5, label="posterior P(w) median")

    ax.fill_between(freqs, flo95, fhi95, color="C0", alpha=0.12)
    ax.plot(freqs, fmed, color="C0", lw=1.5, ls="--", label="fitted DRW prior (Lorentzian)")

    w_ref = np.sqrt(freqs[0] * freqs[-1])
    P_ref = np.interp(w_ref, freqs, pmed)
    w_line = freqs[freqs >= w_ref]
    ax.plot(
        w_line, P_ref * (w_line / w_ref) ** -2, color="C3", lw=1.2, ls=":",
        label=r"$\omega^{-2}$ (random-walk asymptote)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\omega$ [rad/day]")
    ax.set_ylabel(r"$P(\omega)$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    ax.set_title("Driver power spectrum: posterior vs. fitted DRW prior")
    fig.tight_layout()
    return fig, ax


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
        ``C`` -- are summarised by their mean across the vector dimension
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
