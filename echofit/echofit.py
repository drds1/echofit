"""
echofit.py
==========

``EchoFit`` is the main user-facing class: collect light curves, build the
shared frequency/lag grids, run NUTS, and plot the results.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp

from .model import reverberation_model
from .inference import run_mcmc
from .forward_model import response_function, transfer_coeffs, compute_echo, driver_at
from .grid_utils import estimate_dt_min
from . import plotting


class EchoFit:
    """Bayesian AGN reverberation-mapping fit for multi-band light curves.

    Parameters
    ----------
    M_BH : float
        Fixed black hole mass, solar masses. Never inferred.

    Examples
    --------
    >>> ef = EchoFit(M_BH=1e8)
    >>> ef.add_lightcurve("g", wavelength=4770.0, t=t_g, y=y_g, yerr=yerr_g)
    >>> ef.add_lightcurve("i", wavelength=7625.0, t=t_i, y=y_i, yerr=yerr_i)
    >>> ef.build_grid()
    >>> ef.fit(rng_seed=0, num_warmup=500, num_samples=500)
    >>> ef.plot_lightcurve_fits()
    """

    def __init__(self, M_BH: float):
        self.M_BH = float(M_BH)
        self.bands: Dict[str, dict] = {}
        self.freqs: Optional[np.ndarray] = None
        self.tau_grid: Optional[np.ndarray] = None
        self.mcmc = None
        self.samples: Optional[dict] = None

    # ------------------------------------------------------------------
    def add_lightcurve(self, name: str, wavelength: float, t, y, yerr):
        """Register a single band's (possibly irregularly sampled) light curve."""
        t, y, yerr = np.asarray(t, float), np.asarray(y, float), np.asarray(yerr, float)
        order = np.argsort(t)
        self.bands[name] = {
            "t": t[order],
            "y": y[order],
            "yerr": yerr[order],
            "wavelength": float(wavelength),
        }
        return self

    # ------------------------------------------------------------------
    def build_grid(
        self,
        n_freq: int = 60,
        n_tau: int = 400,
        tau_max: Optional[float] = None,
        dt_min: Optional[float] = None,
    ):
        """Build the shared driver-frequency grid and lag grid from the
        currently registered light curves.

        Parameters
        ----------
        n_freq : int
            Number of driver Fourier frequencies.
        n_tau : int
            Number of points on the lag grid used to evaluate/integrate psi.
        tau_max : float, optional
            Maximum lag to consider (days). Defaults to half the observed
            time baseline, which is a generous ceiling for reprocessing
            lags relative to typical monitoring campaigns.
        dt_min : float, optional
            Finest timescale (days) the driver's Fourier series should
            resolve; sets the frequency grid's upper bound
            ``w_max = pi / dt_min``. Defaults to a robust (5th-percentile)
            estimate from the registered light curves' observation gaps
            via :func:`~echofit.grid_utils.estimate_dt_min` -- pass this
            explicitly if you want direct control (e.g. to match a known
            cadence) rather than relying on the data-driven estimate, which
            can be noisy for sparse or highly irregular sampling.
        """
        if not self.bands:
            raise ValueError("Add at least one light curve before build_grid().")

        all_t = np.concatenate([d["t"] for d in self.bands.values()])
        t_span = all_t.max() - all_t.min()
        if dt_min is None:
            dt_min = estimate_dt_min(
                (d["t"] for d in self.bands.values()), t_span=t_span
            )

        w_min = 2.0 * np.pi / t_span
        w_max = np.pi / dt_min
        self.freqs = jnp.asarray(np.geomspace(w_min, w_max, n_freq))

        if tau_max is None:
            tau_max = 0.5 * t_span
        self.tau_grid = jnp.asarray(np.linspace(0.0, tau_max, n_tau))
        return self

    # ------------------------------------------------------------------
    def _model_kwargs(self):
        bands_jax = {
            name: {
                "t": jnp.asarray(d["t"]),
                "y": jnp.asarray(d["y"]),
                "yerr": jnp.asarray(d["yerr"]),
                "wavelength": d["wavelength"],
            }
            for name, d in self.bands.items()
        }
        return dict(freqs=self.freqs, tau_grid=self.tau_grid, M_BH=self.M_BH, bands=bands_jax)

    def fit(
        self,
        rng_seed: int = 0,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 1,
        max_tree_depth: Optional[int] = None,
        chain_method: str = "parallel",
        progress_bar: bool = True,
    ):
        """Run NUTS and store the posterior samples on ``self.samples``.

        ``max_tree_depth`` and ``chain_method`` are passed straight through
        to :func:`~echofit.inference.run_mcmc` -- see its docstring. Useful
        knobs if a fit is spending most samples at the NUTS max-tree-depth
        ceiling (check ``ef.mcmc.get_extra_fields()["num_steps"]`` after a
        run made with ``extra_fields=("num_steps",)``) or if you want
        multiple chains for R-hat/ESS diagnostics without near-linear extra
        wall time (``chain_method="vectorized"``).
        """
        if self.freqs is None or self.tau_grid is None:
            self.build_grid()

        rng_key = jax.random.PRNGKey(rng_seed)
        self.mcmc = run_mcmc(
            reverberation_model,
            self._model_kwargs(),
            rng_key,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            max_tree_depth=max_tree_depth,
            chain_method=chain_method,
            progress_bar=progress_bar,
        )
        self.samples = self.mcmc.get_samples()
        return self

    # ------------------------------------------------------------------
    def plot_raw_lightcurves(self, **kwargs):
        return plotting.plot_raw_lightcurves(self.bands, **kwargs)

    def plot_mcmc_diagnostics(self, param_names=None, **kwargs):
        if self.mcmc is None:
            raise RuntimeError("Call .fit() before plotting diagnostics.")
        samples_by_chain = self.mcmc.get_samples(group_by_chain=True)
        if param_names is None:
            scalar_like = [
                k for k in samples_by_chain
                if not k.startswith("y_pred_") and k not in ("S", "C")
            ]
            param_names = scalar_like
        return plotting.plot_mcmc_diagnostics(samples_by_chain, param_names=param_names, **kwargs)

    def plot_lightcurve_fits(self, n_fine: int = 200, n_pred_samples: int = 200, **kwargs):
        """Draw posterior-predictive light curves and response functions.

        Subsamples up to ``n_pred_samples`` posterior draws for speed, and
        evaluates them (vectorized with ``jax.vmap``) on a dense time grid
        per band plus the shared lag grid.
        """
        if self.samples is None:
            raise RuntimeError("Call .fit() before plotting fits.")

        all_t = np.concatenate([d["t"] for d in self.bands.values()])
        t_fine = jnp.linspace(all_t.min(), all_t.max(), n_fine)

        n_total = self.samples["log_mdot"].shape[0]
        idx = np.random.default_rng(0).choice(
            n_total, size=min(n_pred_samples, n_total), replace=False
        )

        S = jnp.asarray(self.samples["S"])[idx]
        C = jnp.asarray(self.samples["C"])[idx]
        log_mdot = jnp.asarray(self.samples["log_mdot"])[idx]
        inclination = jnp.asarray(self.samples["inclination"])[idx]

        def single_draw(S_s, C_s, log_mdot_s, incl_s, wavelength, S_band_s, C_band_s):
            psi = response_function(
                self.tau_grid, log_mdot=log_mdot_s, wavelength=wavelength,
                inclination=incl_s, M_BH=self.M_BH,
            )
            A, B = transfer_coeffs(self.tau_grid, psi, self.freqs)
            echo = compute_echo(S_s, C_s, self.freqs, A, B, t_fine)
            y_pred = S_band_s * echo + C_band_s
            return y_pred, psi

        y_pred_samples, psi_samples = {}, {}
        for name, d in self.bands.items():
            S_band = jnp.asarray(self.samples[f"S_{name}"])[idx]
            C_band = jnp.asarray(self.samples[f"C_{name}"])[idx]
            y_pred, psi = jax.vmap(
                single_draw, in_axes=(0, 0, 0, 0, None, 0, 0)
            )(S, C, log_mdot, inclination, d["wavelength"], S_band, C_band)
            y_pred_samples[name] = np.asarray(y_pred)
            psi_samples[name] = np.asarray(psi)

        driver_samples = jax.vmap(
            lambda S_s, C_s: driver_at(S_s, C_s, self.freqs, t_fine)
        )(S, C)

        return plotting.plot_lightcurve_fits(
            self.bands, np.asarray(t_fine), y_pred_samples,
            np.asarray(self.tau_grid), psi_samples,
            driver_samples=np.asarray(driver_samples), **kwargs,
        )
