"""
echofit.py
==========

``EchoFit`` is the main user-facing class: collect light curves, build the
shared frequency/lag grids, run NUTS, and plot the results.

Optionally manages on-disk run outputs -- pass ``title=`` to enable it.
``.fit()`` then writes the light curve data, config, periodic checkpoints,
final posterior (as an ArviZ netCDF), and a visual report.html to
``<output_root>/<title>/run_<timestamp>/``. An interrupted fit can be
picked back up with ``EchoFit.resume(title)``. See ``run_manager.py`` for
the on-disk layout and output-directory resolution rules. Without
``title``, EchoFit behaves exactly as a purely in-memory fit -- nothing is
written to disk.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp

from .model import reverberation_model
from .inference import run_mcmc, run_mcmc_chunked
from .forward_model import response_function, transfer_coeffs, compute_echo, driver_at
from .grid_utils import estimate_dt_min
from . import plotting
from . import reporting
from . import run_manager

_UNSET = object()


def _merge_dicts(dicts) -> dict:
    """Concatenate (along axis 0) a list of ``{name: array}`` dicts, skipping
    any that are empty (e.g. "no prior checkpoint to merge in")."""
    dicts = [d for d in dicts if d]
    if not dicts:
        return {}
    return {k: np.concatenate([d[k] for d in dicts], axis=0) for k in dicts[0]}


class EchoFit:
    """Bayesian AGN reverberation-mapping fit for multi-band light curves.

    Parameters
    ----------
    M_BH : float
        Fixed black hole mass, solar masses. Never inferred.
    title : str, optional
        A name for this fit (e.g. an AGN name like ``"ngc_5548"``). If
        given, ``.fit()`` writes its outputs to
        ``<output_root>/<title>/run_<timestamp>/`` -- see
        :func:`~echofit.run_manager.resolve_output_root`. If omitted,
        nothing is written to disk (the original, fully in-memory
        behavior).
    output_dir : str, optional
        Override the output root directory. Only relevant when ``title``
        is given. Otherwise resolved from the ``ECHOFIT_OUTPUT_DIR``
        environment variable, or ``./outputs`` if that's unset too.

    Examples
    --------
    >>> ef = EchoFit(M_BH=1e8, title="ngc_5548")
    >>> ef.add_lightcurve("g", wavelength=4770.0, t=t_g, y=y_g, yerr=yerr_g)
    >>> ef.add_lightcurve("i", wavelength=7625.0, t=t_i, y=y_i, yerr=yerr_i)
    >>> ef.build_grid()
    >>> ef.fit(num_warmup=1000, num_samples=1000)  # writes outputs/ngc_5548/run_.../
    >>> ef.plot_lightcurve_fits()

    If that fit is interrupted (killed, crashed, ...), resume it with::

    >>> ef = EchoFit.resume("ngc_5548")
    >>> ef.fit()  # continues from the last checkpoint, same settings as before
    """

    def __init__(self, M_BH: float, title: Optional[str] = None, output_dir: Optional[str] = None):
        self.M_BH = float(M_BH)
        self.title = title
        self.bands: Dict[str, dict] = {}
        self.freqs: Optional[np.ndarray] = None
        self.tau_grid: Optional[np.ndarray] = None
        self.mcmc = None
        self.samples: Optional[dict] = None
        self.extra_fields: dict = {}

        self._output_root = run_manager.resolve_output_root(output_dir) if title else None
        self.run_dir: Optional[Path] = None
        self._samples_by_chain: Optional[dict] = None
        self._resume_state = None
        self._fit_config: Optional[dict] = None

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

    # ------------------------------------------------------------------
    @classmethod
    def resume(cls, title: str, run_id: str = "latest", output_dir: Optional[str] = None) -> "EchoFit":
        """Load a previous (possibly interrupted) run's data/config/progress
        so ``.fit()`` continues it instead of starting over.

        Parameters
        ----------
        title : str
            The run title it was originally created with.
        run_id : str
            ``"latest"`` (default) picks the most recent run under that
            title, or pass an exact ``"run_<timestamp>"`` string.
        output_dir : str, optional
            Same output-root override as ``EchoFit(..., output_dir=...)``.
        """
        output_root = run_manager.resolve_output_root(output_dir)
        run_dir = run_manager.find_run_dir(output_root, title, run_id)
        manifest = run_manager.load_json(run_dir / "manifest.json")

        ef = cls(M_BH=manifest["M_BH"], title=title, output_dir=output_dir)
        ef.run_dir = run_dir
        ef._fit_config = manifest["fit_config"]

        bands = run_manager.load_bands_npz(run_dir / "data.npz")
        for name, d in bands.items():
            ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])

        grid = run_manager.load_samples_npz(run_dir / "grid.npz")
        ef.freqs = jnp.asarray(grid["freqs"])
        ef.tau_grid = jnp.asarray(grid["tau_grid"])

        checkpoint_dir = run_dir / "checkpoint"
        state_path = checkpoint_dir / "state.pkl"
        if state_path.exists():
            samples = run_manager.load_samples_npz(checkpoint_dir / "samples.npz")
            extra_fields = run_manager.load_samples_npz(checkpoint_dir / "extra_fields.npz")
            n_done = int(next(iter(samples.values())).shape[0])
            ef._resume_state = dict(
                last_state=run_manager.load_state(state_path),
                samples=samples, extra_fields=extra_fields, n_done=n_done,
            )
            print(f"Found checkpoint: {n_done}/{ef._fit_config['num_samples']} samples already collected.")
        else:
            print("No checkpoint found (likely interrupted during warmup) -- restarting this run from scratch.")

        return ef

    # ------------------------------------------------------------------
    def fit(
        self,
        rng_seed=_UNSET,
        num_warmup=_UNSET,
        num_samples=_UNSET,
        num_chains: int = 1,
        max_tree_depth=_UNSET,
        chain_method: str = "parallel",
        checkpoint_every=_UNSET,
        progress_bar: bool = True,
        generate_report: bool = True,
    ):
        """Run NUTS and store the posterior samples on ``self.samples``.

        If this instance has no ``title``, this is a single non-resumable
        in-memory run (the original behavior) -- ``num_chains``/
        ``chain_method`` apply normally.

        If ``title`` was given (directly, or via ``.resume()``), this
        instead runs single-chain NUTS in checkpointed chunks of
        ``checkpoint_every`` samples (``num_chains``/``chain_method`` are
        ignored; a warning is raised if ``num_chains != 1``), saving
        progress after every chunk so an interrupted fit can be resumed
        with ``EchoFit.resume(title)``. When complete, writes the final
        posterior (as ``chains.nc``, an ArviZ ``InferenceData``) and (if
        ``generate_report``) the same plots + report.html as
        ``scripts/smoke_test.py`` to the run directory.

        After ``.resume()``, any arguments left unset here reuse the
        original run's settings (so ``ef.fit()`` with no arguments "just
        continues"); pass a value explicitly to override it.
        """
        if self.freqs is None or self.tau_grid is None:
            self.build_grid()

        if self.title is None:
            rng_seed = 0 if rng_seed is _UNSET else rng_seed
            num_warmup = 1000 if num_warmup is _UNSET else num_warmup
            num_samples = 1000 if num_samples is _UNSET else num_samples
            max_tree_depth = None if max_tree_depth is _UNSET else max_tree_depth
            rng_key = jax.random.PRNGKey(rng_seed)
            self.mcmc = run_mcmc(
                reverberation_model, self._model_kwargs(), rng_key,
                num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                max_tree_depth=max_tree_depth, chain_method=chain_method, progress_bar=progress_bar,
            )
            self.samples = self.mcmc.get_samples()
            self._samples_by_chain = self.mcmc.get_samples(group_by_chain=True)
            self.extra_fields = self.mcmc.get_extra_fields()
            return self

        # -- title given: checkpointed/resumable single-chain path --
        if num_chains != 1:
            warnings.warn(
                f"EchoFit(title=...) checkpointing only supports num_chains=1; "
                f"ignoring num_chains={num_chains}."
            )

        def _pick(value, key, default):
            if value is not _UNSET:
                return value
            if self._fit_config is not None and key in self._fit_config:
                return self._fit_config[key]
            return default

        rng_seed = _pick(rng_seed, "rng_seed", 0)
        num_warmup = _pick(num_warmup, "num_warmup", 1000)
        num_samples = _pick(num_samples, "num_samples", 1000)
        max_tree_depth = _pick(max_tree_depth, "max_tree_depth", None)
        checkpoint_every = _pick(checkpoint_every, "checkpoint_every", 100)

        if self.run_dir is None:
            self.run_dir = run_manager.new_run_dir(self._output_root, self.title)
        checkpoint_dir = self.run_dir / "checkpoint"
        manifest_path = self.run_dir / "manifest.json"

        if self._resume_state is not None:
            init_last_state = self._resume_state["last_state"]
            prev_samples = self._resume_state["samples"]
            prev_extra = self._resume_state["extra_fields"]
            n_already_done = self._resume_state["n_done"]
            print(
                f"Resuming '{self.title}' from {self.run_dir} "
                f"({n_already_done}/{num_samples} samples already collected)."
            )
        else:
            init_last_state = None
            prev_samples, prev_extra, n_already_done = {}, {}, 0
            self._fit_config = dict(
                rng_seed=rng_seed, num_warmup=num_warmup, num_samples=num_samples,
                max_tree_depth=max_tree_depth, checkpoint_every=checkpoint_every,
            )
            if not manifest_path.exists():
                run_manager.save_bands_npz(self.run_dir / "data.npz", self.bands)
                run_manager.save_samples_npz(
                    self.run_dir / "grid.npz",
                    dict(freqs=np.asarray(self.freqs), tau_grid=np.asarray(self.tau_grid)),
                )
                run_manager.save_json(manifest_path, dict(
                    title=self.title, M_BH=self.M_BH,
                    bands={n: d["wavelength"] for n, d in self.bands.items()},
                    fit_config=self._fit_config,
                ))

        chunk_samples_so_far, chunk_extra_so_far = [], []

        def _on_chunk_done(mcmc, last_state, n_done_total):
            chunk_samples_so_far.append(mcmc.get_samples())
            chunk_extra_so_far.append(mcmc.get_extra_fields())
            merged_samples = _merge_dicts([prev_samples] + chunk_samples_so_far)
            merged_extra = _merge_dicts([prev_extra] + chunk_extra_so_far)
            run_manager.save_samples_npz(checkpoint_dir / "samples.npz", merged_samples)
            run_manager.save_samples_npz(checkpoint_dir / "extra_fields.npz", merged_extra)
            run_manager.save_state(checkpoint_dir / "state.pkl", last_state)
            print(f"  checkpoint: {n_done_total}/{num_samples} samples saved.")

        t0 = time.time()
        rng_key = jax.random.PRNGKey(rng_seed)
        samples_this_call, samples_by_chain_this_call, extra_this_call, _ = run_mcmc_chunked(
            reverberation_model, self._model_kwargs(), rng_key,
            num_warmup=num_warmup, num_samples=num_samples, checkpoint_every=checkpoint_every,
            max_tree_depth=max_tree_depth, progress_bar=progress_bar,
            init_last_state=init_last_state, n_already_done=n_already_done,
            on_chunk_done=_on_chunk_done,
        )
        fit_seconds = time.time() - t0

        self.samples = _merge_dicts([prev_samples, samples_this_call])
        self.extra_fields = _merge_dicts([prev_extra, extra_this_call])
        self._samples_by_chain = {k: v[None, ...] for k, v in self.samples.items()}
        self.mcmc = None  # no single mcmc object spans all chunks in this path

        self._save_chains(self.run_dir)
        if generate_report:
            reporting.generate_report(
                self, self.run_dir, fit_seconds=fit_seconds, title=self.title
            )
            print(f"Report written to {self.run_dir / 'report.html'}")

        return self

    def _save_chains(self, run_dir):
        """Save the final posterior. ``chains.npz`` (plain numpy, no extra
        dependencies) is always written; ``chains.nc`` (an ArviZ
        InferenceData, more standard for further MCMC analysis/diagnostics)
        is best-effort -- skipped with a warning if a netCDF backend isn't
        available, rather than failing an otherwise-successful fit."""
        posterior = {
            k: v for k, v in self._samples_by_chain.items() if not k.startswith("y_pred_")
        }
        # drop the leading (num_chains=1) axis: (1, n_samples, ...) -> (n_samples, ...)
        run_manager.save_samples_npz(run_dir / "chains.npz", {k: v[0] for k, v in posterior.items()})
        try:
            import arviz as az

            az.from_dict(posterior=posterior).to_netcdf(str(run_dir / "chains.nc"))
        except Exception as e:
            warnings.warn(f"Could not write chains.nc (ArviZ/netCDF backend issue): {e}")

    # ------------------------------------------------------------------
    def plot_raw_lightcurves(self, **kwargs):
        return plotting.plot_raw_lightcurves(self.bands, **kwargs)

    def plot_power_spectrum(self, **kwargs):
        """Posterior driver power spectrum vs. the fitted DRW prior shape.

        Sanity check that the driver's Fourier coefficients (S, C) are
        actually behaving like a DRW under the posterior, not just the
        prior: P(w) = (S**2 + C**2) / (2*dw) should track the fitted
        Lorentzian (from posterior sigma_drw/tau_drw draws) and flatten
        into a w**-2 slope above 1/tau_drw.
        """
        if self.samples is None:
            raise RuntimeError("Call .fit() before plotting the power spectrum.")
        return plotting.plot_power_spectrum(
            np.asarray(self.freqs),
            np.asarray(self.samples["S"]),
            np.asarray(self.samples["C"]),
            np.asarray(self.samples["sigma_drw"]),
            np.asarray(self.samples["tau_drw"]),
            **kwargs,
        )

    def plot_mcmc_diagnostics(self, param_names=None, **kwargs):
        if self._samples_by_chain is None:
            raise RuntimeError("Call .fit() before plotting diagnostics.")
        samples_by_chain = self._samples_by_chain
        if param_names is None:
            scalar_like = [
                k for k in samples_by_chain
                if not k.startswith("y_pred_") and k not in ("S", "C")
            ]
            param_names = scalar_like
        return plotting.plot_mcmc_diagnostics(samples_by_chain, param_names=param_names, **kwargs)

    def plot_lightcurve_fits(
        self, n_fine: int = 200, n_pred_samples: int = 200, extrapolate_days: float = 30.0, **kwargs
    ):
        """Draw posterior-predictive light curves and response functions.

        Subsamples up to ``n_pred_samples`` posterior draws for speed, and
        evaluates them (vectorized with ``jax.vmap``) on a dense time grid
        per band plus the shared lag grid. This all happens *after* ``.fit()``
        -- extending ``extrapolate_days`` does not slow down NUTS, only the
        (cheap, matrix-multiply) posterior-predictive evaluation here.

        Parameters
        ----------
        extrapolate_days : float
            Extend the plotted time range this far (days) before the first
            and after the last observation, so the credible band's growth
            outside the data is visible (for a DRW-like driver this should
            look roughly like a t^(1/2) widening before saturating).
        """
        if self.samples is None:
            raise RuntimeError("Call .fit() before plotting fits.")

        all_t = np.concatenate([d["t"] for d in self.bands.values()])
        t_fine = jnp.linspace(
            all_t.min() - extrapolate_days, all_t.max() + extrapolate_days, n_fine
        )

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
