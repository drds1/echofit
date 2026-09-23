# echofit

Bayesian modelling of AGN reverberation-mapping light curves as a delayed,
smoothed echo of an unobserved driving (lamppost) X-ray light curve, built on
[JAX](https://github.com/google/jax) + [NumPyro](https://num.pyro.ai/).

## Model

Each band's observed light curve is modelled as

```
y_band(t) = S_band * ∫ X(t - τ) ψ(τ, λ_band, θ) dτ + C_band + ε
```

- **Driver `X(t)`**: a damped random walk (DRW), represented as a truncated
  Fourier series `X(t) = Σ_k [S_k sin(w_k t) + C_k cos(w_k t)]` on a fixed
  frequency grid. The sine/cosine amplitudes `S_k, C_k` are given Gaussian
  priors matching the DRW's Lorentzian power spectrum, so the two DRW
  hyperparameters `sigma_drw` (variability amplitude) and `tau_drw` (damping
  timescale) are inferred directly alongside the amplitudes.
- **Response `ψ(τ, λ, θ)`**: a causal (`τ ≥ 0`), positive, skew-normal
  function. Its mean lag follows the standard thin-disk reprocessing scaling

  ```
  τ_mean ∝ (M_BH)^(2/3) * (Ṁ)^(1/3) * λ^(4/3)
  ```

  with `M_BH` **fixed** (not inferred) and `log_mdot` (mass accretion rate)
  inferred. Inclination controls only the *skewness* of the response, never
  the mean lag.
- **Convolution**: because the driver is exactly a Fourier series, the
  convolution `∫ ψ(τ) X(t-τ) dτ` has a closed form in terms of the response
  function's own Fourier transform, evaluated once per driver frequency
  (`A_k = ∫ ψ cos(w_k τ) dτ`, `B_k = ∫ ψ sin(w_k τ) dτ`). Evaluating the echo
  at any set of observation times is then a single vectorized matrix
  contraction — no loop over `(t_obs, τ)` pairs, and no loop over bands.

Only these are inferred: `log_mdot`, `inclination`, `sigma_drw`, `tau_drw`,
the driver Fourier coefficients `{S_k, C_k}`, and per-band `{S_band, C_band}`.
**`M_BH` is always a fixed input.**

## Package layout

```
echofit/
    __init__.py        public API (EchoFit, forward_model helpers, synthetic data)
    forward_model.py    lag_scaling, response_function, transfer_coeffs, compute_echo
    model.py            NumPyro model (reverberation_model) + DRW prior scale
    grid_utils.py        estimate_dt_min: robust cadence estimate shared by
                          EchoFit.build_grid() and synthetic.py
    inference.py         run_mcmc / run_mcmc_chunked: NUTS wrapper (the
                          latter supports checkpointing/resuming)
    echofit.py           EchoFit: main user-facing class
    plotting.py          plot_raw_lightcurves, plot_lightcurve_fits, plot_power_spectrum,
                          plot_mcmc_diagnostics
    reporting.py          generate_report: shared plots + report.html generation,
                          used by both EchoFit(title=...) and scripts/smoke_test.py
    run_manager.py        on-disk run layout, output-dir resolution, checkpoint
                          save/load (see "Fitting your own light curves" below)
    synthetic.py          generate_synthetic_dataset for tests / the demo notebook
notebooks/
    demo.ipynb            end-to-end synthetic-data demo
scripts/
    smoke_test.py          quick visual sanity check (see below)
tests/
    test_forward_model.py  basic sanity checks on the forward model
    test_recovery.py       end-to-end MCMC recovery test on synthetic data
    test_run_manager.py    checkpointing + resume-after-interruption tests
```

## Install

```bash
pip install -e ".[dev]"
```

Requires a working JAX install (CPU is fine for the demo; see the
[JAX install guide](https://github.com/google/jax#installation) for GPU/TPU).

## Quickstart

```python
from echofit import EchoFit, generate_synthetic_dataset

data = generate_synthetic_dataset(M_BH=1e8)

ef = EchoFit(M_BH=1e8)
for name, d in data["bands"].items():
    ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])

ef.build_grid(n_freq=60, n_tau=400)
ef.fit(rng_seed=0, num_warmup=500, num_samples=500)

ef.plot_raw_lightcurves()
ef.plot_lightcurve_fits()
ef.plot_power_spectrum()
ef.plot_mcmc_diagnostics()
```

See `notebooks/demo.ipynb` for the full walkthrough.

## Fitting your own light curves, with saved/resumable runs

Pass `title=` (e.g. an AGN name) to have `EchoFit` manage on-disk outputs
for the run -- data, config, periodic checkpoints, the final posterior, and
the same visual report as the smoke test:

```python
ef = EchoFit(M_BH=1e8, title="ngc_5548")
ef.add_lightcurve("g", wavelength=4770.0, t=t_g, y=y_g, yerr=yerr_g)
ef.add_lightcurve("i", wavelength=7625.0, t=t_i, y=y_i, yerr=yerr_i)
ef.build_grid()
ef.fit(num_warmup=1000, num_samples=2000)   # writes outputs/ngc_5548/run_<timestamp>/
```

This writes `outputs/ngc_5548/run_YYYYMMDD_HHMMSS/`, containing:

```
manifest.json           run config (for resuming / for your own records)
data.npz                the light curve data you registered
grid.npz                the frequency/lag grids from build_grid()
checkpoint/              progress saved every checkpoint_every samples
chains.npz               final posterior samples (plain numpy, no extra deps)
chains.nc                same, as an ArviZ InferenceData (best-effort --
                          skipped with a warning if no netCDF backend is
                          available)
report.html + *.png      the same visual report as scripts/smoke_test.py
```

**If a fit is interrupted** (killed, crashed, machine restarted), resume it
in a new process with:

```python
ef = EchoFit.resume("ngc_5548")     # picks the latest run by default
ef.fit()                             # continues from the last checkpoint,
                                      # reusing the original run's settings
```

Notes on scope: checkpointing/resuming only covers the *sampling* phase --
if interrupted during warmup (before the first `checkpoint_every` samples
are collected), resuming restarts the fit, reusing the same run directory
rather than creating a new one. It also only supports `num_chains=1` (a
warning is raised otherwise); run multiple independent single-chain fits
if you want chains for R-hat/ESS diagnostics with resume support.

**Output location**, in priority order: the `output_dir=` argument to
`EchoFit`/`EchoFit.resume()`, then the `ECHOFIT_OUTPUT_DIR` environment
variable, then `./outputs` (relative to wherever you run your script from).
`outputs/` is gitignored by default.

Without `title=`, `EchoFit` behaves exactly as in the Quickstart above --
nothing is written to disk.

## Visual smoke test

For a quick "did I break anything" check after touching `forward_model.py`,
`model.py`, or `echofit.py`, run a short fit on synthetic data and save plots
of the raw data, the inferred driving light curve (extended 30 days before/
after the data -- its credible band should widen there before saturating,
since the driver is DRW-like), the posterior-predictive echo fit + response
function per band, the driver's power spectrum against the fitted DRW prior
and the w^-2 random-walk asymptote, and MCMC trace diagnostics:

```bash
python scripts/smoke_test.py                # ~30-50s, 300 warmup + 300 samples
python scripts/smoke_test.py --num-warmup 100 --num-samples 100   # faster, noisier
python scripts/smoke_test.py --no-gaps       # fully uniform random sampling instead
```

By default the synthetic campaign includes two observing gaps (2 weeks from
day 50, 3 weeks from day 150) to make sampling irregular/harder, closer to a
real campaign than uniform random sampling.

Writes PNGs and a `report.html` (open it to see everything in one page) to
`smoke_test_output/`. This is a visual/eyeball check, not a pass/fail test —
for that, see `tests/test_recovery.py`.

## Swapping the response function

`forward_model.response_function` is the single place the response shape
lives. To try a different parametric family (e.g. a top-hat, a Gamma
response, a two-component response), write a new function with the same
signature — `(tau_grid, log_mdot, wavelength, inclination, M_BH, ...) -> psi`
returning a causal, area-normalized array on `tau_grid` — and pass it into
`model.reverberation_model` in place of the default import. Nothing else
(`transfer_coeffs`, `compute_echo`, the plotting code) needs to change.

## Status / caveats

This is a research scaffold, not a validated production pipeline:

- The Fourier-series driver with DRW-matched priors is an approximation to a
  true DRW Gaussian process (a spectral / Hilbert-space GP approximation),
  not an exact DRW likelihood (e.g. via a Kalman filter). It's fast and
  differentiable, which is the point, but you should sanity-check recovered
  `sigma_drw` / `tau_drw` against known DRW literature values for your
  targets.
- The skew-normal response is one reasonable causal, positive, skewable
  parametric family; it is not derived from full disk radiative-transfer
  physics.
- The synthetic test in `generate_synthetic_dataset` uses the *same*
  forward model to generate and fit data (a "self-consistency" check), which
  validates the code but is not a substitute for validation against real
  reverberation-mapping campaigns or independent simulations.
- On the synthetic recovery test in `tests/test_recovery.py`, `log_mdot`
  (which sets the mean lag) recovers well, but `inclination`, `sigma_drw`,
  and `tau_drw` recover only loosely (wide/biased posteriors) even with zero
  divergent transitions — NUTS tends to spend most samples at its
  max-tree-depth ceiling on this model. Treat those three parameters'
  posteriors with extra skepticism on real data until this is investigated
  further; `EchoFit.fit()` exposes `max_tree_depth` and `chain_method` if
  you want to bound worst-case sampling cost or add cheap diagnostic chains
  while doing so.
