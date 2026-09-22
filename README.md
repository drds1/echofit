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
    inference.py         run_mcmc: thin NUTS/MCMC wrapper
    echofit.py           EchoFit: main user-facing class
    plotting.py          plot_raw_lightcurves, plot_lightcurve_fits, plot_mcmc_diagnostics
    synthetic.py          generate_synthetic_dataset for tests / the demo notebook
notebooks/
    demo.ipynb            end-to-end synthetic-data demo
tests/
    test_forward_model.py  basic sanity checks on the forward model
    test_recovery.py       end-to-end MCMC recovery test on synthetic data
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
ef.plot_mcmc_diagnostics()
```

See `notebooks/demo.ipynb` for the full walkthrough.

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
