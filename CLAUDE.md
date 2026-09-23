# CLAUDE.md

Context for Claude (or any future contributor) picking this repo back up.

## What this is

`echofit`: a JAX + NumPyro package that fits multi-band AGN reverberation
light curves as a delayed, smoothed echo of an unobserved driving light
curve. Built from a single detailed spec; see `README.md` for the physics
and package layout.

## Key design decisions (don't relitigate without reason)

1. **Driver = Fourier series, not a literal DRW GP kernel.** `X(t) = Σ_k
   [S_k sin(w_k t) + C_k cos(w_k t)]` on a *fixed* frequency grid built once
   in `EchoFit.build_grid()`. `S_k, C_k` are deterministic transforms
   (`S_k = S_raw_k * prior_scale_k`, non-centred) of unit-Normal
   `S_raw, C_raw` NumPyro sample sites, with `prior_scale` set by the DRW's
   Lorentzian power spectrum (`model.drw_prior_scale`), parameterised by
   inferred `sigma_drw`, `tau_drw`. The Fourier-series driver was requested
   explicitly in the spec and also happens to make the whole model
   analytically convolvable (see next point) instead of needing a GP solve.
   The non-centred form is deliberate — sampling `S, C` directly
   ("centred") creates Neal's-funnel geometry against `sigma_drw`/`tau_drw`
   that pins NUTS near its max-tree-depth ceiling. Don't revert to centred
   without re-checking `tests/test_recovery.py`'s step-count/divergence
   behaviour.

2. **Convolution is closed-form, not numerical double-integration.**
   Because the driver is a sum of sinusoids, `∫ ψ(τ) X(t-τ) dτ` reduces to a
   sum over frequencies weighted by the response function's own Fourier
   transform (`forward_model.transfer_coeffs` → `A_k, B_k`), then
   `forward_model.compute_echo` is a single `(n_obs, n_freq)` matrix
   contraction. **Do not** replace this with a brute-force `(n_obs, n_tau)`
   trapezoidal double loop — that's both slower and was explicitly
   prohibited by the "no unnecessary loops over time" requirement.

3. **`M_BH` is always fixed, never a `numpyro.sample` site.** It's a plain
   Python float passed through `EchoFit(M_BH=...)` into
   `forward_model.lag_scaling` / `response_function` / `model.reverberation_model`.
   If someone asks to infer it later, that's a deliberate scope change —
   flag it, don't just add a prior silently.

4. **Inclination affects skew only, not mean lag.** Mean lag comes from
   `lag_scaling(log_mdot, wavelength, M_BH)` alone. `response_function`'s
   skew-normal `alpha` parameter is a separate function of inclination. Keep
   these decoupled if you touch `forward_model.response_function`.

5. **Response function is swappable by contract, not inheritance.** Any
   replacement must accept `(tau_grid, log_mdot, wavelength, inclination,
   M_BH, ...)` and return a causal (`τ<0 → 0`), area-normalised-on-`tau_grid`
   array. `transfer_coeffs`/`compute_echo`/plotting never assume the skew-
   normal specifically.

6. **On-disk run management (`title=`) is opt-in and single-chain only.**
   `EchoFit(M_BH=..., title=...)` switches `.fit()` from the original
   purely-in-memory path to a checkpointed one (`inference.run_mcmc_chunked`)
   that saves progress every `checkpoint_every` samples and can be resumed
   via `EchoFit.resume(title)`. This only covers the *sampling* phase (not
   warmup) and forces `num_chains=1` -- both deliberate scope limits, not
   oversights; see the "Fitting your own light curves" section of
   `README.md`. Without `title`, behaviour is byte-for-byte the original
   `EchoFit` -- don't let the checkpointed path's bookkeeping leak into it.
   `run_manager.py` owns the on-disk layout/serialisation,
   `reporting.py` owns the shared plots+HTML (used by both this path and
   `scripts/smoke_test.py` -- don't duplicate report-building logic back
   into either call site).

## Known rough edges / things to check before trusting results on real data

- `synthetic.py`'s ground truth is generated with the *same* forward model
  used for fitting — good for verifying the code is self-consistent
  end-to-end, but it is not a substitute for testing on independently
  simulated or real light curves.
- The DRW-Fourier prior (`drw_prior_scale`) is an approximation to an exact
  DRW process. If you need exact DRW likelihoods, consider swapping in a
  Kalman-filter/celerite-style likelihood instead — that's a bigger change
  and would touch `model.py` more than `forward_model.py`.
- `n_freq` / `n_tau` / `tau_max` in `EchoFit.build_grid()` are still simple
  heuristics (log-spaced frequencies from the baseline to a Nyquist-style
  estimate; `tau_max` defaults to half the time baseline). The frequency
  upper bound (`w_max = pi / dt_min`) now comes from
  `grid_utils.estimate_dt_min` — a robust (5th-percentile) estimate of
  observation gaps, shared with `synthetic.py`'s ground-truth grid. This
  replaced an earlier version that used the single *tightest* observed gap,
  which for irregular sampling could blow up `w_max` and put the fit on a
  completely different frequency basis than the data actually supports —
  caught by `tests/test_recovery.py`. Still revisit if fitting real
  campaigns with very different cadences per band; pass `dt_min` explicitly
  to `build_grid()` if the data-driven estimate looks off.
- The pipeline has now been run end-to-end (`pytest`, including an MCMC
  recovery test on synthetic data in `tests/test_recovery.py`), so it's no
  longer purely `py_compile`-checked. One finding from that: NUTS can spend
  most samples pinned at the max-tree-depth ceiling on this model even after
  non-centred reparameterising the driver's `S`/`C` coefficients
  (`model.py`) — `inclination`, `sigma_drw`, `tau_drw` recover only loosely
  in the tested synthetic setup even with zero divergences. `log_mdot` (the
  mean-lag-setting parameter) recovers well. Treat the weaker parameters'
  posteriors with appropriate scepticism until this is investigated further;
  `inference.run_mcmc`/`EchoFit.fit` now expose `max_tree_depth` and
  `chain_method` if you want to bound worst-case cost or add cheap
  diagnostic chains (`chain_method="vectorized"`) while digging in.
- `pyproject.toml` gained `h5netcdf`/`h5py` (a netCDF backend for ArviZ,
  which was already a listed dependency but had no working backend
  installed) so `EchoFit(title=...)` can write `chains.nc`. That write is
  wrapped in a try/except in `echofit.py::_save_chains` and only warns on
  failure -- `chains.npz` (plain numpy) is the dependency-free guaranteed
  artifact, don't remove it even if the netCDF path seems reliable.

## Useful commands

```bash
pip install -e ".[dev]"
pytest                      # forward-model unit tests + end-to-end MCMC
                             # recovery test (tests/test_recovery.py, ~1-2 min)
python scripts/smoke_test.py  # quick visual check: fit + save plots to
                               # smoke_test_output/report.html (~30-50s)
jupyter notebook notebooks/demo.ipynb
```

## Style notes

- Keep files minimal / avoid unnecessary abstraction, per the original spec.
- Prefer `jax.numpy` inside anything that runs under NumPyro's model
  function; plain `numpy` is fine in `synthetic.py` and `plotting.py`, which
  never run inside `jax.jit`/NUTS.
- Physical parameters should stay interpretable (days, Angstrom, degrees,
  solar masses) rather than unit-less/rescaled internally, so priors and
  posterior summaries are directly readable.
