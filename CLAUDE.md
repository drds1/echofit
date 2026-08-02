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
   in `EchoFit.build_grid()`. `S_k, C_k` are free NumPyro parameters with
   priors set by the DRW's Lorentzian power spectrum (`model.drw_prior_scale`),
   parameterized by inferred `sigma_drw`, `tau_drw`. This was requested
   explicitly in the spec and also happens to make the whole model
   analytically convolvable (see next point) instead of needing a GP solve.

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
   M_BH, ...)` and return a causal (`τ<0 → 0`), area-normalized-on-`tau_grid`
   array. `transfer_coeffs`/`compute_echo`/plotting never assume the skew-
   normal specifically.

## Known rough edges / things to check before trusting results on real data

- `synthetic.py`'s ground truth is generated with the *same* forward model
  used for fitting — good for verifying the code is self-consistent
  end-to-end, but it is not a substitute for testing on independently
  simulated or real light curves.
- The DRW-Fourier prior (`drw_prior_scale`) is an approximation to an exact
  DRW process. If you need exact DRW likelihoods, consider swapping in a
  Kalman-filter/celerite-style likelihood instead — that's a bigger change
  and would touch `model.py` more than `forward_model.py`.
- `n_freq` / `n_tau` / `tau_max` in `EchoFit.build_grid()` are currently
  simple heuristics (log-spaced frequencies from the baseline to a Nyquist
  estimate off the tightest per-band sampling; `tau_max` defaults to half
  the time baseline). Revisit if fitting real campaigns with very different
  cadences per band.
- This environment could not `pip install jax`/`numpyro` (no network access
  at the time this repo was generated), so the code was written and
  reasoned through carefully and checked with `python -m py_compile`, but
  **has not been executed end-to-end**. Run `notebooks/demo.ipynb` (or
  `pytest`) first thing after cloning to confirm everything actually runs,
  and fix anything that trips up before relying on it.

## Useful commands

```bash
pip install -e ".[dev]"
pytest                      # forward-model sanity checks
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
