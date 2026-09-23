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
   The non-centred form is deliberate: sampling `S, C` directly
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
   trapezoidal double loop: that's both slower and was explicitly
   prohibited by the "no unnecessary loops over time" requirement.

3. **`M_BH` is always fixed, never a `numpyro.sample` site.** It's a plain
   Python float passed through `EchoFit(M_BH=...)` into
   `forward_model.lag_scaling` / `response_function` / `model.reverberation_model`.
   If someone asks to infer it later, that's a deliberate scope change:
   flag it, don't just add a prior silently.

4. **Inclination affects skew only, not mean lag.** Mean lag comes from
   `lag_scaling(log_mdot, wavelength, M_BH)` alone. `response_function`'s
   skew-normal `alpha` parameter is a separate function of inclination. Keep
   these decoupled if you touch `forward_model.response_function`.

5. **Response function is swappable by contract, not inheritance.** Any
   replacement must accept `(tau_grid, log_mdot, wavelength, inclination,
   M_BH, ...)` and return a causal (`τ<0 → 0`), area-normalised-on-`tau_grid`
   array. `transfer_coeffs`/`compute_echo`/plotting never assume the skew-
   normal specifically. To actually swap it at runtime (e.g. for a quick
   experiment, without editing `forward_model.py`), reassign
   `echofit.model.response_function`, e.g. `import echofit.model as model;
   model.response_function = my_fn`. This is the *only* place that needs
   patching: `echofit.py`'s plotting code reads it via `model.response_function`
   (attribute access on the module, evaluated at call time) rather than its
   own `from .forward_model import response_function`, specifically so a
   swap there is honoured by both fitting and plotting consistently. Don't
   reintroduce a direct `response_function` import in `echofit.py` --that
   was a real bug once (the fit used the swapped function but the plot
   silently kept re-deriving ψ from the original, producing a plot
   inconsistent with what NUTS actually fit).

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

7. **Free-lag bands (`lag_mode="free"`) need a driver light curve to be
   identifiable -- this is exact, not a rule of thumb.** A global shift of
   the driver by any Δ, compensated by shifting every band's response lag
   by -Δ, leaves the predicted light curves *exactly* unchanged (proved
   directly, no MCMC, in `tests/test_shift_degeneracy.py`). The physical
   response (`lag_mode="physical"`, the default) escapes this because every
   such band's lag is tied to one shared `log_mdot` through a fixed
   `λ^(4/3)` scaling law -- a multiplicative rescaling of one shared
   parameter can't mimic an additive common shift once you have 2+ bands at
   different wavelengths. A `lag_mode="free"` band's `tau_{band}` has no
   such tie, so the degeneracy is exact; `add_driver_lightcurve()` (a
   direct, zero-lag `y = S_driver*X(t) + C_driver` observation of the
   driver, via `forward_model.driver_at`) is what anchors it.
   `EchoFit.fit()` warns (doesn't raise) if a free-lag band has no driver
   registered.

   **Gradient trap, already hit once:** `forward_model.tophat_response_free`
   must stay a *smoothed* box (sigmoid edges), never a literal hard
   `jnp.where(|tau - tau_mean| <= half_width, 1, 0)`. A hard top-hat has
   **exactly zero gradient** w.r.t. `tau_mean` almost everywhere (confirmed
   directly with `jax.grad` -- autodiff doesn't backprop through a
   comparison's operands), so NUTS gets no signal at all to move
   `tau_{band}` and the chain just sits near its init. This is invisible
   from `diverging`/acceptance-rate diagnostics alone (0 divergences, looks
   "fine") -- the tell was a posterior with essentially zero width. If you
   add another free-form response family, check its gradient w.r.t.
   whatever parameter NUTS samples before trusting a fit that used it.

8. **`thin_disk_response` and `echofit/responses.py` are a second
   `lag_mode="physical"` response family, not a new mechanism.** They plug
   into the *existing* swap point from decision #5
   (`echofit.model.response_function = ...`) -- `echofit/responses.py` is
   only a small registry (`register_response`/`get_response`) for
   discoverability, it does not change how a response actually gets wired
   into a fit. `thin_disk_response` is a JAX-differentiable, deterministic-
   quadrature adaptation of the Monte-Carlo disk integrator (`tfbx` +
   `tr4visc`/`tr4irad`) in the author's PhD-era CREAM Fortran code
   (`pycecream`'s `cream_f90.f90`): real Shakura-Sunyaev viscous (+ optional
   lamppost-irradiation) temperature profile, the disk's own light-travel-
   time delay surface `tau(r, phi) = r(1 + sin(inclination) cos(phi))`, and
   a Planck-derivative response weighting -- genuine physics, not an
   assumed shape, unlike the default skew-normal. Two adaptation choices
   worth knowing before touching it:
   - It reuses `lag_scaling(log_mdot, wavelength, M_BH)` for its absolute
     lag scale (the "Wien radius"), rather than independently deriving an
     Eddington-ratio-to-Mdot conversion from scratch. This was deliberate:
     an independent derivation would give `log_mdot` a second, incompatible
     meaning depending which response a band used. Only the inner (ISCO)
     radius uses real physical constants (G, c, M_sun) directly, since that
     conversion doesn't need any extra accretion-rate calibration.
   - Like `tophat_response_free`, it must stay a *smoothed* (Gaussian
     kernel) deposit of disk-grid points onto `tau_grid`, never a hard
     histogram/binning -- same zero-gradient trap as decision #7's "already
     hit once", confirmed again here with `jax.grad` before considering the
     function done (see `tests/test_thin_disk_response.py`).
   - It costs `O(n_r * n_phi * n_tau)` per evaluation (a real
     radius/azimuth integral) versus the skew-normal's closed form, so it's
     an optional, heavier alternative -- not a default-swap candidate for
     routine fits without checking the cost is acceptable.
   - `n_r`/`n_phi` default to 50/64, not something smaller -- an earlier
     40/24 default looked fine on the causality/normalisation/gradient
     tests (none of which check smoothness) but produced a visibly jagged,
     under-converged psi once actually plotted for `docs/thin_disk_response.md`
     (each radius only contributing 24 distinct azimuth samples). Confirmed
     by comparing 40x24/50x64/60x120 side by side: 50x64 is already
     converged onto the same curve as 60x120. Don't drop below ~50x64
     without re-checking a plotted psi, not just the numeric tests.
   - `r_max` is capped at `r_max_factor * tau_ref` (default 20x), not left
     as the uncapped `tau_grid[-1] / (1 - sin(inclination))` geometric
     formula -- that formula alone made high-inclination curves ~66x-100x
     wider in radial domain than a face-on one on the same `tau_grid` (at
     80/89 degrees respectively), spreading the same log-spaced `n_r`
     across a domain two orders of magnitude bigger and leaving it visibly
     wavy right where the response has weight, even at the resolution that
     already looked fine face-on. Found by actually plotting high-
     inclination curves for `docs/thin_disk_response.md`, not by the unit
     tests. The cap is safe (confirmed against an uncapped, far-higher-
     resolution reference: max absolute difference ~2e-4) because the
     Planck-derivative response weight decays exponentially in radius, so
     nothing beyond ~20x `tau_ref` has any real weight to lose. If you
     touch the radial grid again, re-run this comparison rather than
     trusting the numeric tests alone -- they don't check smoothness.

9. **`build_thin_disk_response_table`/`build_thin_disk_response_fast`
   trade `thin_disk_response`'s accuracy for MCMC-usable speed, the same
   way the author's PhD-era CREAM Fortran code did.** `thin_disk_response`
   costs `O(n_r * n_phi * n_tau)` per call, recomputed on every NUTS
   leapfrog step if used directly -- these precompute a table of responses
   across an inclination grid *once* (at one reference `log_mdot`/
   `wavelength`), then get any other inclination via linear interpolation
   and any other `log_mdot`/`wavelength` by *stretching* the lag axis
   according to `lag_scaling`'s own scaling law (`s = lag_scaling(...) /
   tau_ref_reference`, evaluate the template at `tau_grid / s`, divide by
   `s` to keep the area normalised). Confirmed ~90x faster per call at
   matched resolution, with `jax.grad` still non-zero w.r.t. `log_mdot`
   and `inclination` (checked on and off the precomputed inclination grid
   points, same discipline as decision #7's gradient trap).

   The stretch is a genuine approximation, not an identity: the ISCO
   (`r_in`) is a fixed absolute length that doesn't stretch along with
   everything else, so the ratio `r_in / tau_ref` -- and with it, how much
   the inner-boundary term shapes the response -- differs between the
   table's reference point and wherever a fit actually queries it. This
   bites hardest exactly where the response is sharpest: high inclination,
   far from the reference wavelength/`log_mdot`. Confirmed directly:
   `inclination=85`, `wavelength=7000` (table built with the default
   `reference_wavelength=5000`) is off by ~35% at the near-zero-lag spike's
   *peak*, while the mean lag still tracks well and everywhere away from
   the spike matches closely -- see `docs/thin_disk_response.md` section 5
   and `tests/test_thin_disk_response_fast.py`'s
   `test_fast_response_approximation_degrades_away_from_reference`. This
   is a real, documented tradeoff to make deliberately (build the table
   with a `reference_wavelength` close to the run's actual bands if the
   posterior is expected to favour high inclination), not a bug to chase.

## Known rough edges / things to check before trusting results on real data

- `synthetic.py`'s ground truth is generated with the *same* forward model
  used for fitting: good for verifying the code is self-consistent
  end-to-end, but it is not a substitute for testing on independently
  simulated or real light curves.
- The DRW-Fourier prior (`drw_prior_scale`) is an approximation to an exact
  DRW process. If you need exact DRW likelihoods, consider swapping in a
  Kalman-filter/celerite-style likelihood instead: that's a bigger change
  and would touch `model.py` more than `forward_model.py`.
- `n_freq` / `n_tau` / `tau_max` in `EchoFit.build_grid()` are still simple
  heuristics (log-spaced frequencies from the baseline to a Nyquist-style
  estimate; `tau_max` defaults to half the time baseline). The frequency
  upper bound (`w_max = pi / dt_min`) now comes from
  `grid_utils.estimate_dt_min`, a robust (5th-percentile) estimate of
  observation gaps, shared with `synthetic.py`'s ground-truth grid. This
  replaced an earlier version that used the single *tightest* observed gap,
  which for irregular sampling could blow up `w_max` and put the fit on a
  completely different frequency basis than the data actually supports;
  caught by `tests/test_recovery.py`. Still revisit if fitting real
  campaigns with very different cadences per band; pass `dt_min` explicitly
  to `build_grid()` if the data-driven estimate looks off.
- The pipeline has now been run end-to-end (`pytest`, including an MCMC
  recovery test on synthetic data in `tests/test_recovery.py`), so it's no
  longer purely `py_compile`-checked. One finding from that: NUTS can spend
  most samples pinned at the max-tree-depth ceiling on this model even after
  non-centred reparameterising the driver's `S`/`C` coefficients
  (`model.py`): `inclination`, `sigma_drw`, `tau_drw` recover only loosely
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
- **A single NUTS chain is not sufficient evidence for `lag_mode="free"`
  recovery, even with a driver and zero divergences.** A broad
  `Uniform(0, tau_max)` prior on each `tau_{band}`, combined with the
  driver's own stochastic autocorrelation structure, gives a genuinely
  multimodal likelihood: confirmed directly by running the same fit from 3
  different single-chain seeds at a sparse data budget (35 obs/line,
  `noise_level=0.05`) -- 2 of 3 converged (confidently, 0 divergences,
  tight posterior) to the true lags almost exactly, and the third converged
  just as confidently to values 3-4x too large. Don't trust a single
  chain's point estimate for free-lag bands; run several chains
  (`num_chains=4, chain_method="vectorized"` is cheap on top of a single
  chain) and check Gelman-Rubin R-hat (`numpyro.diagnostics.summary`)
  before trusting any of them -- see
  `tests/test_free_lag_mode.py::test_free_lag_recovery_with_driver_anchor`,
  which does exactly this and documents the sparse-data failure mode in
  its docstring. With more/cleaner data (60 obs/line, `noise_level=0.02`)
  4 chains converge cleanly (R-hat ~1.0) to the true lags -- multimodality
  risk trades off against how constraining the data actually is.
- **Running `pytest` in a background/headless shell can crash on exit if
  matplotlib's default backend is interactive.** `tests/test_response_function_swap.py`
  calls `EchoFit.plot_lightcurve_fits()` without ever closing the returned
  figure; on a machine where matplotlib's default backend is `TkAgg` (true
  on at least one contributor's Mac), this creates real Tk windows against
  the active display even from a non-interactive test run. If that process
  is then killed (e.g. a `timeout` wrapper cutting off a background run
  before the ~10-minute full suite finishes), Python's interpreter teardown
  can hit a Tcl/Tk finalisation bug (`PyEval_RestoreThread: NULL tstate`)
  and abort with `SIGABRT`, a visible crash dialog with no connection to
  whatever test was actually running. Run `pytest` with `MPLBACKEND=Agg`
  set (and give it enough time to finish) in any headless/background
  context to avoid this; it's an invocation-time fix, not a reason to
  force a non-interactive backend inside `plotting.py` itself, which would
  break interactive use from the notebook.
- **Dependency versions matter more than they look like they should for
  this stack.** `jax`/`jaxlib` are pinned `>=0.4.28,<0.5` (not just
  floored) because an unconstrained range let `poetry install` resolve to
  `jaxlib==0.10.2`, which has no published wheel for this machine's
  platform/Python combination and fails outright; `pip install` happened
  to land on `0.4.38` instead via an indirect constraint from `numpyro`,
  but that's not something to rely on. A `poetry.lock` is committed so
  `poetry install` is fully reproducible regardless -- regenerate it
  (`poetry lock`) if `pyproject.toml`'s dependencies change, don't hand-edit
  it. Separately, `np.trapz` (plain NumPy, not `jnp.trapz`) was removed in
  NumPy 2.x; every plain-NumPy trapezoidal-integral call site now uses the
  same `np.trapezoid if hasattr(np, "trapezoid") else np.trapz` fallback
  already established for `jnp` in `forward_model.py` -- keep using that
  pattern for any new one rather than calling `np.trapz`/`jnp.trapz` bare.

## Useful commands

Poetry-managed (see README.md's "Install" for a from-scratch walkthrough);
`poetry install --extras dev` once, then prefix commands with `poetry run`,
or `poetry shell` (needs the `shell` plugin in Poetry 2.x) to avoid
repeating it. Plain `pip install -e ".[dev]"` also works without Poetry, it
just skips the `poetry.lock` version pinning.

```bash
poetry install --extras dev
poetry run pytest             # forward-model unit tests + end-to-end MCMC
                               # recovery test (tests/test_recovery.py, ~1-2 min --
                               # the full suite, including the free-lag-mode
                               # multi-chain recovery test, is more like 10 min)
poetry run python scripts/smoke_test.py  # quick visual check: fit + save
                                          # plots to smoke_test_output/report.html (~30-50s)
poetry run jupyter notebook notebooks/demo.ipynb
```

## Style notes

- Keep files minimal / avoid unnecessary abstraction, per the original spec.
- Prefer `jax.numpy` inside anything that runs under NumPyro's model
  function; plain `numpy` is fine in `synthetic.py` and `plotting.py`, which
  never run inside `jax.jit`/NUTS.
- Physical parameters should stay interpretable (days, Angstrom, degrees,
  solar masses) rather than unit-less/rescaled internally, so priors and
  posterior summaries are directly readable.
