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
   these decoupled if you touch `forward_model.response_function`. This is
   exact for `response_function` by construction. `thin_disk_response`
   satisfies it exactly too, analytically -- *without* its default
   smoothing (`smoothing_days=0.0`); with the default smoothing on (see
   decision #8), it holds only approximately (~10% mean-lag drift face-on
   to 80 degrees), a deliberate, documented trade-off, not an oversight.

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
   into a fit. `thin_disk_response` follows Starkey, Horne & Villforth
   (2016, MNRAS 456, 1960; arXiv:1511.06162, the CREAM paper -- confirmed
   directly against the paper's own equations, not just the Fortran): real
   Shakura-Sunyaev viscous + lamppost-irradiation temperature profile
   (their eq. 2), the disk's own light-travel-time delay surface
   `tau(r, phi) = r(1 + cos(phi) sin(inclination))` (their eq. 5), and a
   Planck-derivative response weighting -- genuine physics, not an assumed
   shape, unlike the default skew-normal.

   **It is now an exact analytic reduction to a 1-D integral over azimuth,
   not a 2-D radius/azimuth quadrature (the original implementation, and
   what the Fortran/Starkey+2016 both do numerically).** At fixed `phi`,
   `tau(r, phi)` is linear in `r`, so the disk integral's delta function
   collapses the radial integral exactly: `r*(phi, tau) = tau / (1 +
   cos(phi) sin(inclination))`, giving `psi_raw(tau) = tau * integral
   weight(r*(phi,tau)) / (1 + cos(phi) sin(inclination))**2 dphi`. No
   radial grid, no domain cutoff -- `docs/thin_disk_response.md` section 4
   has the full derivation. This was a direct response to two things: a
   user report that high-inclination curves still looked wrong even after
   the `r_max` cap fix below (turned out to be real residual radial-grid
   artefacts, now gone entirely), and a direct request to derive an
   analytic form for speed. Removing the radial dimension cut a real
   `O(n_r * n_phi * n_tau)` compute cost down to `O(n_phi * n_tau)` --
   an exact multiple wasn't re-verified under `jax.jit` before the old
   grid version was deleted (see decision #9's "a methodology correction"
   paragraph for why an eager-mode number here would be unreliable
   regardless), `jax.grad` still non-zero w.r.t. `log_mdot` and
   `inclination` (checked including at `inclination=89`, near the `sin=1`
   edge). With smoothing off (`smoothing_days=0.0`, see below), the
   mean-lag inclination-independence from decision #4 is exact to <1%
   drift face-on to 80 degrees (was flat to only 4 significant figures,
   with drift purely from quadrature error, before this rewrite) --
   **with the default smoothing on, this loosens to ~10%, a deliberate
   trade-off, not a regression** (see the smoothing paragraph below).

   **Gaussian smoothing is back, but for a different reason and by a
   different mechanism than before.** A second user report -- the
   high-inclination curves still didn't look like Starkey+2016 Figure 3's
   skewed-Gaussian-like shapes, showing an abrupt change in decay rate
   (a "shoulder") near tau~1-2 days instead -- led to checking the Fortran
   again, at the user's prompting, for a smoothing step. Confirmed present
   in *both* response-function subroutines: `tfbx`'s `sig_gaus = dtau` and
   the older `tfb`'s `sig_g = dtau/2`, each applied not as artefact
   clean-up but as part of the physical model itself -- every individual
   disk element's contribution is deposited as a Gaussian spread across
   several tau bins (`tfbx` lines ~12604-12714), not a single delta-function
   spike at its own exact delay. Because that smoothing is linear and
   `sig_gaus` is one constant for the whole disk, smoothing each element's
   contribution and summing is mathematically identical to computing the
   exact unsmoothed integral once and convolving *that* with the same
   Gaussian -- confirmed directly, and implemented as the latter (cheaper).
   The Fortran's literal `sig_gaus = dtau` convention doesn't transplant
   directly, though: it only smooths meaningfully because the Fortran's
   typical output grids were coarse (`dtau` incidentally ~0.3-1 day); for a
   finer `tau_grid` (confirmed with 800 points: `dtau` ~0.01 days) it's
   negligible, and ties the disk's own physical smoothing to an unrelated
   resolution choice regardless. `smoothing_frac` (default 0.4) instead
   scales the width with `tau_ref` (from `lag_scaling`), chosen by directly
   comparing rendered curves against Starkey+2016 Figure 3 -- see
   `docs/thin_disk_response.md` section 4.

   **This reintroduces a real, quantified trade-off, and it was resolved
   with the user directly rather than picked unilaterally**: enough
   smoothing to match the published shape (`smoothing_frac=0.4`) costs
   ~10% mean-lag drift face-on to 80 degrees (0% unsmoothed, ~3% at 0.1,
   ~6% at 0.2 -- monotonic in `smoothing_frac`). A reflecting-boundary
   kernel (folding smoothed-away mass at `tau=0` back in, rather than
   discarding it) was tried and does *not* fix this -- the drift is
   inherent to any causally-respecting smoothing near a boundary that
   high-inclination responses sit much closer to than face-on ones, not an
   artefact of discarding vs redistributing leaked mass. Presented this
   quantified trade-off to the user directly; they chose to keep
   `smoothing_frac=0.4` as the default (matching the published look) over
   a smaller value or defaulting to exact. `smoothing_days=0.0` remains
   available for anyone who wants the decision-#4-exact behaviour back.
   See `tests/test_thin_disk_response.py`'s
   `test_thin_disk_response_smoothing_days_zero_gives_exact_mean_lag_independence`
   and `test_thin_disk_response_default_smoothing_mean_lag_drift_is_bounded`.

   The **irradiation term** (`include_irradiation=True`) was also corrected
   while cross-checking against Starkey+2016 eq. 2: it's now the true
   lamppost geometry `T_irr**4(r) ~ h_x / (r**2 + h_x**2)**1.5`
   (`lamppost_height_rs`, default 3.0 Schwarzschild radii, matching the
   paper's own illustrative value and the Fortran's `x_height=-3.0`), not
   the `~1/r**3` far-field approximation used before -- the two only agree
   for `r >> h_x`, and `h_x` is the same order of magnitude as `r_in`
   (also 3 Schwarzschild radii, confirmed against both the paper's explicit
   statement and the Fortran's actual default, `rinsch=1.0`, not the
   commented-out `=3.0` that would give a different value), i.e. exactly
   the regime this function spends the most time in. `include_irradiation`
   still defaults to `False` (pure viscous), so this only affects opt-in
   use.

   Two adaptation choices still worth knowing:
   - It reuses `lag_scaling(log_mdot, wavelength, M_BH)` for its absolute
     lag scale (the "Wien radius"), rather than independently deriving an
     Eddington-ratio-to-Mdot conversion from scratch. This was deliberate:
     an independent derivation would give `log_mdot` a second, incompatible
     meaning depending which response a band used. Only the inner (ISCO)
     radius and the lamppost height use real physical constants (G, c,
     M_sun) directly, since that conversion needs no extra accretion-rate
     calibration.
   - The `r* < r_in` (inside the ISCO) mask is a smooth sigmoid, not a hard
     `jnp.where` -- same zero-gradient trap as decision #7's "already hit
     once", since `r*` depends on `inclination`, a sampled parameter.

   The superseded (pre-analytic-rewrite) `n_r`/`r_max_factor` machinery,
   and the debugging trail that led to it (a `40x24` default that passed
   every numeric test but looked jagged once plotted; then an `r_max`
   geometric-formula bug that made high-inclination curves ~66x-100x wider
   in radial domain than face-on ones on the same grid), is preserved in
   git history and `docs/thin_disk_response.md`'s section 4 as a record of
   what was tried, not as current guidance -- neither exists in the
   function any more. `smoothing_days` does still exist, but reintroduced
   for a completely different reason (see above) -- don't assume it's the
   same numerical-artefact-management parameter it used to be.

9. **`build_thin_disk_response_table`/`build_thin_disk_response_fast`
   trade a little more accuracy for MCMC-usable speed, the same way the
   author's PhD-era CREAM Fortran code did (confirmed directly in
   `cream_f90.f90`: a `psistore(:,ist)` array precomputed once across an
   inclination grid at one fixed reference `umdotref`/`wavref`, exactly
   this mechanism).** These precompute a table of responses across an
   inclination grid *once* (at one reference `log_mdot`/`wavelength`), then
   get any other inclination via linear interpolation and any other
   `log_mdot`/`wavelength` by *stretching* the lag axis according to
   `lag_scaling`'s own scaling law (`s = lag_scaling(...) /
   tau_ref_reference`, evaluate the template at `tau_grid / s`, divide by
   `s` to keep the area normalised). `jax.grad` still non-zero w.r.t.
   `log_mdot` and `inclination` (checked on and off the precomputed
   inclination grid points, same discipline as decision #7's gradient
   trap).

   **A methodology correction, worth flagging explicitly:** every speed
   number in this file and `docs/thin_disk_response.md` up to this point
   was measured with plain, eager (non-`jax.jit`) repeated Python calls --
   which is *not* what happens inside a real NUTS fit, where NumPyro
   `jax.jit`-compiles the whole log-density function once and every
   leapfrog step reuses that compiled executable with none of the
   per-call Python dispatch overhead eager timing includes. Checked
   directly: at `n_tau=600`, an eager call to `thin_disk_response` took
   ~134ms; the *same* call through `jax.jit` took ~2.6ms -- a ~50x gap
   that has nothing to do with the function's real cost. This means the
   ~90x/~4x/~25x/~1.3-1.5x figures this file used to carry for the fast
   path's speedup at various points in its history are not reliable, and
   in fact went the *wrong direction*: re-measured under `jax.jit` at the
   point right after decision #8's analytic rewrite but before this
   section's smoothing was reintroduced, the fast path was actually
   **~13x faster** (not the ~4x the eager measurement had claimed).
   **Properly measured now** (`jax.jit`, `n_tau=400`, `build_grid()`'s
   own default): `thin_disk_response` (with default smoothing) costs
   ~1.5ms per call, the templated fast path ~0.55ms -- **~2.8x faster**,
   with the range ~2x-5x depending on `n_tau` (checked at 150/400/600).
   Smoothing itself adds real cost when jitted (confirmed: not negligible
   the way it looked eagerly), ~1.5x over `smoothing_days=0.0`. Both
   `thin_disk_response` variants remain substantially more expensive than
   `response_function` even fast and jitted -- ~12x for the templated
   path, ~34x for the plain one, at the same settings -- confirming this
   is an inherent cost of doing a real disk integral (any form) rather
   than an assumed closed-form shape, not something either optimisation
   removes; still a real, essentially-free win at call time over the
   plain version, worth reaching for on long runs where every bit of
   per-step cost compounds, but not a way to make `thin_disk_response`
   competitive with the skew-normal's cost.

   A banded/truncated smoothing kernel (only summing over the `~5 sigma`
   nearest tau bins instead of the full dense `n_tau x n_tau` matrix) was
   prototyped when investigating this and found genuinely faster under
   `jax.jit` (~1.65x on the smoothing step alone) but with a real
   edge-handling bug (clipping out-of-range neighbour indices to the
   boundary over-weights the last few bins rather than properly excluding
   them) that would need fixing before it's trustworthy -- not implemented,
   given the smoothing step's absolute cost is already small once
   properly jitted and this fit's overall per-step cost has other
   components (Fourier transfer, per-band likelihoods) not profiled here
   that may dominate regardless. Worth revisiting if profiling a real fit
   shows the disk response specifically as the bottleneck.

   **Templates are built unsmoothed (`smoothing_days=0.0`
   internally), and smoothing is applied once, after stretching, at each
   query's own `tau_ref` -- not baked into the table.** Smoothing the
   templates *before* stretching them onto a different `log_mdot`/
   `wavelength` would stretch the smoothing bandwidth right along with
   everything else, silently coupling two things that should be
   independent -- confirmed directly to matter a lot: doing it the wrong
   way around made even the *near-reference-point* case (which should be
   almost exact) off by ~13% at the peak; doing it the right way (as
   implemented) brings that back under 1%, and the previously-hardest
   tested case (`inclination=85`, `wavelength=7000`, table built with the
   default `reference_wavelength=5000`) down to ~0.7% (was ~35%
   pre-analytic-rewrite, ~4% immediately after it, before smoothing was
   reintroduced). `thin_disk_response_from_table`/
   `build_thin_disk_response_fast` both take their own
   `smoothing_frac`/`smoothing_days`, independent of whatever
   `build_thin_disk_response_table`'s `**table_kwargs` were.

   The stretch is still a genuine approximation, not an identity: the ISCO
   (`r_in`) is a fixed absolute length that doesn't stretch along with
   everything else, so the ratio `r_in / tau_ref` -- and with it, how much
   the inner-boundary term shapes the response -- differs between the
   table's reference point and wherever a fit actually queries it. This
   still bites hardest exactly where the response is sharpest: high
   inclination, far from the reference wavelength/`log_mdot` -- now a much
   smaller effect in practice than the smoothing-order bug above was, but
   the underlying asymmetry hasn't gone away, and is worth remembering if
   accuracy at extreme parameter combinations matters -- see
   `docs/thin_disk_response.md` section 7 and
   `tests/test_thin_disk_response_fast.py`'s
   `test_fast_response_approximation_degrades_away_from_reference`. This
   is a real, documented tradeoff to make deliberately (build the table
   with a `reference_wavelength` close to the run's actual bands if the
   posterior is expected to favour high inclination), not a bug to chase.

10. **`plot_corner`/`plot_corner_bands`/`plot_corner_free_lag`/
    `plot_fourier_correlation` (`plotting.py`) were added directly in
    response to looking at Starkey+2016's own diagnostic figures (Figure 6:
    a `log_mdot`/inclination corner plot, 3 chains overlaid, crosshair at
    the true value; Figure 4: individual-frequency power spectrum points,
    not just a smoothed line -- the latter is why `plot_power_spectrum`'s
    median line also got marker dots at each frequency).** `plot_corner`
    is intentionally a small, hand-rolled matplotlib implementation, not
    `arviz.plot_pair` (already a project dependency, used elsewhere for
    `chains.nc`): `arviz.plot_pair` doesn't distinguish chains by colour in
    scatter mode, and per-chain colouring is exactly the point here --
    chains landing in visibly different places is the same thing a
    Gelman-Rubin R-hat check would flag (see the `lag_mode="free"`
    rough-edge below), made visible in a plot instead of a single number.
    `S_band`/`C_band`/`tau_{band}` are individually-named scalar sites, so
    `plot_corner_bands`/`plot_corner_free_lag` just auto-detect names from
    `self.bands` and call the general `plot_corner`.

    `S`/`C` (the driver's Fourier coefficients) are different in kind --
    one vector-valued site each, from `numpyro.plate("freq", n_freq)`, not
    `n_freq` individually-named scalars -- and `n_freq` is often in the
    tens, where an `n_freq x n_freq` scatter-matrix corner plot would be
    both unreadable and slow. `plot_fourier_correlation` shows the
    posterior correlation matrix as a heatmap instead (chains pooled, not
    coloured separately, since a correlation matrix is already a
    per-sample summary): the same "is the posterior geometry sane"
    question a corner plot would answer, `O(n_freq^2)` pixels instead of
    `O(n_freq^2)` subplots, so it scales to any `n_freq` without changing
    shape. All four are wired into `reporting.generate_report` --
    `plot_corner_bands`/`plot_fourier_correlation` unconditionally,
    `plot_corner`/`plot_corner_free_lag` only when the fit actually has
    `log_mdot`/`inclination` or any `lag_mode="free"` band respectively
    (checked via `ef.samples`/`ef.bands`, not assumed), so a report never
    errors on a fit that doesn't have the relevant parameters.

11. **Badness-of-Fit (`plot_bof`) piggybacks on NUTS's own
    `potential_energy`, and doesn't need any new computation.** Starkey,
    Horne & Villforth (2016) eq. 12 defines `BOF = chi**2 +
    sum(ln(sigma_i**2)) - 2*ln(P(Theta)) + const`; NumPyro's NUTS already
    computes `potential_energy = -log(likelihood * prior)` on every
    leapfrog step to decide whether to accept a proposal, and for a
    standard Gaussian log-likelihood `-2*ln(likelihood)` is exactly
    `chi**2 + sum(ln(sigma_i**2)) + const`, so `BOF = 2 *
    potential_energy` up to that additive constant -- confirmed by direct
    derivation, not assumed. `inference.run_mcmc`/`run_mcmc_chunked` both
    request `extra_fields=("potential_energy",)` (NumPyro only returns
    `"diverging"` unless asked), and `EchoFit` now tracks
    `self._extra_fields_by_chain` the same way it already tracked
    `self._samples_by_chain`. `EchoFit.plot_bof()` /
    `plotting.plot_bof` draw one BOF trace per chain; wired into
    `reporting.generate_report` conditionally on `"potential_energy" in
    ef.extra_fields`, since a checkpoint resumed from before this feature
    existed (`_merge_dicts` only iterates the first chunk's keys, so an
    old `prev_extra` without `potential_energy` silently drops it from the
    merge rather than erroring) won't have it -- a narrow, accepted edge
    case, not one worth extra machinery for.

12. **`EchoFit.fit(title=..., report_every=...)` can refresh
    `report.html` partway through a long checkpointed run, not just once
    at the end.** `report_every` (in samples, like `checkpoint_every`) is
    checked inside the existing `_on_chunk_done` callback
    (`inference.run_mcmc_chunked`'s per-chunk hook, previously used only
    for on-disk persistence): once enough new samples have accumulated
    since the last refresh, it updates `self.samples`/`self.extra_fields`/
    `self._samples_by_chain`/`self._extra_fields_by_chain` from the
    merged-so-far checkpoint data and calls `reporting.generate_report`
    again, so the report reflects genuinely fresh state rather than a
    stale write. Off by default (`None`): re-rendering the full plot set
    (multiple corner plots, posterior-predictive fits with credible
    bands, ...) has real cost, and most callers won't be watching a run
    live. `self.mcmc` stays `None` throughout the checkpointed path
    regardless (decision #6) -- `reporting.py` never reads it, only
    `ef.samples`/`ef.extra_fields`/`ef._samples_by_chain`/
    `ef._extra_fields_by_chain`/`ef.bands`, which is what makes updating
    those mid-fit sufficient for a mid-fit report to work at all.

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
