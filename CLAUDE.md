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

13. **`sigma_drw`'s prior scale is anchored to the registered data's own
    amplitude (`EchoFit._sigma_drw_prior_scale`), not a fixed constant --
    found via `scripts/make_fit_animation.py`'s GIF, which made a real,
    previously-unnoticed degeneracy visible.** Rescaling the driver by any
    `lambda != 0` (`S, C -> lambda*S, lambda*C`), compensated by rescaling
    every band's gain by `1/lambda` (`S_band -> S_band/lambda` for every
    band), leaves every predicted light curve -- and hence the
    likelihood -- exactly unchanged. Unlike the shift degeneracy in
    decision #7, this holds regardless of `lag_mode` or band count: it's a
    property of the model's linear driver-amplitude/per-band-gain
    separability, not something the disk physics (shared `log_mdot`, etc.)
    can break. With a fixed `HalfNormal(2.0)` prior on `sigma_drw`
    (unrelated to the actual light curves' units) and `S_band ~
    LogNormal(0, 1)`, this direction is only weakly regularised by the
    priors -- visible directly in the animation as the inferred driver
    growing/shrinking with little constraint early in a chain, before the
    (still fairly broad) prior eventually reins it in.

    Checked directly against the author's PhD-era CREAM Fortran code
    (`cream_f90.f90`) for how it handled this at the time: it has an
    explicit, optional Gaussian prior directly on `P0`, the power-spectrum
    normalisation (`sigma_drw`'s counterpart), gated by
    `sigp0square`/`siglogp0` (the `bof4` term, off by default). Same fix,
    same mechanism, applied here: `EchoFit._model_kwargs()` now computes a
    data-anchored `sigma_drw_prior_scale` (prefers a registered driver
    light curve's own std if one exists, via `add_driver_lightcurve`,
    since that's the most direct available observation of the driver;
    otherwise the largest std across the registered bands, since the
    least-reprocessed band is the closest available proxy for the driver's
    own amplitude -- a reprocessed echo is usually damped relative to what
    drives it, not amplified) and passes it to
    `reverberation_model(..., sigma_drw_prior_scale=...)`, which now takes
    that as a parameter instead of hardcoding `2.0`. The old fixed value
    remains the function's own default, for direct/standalone calls to
    `reverberation_model` outside `EchoFit`.

14. **`scripts/make_fit_animation.py` and `forward_model.disk_temperature_profile`
    are a demo/visualisation aid, not part of the fitting pipeline itself.**
    The GIF embedded in README.md's opener renders driver/echo/response-
    function panels (response-function x-axis capped at 30 days -- the
    response itself is always much narrower than the full lag grid it's
    evaluated on -- and the light-curve column given more width via
    `gridspec_kw={"width_ratios": [3, 1, 1]}`, since that's the panel
    worth the most screen space) plus two disk panels, evolving sample by
    sample early in a deliberately short-warmup chain -- the same
    short-warmup chain whose driver amplitude swings motivated decision
    #13 above. `disk_temperature_profile` and `thin_disk_response`'s
    viscous term now share the one actual formula, via `_viscous_t4_shape`
    (same formula, same Wien's-law reference point -- see
    `tests/test_thin_disk_response.py`'s
    `test_disk_temperature_profile_matches_wien_law_at_the_reference_radius`).
    Originally deliberately *not* shared, to avoid regression risk to the
    already-validated `thin_disk_response` for what was then a purely
    cosmetic reuse -- revisited and done anyway after a direct "best not to
    have duplication of code" request, once there was a second real
    consumer of the same physics (this decision, the animation) and the
    existing `thin_disk_response` test suite (20 tests, all passing
    unchanged after the refactor) to catch a regression if the shared
    helper got it wrong. `_viscous_t4_shape` takes the two callers'
    already-clipped radius and the constants they'd already computed
    (`tau_ref`, `r_in`) rather than clipping internally, since the two
    callers need genuinely different clipping strategies at `r <= r_in`
    (`thin_disk_response`'s `r_star` depends on a sampled parameter and
    needs a smooth cutoff -- handled separately via its own sigmoid mask,
    same zero-gradient concern as decision #7; `disk_temperature_profile`
    doesn't sample `r`, so a plain hard clip is fine there) -- the formula
    is shared, the clipping decision deliberately isn't. Always uses the
    viscous-only profile at a fixed illustrative reference wavelength
    (5000 Angstrom), regardless of which response function the fit itself
    used or what wavelengths its bands are at -- only the thin-disk
    response family has a literal disk geometry to show, and the picture
    is meant to convey the physics qualitatively, not represent the
    specific fit's own bands.

    **The disk view is split into two panels, not one, after a direct
    user correction:** an earlier single-panel version squashed the
    face-on disk vertically by `cos(inclination)` *and* moved the
    observer icon's position to suggest the viewing angle, which read as
    the observer itself moving rather than the disk tilting. Now:
    face-on temperature panel (fixed geometry, colour-only updates via
    `set_array` -- cheaper than the old per-frame `pcolormesh` rebuild,
    titled each frame with the current `log_mdot`/`inclination` values)
    and a separate side-on schematic panel where a line through the
    centre rotates with inclination (vertical at `inclination=0`/face-on,
    rotating toward horizontal/aligned with the fixed observer's line of
    sight as inclination approaches edge-on) while the eye-on-a-sphere
    observer and its dashed sightline are drawn once and never move.
    Both remain a simplified 2-D schematic, not a 3-D/raytraced render.

    **Every band/driver panel also shows an expanding-window 68%/95%
    credible envelope** (`_expanding_percentiles`): frame `i`'s envelope
    is the percentiles of samples `0..i` only, not the full run, so it
    starts at zero width (one sample has no spread), can widen as the
    chain starts genuinely exploring, and narrows towards the converged
    posterior's own width as more samples dilute the influence of any
    early, still-unconverged ones -- the same shaded-band convention as
    the standard (non-animated) `plot_lightcurve_fits`, just computed
    cumulatively per frame instead of once over the whole chain. Percentiles
    of a samples-so-far prefix are mathematically bounded by the full run's
    own min/max, so the existing axis limits (set from the full arrays)
    already contain every frame's envelope with no extra padding needed.
    `fill_between` has no in-place update method, so these artists are
    removed and redrawn every frame (unlike the disk-temperature mesh's
    `set_array`).

    **Bands are coloured by real-world wavelength
    (`plotting.wavelength_to_colour`), not an arbitrary per-plot palette,
    in both this animation and every standard light-curve plot
    (`plot_raw_lightcurves`/`plot_lightcurve_fits`) -- another direct user
    request.** X-ray (< 100 A) black, UV (100-3800 A) a strong violet,
    the visible range (3800-7500 A) its actual approximate spectral
    colour (`_visible_wavelength_to_rgb`, the standard Bruton-style
    piecewise-linear wavelength-to-RGB approximation), and IR and beyond
    (> 7500 A) a reddish colour -- matching the lamppost picture of a
    short-wavelength driver reprocessed into progressively redder bands
    further out in the disk. The driving light curve's own line is plain
    `"black"` throughout, consistent with "X-ray driving light curves are
    black" regardless of whether a real driver light curve was registered.
    `_band_colours`'s old behaviour (a `plasma_r` colormap normalised to
    whatever wavelength range happened to be in the current fit) is gone
    entirely -- colours are now absolute and comparable across different
    fits/reports, not just internally consistent within one.

    **Higher resolution (`--dpi`, default 150, was a fixed 80) after a
    direct "looks blurry when zoomed in" correction** -- also bumps the
    committed GIF from ~3-4MB to ~10MB; accepted as the direct cost of
    that request rather than silently re-compressed back down, since
    colour-count reduction was checked and barely helps (96 to 64 colours
    saved well under 1MB) without visibly hurting the smooth gradients.

    **Gridlines on every light-curve/psi/BOF panel (`alpha=0.6`, both here
    and in `plotting.py`) after direct feedback that the charts "don't
    have gridlines"** -- they technically did (`alpha=0.25`), but
    `grid.color`'s default (`#b0b0b0`, already a light grey) at that alpha
    on a white background turned out to be essentially invisible once
    actually rendered, not just subtle; confirmed by rendering comparison
    swatches at several alpha values before picking `0.6`. Also added a
    sixth panel, Badness of Fit vs. sample (`2 * potential_energy`,
    matching `plotting.plot_bof`) as a growing trace extending one point
    per frame -- reuses `ef.extra_fields["potential_energy"]`, already
    collected by every fit by default (decision #11), so no extra
    computation, same as the standalone `plot_bof`.

15. **`inclination` is sampled uniform in `cos(inclination)`, not
    `inclination` itself, and `EchoFit(fixed_params={...})` generalises
    decision #3 ("`M_BH` is always fixed") to any scalar site.** Both
    direct user requests, both in `model.py` -- see its "inclination note"
    and "fixed-parameter note" docstring sections for the full mechanism
    (`_param`: substitute a `numpyro.deterministic` constant for the
    `numpyro.sample` call when a name is in `fixed_params`; `inclination`'s
    non-fixed path samples `cos_inclination ~ Uniform(cos(80deg), 1)` and
    derives `inclination = arccos(...)` as a deterministic). Two things
    worth knowing that aren't obvious from the mechanism alone:

    - **"Step in cos(i), uniform there" and "an explicit prior on
      inclination" are the same thing, not two different options.**
      Directly clarified with the user after they asked whether stepping
      in cos-space could "mimic the effect of" a prior without one being
      declared: it can't, and doesn't need to try to -- sampling
      `cos_inclination` uniformly and setting `inclination =
      arccos(cos_inclination)` *is*, via the transform's Jacobian, exactly
      the standard isotropic-orientation prior `p(inclination) ∝
      sin(inclination)`. There's no version of this that changes only how
      NUTS steps without also changing the marginal prior on inclination.
    - **`EchoFit._init_strategy()`** (data-anchored starting guesses for
      each band's `S_{band}`/`C_{band}` -- `C_band` the band's own mean,
      `S_band` its std relative to `_sigma_drw_prior_scale`, the same
      CREAM-Fortran-inspired idea as decision #13's prior anchoring, via
      `numpyro.infer.init_to_value`) **must stay disabled whenever
      `num_chains != 1`.** Found the hard way, not anticipated in advance:
      applying it unconditionally made
      `tests/test_free_lag_mode.py::test_free_lag_recovery_with_driver_anchor`'s
      4-chain R-hat check on the free-lag `tau_{band}` sites blow up to
      ~1000 (down from ~1.0 with it disabled) -- `init_to_value` gives
      *every* chain the identical starting point, which defeats the
      independently-initialised-chains premise that whole R-hat-based
      convergence-checking methodology depends on (see the "rough edges"
      note below on why that independence matters for this model's
      free-lag multimodality risk specifically). `_init_strategy(num_chains=1)`
      is safe (and the checkpointed path is always single-chain regardless,
      decision #6); anything else returns `None` (NumPyro's own
      `init_to_uniform` default) so multi-chain fits keep genuinely
      independent starts. `fixed_params` is persisted to/restored from
      `manifest.json` across `EchoFit.resume()` -- forgetting this would
      silently change the model structure (which sites get sampled at all)
      partway through a checkpointed run's chunks.

16. **Test coverage is measured (`pytest-cov`), enforced locally via
    pre-commit hooks, and run in full by CI -- all added directly in
    response to "how is our test coverage".** See README's "Tests and
    coverage" for the commands; the summary here is what changed and why.
    Coverage sat at ~90% line coverage before any of this was added (the
    codebase already had substantial tests throughout this file's earlier
    decisions), concentrated weakest in `run_manager.py` (68%) and
    `synthetic.py` (75%) -- their less-common paths (default arguments,
    observing-gap handling, resume edge cases) rather than anything
    central to the model. `tests/test_validation_and_utils.py` adds a
    first pass at those, plus EchoFit's own input-validation/guard-clause
    paths (bad `lag_mode`, `build_grid()` with no bands, every `plot_*`
    method's "call `.fit()` first" check) and a couple of branches nothing
    else happened to exercise (`max_tree_depth` actually reaching NUTS,
    `EchoFit.resume()` with a registered driver light curve, the
    `chains.nc` best-effort fallback documented above actually working
    when netCDF writing fails, not just having a `try`/`except` around it).

    **A genuine bug found this way, not a hypothetical one:**
    `grid_utils.estimate_dt_min`'s "no band has 2+ points" fallback
    (`dt_min = t_span if t_span is not None else 1.0`) was unreachable
    dead code -- `np.concatenate([])` on an empty list of per-band gap
    arrays raises `ValueError` immediately, before the code ever reaches
    the check that was supposed to handle exactly that case. Fixed by
    checking the list of per-band diffs directly, before concatenating,
    instead of checking the (never successfully computed) concatenated
    result's size. This is exactly the value a coverage report is for --
    "line never executed" flagged a branch that could never execute given
    how it was reached, not just one nobody had gotten around to testing.

    **Pre-commit hooks run file hygiene only (trailing whitespace,
    large-file checks, valid YAML/TOML), deliberately *not* any tests** --
    see `.pre-commit-config.yaml`, and its own comment for why, since this
    was a real course-correction worth recording rather than the first
    design that shipped. `@pytest.mark.slow` (registered in
    `pyproject.toml`) exists and marks the handful of tests actually
    responsible for the bulk of the full suite's ~15-minute runtime
    (decided from real `pytest --durations=0` data, not a guess -- one of
    them, the free-lag 4-chain recovery test, is a large fraction of it by
    itself), and the plan going in was a local `pytest -m "not slow"` hook
    at roughly that time saved back. Measured directly instead of assumed:
    that "fast" subset still took **~10 minutes**, because almost every
    other test in this suite also fits a real (if small) NUTS chain, each
    paying its own one-time JAX JIT-compilation cost -- there's no large
    genuinely-fast subset available short of cutting down to only the
    handful of pure-unit (no-fit) tests, which would defeat the point of
    a pre-commit smoke check. Large-file checks exclude
    `docs/images/fit_animation.gif` and `smoke_test_output/`, both
    intentional, regenerated-on-purpose binary assets, not accidents.
    Tests instead run automatically via CI (`.github/workflows/tests.yml`,
    full suite, no marker filter, every push/PR), where a few extra
    minutes doesn't block anyone's local `git commit`; run
    `poetry run pytest -m "not slow"` manually before pushing for that
    feedback sooner, accepting the ~10 minutes.

17. **`dense_mass=True` (a NUTS kernel option, now exposed on `EchoFit.fit()`) is the single
    highest-leverage, accuracy-preserving speed lever found so far, discovered directly from
    `scripts/profile_pipeline.py`'s own numbers, not guessed.** That script's per-iteration chart showed
    every individual forward-model component (response function, `transfer_coeffs`+`compute_echo`,
    potential-energy eval/grad) costing well under a millisecond, while NUTS's own measured per-sample cost
    was ~47ms, a ~120x gap that only makes sense if a single NUTS sample is taking on the order of 100+
    leapfrog steps. Checked directly against NUTS's own `num_steps` extra field (not inferred, measured):
    with the default diagonal mass matrix, this model's NUTS chain spends essentially every sample pinned at
    `max_tree_depth`'s default ceiling (2^10-1 = 1023 steps) -- mean 857.9, median exactly 1023, minimum 511,
    with **zero divergences**. That "maxed-out trajectory length with zero divergences" pattern is the
    textbook signature of NUTS being cut off by the step budget before it can find a genuine U-turn, not
    genuinely difficult/multimodal geometry -- exactly the kind of correlated-parameter geometry a diagonal
    mass matrix (which only rescales each parameter independently) can't correct for, but a full
    covariance-based one can.

    `dense_mass=True` on NumPyro's `NUTS` kernel (estimates a full covariance matrix during warmup instead of
    per-parameter variances) fixed it directly: mean leapfrog steps per sample dropped **~7.5x** (857.9 ->
    113.6), with `log_mdot` recovery unchanged (if anything marginally tighter: posterior std 0.023 ->
    0.021). The one real cost is that a dense mass matrix has O(P^2) entries to estimate during warmup
    instead of O(P), so it needs more warmup than the default to adapt properly -- confirmed directly: 200
    warmup samples gave a real, if modest, divergence rate (12/200, 6%), which 800 warmup samples brought
    down to a healthy 1.5% (3/200) with the same steps-per-sample improvement. This is a warmup-phase
    (one-off) cost, not a per-sample (recurring) one, so it doesn't erode the wall-time win on any run long
    enough for the sampling phase to dominate, which `scripts/profile_pipeline.py`'s own pipeline breakdown
    shows is true for any run past a few hundred samples.

    Off by default (`dense_mass=False`) to keep existing behaviour/results reproducible for anyone already
    relying on it; exposed as a plain passthrough on `EchoFit.fit()` (both the in-memory and `title=`
    checkpointed paths -- persisted in the checkpointed path's `_fit_config`/`manifest.json` the same way
    `max_tree_depth` already was, so a resumed run keeps using it rather than silently reverting to diagonal
    partway through) and on `scripts/fit_lightcurves.py`'s `--dense-mass` flag. Not applied automatically,
    because the extra warmup it needs is a real, user-facing tradeoff (more warmup samples means more one-off
    wall time before sampling starts) that's better left as an explicit choice than a silent default change.
    `docs/mcmc_implementation.md` covers the whole mechanism (the mass matrix, why NUTS needs gradients at
    all, JIT compilation) with real before/after charts (`scripts/plot_dense_mass_comparison.py` regenerates
    them), and a comparison to the original CREAM Fortran implementation checked directly against
    `cream_f90.f90`'s own sampling loop, not assumed: its default (`mcmcmulti_iteration`) is single-site
    random-scan Metropolis-Hastings (a Gaussian random-walk proposal on one parameter at a time, standard
    Metropolis accept/reject, crude accept/reject-streak step-size doubling/halving), and it turns out to
    have a genuine, independently-arrived-at analogue to `dense_mass` (`affine_step`, opt-in via a
    `cream_affine.par` file, only active every other iteration, only for a hand-picked parameter subset):
    eigendecompose that subset's empirical covariance and propose a joint step along its principal axes,
    the same "align the proposal with correlations, not just per-parameter scale" idea `dense_mass` applies
    automatically to every parameter. The difference that remains is gradients: `affine_step` is still a
    blind random draw within the right-shaped geometry, while NUTS's leapfrog dynamics move along the
    log-posterior's gradient at every step, which is the standard, well-established reason HMC/NUTS-family
    samplers need far fewer posterior evaluations than Metropolis-Hastings-family ones in a correlated,
    moderate-to-high-dimensional posterior like this one's (tens of driver Fourier coefficients alone). No
    head-to-head `echofit`-vs-`pycecream` wall-clock benchmark has been run; the comparison above is of the
    two mechanisms, verified against the Fortran source, not a benchmark of the two actual codebases.

18. **Per-light-curve error rescaling (`fit_error_model`) is a direct, checked adaptation of the author's
    PhD-era CREAM Fortran code's `sigexpand`/`varexpand` nuisance parameters, found by reading
    `cream_f90.f90` while answering "is there anything to learn from the Fortran implementation" (see
    `docs/mcmc_implementation.md`'s comparison section).** Confirmed directly at `cream_f90.f90:4284`:
    `ernew2 = (er(it)*fnow)**2 + varnow`, i.e. the reported error is treated as only approximately correct
    and combined with a multiplicative rescale (`fnow`) and an additive jitter variance (`varnow`), both
    themselves fitted nuisance parameters. `echofit` previously had no equivalent: `model.py`'s likelihood
    used `d["yerr"]` verbatim (`dist.Normal(y_pred, d["yerr"])`), which silently assumes every quoted
    uncertainty is exactly right -- a real risk on actual (not synthetic) data, where an overconfident
    likelihood from underestimated errors makes everything else (the BOF, the other posteriors) look more
    constrained than it should.

    `EchoFit.add_lightcurve(..., fit_error_model=True)` / `add_driver_lightcurve(..., fit_error_model=True)`
    turn this on **per light curve**, off by default (`False`) so existing/synthetic-data fits keep exactly
    today's likelihood unless explicitly opted in -- a deliberate, explicit request when this was
    implemented, not an incidental default. When on, that light curve's `sigma_scale_{name}`
    (`LogNormal(0, 0.5)`, median 1, "no rescaling" is the prior's own centre) and `sigma_jitter_{name}`
    (`HalfNormal(mean(yerr))`, anchored to that light curve's own typical quoted error, the same
    data-anchoring philosophy as `sigma_drw`'s prior, decision #13) combine as `sigma_eff =
    sqrt((sigma_scale*yerr)**2 + sigma_jitter**2)` in place of the raw `yerr`. Both new sites go through the
    existing `_param`/`fixed_params` mechanism (decision #15) automatically, so e.g.
    `fixed_params={"sigma_jitter_g": 0.0}` pins just the jitter term while still fitting the rescale factor
    for that band, without any new plumbing.

    **A real bug found and fixed while wiring this up, not a hypothetical one:** `run_manager.save_bands_npz`/
    `save_driver_npz` were hardcoded to a fixed set of keys (`t`, `y`, `yerr`, `wavelength`, `lag_mode`) and
    would have silently dropped `fit_error_model` across a checkpointed run's resume cycle -- the exact same
    class of bug decision #15 hit once already with `fixed_params` not persisting. Fixed by adding
    `fit_error_model` to both save/load functions (backward-compatible: an old checkpoint file without the
    key loads as `False`, not an error). A *second*, separate instance of the same class of bug was caught by
    the resume-persistence test itself: `EchoFit.resume()` correctly loaded `fit_error_model` via the fixed
    loader but never passed it through to the `add_lightcurve()`/`add_driver_lightcurve()` calls that
    reconstruct `self.bands`/`self.driver_data` -- two independent places the same value had to flow through
    correctly, both needed fixing, both are covered by
    `tests/test_error_model.py::test_error_model_persists_across_resume`.

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
poetry run pytest             # full suite, ~15 min (see decision #16 for
                               # why -- pytest -m "not slow" for ~2 min)
poetry run pytest --cov=echofit --cov-report=term-missing  # + coverage
poetry run pre-commit install  # one-time: run the fast subset on every commit
poetry run python scripts/smoke_test.py  # quick visual check: fit + save
                                          # plots to smoke_test_output/report.html (~30-50s)
poetry run python scripts/profile_pipeline.py  # one-off perf snapshot: where
                                                # wall time goes across the whole
                                                # pipeline -> profiling_output/
                                                # (gitignored, ~a few minutes)
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
