# The thin-disk accretion response function

`forward_model.thin_disk_response` is a physically-motivated alternative to
the default skew-normal `response_function` (see the README's "Swapping the
response function" section and `CLAUDE.md`'s design decisions #5, #8 and
#9). This document works through exactly how it is computed, step by step,
shows the two scaling checks worth seeing on a chart rather than taking on
faith (that inclination reshapes the response without moving its mean lag,
and that the mean lag grows with accretion rate the way thin-disk theory
says it should), and covers the precomputed-template fast path
(`build_thin_disk_response_fast`, section 5).

The physics follows Starkey, Horne & Villforth (2016, MNRAS 456, 1960;
[arXiv:1511.06162](https://arxiv.org/abs/1511.06162), the CREAM paper),
which in turn cites Cackett, Horne & Winkler (2007) for the response
function derivation, and the author's PhD-era Fortran CREAM code
(`pycecream`'s `cream_f90.f90`, the `tfbx`/`tr4visc`/`tr4irad` subroutines)
for the original numerical (Monte Carlo) implementation. This is not a
line-by-line translation of either: the Fortran evaluates the response by
random sampling and Gaussian smoothing (not reproducible/differentiable in
a NUTS-friendly way), and Starkey+2016 itself evaluates it numerically too
-- section 4 below works through an exact **analytic** reduction instead,
found while implementing this, that removes the need for any of that.

## 1. The physical picture

A thin accretion disk reprocesses the AGN's variable ionising continuum
into thermal (viscous) and reprocessed (irradiation-heated) emission at
every radius `r` and azimuth `phi`. A patch of disk at `(r, phi)` sees the
driving continuum after a light-travel-time delay set by the disk's
geometry, and re-emits at a wavelength set by its own temperature. Summed
over the whole disk, this gives a causal, positive transfer function
`psi(tau)`, the fraction of the reprocessed response arriving with delay
`tau`, exactly what `response_function` also computes, just with the
skew-normal shape swapped out for the real geometry and temperature
physics.

## 2. Temperature profile

Two heating mechanisms are combined, each raised to the 4th power of
temperature (as radiative flux balance requires) and summed -- this is
Starkey+2016 eq. 2 exactly:

```
T**4(r) = 3GM*Mdot/(8 pi sigma r**3) * (1 - sqrt(r_in/r))     [viscous]
          + L_b*(1-a)*h_x / (4 pi sigma x**3)                  [irradiation]
```

with `x = sqrt(r**2 + h_x**2)` the distance from the lamppost (at height
`h_x` above the disk plane) to the surface element, `L_b = eta*Mdot*c**2`
the lamppost's bolometric luminosity, `a` the disk albedo, and `r_in` the
innermost stable circular orbit, **3 Schwarzschild radii for a
non-spinning black hole** -- confirmed against both the published paper's
own statement of this ("`rin` is the innermost stable circular orbit
(3 rs for a Schwarzschild black hole)") and the Fortran's own default
(`rinld = 3*f_rs2ld(rinsch, embh)` with the active default `urin = 1.0`,
i.e. `rinsch=1` giving exactly 3 Schwarzschild radii too -- there's a
separate, *commented-out* `urin = 3.0` line in the Fortran that would give
9 Rs instead, but it isn't the one actually used).

**Viscous heating**, with an inner boundary term that drives the
temperature smoothly to zero at `r_in`:

```
T_visc**4(r) = K_visc * (1 - sqrt(r_in / r)) / r**3
```

**Lamppost irradiation** (optional, `include_irradiation=True`): the *true*
lamppost geometry above, not the `~1/r**3` far-field approximation this
function used before cross-checking against Starkey+2016 eq. 2 (a real
correction this document's earlier draft got wrong -- the two forms only
agree for `r >> h_x`, and `h_x` defaults to 3 Schwarzschild radii, the same
order of magnitude as `r_in` itself, so the difference matters exactly in
the regime this function spends the most time in):

```
T_irr**4(r) = K_irr * h_x / (r**2 + h_x**2)**1.5
```

`h_x` (`lamppost_height_rs`, default 3.0, Starkey+2016's own illustrative
value) is expressed in Schwarzschild radii and converted to light-days via
the same `M_BH`-based helper as `r_in`. The two terms are combined as
`T**4(r) = w * T_irr**4(r) + (1-w) * T_visc**4(r)` for a mixing weight `w`
(`irradiation_weight`), or just the viscous term alone by default.

Both `K_visc` and `K_irr`'s radial *power-law index* (`viscous_slope`,
`irradiation_slope`, default 0.75 each, the standard thin-disk value, and
what eq. 2 reduces to for `r >> r_in, h_x`) are fixed, not inferred,
matching `response_function`'s existing convention of fixed shape
hyperparameters.

**Absolute calibration -- a deliberate departure from the Fortran.** The
Fortran derives `K_visc`/`K_irr` from `G`, the Stefan-Boltzmann constant,
and a physical `Mdot` (itself derived from an Eddington ratio via
`myedrat2mdot`). Doing the same here would give `log_mdot` two different,
incompatible physical meanings depending which response function a band
used. Instead, `thin_disk_response` reuses
`forward_model.lag_scaling(log_mdot, wavelength, M_BH)` -- the same
function `response_function` uses -- to fix a characteristic radius
`r_ref` (in light-days), and Wien's displacement law,
`wavelength * T = b`, to fix the temperature at that radius. This ties the
two response families together: switching a band from `response_function`
to `thin_disk_response` keeps `log_mdot`'s meaning the same. Only the
disk's inner edge and the lamppost height use real physical constants (G,
c, M_sun) directly, since that conversion needs no accretion-rate
calibration at all.

One consequence worth being explicit about: because `r_ref` is only a
*characteristic* radius, not a promise about the resulting response's
actual mean, `thin_disk_response`'s empirical mean lag comes out somewhat
larger than `lag_scaling`'s pivot value (the physical response has real
weight extending well beyond `r_ref`) -- see the worked numbers in
section 6. `response_function`'s mean lag isn't exactly the pivot value
either, for the same reason (a skew-normal's mean isn't its `loc`
parameter once it's skewed). `lag_scaling` fixes the *scale*, not the
*exact* mean, for either response family.

## 3. Delay surface

A patch of disk at radius `r` and azimuth `phi` (measured from the
observer's line of sight projected onto the disk plane) is farther from
the observer, by light-travel time, than the disk's centre by (Starkey+2016
eq. 5):

```
tau(r, phi) = r * (1 + cos(phi) * sin(inclination))
```

Face-on (`inclination = 0`), every azimuth has the same delay `r`,
whatever the radius, so the whole ring at radius `r` contributes at a
single lag. Inclined, the near side (`cos(phi) = -1`) arrives earlier and
the far side (`cos(phi) = +1`) later than the ring's mean -- this is what
gives the response its inclination-dependent skew, exactly as
Starkey+2016 describes it: "Tilting the disc makes the response function
more skewed; it peaks at shorter lags and develops a tail toward large
lags." Critically, `integral_0^{2 pi} cos(phi) dphi = 0`, so **the ring's
mean delay is `r` regardless of inclination**: inclination reshapes the
response around a fixed mean -- the same requirement `response_function`
satisfies by construction (`CLAUDE.md` decision #4), and the paper states
explicitly too ("The mean delay... is independent of inclination"). Note
that `h_x` (the lamppost height) does *not* appear in the delay surface --
matching eq. 5 exactly -- only in the temperature profile above.

## 4. Response weight and the analytic disk integral

Each `(r, phi)` patch's contribution to `psi(tau)` is weighted by how
strongly its emission responds, in the observing band, to a small heating
perturbation: the Planck function's derivative with respect to
temperature,

```
X = hc / (k * wavelength * T(r))
weight(r) = X**5 * exp(X) / (exp(X) - 1)**2
```

times the disk-plane area element `r dr dphi`. Putting the last three
sections together, the full disk integral is:

```
psi_raw(tau) = integral_{r_in}^{infinity} integral_0^{2 pi}
                   weight(r) * delta(tau - tau(r, phi)) * r dr dphi
```

with `delta` a Dirac delta picking out exactly the `(r, phi)` patches that
land at each `tau`.

**This integral has an exact closed form in `r`, with no approximation
needed.** At *fixed* `phi`, `tau(r, phi) = r * (1 + cos(phi)
sin(inclination))` is *linear* in `r`, so the delta function collapses the
radial integral onto a single root,

```
r*(phi, tau) = tau / (1 + cos(phi) * sin(inclination))
```

with Jacobian `d(tau)/dr = 1 + cos(phi) sin(inclination)` (constant in
`r`, at fixed `phi`), leaving:

```
psi_raw(tau) = tau * integral_0^{2 pi}
                   weight(r*(phi, tau)) / (1 + cos(phi) sin(inclination))**2 dphi
```

a plain 1-D integral over a *fixed* `phi` grid (`n_phi`, evenly spaced),
evaluated the same way for every `tau_grid` point via a uniform Riemann
sum -- exact for a periodic integrand, no `trapz` edge correction needed.
`psi` is this normalised to unit area on `tau_grid`, and zeroed for
`tau < 0` (automatically satisfied by the geometry: `tau(r, phi) >= r(1 -
sin(inclination)) >= 0` for `inclination <= 90` degrees, same as before).
The only masking needed is excluding `r* < r_in` (no disk material inside
the ISCO), done with a *smooth* sigmoid rather than a hard cutoff, since
`r*` depends on `inclination`, a sampled parameter, and a hard `jnp.where`
there would carry the same zero-gradient risk already documented for
`tophat_response_free` (`CLAUDE.md` decision #7).

**This replaces an earlier two-radial-grid, Gaussian-kernel-deposit
implementation entirely**, which needed a separate log-spaced radial grid
(`n_r`), a domain cutoff (`r_max_factor`) and a smoothing bandwidth
(`smoothing_days`) to avoid visible quadrature artefacts -- worst exactly
at high inclination, where the naive radial domain a face-on response
needs blows up by two to three orders of magnitude (documented at length
in git history and the superseded parts of `CLAUDE.md` decision #8, kept
there as a record of what was tried and why it wasn't good enough, not as
current guidance). None of that machinery is needed here: there is no
radial grid to under-resolve, no smoothing bandwidth to tune, and no
truncation to get wrong, because the radial integral was never
approximated -- it was solved. Removing the radial dimension also drops
the cost from `O(n_r * n_phi * n_tau)` to `O(n_phi * n_tau)`, confirmed
**~25x faster per call** at matched quality, with `n_phi=200` (the current
default) already fully converged where the old implementation needed
`n_r=400` *and* still showed residual artefacts at extreme inclination or
accretion rate.

## 5. A precomputed, interpolated fast path for MCMC

Even at `O(n_phi * n_tau)`, NUTS calls a band's response function on every
leapfrog step of every sample, so there is still a real cost to
recomputing the full integral that often when only `log_mdot` and
`inclination` change step to step, not the disk's physics.

`build_thin_disk_response_table` / `build_thin_disk_response_fast`
implement the same fix the author used for this in the PhD-era CREAM
Fortran code (confirmed directly in `cream_f90.f90`: a `psistore(:,ist)`
array precomputed once across an inclination grid, `degstore = (ist-1) *
ddeginc`, at one fixed reference `umdotref`/`wavref`): precompute the
response once, on a grid of inclinations, at one reference accretion rate;
then get any other inclination by interpolating that grid, and any other
accretion rate (or wavelength) by *stretching* the lag axis according to
the `mdot**(1/3)` / `wavelength**(4/3)` scaling `lag_scaling` already uses,
rather than recomputing the disk integral at all:

```python
import echofit.model as model
from echofit.forward_model import build_thin_disk_response_fast

model.response_function = build_thin_disk_response_fast(M_BH=1e8)
```

Concretely, for a given `(log_mdot, wavelength, inclination)`:

1. **Interpolate over inclination** (`jnp.interp`, linear, differentiable
   in `inclination` almost everywhere) against the precomputed template
   family, giving one dimensionless template on the table's own lag axis.
2. **Stretch** that template by `s = lag_scaling(log_mdot, wavelength,
   M_BH) / tau_ref_reference`: evaluate it at `tau_grid / s` and divide by
   `s` to keep the area normalised to 1 under that change of variables.

Both steps are lookups against fixed tables, `O(n_u + n_tau)` rather than
`O(n_phi * n_tau)`. Since section 4's analytic reduction already made the
plain disk integral itself much cheaper, the fast path's relative payoff
shrank along with it -- confirmed directly: **~4x faster per call** now
(precomputing the table itself takes ~3 seconds), down from the ~90x this
document reported before the analytic rewrite, when the plain version was
still the slow O(n_r * n_phi * n_tau) grid integral. It is still a real,
free win (no radial/azimuthal quadrature at all at call time, just two
`jnp.interp` lookups), just no longer the dramatic, close-to-mandatory
speedup it used to be -- for most fits, calling `thin_disk_response`
directly is now fast enough on its own, and the fast path is worth
reaching for mainly on long runs where every bit of per-step cost compounds.
Gradients w.r.t. `log_mdot` and `inclination` are still non-zero (checked
with `jax.grad`, including at an inclination *not* on the precomputed
grid, the same discipline used for `thin_disk_response` and
`tophat_response_free`'s gradients elsewhere in this codebase).

**The stretch is an approximation, not identity, and it is honest to say
so** -- though a much smaller one now that the underlying templates are
exact rather than grid-noisy. The disk isn't *exactly* self-similar under
it: `r_in` (the ISCO) is a fixed absolute length, so it doesn't stretch
along with everything else, meaning `r_in / tau_ref` -- and with it, how
much the inner boundary term shapes the response -- genuinely differs
between the table's reference point and wherever a fit actually queries
it. In practice this still shows up worst exactly where the response is
sharpest: at high inclination, near the near-zero-lag spike from the
disk's near side.

![thin_disk_response, fast vs exact](images/thin_disk_response_fast_vs_slow.png)

Near the table's own reference point (`log_mdot=0`, `wavelength=5000`
Angstrom), the two curves are visually indistinguishable (max absolute
difference ~0.0001, ~0.01% of the peak). Far from it (`inclination=85`
degrees, `wavelength=7000` Angstrom, `log_mdot=-0.5`), the two are now
close even zoomed right into the spike -- max absolute difference ~0.08,
about 4% of the peak height, down from ~35% before the analytic rewrite --
and the mean lag still tracks well (1.9935 exact vs 1.9902 fast). If a
fit's posterior is expected to live mostly at high inclination and/or
spans multiple bands at very different wavelengths, building the table
with `reference_wavelength` set closer to the run's own wavelength(s)
narrows this further; using the exact `thin_disk_response` directly
removes it entirely, at the now-modest ~4x per-step cost from section 4.
This tradeoff, not a hidden bug, is exactly what
`tests/test_thin_disk_response_fast.py::test_fast_response_approximation_degrades_away_from_reference`
checks for.

## 6. Verification: the two scaling-law charts

Both figures use a case-study disk with `M_BH = 1e8` solar masses and
`wavelength = 5000` Angstrom (the same pivot values used throughout
`forward_model.py` and the test suite), generated by
`scripts/plot_thin_disk_response_scalings.py`; rerun it to regenerate
these PNGs if `thin_disk_response`'s physics or defaults change.

### Inclination sweep (fixed accretion rate)

`log_mdot = 0` throughout; only inclination varies.

![thin_disk_response, varying inclination](images/thin_disk_response_inclination_sweep.png)

| inclination (deg) | mean lag (days, numerically integrated) |
|---:|---:|
| 0  | 1.880500 |
| 20 | 1.880500 |
| 40 | 1.880469 |
| 60 | 1.880336 |
| 80 | 1.875719 |

The response visibly reshapes (a broad, near-symmetric peak face-on;
an increasingly sharp near-zero-lag spike at high inclination, from the
disk's near side), while the dashed vertical lines, each curve's own
numerically integrated mean lag, sit essentially on top of each other:
flat to 5 significant figures out to 60 degrees, and within 0.3% even at
80 degrees. This is section 3's `integral cos(phi) dphi = 0` argument and
Starkey+2016's own inclination-independence claim, both confirmed
numerically rather than just algebraically -- and, since section 4's
reduction is now exact rather than grid-based, this flatness reflects the
underlying physics directly rather than partly being a side effect of
quadrature resolution (an earlier version of this table, before the
analytic rewrite, was flat to only 4 significant figures and drifted to
1.6% by 80 degrees, for exactly that reason).

### Accretion-rate sweep (fixed face-on inclination)

`inclination = 0` throughout; only `log_mdot` varies.

![thin_disk_response, varying mdot](images/thin_disk_response_mdot_sweep.png)

| log_mdot | mdot | mean lag (days) | `lag_scaling` reference radius (days) |
|---:|---:|---:|---:|
| -1.0 | 0.10 | 0.889 | 0.464 |
| -0.5 | 0.32 | 1.292 | 0.681 |
| +0.0 | 1.00 | 1.881 | 1.000 |
| +0.5 | 3.16 | 2.743 | 1.468 |
| +1.0 | 10.00 | 3.982 | 2.154 |

The mean lag grows monotonically with `mdot`, close to the
`mdot**(1/3)` scaling `lag_scaling` uses (a factor of 4.48x from
`log_mdot=-1` to `+1`, against an ideal `100**(1/3) = 4.64x` -- the small
difference is the inner-boundary term in section 2 bending the pure power
law, exactly as it would for a real disk). As section 2 already flags, the
empirical mean lag sits above the `lag_scaling` reference radius at every
`mdot` -- `lag_scaling` fixes the response's characteristic scale, not its
exact mean.

## 7. API summary

```python
from echofit.forward_model import thin_disk_response, build_thin_disk_response_fast
from echofit.responses import get_response
import echofit.model as model

# use the exact disk integral directly
psi = thin_disk_response(tau_grid, log_mdot, wavelength, inclination, M_BH)

# or swap it in for "physical"-mode bands (see CLAUDE.md decision #5)
model.response_function = get_response("thin_disk")

# or use the precomputed-template fast path (section 5) for a real fit --
# now a smaller win than it used to be, since thin_disk_response itself
# got ~25x cheaper, but still free performance with a documented accuracy
# tradeoff
model.response_function = build_thin_disk_response_fast(M_BH=1e8)
```

See `thin_disk_response`'s own docstring in `forward_model.py` for the
full parameter list (`viscous_slope`, `include_irradiation`,
`irradiation_slope`, `irradiation_weight`, `lamppost_height_rs`, `n_phi`),
`build_thin_disk_response_table`'s docstring for the fast path's own
parameters (`incl_grid`, `reference_log_mdot`, `reference_wavelength`,
`u_max_factor`, `n_u`), and `README.md`'s "Swapping the response function"
section for how these relate to the default skew-normal and to
`echofit/responses.py`'s registry.
