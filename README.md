# echofit

`echofit` is an AGN light curve fitting code that uses MCMC (via
[JAX](https://github.com/google/jax) + [NumPyro](https://num.pyro.ai/)) to
model multi-band light curves as a lagged echo of a lamppost-driven
accretion disk, inferring the posterior probability distributions of a
physically motivated disk model's parameters: accretion rate (`mdot`),
inclination, and temperature profile. Read on for install instructions,
tests on synthetic data, and example usage.

## Contents

- [Background](#background)
- [Model](#model)
- [Package layout](#package-layout)
- [Install](#install)
  - [1. Check you have Python 3.10 or newer](#1-check-you-have-python-310-or-newer)
  - [2. Install Poetry](#2-install-poetry)
  - [3. Get the code and install its dependencies](#3-get-the-code-and-install-its-dependencies)
  - [4. Run things with `poetry run`](#4-run-things-with-poetry-run)
- [Quickstart](#quickstart)
- [Fitting your own light curves, with saved/resumable runs](#fitting-your-own-light-curves-with-savedresumable-runs)
- [Visual smoke test](#visual-smoke-test)
- [Swapping the response function](#swapping-the-response-function)
- [Emission-line / free-lag mode and driver light curves](#emission-line--free-lag-mode-and-driver-light-curves)
- [Status / caveats](#status--caveats)

## Background

Active galactic nuclei (AGN) are powered by gas accreting onto a
supermassive black hole through a hot, luminous disk. That disk doesn't
shine steadily: its continuum flux flickers stochastically on timescales
of days to months, and the flickering isn't synchronized across colour:
shorter (bluer) wavelengths vary first, and longer (redder) wavelengths
echo them with a delay of hours to days. The standard picture is a
**lamppost** geometry: a compact, hard X-ray/UV-emitting corona above the
disk irradiates it, each annulus reprocesses that irradiation and
re-emits thermally at a wavelength set by its temperature (hotter, so
bluer, closer in), and a cooler ring further out reprocesses more slowly,
so its light reaches the observer later. The lag between bands is
therefore a direct, geometry-independent probe of the disk's temperature
profile and physical size.

Measuring those lags precisely (**continuum reverberation mapping**) is
one of the few ways to measure an accretion disk's size directly, rather
than inferring it from a spectral model, and it's turned out to be a
genuinely useful stress test for disk theory: continuum RM campaigns (e.g.
AGN STORM, the SDSS Reverberation Mapping project) have repeatedly found
disks several times larger than standard thin-disk theory predicts for the
same black hole mass and accretion rate: a persistent "disk size problem"
that better lag measurements, not just better spectra, can help resolve.

It's a genuinely hard fitting problem, though: real light curves are noisy
and irregularly sampled with observing gaps, and each band's flux is
correlated red noise rather than independent points, so pairwise
cross-correlation of light curves can be misleading, and what you really
want is one joint statistical model of *all* bands at once that propagates
uncertainty properly through to the physical parameters (black hole mass,
accretion rate, inclination), not just to a best-fit lag per band pair.

`echofit` is a specific, opinionated take on that joint fit:

- **Fully Bayesian, one model, every band at once.** A single NumPyro model
  jointly infers the shared driving light curve, the disk response per
  band, and the physical parameters behind it, rather than fitting each
  band's lag independently and combining the results afterwards.
- **Closed-form and differentiable.** The driving light curve is
  represented as a finite Fourier series rather than a literal Gaussian
  process, which makes the disk-reprocessing convolution analytically
  closed-form (see "Model" below) instead of a numerical double integral;
  and because everything is written in JAX, gradients come for free, so
  fitting uses gradient-guided Hamiltonian Monte Carlo (NUTS) rather than a
  gradient-free sampler.
- **Built for running a campaign, not just a demo script.** Managed,
  resumable fit runs (`EchoFit(title=...)`, checkpointing,
  `EchoFit.resume()`) and a one-command visual sanity-check report
  (`scripts/smoke_test.py`), because real fitting campaigns get
  interrupted midway and real results need to be eyeballed before they're
  trusted, not just produced.

This continues a line of continuum- and line-reverberation-mapping
software (JAVELIN, PyROA, CREAM/MICA among others) rather than starting
from nothing: see "Status / caveats" below for what hasn't been validated
against real campaigns yet.

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
  at any set of observation times is then a single vectorised matrix
  contraction: no loop over `(t_obs, τ)` pairs, and no loop over bands.

Only these are inferred: `log_mdot`, `inclination`, `sigma_drw`, `tau_drw`,
the driver Fourier coefficients `{S_k, C_k}`, and per-band `{S_band, C_band}`.
**`M_BH` is always a fixed input.**

## Package layout

```
echofit/
    __init__.py        public API (EchoFit, forward_model helpers, synthetic data)
    forward_model.py    lag_scaling, response_function, thin_disk_response
                          (accretion-disk physical response), build_thin_disk_response_fast
                          (precomputed-template fast path for thin_disk_response),
                          tophat_response_free (free-lag mode), transfer_coeffs,
                          compute_echo, driver_at
    responses.py         a small registry (register_response/get_response) for
                          swapping in a built-in or custom physical response
    model.py            NumPyro model (reverberation_model) + DRW prior scale
    grid_utils.py        estimate_dt_min: robust cadence estimate shared by
                          EchoFit.build_grid() and synthetic.py
    inference.py         run_mcmc / run_mcmc_chunked: NUTS wrapper (the
                          latter supports checkpointing/resuming)
    echofit.py           EchoFit: main user-facing class
    plotting.py          plot_raw_lightcurves, plot_lightcurve_fits, plot_power_spectrum,
                          plot_mcmc_diagnostics, plot_corner, plot_fourier_correlation
    reporting.py          generate_report: shared plots + report.html generation,
                          used by both EchoFit(title=...) and scripts/smoke_test.py
    run_manager.py        on-disk run layout, output-dir resolution, checkpoint
                          save/load (see "Fitting your own light curves" below)
    synthetic.py          generate_synthetic_dataset (physical bands) and
                          generate_free_lag_dataset (free-lag bands + driver)
                          for tests / the demo notebook
docs/
    thin_disk_response.md  how thin_disk_response is computed, with
                          scaling-law verification charts
notebooks/
    demo.ipynb            end-to-end synthetic-data demo
scripts/
    smoke_test.py          quick visual sanity check (see below)
    plot_thin_disk_response_scalings.py  regenerates docs/thin_disk_response.md's charts
tests/
    test_forward_model.py  basic sanity checks on the forward model
    test_recovery.py       end-to-end MCMC recovery test on synthetic data
    test_run_manager.py    checkpointing + resume-after-interruption tests
    test_response_function_swap.py  swapped response_function reflected in
                          both fit and plot
    test_thin_disk_response.py  causality/normalisation/gradient checks on
                          thin_disk_response, plus the responses.py registry
    test_thin_disk_response_fast.py  the precomputed-template fast path:
                          accuracy near/far from its reference point, gradients, speed
    test_shift_degeneracy.py  deterministic proof of the free-lag identifiability
                          claim above
    test_free_lag_mode.py  validation (M_BH/driver requirements) + a real
                          driver-anchored free-lag recovery test
```

## Install

The steps below assume nothing is already set up beyond a normal Linux (or
macOS) shell: no Python environment, no Poetry, nothing. If you already
have both, skip to step 3.

### 1. Check you have Python 3.10 or newer

```bash
python3 --version
```

If that prints `Python 3.10.x` or higher, move on to step 2. If `python3`
isn't found, or the version is older than 3.10, install one with your
distribution's package manager:

```bash
# Debian / Ubuntu
sudo apt update && sudo apt install python3 python3-venv

# Fedora / RHEL
sudo dnf install python3

# Arch
sudo pacman -S python

# macOS (with Homebrew)
brew install python@3.11
```

### 2. Install Poetry

This project uses [Poetry](https://python-poetry.org/) to manage its
Python environment and dependencies, so you don't have to. Install it with
its official installer:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

This puts the `poetry` command in `~/.local/bin`. If `poetry --version`
below doesn't work, that directory probably isn't on your `PATH` yet: add
`export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` (or `~/.zshrc`),
then open a new terminal (or run `source ~/.bashrc`) and try again.

```bash
poetry --version
```

### 3. Get the code and install its dependencies

```bash
git clone https://github.com/drds1/echofit.git
cd echofit
poetry install --extras dev
```

`poetry install` creates a self-contained Python environment for this
project only (it won't touch or conflict with anything else on your
machine) and installs the exact package versions recorded in the
committed `poetry.lock`, the same ones this codebase is developed and
tested against. `--extras dev` is needed, not `--with dev`: the optional
`pytest`/Jupyter dependencies are declared the standard (PEP 621) way in
`pyproject.toml`, as an "extra" rather than a Poetry-specific dependency
group. This step downloads a few hundred MB (mostly JAX) and can take a
few minutes the first time.

### 4. Run things with `poetry run`

There's no separate "activate the environment" step to remember. Prefix
any command that needs this project's packages with `poetry run`:

```bash
poetry run pytest                          # confirm the install actually works
poetry run python scripts/smoke_test.py    # a quick visual check, see below
poetry run jupyter notebook notebooks/demo.ipynb
```

<details>
<summary>Prefer plain pip? (for anyone who already manages their own Python environment)</summary>

```bash
pip install -e ".[dev]"
```

This doesn't get the version pinning `poetry.lock` provides, but installs
the same package.
</details>

CPU is fine for everything above; see the
[JAX install guide](https://github.com/google/jax#installation) if you
want GPU/TPU support instead.

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
ef.plot_corner()              # log_mdot / inclination, coloured per chain
ef.plot_corner_bands()        # S_band / C_band (offset/stretch) for every band
ef.plot_corner_free_lag()     # tau_band, only if any band used lag_mode="free"
ef.plot_fourier_correlation() # driver Fourier coefficient correlation heatmap
ef.plot_bof()                 # Badness-of-Fit (2 x potential energy) vs. sample, one line per chain
```

`plot_corner`/`plot_corner_bands`/`plot_corner_free_lag` are pairwise
posterior corner plots (joint scatter + 1-D marginals), coloured per chain
and, for synthetic data, overlaid with the true value -- in the style of
Starkey, Horne & Villforth (2016) Figure 6. `plot_fourier_correlation` is
the scalable stand-in for a corner plot on the driver's Fourier
coefficients (`S`/`C`), which are one vector-valued site each rather than
individually-named scalars and often number in the tens -- a correlation
heatmap answers the same "is the posterior geometry sane" question a
corner plot would, without needing `n_freq` scatter panels. `plot_bof`
tracks Starkey, Horne & Villforth (2016) eq. 12's Badness-of-Fit, which
turns out to be exactly `2 x potential_energy` (NUTS's own `-log(likelihood
x prior)`, already computed on every step) up to an additive constant, so
it costs nothing extra to expose -- a chain that's still trending down at
the end of a run hasn't finished burning in. All five are included
automatically in `report.html` (see below); the disk-parameter and
free-lag corners only when the fit actually has those parameters, and the
BOF plot only when `extra_fields` has `potential_energy` (missing only for
a checkpoint resumed from before this feature existed).

See `notebooks/demo.ipynb` for the full walkthrough.

## Fitting your own light curves, with saved/resumable runs

**From the terminal, with no Python required:** `scripts/fit_lightcurves.py`
wraps everything below as a command-line tool, reading each band's light
curve from a plain text file, three columns `t y yerr` (whitespace- or
comma-separated, one observation per line, `#`-prefixed lines ignored --
time in days, on a consistent zero-point across every band):

```bash
python scripts/fit_lightcurves.py \
    --title ngc_5548 --m-bh 1e8 \
    --band g 4770 data/g_band.txt --band i 7625 data/i_band.txt \
    --num-warmup 1000 --num-samples 2000 --checkpoint-every 200
```

`--band NAME WAVELENGTH PATH` is repeatable (one per band); the full
argument list also covers `--free-lag-band`/`--driver` (see "Emission-line
/ free-lag mode" below), `--num-chains`/`--chain-method`/`--max-tree-depth`,
`--report-every`, `--output-dir`, `--n-freq`/`--n-tau`/`--tau-max`, and
`--resume` -- run `python scripts/fit_lightcurves.py --help` for all of it.

**`scripts/run_example_fit.sh` is a complete, runnable command file** built
on that CLI -- a template to copy and adapt for a real campaign rather than
writing one from scratch. Run it directly:

```bash
./scripts/run_example_fit.sh
```

It runs four steps in sequence, each one a plain `python scripts/...`
invocation you can copy out individually:

1. `scripts/make_example_data.py` writes a synthetic two-band dataset to
   `example_data/*.txt` in the CLI's expected format -- swap this step out
   for your own light curve files.
2. A managed/checkpointed fit (`--title example_ngc`, `--checkpoint-every
   100 --report-every 200`), writing everything to
   `outputs/example_ngc/run_<timestamp>/` as described below.
3. `--resume`-ing that same run (a no-op here since step 2 already finished
   -- this is the command to re-run after a crash or interruption instead).
4. A quick, non-checkpointed 4-chain diagnostic fit (`--num-chains 4
   --chain-method vectorized`, no `--title`), for R-hat/ESS convergence
   checks rather than a production run, writing a one-off report to
   `diagnostic_run/`.

Each step prints where its `report.html` landed; open it in a browser to
see the fit.

**From Python directly:** pass `title=` (e.g. an AGN name) to have
`EchoFit` manage on-disk outputs for the run -- data, config, periodic
checkpoints, the final posterior, and the same visual report as the smoke
test:

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
report.html + *.png      the same visual report as scripts/smoke_test.py,
                          including bof.png (Badness-of-Fit vs. sample)
```

Pass `report_every=` to refresh `report.html` (and its PNGs) partway through
a long fit, instead of only once at the end -- useful for watching a
long-running checkpointed fit converge from another terminal:

```python
ef.fit(num_warmup=1000, num_samples=5000, checkpoint_every=200, report_every=1000)
# report.html refreshes every 1000 new samples, in addition to the final write
```

`report_every` only applies to the checkpointed (`title=`) path, and is off
by default -- re-rendering the full plot set (corner plots,
posterior-predictive fits, ...) on every checkpoint would add real overhead
for a short `checkpoint_every`. The final report is always written
regardless of this setting.

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
`smoke_test_output/`. This is a visual/eyeball check, not a pass/fail test;
for that, see `tests/test_recovery.py`.

## Swapping the response function

`forward_model.response_function` is the single place the physical
(`lag_mode="physical"`, see below) response shape lives. Any replacement
must have the same signature, `(tau_grid, log_mdot, wavelength,
inclination, M_BH, ...) -> psi`, and return a causal, area-normalised array
on `tau_grid`. Two are built in:

* `response_function` (the default): an ad-hoc but cheap skew-normal shape,
  fast to evaluate every NUTS step.
* `thin_disk_response`: a physically-motivated accretion-disk response
  following Starkey, Horne & Villforth (2016, MNRAS 456, 1960;
  [arXiv:1511.06162](https://arxiv.org/abs/1511.06162), the CREAM paper),
  cross-checked directly against that paper's equations and against the
  author's PhD-era CREAM Fortran code
  ([`pycecream`](https://github.com/drds1/pycecream)`/cream_f90.f90`'s
  `tfbx`/`tr4visc`/`tr4irad`). It combines a genuine Shakura-Sunyaev
  viscous + lamppost-irradiation temperature profile with the disk's own
  light-travel-time delay surface, weighted by the Planck-function
  temperature derivative -- giving inclination-driven skew and a hard
  causal edge from the geometry itself, rather than an assumed shape.
  Unlike the closed-form skew-normal, this is a genuine disk integral, but
  an **exact analytic one**: the two radius/azimuth integral reduces, via a
  delta-function argument, to a single 1-D integral over azimuth (no
  radial grid, no truncation). A single Gaussian smoothing convolution is
  then applied on top (matching a genuine, physically-motivated part of
  the original Fortran, not just numerical clean-up -- see the docs below
  for why), exposed via `echofit.responses` for discoverability:

  ```python
  import echofit.model as model
  from echofit.responses import get_response

  model.response_function = get_response("thin_disk")
  ```

  See [`docs/thin_disk_response.md`](docs/thin_disk_response.md) for
  exactly how this is computed (temperature profile, delay surface,
  response weighting, the analytic azimuthal-integral derivation, and the
  smoothing step and the mean-lag-vs-inclination trade-off it brings),
  plus charts verifying that the mean lag scales with accretion rate the
  way thin-disk theory predicts.

  There's also a fast path, `build_thin_disk_response_fast`, for the
  common case of NUTS calling a band's response function on every leapfrog
  step: it precomputes `thin_disk_response` once across a grid of
  inclinations, then gets any other inclination via interpolation and any
  other accretion rate/wavelength by *stretching* the lag axis according
  to `lag_scaling`'s own `mdot**(1/3)`/`wavelength**(4/3)` law, the same
  precompute-and-stretch trick used in the author's PhD-era CREAM code --
  confirmed (properly, via `jax.jit`, matching how NUTS actually calls it --
  see `docs/thin_disk_response.md` section 7 for why that distinction
  matters) roughly 2-5x faster per call depending on grid size, though
  both `thin_disk_response` variants remain tens of times more expensive
  than the closed-form skew-normal even so, an inherent cost of a real
  disk integral rather than something either optimisation removes:

  ```python
  from echofit.forward_model import build_thin_disk_response_fast

  model.response_function = build_thin_disk_response_fast(M_BH=1e8)
  ```

  The stretch is an approximation (the disk's inner edge is a fixed
  absolute radius, so it doesn't stretch too), worst at high inclination
  far from the table's reference accretion rate/wavelength -- see
  `docs/thin_disk_response.md` section 7 for exactly how much that costs
  in accuracy and when to use the exact `thin_disk_response` instead.

`echofit.responses.register_response(name, fn)` registers your own
response under a name for `get_response` to find; `available_responses()`
lists what's registered. Registering doesn't by itself change what a fit
uses -- reassigning `model.response_function` (as above) is the one place
to patch, since `echofit.py`'s plotting code reads it the same way
(module-attribute access, not its own import), so a swap is honoured
consistently by both fitting and plotting.

## Emission-line / free-lag mode and driver light curves

Every band defaults to `lag_mode="physical"`: its mean lag comes from
`lag_scaling(log_mdot, wavelength, M_BH)`, tied to every other physical
band through the one shared `log_mdot`. Pass `lag_mode="free"` to
`add_lightcurve()` instead for a band whose lag isn't physically tied to
the others at all, e.g. an emission line reverberating the continuum,
where each line's lag is its own independent quantity, not a point on a
shared `λ^(4/3)` curve. A free-lag band gets its own inferred `tau_{name}`
and a smoothed top-hat response (`forward_model.tophat_response_free`)
centred on it.

**This needs a driver light curve to be identifiable.** A global shift of
the driver, compensated by shifting every band's lag the same amount,
leaves the predicted light curves exactly unchanged (worked through in the
"Background" section above, and proved directly, with no MCMC and no
sampling noise, in `tests/test_shift_degeneracy.py`). The physical response
escapes this because the shared `log_mdot` can only *rescale* every band's
lag together, not shift them by a common additive amount; a free-lag
band's `tau_{name}` has no such tie, so without an anchor the fit is
exactly degenerate in the absolute lag origin.

`add_driver_lightcurve(t, y, yerr)` registers a light curve that directly
(zero-lag) observes the driver itself (an X-ray/lamppost continuum, or a
directly monitored AGN continuum anchoring an emission-line fit), modelled
as `y(t) = S_driver * X(t) + C_driver` (its own scale/offset, no
convolution). `EchoFit.fit()` warns if any `lag_mode="free"` band is
registered without one:

```python
ef = EchoFit(M_BH=None)  # M_BH only matters for lag_mode="physical" bands
ef.add_driver_lightcurve(t=t_x, y=y_x, yerr=yerr_x)  # e.g. an X-ray continuum
ef.add_lightcurve("Hbeta", wavelength=4861.0, t=t_hb, y=y_hb, yerr=yerr_hb, lag_mode="free")
ef.add_lightcurve("Halpha", wavelength=6563.0, t=t_ha, y=y_ha, yerr=yerr_ha, lag_mode="free")
ef.build_grid()
ef.fit(num_warmup=1000, num_samples=1000)
```

`plot_raw_lightcurves()`/`plot_lightcurve_fits()` show the driver in its
own panel: the latter overlays the driver's own data (back-transformed
through the posterior-mean `S_driver`/`C_driver`) on the inferred driving
light curve panel, a direct visual check that the two agree.

**A single chain isn't enough to trust a free-lag fit, even with a driver.**
Each `tau_{name}` has a broad `Uniform(0, tau_max)` prior, and the driver's
own stochastic structure can make the likelihood genuinely multimodal: one
chain can converge confidently (0 divergences, tight posterior) to a
plausible-looking but wrong value. Run multiple chains
(`num_chains=4, chain_method="vectorized"`) and check they agree
(Gelman-Rubin R-hat, via `numpyro.diagnostics.summary`) before trusting the
result; more/cleaner data also reduces the risk. See
`tests/test_free_lag_mode.py::test_free_lag_recovery_with_driver_anchor`
for a worked example.

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
  divergent transitions: NUTS tends to spend most samples at its
  max-tree-depth ceiling on this model. Treat those three parameters'
  posteriors with extra scepticism on real data until this is investigated
  further; `EchoFit.fit()` exposes `max_tree_depth` and `chain_method` if
  you want to bound worst-case sampling cost or add cheap diagnostic chains
  while doing so.
