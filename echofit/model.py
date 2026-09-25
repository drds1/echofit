"""
model.py
========

The NumPyro probabilistic model tying together:

* a stochastic driver, represented as a truncated Fourier series whose
  sine/cosine amplitudes are given Gaussian priors matching a target power
  spectrum -- by default a pure random-walk (RW) power law (``drw_prior=False``,
  ``sigma_drw`` the only hyperparameter, inferred), or, when ``drw_prior=True``,
  a damped random walk (DRW)'s Lorentzian power spectrum (``sigma_drw`` and
  ``tau_drw`` both inferred) -- see the "random-walk prior" note below;
* an optional driver light curve, a direct (zero-lag) observation of the
  driver itself -- see the module docstring note on identifiability below;
* a per-band causal response function -- either the physical, thin-disk-
  scaling one (``forward_model.response_function``, tied to a single shared
  ``log_mdot``/``inclination``/``M_BH`` across all such bands) or a free-lag
  one (``forward_model.tophat_response_free``, an independently inferred
  lag per band, e.g. for emission-line reverberation mapping) -- that
  convolves the driver into each band's echo;
* a Gaussian observation likelihood for irregularly sampled multi-band light
  curves.

Inferred, always: sigma_drw, {S_k, C_k} driver Fourier coefficients,
{S_band, C_band} per band; tau_drw too if ``drw_prior=True``. Inferred if
any band uses the physical response: log_mdot, inclination (shared across
those bands). Inferred per band using the free-lag response: tau_{band}.
Inferred if a driver light curve is given: S_driver, C_driver.

M_BH is a fixed input, never a latent variable.

Identifiability note: a global shift of the driver by any Δ, compensated by
shifting every band's response lag by -Δ, leaves the likelihood exactly
unchanged (see CLAUDE.md) -- the absolute lag origin is not identifiable
from the echoes alone. The physical response ties every such band's lag to
one shared ``log_mdot`` through a fixed, monotonic wavelength scaling
(``lag_scaling``), which breaks that degeneracy across 2+ bands at
different wavelengths -- a multiplicative rescaling of the shared parameter
can't mimic an additive common shift. The free-lag response has no such
tie: each band's lag is independent, so the degeneracy is exact, and a
fit using it for any band needs a driver light curve (or some other
external anchor) to be identifiable.

Driver amplitude note: a global rescale of the driver by any lambda != 0
(``S, C -> lambda*S, lambda*C``), compensated by rescaling every band's
gain by ``1/lambda`` (``S_band -> S_band/lambda`` for every band), also
leaves every predicted light curve, and hence the likelihood, exactly
unchanged -- unlike the shift degeneracy above, this holds regardless of
lag_mode or how many bands there are, since it's a property of the model's
linear driver-amplitude/per-band-gain separability, not the disk physics.
It's broken only by the priors, not the data: with `sigma_drw`'s prior
scale a fixed constant unrelated to the actual light curves' units, this
direction is only weakly regularised, which shows up as the inferred
driver visibly growing/shrinking with little constraint early in a chain
(see the GIF in README.md). ``EchoFit`` anchors `sigma_drw`'s prior scale
to the registered data's own amplitude instead (see
``EchoFit._sigma_drw_prior_scale``) -- the same fix, by the same
mechanism, as the author's PhD-era CREAM Fortran code's optional Gaussian
prior on its power-spectrum normalisation `P0` (`cream_f90.f90`'s `bof4`,
gated by `sigp0square`/`siglogp0`), which plays the same role as
`sigma_drw` here.

Random-walk prior note: `drw_prior` (`EchoFit(drw_prior=...)`) chooses
between a pure random-walk (RW) driver prior, `rw_prior_scale` (default,
`P(w) = sigma_drw**2 / w**2`, no damping timescale, no `tau_drw` site at
all), and the original damped random walk (DRW) prior, `drw_prior_scale`
(`P(w) = sigma_drw**2 * tau_drw / (1 + (w*tau_drw)**2)`, with `tau_drw`
inferred alongside `sigma_drw`). RW is the default: for the short,
irregularly-sampled campaigns this package targets, `tau_drw` is often
weakly identified anyway (see "Known rough edges" in CLAUDE.md), and a
plain power law is one fewer hyperparameter to identify for the same
qualitative "smooth stochastic variability" driver.

This is deliberately *not* implemented as "fix `tau_drw` to a large
constant inside `drw_prior_scale`" (the seemingly obvious way to remove
the turnover) -- that limit sends the power spectrum to *zero* everywhere
at fixed `sigma_drw` (`sigma_drw**2 * tau_drw / (w*tau_drw)**2 =
sigma_drw**2 / (tau_drw * w**2) -> 0` as `tau_drw -> infinity`), not to a
finite power law. Getting a genuine, non-trivial `1/w**2` law in that
limit needs `sigma_drw**2 / tau_drw` held fixed as `tau_drw` grows, i.e.
`sigma_drw`'s own prior would need rescaling to compensate for whatever
constant `tau_drw` was fixed to -- confirmed directly, not assumed.
Writing `rw_prior_scale`'s `1/w**2` form directly sidesteps this
entirely: exact, no large-constant tuning, no `tau_drw` site or rescaling
needed, and `sigma_drw` keeps the *same* site name and the *same*
data-anchored prior (`EchoFit._sigma_drw_prior_scale`) in both cases,
even though its physical meaning differs (a saturating asymptotic
variability scale for the DRW; a non-saturating power-law amplitude for
the RW, since a pure random walk's variance grows without bound).

Inclination note: `inclination` is sampled as `cos_inclination ~
Uniform(cos(INCLINATION_MAX_DEG), 1)`, not `inclination ~
Uniform(0, INCLINATION_MAX_DEG)` directly, with `inclination` itself a
`numpyro.deterministic` transform (`arccos`) of it. This is the standard
"isotropic orientation" prior: solid angle `dOmega = sin(i) di dphi`
integrates to a flat density in `cos(i)`, not in `i` itself -- a uniform
prior in `i` directly over-weights edge-on orientations relative to a
population with no preferred axis. `inclination` (in degrees) remains the
name every other function reads (plotting, `response_function`, the disk
visualisation) -- only the sampling parameterisation changed, not the
site's public meaning.

Fixed-parameter note: `fixed_params` (`EchoFit(fixed_params={...})`) lets
any of this model's scalar sites (`sigma_drw`, `tau_drw` if `drw_prior=True`
(it isn't a site at all otherwise), `log_mdot`,
`inclination`, `S_driver`, `C_driver`, `S_{band}`, `C_{band}`, free-lag
bands' `tau_{band}`, and any band/driver's `sigma_scale_{name}`/
`sigma_jitter_{name}` if its error model is turned on, see below) be held
at a known value instead of inferred, e.g. `fixed_params={"inclination":
0.0}` to assume a face-on disk while fitting everything else. This
generalises decision #3's "`M_BH` is always fixed" to any parameter a
caller already knows or wants to hold fixed for a particular fit, via the
same mechanism throughout (`_param`: substitute a `numpyro.deterministic`
constant for the `numpyro.sample` call) rather than a bespoke flag per
parameter. A fixed `inclination` bypasses the `cos_inclination`
reparameterisation above entirely -- the fixed value is used directly, in
degrees, matching how every other caller of `inclination` already expects
it.

Error-model note: each band (and the driver light curve, if registered)
can optionally fit its own error rescaling, via `EchoFit.add_lightcurve(...,
fit_error_model=True)` (default `False`, off, exactly today's behaviour --
this is opt-in per light curve, not a global switch). When on, the
band's/driver's reported `yerr` is treated as only approximately correct
and combined with two extra nuisance parameters into an effective sigma:

    sigma_eff = sqrt((sigma_scale * yerr)**2 + sigma_jitter**2)

`sigma_scale_{name}` (`LogNormal(0, 0.5)`, median 1: "no rescaling" is the
prior's own central value) multiplicatively rescales the whole error
array, and `sigma_jitter_{name}` (`HalfNormal(mean(yerr))`, i.e. anchored
to that light curve's own typical quoted error, the same data-anchoring
philosophy as `sigma_drw`'s prior, decision #13) adds a constant "floor"
variance term. This is a direct, checked adaptation of the author's
PhD-era CREAM Fortran code's own `sigexpand`/`varexpand` nuisance
parameters (`cream_f90.f90`, `ernew2 = (er(it)*fnow)**2 + varnow`,
confirmed by reading the source -- see decision #18 and
`docs/mcmc_implementation.md`), for the same reason it exists there: real
quoted photometric/measurement errors are often mis-calibrated (too small
or too large), and fitting the rescaling instead of trusting `yerr`
verbatim avoids an overconfident (or underconfident) posterior on
everything else. Off by default per light curve so synthetic-data fits
(where `yerr` genuinely is correct by construction) and any existing
analysis keep exactly today's likelihood unless explicitly opted in.
"""

from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .forward_model import response_function, tophat_response_free, transfer_coeffs, compute_echo, driver_at

INCLINATION_MAX_DEG = 80.0


def drw_prior_scale(freqs: jnp.ndarray, sigma_drw, tau_drw) -> jnp.ndarray:
    """Standard deviation of each Fourier coefficient under a DRW prior.

    The DRW has a Lorentzian power spectrum in angular frequency w:

        P(w) = sigma_drw**2 * tau_drw / (1 + (w * tau_drw)**2)

    Discretising onto the fixed frequency grid ``freqs`` with local spacing
    ``dw_k``, the implied standard deviation of each independent
    sine/cosine amplitude is ``sqrt(P(w_k) * dw_k)``. This turns the
    "arbitrary sinusoids" driver into a proper (approximate) DRW Gaussian
    process, with ``sigma_drw`` (long-term variability amplitude) and
    ``tau_drw`` (damping timescale, days) as the two inferred hyperparameters.
    """
    power = sigma_drw ** 2 * tau_drw / (1.0 + (freqs * tau_drw) ** 2)
    # local frequency spacing (grid is expected to be sorted, positive)
    dw = jnp.gradient(freqs)
    dw = jnp.clip(dw, 1e-8, None)
    return jnp.sqrt(power * dw)


def rw_prior_scale(freqs: jnp.ndarray, sigma_drw) -> jnp.ndarray:
    """Standard deviation of each Fourier coefficient under a pure
    random-walk (RW) prior -- the "no damping timescale" limit of the DRW
    above, with a genuine power-law power spectrum at every frequency on
    the grid:

        P(w) = sigma_drw**2 / w**2

    Written directly rather than by taking ``tau_drw -> infinity`` in
    ``drw_prior_scale``: that limit sends the power spectrum to *zero*
    everywhere at fixed ``sigma_drw`` (``sigma_drw**2 * tau_drw / (w*tau_drw)**2
    = sigma_drw**2 / (tau_drw * w**2) -> 0`` as ``tau_drw -> infinity``),
    not to a finite power law -- getting a non-trivial limit needs
    ``sigma_drw**2 / tau_drw`` held fixed as ``tau_drw`` grows, which is
    exactly what writing the ``1/w**2`` form directly does, with no
    ``tau_drw`` site or rescaling required. ``sigma_drw`` keeps the same
    site name and the same data-anchored prior (``EchoFit._sigma_drw_prior_scale``)
    as the DRW case, but its physical meaning changes: it's now this
    power law's amplitude, not a saturating asymptotic variability scale
    (a pure random walk's variance grows without bound, it never
    saturates the way a DRW's does).
    """
    dw = jnp.gradient(freqs)
    dw = jnp.clip(dw, 1e-8, None)
    power = sigma_drw ** 2 / freqs ** 2
    return jnp.sqrt(power * dw)


def reverberation_model(
    freqs: jnp.ndarray,
    tau_grid: jnp.ndarray,
    M_BH: Optional[float],
    bands: Dict[str, dict],
    driver: Optional[dict] = None,
    sigma_drw_prior_scale: float = 2.0,
    fixed_params: Optional[Dict[str, float]] = None,
    drw_prior: bool = False,
):
    """NumPyro model for multi-band reverberation-mapped light curves.

    Parameters
    ----------
    freqs : (n_freq,) array
        Fixed grid of driver angular frequencies (rad/day).
    tau_grid : (n_tau,) array
        Fixed grid of lags (days) used to evaluate/normalise psi and its
        Fourier transform.
    M_BH : float, optional
        Fixed black hole mass (solar masses). Not inferred. Only needed
        (may be ``None`` otherwise) if at least one band uses
        ``lag_mode="physical"``.
    bands : dict
        Mapping ``band_name -> {"t", "y", "yerr", "wavelength", "lag_mode",
        "fit_error_model"}`` for each observed light curve. ``lag_mode`` is
        ``"physical"`` (mean lag tied to the shared ``log_mdot`` via
        ``lag_scaling``) or ``"free"`` (an independently inferred
        ``tau_{band_name}``) -- see the module docstring's identifiability
        note for when ``"free"`` needs a ``driver`` to be identifiable.
        ``fit_error_model`` (optional, default ``False``) turns on that
        band's ``sigma_scale_{band_name}``/``sigma_jitter_{band_name}`` --
        see the module docstring's "error-model" note.
    driver : dict, optional
        ``{"t", "y", "yerr", "fit_error_model"}`` for a light curve that
        directly (zero-lag) observes the driver itself, e.g. an
        X-ray/lamppost continuum, or a directly-monitored AGN continuum
        anchoring an emission-line fit. ``fit_error_model`` works the same
        way as for a band, turning on ``sigma_scale_driver``/
        ``sigma_jitter_driver``.
    sigma_drw_prior_scale : float
        Scale of ``sigma_drw``'s ``HalfNormal`` prior. ``EchoFit`` sets this
        from the registered light curves' own data (see
        ``EchoFit._sigma_drw_prior_scale``) rather than leaving it a fixed
        constant -- see the module docstring's "driver amplitude" note for
        why. Defaults to the old fixed value only for direct/standalone
        calls to this function.
    fixed_params : dict, optional
        ``{site_name: value}`` for any scalar site this model would
        otherwise sample -- see the module docstring's "fixed-parameter"
        note. Unrecognised keys are silently unused (``EchoFit`` validates
        them against the actual registered bands/driver before fitting).
    drw_prior : bool
        ``False`` (default): the driver's Fourier coefficients follow a
        pure random-walk (RW) prior, ``rw_prior_scale`` -- no ``tau_drw``
        site at all. ``True``: the original damped random walk (DRW)
        prior, ``drw_prior_scale``, with ``tau_drw`` inferred alongside
        ``sigma_drw``. See the module docstring's "random-walk prior" note.
    """
    fixed_params = fixed_params or {}

    def _param(name, dist_obj):
        """Sample ``name``, unless ``fixed_params`` pins it to a constant."""
        if name in fixed_params:
            return numpyro.deterministic(name, jnp.asarray(fixed_params[name], dtype=jnp.float32))
        return numpyro.sample(name, dist_obj)

    # -- shared driving-source hyperparameters ---------------------------
    sigma_drw = _param("sigma_drw", dist.HalfNormal(sigma_drw_prior_scale))
    if drw_prior:
        tau_drw = _param("tau_drw", dist.LogNormal(loc=jnp.log(20.0), scale=1.0))
        prior_scale = drw_prior_scale(freqs, sigma_drw, tau_drw)
    else:
        prior_scale = rw_prior_scale(freqs, sigma_drw)
    n_freq = freqs.shape[0]

    # Non-centred parameterisation: S, C's scale is itself a sampled
    # hyperparameter (via prior_scale(sigma_drw[, tau_drw])), which produces
    # a Neal's-funnel geometry if sampled directly ("centred") -- NUTS then
    # can't find one step size that works both where prior_scale is small
    # and where it's large, and every trajectory runs to max tree depth.
    # Sampling unit-scale S_raw/C_raw and pushing the hyperparameter
    # dependence into a deterministic transform removes that coupling.
    with numpyro.plate("freq", n_freq):
        S_raw = numpyro.sample("S_raw", dist.Normal(0.0, 1.0))
        C_raw = numpyro.sample("C_raw", dist.Normal(0.0, 1.0))
    S = numpyro.deterministic("S", S_raw * prior_scale)
    C = numpyro.deterministic("C", C_raw * prior_scale)

    # -- driver light curve: a direct, zero-lag anchor on X(t) itself ----
    if driver is not None:
        S_driver = _param("S_driver", dist.LogNormal(0.0, 1.0))
        C_driver = _param("C_driver", dist.Normal(0.0, 5.0))
        y_pred_driver = S_driver * driver_at(S, C, freqs, driver["t"]) + C_driver
        numpyro.deterministic("y_pred_driver", y_pred_driver)
        if driver.get("fit_error_model", False):
            sigma_scale_driver = _param("sigma_scale_driver", dist.LogNormal(0.0, 0.5))
            sigma_jitter_driver = _param("sigma_jitter_driver", dist.HalfNormal(jnp.mean(driver["yerr"])))
            sigma_eff_driver = jnp.sqrt((sigma_scale_driver * driver["yerr"]) ** 2 + sigma_jitter_driver ** 2)
        else:
            sigma_eff_driver = driver["yerr"]
        numpyro.sample("obs_driver", dist.Normal(y_pred_driver, sigma_eff_driver), obs=driver["y"])

    # -- shared physical reprocessing parameters (physical-mode bands only) --
    if any(d["lag_mode"] == "physical" for d in bands.values()):
        log_mdot = _param("log_mdot", dist.Normal(0.0, 1.0))
        if "inclination" in fixed_params:
            inclination = numpyro.deterministic(
                "inclination", jnp.asarray(fixed_params["inclination"], dtype=jnp.float32)
            )
        else:
            # Uniform in cos(inclination), not inclination itself -- see the
            # module docstring's "inclination note". inclination stays the
            # public (degrees) site every other function reads.
            cos_incl_min = jnp.cos(jnp.deg2rad(INCLINATION_MAX_DEG))
            cos_inclination = numpyro.sample("cos_inclination", dist.Uniform(cos_incl_min, 1.0))
            inclination = numpyro.deterministic("inclination", jnp.rad2deg(jnp.arccos(cos_inclination)))

    # tau_grid[-1], not float(...): under NUTS's internal while_loop tracing
    # tau_grid can be an abstract tracer, and dist.Uniform accepts a JAX
    # scalar directly -- no need to (and, when traced, can't) concretise it.
    tau_max = tau_grid[-1]

    # -- per-band amplitude / offset + likelihood ------------------------
    for band_name, d in bands.items():
        S_band = _param(f"S_{band_name}", dist.LogNormal(0.0, 1.0))
        C_band = _param(f"C_{band_name}", dist.Normal(0.0, 5.0))

        if d["lag_mode"] == "physical":
            psi = response_function(
                tau_grid,
                log_mdot=log_mdot,
                wavelength=d["wavelength"],
                inclination=inclination,
                M_BH=M_BH,
            )
        else:
            tau_band = _param(f"tau_{band_name}", dist.Uniform(0.0, tau_max))
            psi = tophat_response_free(tau_grid, tau_mean=tau_band)

        A, B = transfer_coeffs(tau_grid, psi, freqs)
        echo = compute_echo(S, C, freqs, A, B, d["t"])
        y_pred = S_band * echo + C_band

        numpyro.deterministic(f"y_pred_{band_name}", y_pred)
        if d.get("fit_error_model", False):
            sigma_scale = _param(f"sigma_scale_{band_name}", dist.LogNormal(0.0, 0.5))
            sigma_jitter = _param(f"sigma_jitter_{band_name}", dist.HalfNormal(jnp.mean(d["yerr"])))
            sigma_eff = jnp.sqrt((sigma_scale * d["yerr"]) ** 2 + sigma_jitter ** 2)
        else:
            sigma_eff = d["yerr"]
        numpyro.sample(
            f"obs_{band_name}",
            dist.Normal(y_pred, sigma_eff),
            obs=d["y"],
        )
