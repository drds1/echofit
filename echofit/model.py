"""
model.py
========

The NumPyro probabilistic model tying together:

* a damped-random-walk (DRW) driver, represented as a truncated Fourier
  series whose sine/cosine amplitudes are given Gaussian priors matching the
  DRW's Lorentzian power spectrum (so ``sigma_drw`` and ``tau_drw`` are
  genuine, interpretable DRW hyperparameters that get inferred);
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

Inferred, always: sigma_drw, tau_drw, {S_k, C_k} driver Fourier coefficients,
{S_band, C_band} per band. Inferred if any band uses the physical response:
log_mdot, inclination (shared across those bands). Inferred per band using
the free-lag response: tau_{band}. Inferred if a driver light curve is
given: S_driver, C_driver.

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
"""

from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .forward_model import response_function, tophat_response_free, transfer_coeffs, compute_echo, driver_at


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


def reverberation_model(
    freqs: jnp.ndarray,
    tau_grid: jnp.ndarray,
    M_BH: Optional[float],
    bands: Dict[str, dict],
    driver: Optional[dict] = None,
    sigma_drw_prior_scale: float = 2.0,
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
        Mapping ``band_name -> {"t", "y", "yerr", "wavelength", "lag_mode"}``
        for each observed light curve. ``lag_mode`` is ``"physical"`` (mean
        lag tied to the shared ``log_mdot`` via ``lag_scaling``) or
        ``"free"`` (an independently inferred ``tau_{band_name}``) -- see
        the module docstring's identifiability note for when ``"free"``
        needs a ``driver`` to be identifiable.
    driver : dict, optional
        ``{"t", "y", "yerr"}`` for a light curve that directly (zero-lag)
        observes the driver itself, e.g. an X-ray/lamppost continuum, or a
        directly-monitored AGN continuum anchoring an emission-line fit.
    sigma_drw_prior_scale : float
        Scale of ``sigma_drw``'s ``HalfNormal`` prior. ``EchoFit`` sets this
        from the registered light curves' own data (see
        ``EchoFit._sigma_drw_prior_scale``) rather than leaving it a fixed
        constant -- see the module docstring's "driver amplitude" note for
        why. Defaults to the old fixed value only for direct/standalone
        calls to this function.
    """
    # -- shared driving-source (DRW) hyperparameters --------------------
    sigma_drw = numpyro.sample("sigma_drw", dist.HalfNormal(sigma_drw_prior_scale))
    tau_drw = numpyro.sample("tau_drw", dist.LogNormal(loc=jnp.log(20.0), scale=1.0))

    prior_scale = drw_prior_scale(freqs, sigma_drw, tau_drw)
    n_freq = freqs.shape[0]

    # Non-centred parameterisation: S, C's scale is itself a sampled
    # hyperparameter (via prior_scale(sigma_drw, tau_drw)), which produces
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
        S_driver = numpyro.sample("S_driver", dist.LogNormal(0.0, 1.0))
        C_driver = numpyro.sample("C_driver", dist.Normal(0.0, 5.0))
        y_pred_driver = S_driver * driver_at(S, C, freqs, driver["t"]) + C_driver
        numpyro.deterministic("y_pred_driver", y_pred_driver)
        numpyro.sample("obs_driver", dist.Normal(y_pred_driver, driver["yerr"]), obs=driver["y"])

    # -- shared physical reprocessing parameters (physical-mode bands only) --
    if any(d["lag_mode"] == "physical" for d in bands.values()):
        log_mdot = numpyro.sample("log_mdot", dist.Normal(0.0, 1.0))
        inclination = numpyro.sample("inclination", dist.Uniform(0.0, 80.0))

    # tau_grid[-1], not float(...): under NUTS's internal while_loop tracing
    # tau_grid can be an abstract tracer, and dist.Uniform accepts a JAX
    # scalar directly -- no need to (and, when traced, can't) concretise it.
    tau_max = tau_grid[-1]

    # -- per-band amplitude / offset + likelihood ------------------------
    for band_name, d in bands.items():
        S_band = numpyro.sample(f"S_{band_name}", dist.LogNormal(0.0, 1.0))
        C_band = numpyro.sample(f"C_{band_name}", dist.Normal(0.0, 5.0))

        if d["lag_mode"] == "physical":
            psi = response_function(
                tau_grid,
                log_mdot=log_mdot,
                wavelength=d["wavelength"],
                inclination=inclination,
                M_BH=M_BH,
            )
        else:
            tau_band = numpyro.sample(f"tau_{band_name}", dist.Uniform(0.0, tau_max))
            psi = tophat_response_free(tau_grid, tau_mean=tau_band)

        A, B = transfer_coeffs(tau_grid, psi, freqs)
        echo = compute_echo(S, C, freqs, A, B, d["t"])
        y_pred = S_band * echo + C_band

        numpyro.deterministic(f"y_pred_{band_name}", y_pred)
        numpyro.sample(
            f"obs_{band_name}",
            dist.Normal(y_pred, d["yerr"]),
            obs=d["y"],
        )
