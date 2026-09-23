"""
forward_model.py
=================

Deterministic pieces of the reverberation-mapping forward model:

1. ``lag_scaling``       -- physically motivated mean-lag scaling law.
2. ``response_function`` -- causal, positive transfer function psi(tau).
3. ``transfer_coeffs``   -- Fourier cosine/sine transform of psi on a fixed
                            frequency grid (used to convolve analytically).
4. ``compute_echo``      -- evaluate the convolution of a Fourier-series
                            driver with psi at arbitrary observation times.

Design notes
------------
The driver is represented as a truncated Fourier series::

    X(t) = sum_k [ S_k sin(w_k t) + C_k cos(w_k t) ]

Because the driver is *exactly* a sum of sinusoids, the convolution with the
response function has a closed form in terms of the response function's
Fourier transform evaluated at each driver frequency w_k:

    A_k = int psi(tau) cos(w_k tau) dtau
    B_k = int psi(tau) sin(w_k tau) dtau

    int psi(tau) X(t - tau) dtau
        = sum_k [ sin(w_k t) (S_k A_k + C_k B_k)
                  + cos(w_k t) (C_k A_k - S_k B_k) ]

This avoids ever looping over observation times and tau jointly (an
O(n_obs * n_tau) operation); instead the expensive tau-integral is done once
per frequency (O(n_freq * n_tau)) to get (A_k, B_k), and then the echo at any
set of times is a single O(n_obs * n_freq) matrix contraction. Everything is
written with ``jax.numpy`` so it vectorises and is differentiable / jittable
end to end, which is what NumPyro's NUTS sampler needs.

The response function is intentionally isolated in ``response_function`` so
it can be swapped for a different parametric family later without touching
``compute_echo`` or the NumPyro model.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import erf as _erf

# ----------------------------------------------------------------------
# Pivot / calibration constants for the lag-scaling law.
# These set the overall day-scale and are not inferred.
# ----------------------------------------------------------------------
TAU0_DAYS = 1.0          # normalisation: lag (days) at the pivot point below
M_BH_PIVOT = 1.0e8       # pivot black hole mass, solar masses
MDOT_PIVOT = 1.0         # pivot dimensionless accretion rate (10**log_mdot)
LAMBDA_PIVOT = 5000.0    # pivot wavelength, Angstrom


def lag_scaling(log_mdot, wavelength, M_BH):
    """Mean disk-reprocessing lag, in days.

    Follows the standard lamppost/thin-disk scaling for the radius at which
    the disk temperature matches a given observing wavelength,
    R(lambda) ~ M_BH^(2/3) Mdot^(1/3) lambda^(4/3), converted to a light
    travel time. Black hole mass is a *fixed input*, not inferred.

    Parameters
    ----------
    log_mdot : array_like
        log10 of the (dimensionless) mass accretion rate. Inferred.
    wavelength : array_like
        Observing wavelength in Angstrom (rest frame).
    M_BH : float
        Black hole mass in solar masses. Fixed, not inferred.

    Returns
    -------
    tau_mean : array_like
        Mean lag in days, same broadcast shape as (log_mdot, wavelength).
    """
    mdot = 10.0 ** log_mdot
    tau_mean = (
        TAU0_DAYS
        * (M_BH / M_BH_PIVOT) ** (2.0 / 3.0)
        * (mdot / MDOT_PIVOT) ** (1.0 / 3.0)
        * (wavelength / LAMBDA_PIVOT) ** (4.0 / 3.0)
    )
    return tau_mean


def _skew_normal_pdf(x, loc, scale, alpha):
    """Standard skew-normal pdf (unnormalised area beyond truncation)."""
    z = (x - loc) / scale
    phi = jnp.exp(-0.5 * z ** 2) / jnp.sqrt(2.0 * jnp.pi)
    Phi = 0.5 * (1.0 + _erf(alpha * z / jnp.sqrt(2.0)))
    return (2.0 / scale) * phi * Phi


def response_function(
    tau_grid,
    log_mdot,
    wavelength,
    inclination,
    M_BH,
    width_frac: float = 0.35,
    skew_offset: float = 1.0,
    skew_incl_slope: float = 4.0,
):
    """Causal, positive transfer function psi(tau, lambda; theta).

    A skew-normal response, truncated to tau >= 0 and re-normalised on the
    supplied ``tau_grid`` so that ``trapz(psi, tau_grid) == 1``. The mean lag
    is set purely by :func:`lag_scaling` (accretion rate, wavelength, fixed
    mass). Inclination controls only the *skewness*, not the mean lag, per
    the physical requirement that inclination reshapes the iso-delay surface
    without changing the mean reprocessing radius.

    Parameters
    ----------
    tau_grid : array_like, shape (n_tau,)
        Non-negative lag grid (days) the response is evaluated / normalised
        on.
    log_mdot, wavelength, inclination : array_like
        Physical parameters. ``inclination`` in degrees, 0 = face-on.
    M_BH : float
        Fixed black hole mass (solar masses).
    width_frac : float
        Response width as a fraction of the mean lag.
    skew_offset, skew_incl_slope : float
        alpha = skew_offset + skew_incl_slope * sin(inclination); this keeps
        the response symmetric-ish face-on and increasingly skewed at high
        inclination, without touching the mean lag.

    Returns
    -------
    psi : array_like, shape (n_tau,)
        Normalised response evaluated at ``tau_grid``, zero for tau < 0.
    """
    tau_mean = lag_scaling(log_mdot, wavelength, M_BH)
    scale = jnp.clip(width_frac * tau_mean, 1e-3, None)
    alpha = skew_offset + skew_incl_slope * jnp.sin(jnp.deg2rad(inclination))

    raw = _skew_normal_pdf(tau_grid, loc=tau_mean, scale=scale, alpha=alpha)
    raw = jnp.where(tau_grid >= 0.0, raw, 0.0)

    area = jnp.trapezoid(raw, tau_grid) if hasattr(jnp, "trapezoid") else jnp.trapz(raw, tau_grid)
    psi = raw / jnp.clip(area, 1e-12, None)
    return psi


def tophat_response_free(tau_grid, tau_mean, width_frac: float = 0.3, edge_softness_frac: float = 0.15):
    """Causal top-hat-*like* response centred directly on ``tau_mean``, with
    smoothed (not hard) edges -- see "Why not a literal top-hat" below.

    Unlike :func:`response_function`, ``tau_mean`` is taken as given rather
    than derived from ``lag_scaling(log_mdot, wavelength, M_BH)`` -- this is
    the response used for a band in "free" lag mode (see
    ``model.reverberation_model``'s ``bands[...]["lag_mode"]``), where the
    lag itself is an independently inferred parameter rather than tied to
    the other bands through the shared thin-disk scaling law. That
    independence is exactly what makes a *single* such band's lag
    unidentifiable from its light curve alone (a global shift of the driver
    and an equal shift of the lag leave the data unchanged -- see
    CLAUDE.md); a fit using this response for any band should also register
    a driver light curve (:meth:`~echofit.echofit.EchoFit.add_driver_lightcurve`)
    to anchor the absolute lag scale, or tie multiple free-lag bands
    together some other way.

    Why not a literal top-hat: a hard ``jnp.where(|tau - tau_mean| <=
    half_width, 1, 0)`` box has **exactly zero gradient** with respect to
    ``tau_mean`` everywhere except the measure-zero edge (autodiff doesn't
    backprop through a comparison's operands) -- since ``tau_mean`` is a
    ``numpyro.sample`` site here, NUTS would see zero gradient signal and
    be unable to move it at all (confirmed directly: ``jax.grad`` of the
    hard version w.r.t. ``tau_mean`` is identically ``0.0``). This uses a
    sigmoid-smoothed edge instead -- still a box in the limit
    ``edge_softness_frac -> 0``, but differentiable everywhere.

    Parameters
    ----------
    tau_grid : array_like, shape (n_tau,)
    tau_mean : array_like
        The lag (days), e.g. a ``numpyro.sample`` site rather than a fixed
        constant.
    width_frac : float
        Half-width as a fraction of ``tau_mean`` (fixed, not inferred --
        matches ``response_function``'s ``width_frac`` convention). Note
        this width-scales-with-the-lag convention means the shift
        degeneracy above is not perfectly exact when ``tau_mean`` itself
        changes (the width changes slightly with it, unlike a true
        ``psi'(tau) = psi(tau + Delta)`` shift, which preserves width) --
        see ``tests/test_shift_degeneracy.py`` for the width-independent
        version used to verify the degeneracy claim exactly.
    edge_softness_frac : float
        Edge transition width as a fraction of the half-width (fixed, not
        inferred). Smaller is closer to a true top-hat but with a narrower
        region of usable gradient for NUTS to find the edges through.

    Returns
    -------
    psi : array_like, shape (n_tau,)
        Normalised response evaluated at ``tau_grid``, ~zero well outside
        ``[tau_mean - half_width, tau_mean + half_width]`` and for tau < 0.
    """
    half_width = jnp.clip(width_frac * tau_mean, 1e-3, None)
    edge_softness = jnp.clip(edge_softness_frac * half_width, 1e-3, None)
    left_edge = tau_mean - half_width
    right_edge = tau_mean + half_width
    rising = jax.nn.sigmoid((tau_grid - left_edge) / edge_softness)
    falling = jax.nn.sigmoid((right_edge - tau_grid) / edge_softness)
    raw = rising * falling
    raw = jnp.where(tau_grid >= 0.0, raw, 0.0)

    trapz = jnp.trapezoid if hasattr(jnp, "trapezoid") else jnp.trapz
    area = trapz(raw, tau_grid)
    psi = raw / jnp.clip(area, 1e-12, None)
    return psi


def transfer_coeffs(tau_grid, psi, freqs):
    """Fourier cosine/sine transform of psi at each driver frequency.

    A_k = int psi(tau) cos(w_k tau) dtau
    B_k = int psi(tau) sin(w_k tau) dtau

    Parameters
    ----------
    tau_grid : array_like, shape (n_tau,)
    psi : array_like, shape (n_tau,)
    freqs : array_like, shape (n_freq,)
        Angular frequencies w_k of the driver's Fourier basis.

    Returns
    -------
    A, B : array_like, shape (n_freq,)
    """
    # outer product: (n_freq, n_tau)
    phase = freqs[:, None] * tau_grid[None, :]
    trapz = jnp.trapezoid if hasattr(jnp, "trapezoid") else jnp.trapz
    A = trapz(psi[None, :] * jnp.cos(phase), tau_grid, axis=-1)
    B = trapz(psi[None, :] * jnp.sin(phase), tau_grid, axis=-1)
    return A, B


def compute_echo(S, C, freqs, A, B, t_obs):
    """Evaluate int psi(tau) X(t_obs - tau) dtau for a Fourier-series driver.

    Parameters
    ----------
    S, C : array_like, shape (n_freq,)
        Driver sine/cosine amplitudes.
    freqs : array_like, shape (n_freq,)
        Driver angular frequencies w_k.
    A, B : array_like, shape (n_freq,)
        Response transfer coefficients from :func:`transfer_coeffs`.
    t_obs : array_like, shape (n_obs,)
        Observation times at which to evaluate the echo.

    Returns
    -------
    echo : array_like, shape (n_obs,)
    """
    wt = freqs[None, :] * t_obs[:, None]           # (n_obs, n_freq)
    sin_coef = S * A + C * B                        # (n_freq,)
    cos_coef = C * A - S * B                        # (n_freq,)
    echo = jnp.sin(wt) @ sin_coef + jnp.cos(wt) @ cos_coef
    return echo


def driver_at(S, C, freqs, t):
    """Evaluate the raw driver X(t) = sum_k S_k sin(w_k t) + C_k cos(w_k t).

    Convenience function for plotting / diagnostics (not used in the echo
    convolution itself, which uses the closed-form ``compute_echo``).
    """
    wt = freqs[None, :] * t[:, None]
    return jnp.sin(wt) @ S + jnp.cos(wt) @ C
