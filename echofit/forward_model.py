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

from typing import Callable, NamedTuple

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


# ----------------------------------------------------------------------
# Physical constants for thin_disk_response. Only used to convert the
# black hole mass into an inner (ISCO) radius -- the overall lag scale
# still comes from lag_scaling, see thin_disk_response's docstring.
# ----------------------------------------------------------------------
_G = 6.674e-11                              # m^3 kg^-1 s^-2
_C_LIGHT = 2.99792458e8                     # m / s
_M_SUN = 1.98855e30                         # kg
_LIGHT_DAY_M = _C_LIGHT * 86400.0           # metres per light-day
_HC_OVER_K_ANGSTROM = 1.4387773538277204e8  # hc / k_B, in Angstrom * Kelvin
_WIEN_X_PEAK = 4.965114231744276            # solution of x = 5(1 - e^-x)
_WIEN_B_ANGSTROM_KELVIN = _HC_OVER_K_ANGSTROM / _WIEN_X_PEAK


def _isco_light_days(M_BH):
    """Innermost stable circular orbit for a non-spinning black hole, in
    light-days: r_ISCO = 6 GM / c^2 (i.e. 3 Schwarzschild radii)."""
    r_g_m = _G * (M_BH * _M_SUN) / _C_LIGHT ** 2
    return 6.0 * r_g_m / _LIGHT_DAY_M


def thin_disk_response(
    tau_grid,
    log_mdot,
    wavelength,
    inclination,
    M_BH,
    viscous_slope: float = 0.75,
    include_irradiation: bool = False,
    irradiation_slope: float = 0.75,
    irradiation_weight: float = 0.5,
    n_r: int = 50,
    n_phi: int = 64,
    r_max_factor: float = 20.0,
    smoothing_days: float | None = None,
):
    """Causal, area-normalised transfer function from thin-disk reprocessing
    theory, integrated over a genuine 2-D (radius, azimuth) disk grid.

    Adapted (deterministic-quadrature, JAX-differentiable -- not a literal
    line-by-line translation) from the Monte-Carlo disk integrator
    ``tfbx``/``tr4visc``/``tr4irad`` in the author's PhD-era Fortran CREAM
    code (``pycecream``'s ``cream_f90.f90``). Physics kept:

    * Shakura-Sunyaev viscous temperature profile with an inner boundary
      term, ``T_visc^4(r) ~ (1 - sqrt(r_in/r)) / r**3``.
    * Optional lamppost irradiation temperature, ``T_irr^4(r) ~ 1/r**3``
      (same radial power as viscous, in the flat-disk limit).
    * The disk light-travel-time delay surface
      ``tau(r, phi) = r * (1 + sin(inclination) * cos(phi))`` -- this is
      what gives the response a genuine, inclination-driven skew and a hard
      causal edge at tau=0, rather than :func:`response_function`'s ad-hoc
      skew-normal shape.
    * A response weight equal to the Planck-function derivative with
      respect to temperature, ``X**5 * e**X / (e**X - 1)**2`` with
      ``X = hc / (k * lambda * T(r))``, i.e. how strongly a patch of disk
      at temperature ``T(r)`` responds, in the observing band, to a small
      heating perturbation.

    What is *not* replicated: the Fortran integrates by Monte Carlo
    sampling of (r, phi) on every call and smooths the result onto the tau
    grid with a Gaussian kernel -- not reproducible/differentiable in a
    NUTS-friendly way. This uses a fixed deterministic (log-radius x
    azimuth) quadrature grid instead, and deposits each grid point's
    contribution onto ``tau_grid`` with that same Gaussian-kernel idea --
    which also happens to be exactly what keeps this differentiable (a hard
    histogram deposit has the same zero-gradient problem as a hard top-hat;
    see :func:`tophat_response_free`'s docstring).

    Absolute normalisation: rather than independently deriving a physical
    Eddington-ratio-to-accretion-rate conversion (which would give
    ``log_mdot`` a second, incompatible meaning depending which response
    function a band uses), the characteristic (Wien-law) radius is taken
    directly from :func:`lag_scaling` -- so switching a band between
    ``response_function`` and this function preserves what ``log_mdot``
    means. Only the disk's *inner* edge (the ISCO) uses real physical
    constants (G, c, M_sun), since that conversion needs no separate
    accretion-rate calibration.

    Parameters
    ----------
    tau_grid : array_like, shape (n_tau,)
    log_mdot, wavelength, inclination : array_like
        Same meaning and units as in :func:`response_function`.
    M_BH : float
        Fixed black hole mass (solar masses).
    viscous_slope : float
        Temperature-radius power-law index (0.75 = standard thin-disk;
        matches the exponent implicit in ``lag_scaling``'s
        ``wavelength**(4/3)``). Fixed, not inferred.
    include_irradiation : bool
        If True, mix in a separate lamppost-irradiation temperature
        component (its own slope, weighted by ``irradiation_weight``)
        instead of a single-temperature-component viscous disk.
    irradiation_slope : float
        Only used if ``include_irradiation``.
    irradiation_weight : float
        Fraction of T**4 attributed to irradiation vs viscous heating at
        the characteristic radius, if ``include_irradiation``.
    n_r, n_phi : int
        Radial / azimuthal quadrature resolution (fixed, not inferred).
        Higher is more accurate and more expensive: this response is
        integrated over ``n_r * n_phi`` points per evaluation, unlike
        :func:`response_function`'s closed-form evaluation.
    r_max_factor : float
        Caps the radial integration domain at ``r_max_factor * tau_ref``
        (``tau_ref`` from :func:`lag_scaling`), instead of extending all
        the way to the geometrically-largest radius that could in
        principle contribute at ``tau_grid[-1]`` (``tau_grid[-1] / (1 -
        sin(inclination))``, which blows up at high inclination -- e.g.
        ~1000x ``tau_ref`` at 80 degrees for a typical grid). The physical
        response weight decays exponentially in radius (see the Planck
        derivative below) and is negligible far below that geometric
        extreme, so the cap loses essentially no real signal (confirmed
        directly against an uncapped, far-higher-resolution reference: max
        absolute difference ~2e-4) while keeping the log-spaced radial grid
        concentrated where the response actually has weight. Without this,
        the same ``n_r`` that resolves a face-on response smoothly leaves a
        high-inclination one visibly jagged, since log-spacing over a
        ~1000x larger domain is correspondingly ~1000x coarser near
        ``tau_ref`` -- this was caught by actually plotting the response at
        high inclination (see `docs/thin_disk_response.md`), not by the
        unit tests, which don't check smoothness.
    smoothing_days : float, optional
        Gaussian deposit bandwidth (days) used to bin disk-grid points onto
        ``tau_grid``. Defaults to 1.5x the ``tau_grid`` spacing.

    Returns
    -------
    psi : array_like, shape (n_tau,)
        Normalised response evaluated at ``tau_grid``, zero for tau < 0.
    """
    tau_ref = lag_scaling(log_mdot, wavelength, M_BH)
    r_in = jnp.clip(_isco_light_days(M_BH), 1e-4, 0.4 * tau_ref)
    r_max_geometric = tau_grid[-1] / jnp.clip(1.0 - jnp.sin(jnp.deg2rad(inclination)), 1e-2, None)
    r_max = jnp.minimum(r_max_geometric, r_max_factor * tau_ref)
    r_max = jnp.clip(r_max, 2.0 * tau_ref, None)

    # log-spaced radial quadrature grid; jnp.gradient gives each point's
    # local spacing for the area-element weight (same pattern as
    # model.drw_prior_scale's dw).
    r = jnp.exp(jnp.linspace(jnp.log(r_in), jnp.log(r_max), n_r))
    dr = jnp.gradient(r)
    phi = (jnp.arange(n_phi) + 0.5) * (2.0 * jnp.pi / n_phi)
    dphi = 2.0 * jnp.pi / n_phi

    # T**4(r), shape-only (normalised to 1 at r = tau_ref, the Wien radius).
    inner_term = (1.0 - jnp.sqrt(r_in / r)) / jnp.clip(
        1.0 - jnp.sqrt(r_in / tau_ref), 1e-6, None
    )
    t4_visc = (tau_ref / r) ** (4.0 * viscous_slope) * inner_term
    if include_irradiation:
        t4_irad = (tau_ref / r) ** (4.0 * irradiation_slope)
        t4_shape = irradiation_weight * t4_irad + (1.0 - irradiation_weight) * t4_visc
    else:
        t4_shape = t4_visc

    # Absolute temperature via Wien's law: T(tau_ref) satisfies
    # lambda * T = b exactly, tying the disk's temperature scale to the
    # observing band without any extra free calibration constant.
    t_ref_kelvin = _WIEN_B_ANGSTROM_KELVIN / wavelength
    T = t_ref_kelvin * jnp.clip(t4_shape, 1e-12, None) ** 0.25
    X = jnp.clip(_HC_OVER_K_ANGSTROM / (wavelength * T), None, 50.0)
    eX = jnp.exp(X)
    planck_deriv = X ** 5 * eX / (eX - 1.0) ** 2  # (n_r,)

    sininc = jnp.sin(jnp.deg2rad(inclination))
    tau_rphi = r[:, None] * (1.0 + sininc * jnp.cos(phi)[None, :])       # (n_r, n_phi)
    # the per-radius weight has no phi-dependence, so broadcast it across
    # the azimuth axis explicitly rather than relying on a (n_r, 1) shape
    # to survive the later .reshape(-1) as if it were (n_r, n_phi).
    weight_rphi = jnp.broadcast_to((planck_deriv * r * dr)[:, None] * dphi, (n_r, n_phi))

    tau_flat = tau_rphi.reshape(-1)
    weight_flat = weight_rphi.reshape(-1)

    if smoothing_days is None:
        sigma = 1.5 * jnp.mean(jnp.diff(tau_grid))
    else:
        sigma = smoothing_days

    kernel = jnp.exp(-0.5 * ((tau_grid[:, None] - tau_flat[None, :]) / sigma) ** 2)
    raw = kernel @ weight_flat
    raw = jnp.where(tau_grid >= 0.0, raw, 0.0)

    trapz = jnp.trapezoid if hasattr(jnp, "trapezoid") else jnp.trapz
    area = trapz(raw, tau_grid)
    psi = raw / jnp.clip(area, 1e-12, None)
    return psi


class ThinDiskResponseTable(NamedTuple):
    """Precomputed :func:`thin_disk_response` templates across inclination,
    for :func:`thin_disk_response_from_table` / :func:`build_thin_disk_response_fast`
    -- see those for why this exists."""

    M_BH: float
    incl_grid: jnp.ndarray       # (n_incl,) degrees
    u_grid: jnp.ndarray          # (n_u,) days, dimensionless-lag axis at the reference mdot/wavelength
    templates: jnp.ndarray       # (n_incl, n_u)
    tau_ref_reference: float     # lag_scaling(reference_log_mdot, reference_wavelength, M_BH)


def build_thin_disk_response_table(
    M_BH,
    incl_grid=None,
    reference_log_mdot: float = 0.0,
    reference_wavelength: float = 5000.0,
    u_max_factor: float = 15.0,
    n_u: int = 1200,
    n_r: int = 400,
    n_phi: int = 96,
    **thin_disk_kwargs,
) -> ThinDiskResponseTable:
    """Precompute a lookup table of :func:`thin_disk_response` shapes across
    inclination, at high quadrature resolution, once -- so a fit can look
    the response up (:func:`thin_disk_response_from_table`) instead of
    re-running the full O(n_r * n_phi * n_tau) disk integral on every NUTS
    step.

    This is a JAX-differentiable version of the precompute-and-interpolate
    trick used in the author's PhD-era CREAM Fortran code: precompute the
    response at a fixed accretion rate across an inclination grid once at
    the start of a run, then get any other inclination by interpolating the
    table, and any other accretion rate (or wavelength) by *stretching* the
    lag axis according to the ``mdot**(1/3)`` (and ``wavelength**(4/3)``)
    scaling :func:`lag_scaling` already uses. Section 5 of
    ``docs/thin_disk_response.md`` explains why that stretch is only
    approximate (a fixed absolute inner radius means the disk isn't
    *exactly* self-similar under it), and it is the same approximation the
    original Fortran made, not something new introduced here.

    The returned table is a plain, fixed set of arrays, meant to be built
    *before* a fit (M_BH is fixed input anyway, per CLAUDE.md decision #3)
    and passed to :func:`build_thin_disk_response_fast` to get an
    interpolation-based response function ready to assign to
    ``echofit.model.response_function``.

    Parameters
    ----------
    M_BH : float
        Fixed black hole mass, matching the run this table is for.
    incl_grid : array_like, optional
        Inclinations (degrees) to precompute templates at. Defaults to
        every 2.5 degrees from 0 to 90 (37 templates).
    reference_log_mdot, reference_wavelength : float
        The one (log_mdot, wavelength) pair templates are actually computed
        at; every other (log_mdot, wavelength) is reached by stretching.
    u_max_factor : float
        The template's own lag axis spans ``[-0.05, u_max_factor] *
        tau_ref_reference``. Must be generous enough that, after stretching,
        it still covers whatever part of the real ``tau_grid`` matters for
        the (log_mdot, wavelength) values a fit actually visits -- lookups
        outside this range return 0 (see :func:`thin_disk_response_from_table`),
        which is safe but silently loses accuracy in the tail if too small.
    n_u : int
        Resolution of the template's own lag axis.
    n_r, n_phi : int
        Passed through to :func:`thin_disk_response` for the (one-off,
        so affordably high-resolution) template computation.
    **thin_disk_kwargs
        Any other :func:`thin_disk_response` keyword (``viscous_slope``,
        ``include_irradiation``, ...), applied identically to every
        inclination in the table.
    """
    if incl_grid is None:
        incl_grid = jnp.arange(0.0, 90.001, 2.5)
    else:
        incl_grid = jnp.asarray(incl_grid)

    tau_ref_reference = float(lag_scaling(reference_log_mdot, reference_wavelength, M_BH))
    u_grid = jnp.linspace(-0.05 * tau_ref_reference, u_max_factor * tau_ref_reference, n_u)

    templates = jnp.stack([
        thin_disk_response(
            u_grid, reference_log_mdot, reference_wavelength, float(incl), M_BH,
            n_r=n_r, n_phi=n_phi, **thin_disk_kwargs,
        )
        for incl in incl_grid
    ])

    return ThinDiskResponseTable(
        M_BH=M_BH, incl_grid=incl_grid, u_grid=u_grid,
        templates=templates, tau_ref_reference=tau_ref_reference,
    )


def thin_disk_response_from_table(table: ThinDiskResponseTable, tau_grid, log_mdot, wavelength, inclination):
    """Fast, interpolated stand-in for :func:`thin_disk_response`, using a
    precomputed :class:`ThinDiskResponseTable` (see
    :func:`build_thin_disk_response_table`) instead of recomputing the disk
    integral. Two cheap lookups replace it:

    1. Interpolate the template family over ``inclination`` (linear,
       differentiable in ``inclination`` almost everywhere, same as any
       other lookup-table use in JAX) to get one dimensionless template for
       this exact inclination, still on the table's own ``u_grid``.
    2. Stretch that template onto the real ``tau_grid`` by
       ``s = lag_scaling(log_mdot, wavelength, M_BH) / table.tau_ref_reference``
       -- i.e. evaluate it at ``tau_grid / s`` and divide by ``s`` to keep
       the area normalised to 1 under that change of variables -- which is
       exactly the "stretch to a different mdot" trick from
       :func:`build_thin_disk_response_table`'s docstring.

    Both steps are ``jnp.interp`` against a fixed table, so this is `O(n_u +
    n_tau)` per call rather than :func:`thin_disk_response`'s `O(n_r * n_phi
    * n_tau)` disk integral -- the whole point for use inside NUTS. Queries
    landing outside the table's ``u_grid`` (an accretion rate/wavelength
    combination far from what the table was built for) return 0 rather than
    extrapolating, which is safe but a sign the table needs a larger
    ``u_max_factor`` or a reference point closer to where the fit actually
    lives.
    """
    tau_ref = lag_scaling(log_mdot, wavelength, table.M_BH)
    stretch = jnp.clip(tau_ref / table.tau_ref_reference, 1e-6, None)

    psi_u = jax.vmap(lambda column: jnp.interp(inclination, table.incl_grid, column))(table.templates.T)

    u_query = tau_grid / stretch
    raw = jnp.interp(u_query, table.u_grid, psi_u, left=0.0, right=0.0) / stretch
    raw = jnp.where(tau_grid >= 0.0, raw, 0.0)

    trapz = jnp.trapezoid if hasattr(jnp, "trapezoid") else jnp.trapz
    area = trapz(raw, tau_grid)
    return raw / jnp.clip(area, 1e-12, None)


def build_thin_disk_response_fast(M_BH, **table_kwargs) -> Callable:
    """Build and return a ready-to-use, interpolation-based response
    function matching the standard ``(tau_grid, log_mdot, wavelength,
    inclination, M_BH, ...)`` contract (CLAUDE.md decision #5) -- the
    one-call convenience wrapper around :func:`build_thin_disk_response_table`
    + :func:`thin_disk_response_from_table`::

        import echofit.model as model
        from echofit.forward_model import build_thin_disk_response_fast

        model.response_function = build_thin_disk_response_fast(M_BH=1e8)

    The underlying table (useful for inspecting/plotting what got
    precomputed) is attached as ``.table`` on the returned function.
    """
    table = build_thin_disk_response_table(M_BH, **table_kwargs)

    def _response(tau_grid, log_mdot, wavelength, inclination, M_BH=None, **kwargs):
        return thin_disk_response_from_table(table, tau_grid, log_mdot, wavelength, inclination)

    _response.table = table
    return _response


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
