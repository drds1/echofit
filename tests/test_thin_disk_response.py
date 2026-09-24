"""
Tests for forward_model.thin_disk_response (the accretion-disk response
ported from the PhD-era CREAM Fortran, see CLAUDE.md) and the small
echofit.responses registry that makes it (and any custom response) a
one-line, discoverable swap-in for "physical"-mode bands.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from echofit.forward_model import thin_disk_response, disk_temperature_profile, lag_scaling, _schwarzschild_radius_light_days, _WIEN_B_ANGSTROM_KELVIN
from echofit.responses import available_responses, get_response, register_response

_np_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def test_thin_disk_response_is_causal_and_normalised():
    tau_grid = jnp.linspace(-5.0, 80.0, 300)
    psi = thin_disk_response(
        tau_grid, log_mdot=0.0, wavelength=5000.0, inclination=30.0, M_BH=1e8
    )
    psi_np = np.asarray(psi)
    tau_np = np.asarray(tau_grid)
    assert not np.isnan(psi_np).any()
    assert np.all(psi_np[tau_np < 0] == 0.0)
    area = _np_trapz(psi_np, tau_np)
    assert np.isclose(area, 1.0, atol=1e-2)


def test_thin_disk_response_peak_lag_scales_with_wavelength():
    tau_grid = jnp.linspace(-5.0, 150.0, 400)

    def peak_lag(wavelength):
        psi = thin_disk_response(
            tau_grid, log_mdot=0.0, wavelength=wavelength, inclination=20.0, M_BH=1e8
        )
        psi_np = np.asarray(psi)
        return float(np.asarray(tau_grid)[np.argmax(psi_np)])

    assert peak_lag(3000.0) < peak_lag(9000.0)


def test_thin_disk_response_has_nonzero_gradients():
    """The same failure mode as tophat_response_free's hard edges: a
    zero-gradient response would look fine (finite, normalised) but leave
    NUTS unable to move log_mdot/inclination at all."""
    tau_grid = jnp.linspace(0.0, 80.0, 200)

    def mean_lag(log_mdot, inclination):
        psi = thin_disk_response(
            tau_grid, log_mdot, wavelength=5000.0, inclination=inclination, M_BH=1e8
        )
        return jnp.sum(psi * tau_grid)

    grad_mdot = jax.grad(mean_lag, argnums=0)(0.0, 30.0)
    grad_incl = jax.grad(mean_lag, argnums=1)(0.0, 30.0)
    assert float(grad_mdot) != 0.0
    assert float(grad_incl) != 0.0


def test_thin_disk_response_inclination_increases_skew_not_just_shifts_mean():
    """Mirrors CLAUDE.md decision #4 (checked already for response_function):
    inclination should reshape the response, not simply translate it -- a
    face-on disk's iso-delay surface is flatter than an inclined one's."""
    tau_grid = jnp.linspace(0.0, 100.0, 500)
    tau_np = np.asarray(tau_grid)

    psi_face_on = np.asarray(
        thin_disk_response(tau_grid, log_mdot=0.0, wavelength=5000.0, inclination=1.0, M_BH=1e8)
    )
    psi_inclined = np.asarray(
        thin_disk_response(tau_grid, log_mdot=0.0, wavelength=5000.0, inclination=70.0, M_BH=1e8)
    )

    def skewness(psi):
        mean = _np_trapz(psi * tau_np, tau_np)
        var = _np_trapz(psi * (tau_np - mean) ** 2, tau_np)
        third = _np_trapz(psi * (tau_np - mean) ** 3, tau_np)
        return third / var ** 1.5

    assert skewness(psi_inclined) > skewness(psi_face_on)


def test_thin_disk_response_smoothing_days_zero_gives_exact_mean_lag_independence():
    """With smoothing off (smoothing_days=0.0), the analytic azimuthal
    integral is exact, so decision #4's mean-lag-independent-of-inclination
    claim should hold tightly, not just approximately -- this is the
    "escape hatch" for anyone who wants that exactness back at the cost of
    the visible kink the default smoothing exists to remove (see the next
    test)."""
    tau_grid = jnp.linspace(-1.0, 8.0, 800)
    tau_np = np.asarray(tau_grid)

    lags = []
    for inclination in [0.0, 40.0, 80.0]:
        psi = np.asarray(thin_disk_response(
            tau_grid, log_mdot=0.0, wavelength=4000.0, inclination=inclination,
            M_BH=1e8, smoothing_days=0.0,
        ))
        lags.append(_np_trapz(psi * tau_np, tau_np))

    drift = abs(lags[-1] - lags[0]) / lags[0]
    assert drift < 0.01, f"expected near-exact mean-lag independence with smoothing off, got {drift:.4f} drift"


def test_thin_disk_response_default_smoothing_mean_lag_drift_is_bounded():
    """The default smoothing_frac=0.4 (chosen to match Starkey+2016 Figure
    3's shape, see docs/thin_disk_response.md section 4) trades some of
    the smoothing_days=0.0 exactness above for a smoother curve: causal
    smoothing near a boundary that a high-inclination response sits much
    closer to than a face-on one inherently breaks perfect mean-lag
    independence. This is a deliberate, discussed trade-off (not a bug),
    but the drift should stay in the ballpark it was chosen at (~10%
    face-on to 80 degrees) -- this catches an accidental regression to
    something much larger, not a change in the trade-off itself."""
    tau_grid = jnp.linspace(-1.0, 8.0, 800)
    tau_np = np.asarray(tau_grid)

    lags = []
    for inclination in [0.0, 40.0, 80.0]:
        psi = np.asarray(thin_disk_response(
            tau_grid, log_mdot=0.0, wavelength=4000.0, inclination=inclination, M_BH=1e8,
        ))
        lags.append(_np_trapz(psi * tau_np, tau_np))

    drift = abs(lags[-1] - lags[0]) / lags[0]
    assert 0.03 < drift < 0.2, (
        f"expected the default smoothing's mean-lag drift to stay near the ~10% it was "
        f"chosen at (face-on to 80 degrees), got {drift:.4f} -- see docs/thin_disk_response.md"
    )


def test_registry_looks_up_built_ins():
    assert "thin_disk" in available_responses()
    assert "skew_normal" in available_responses()
    assert get_response("thin_disk") is thin_disk_response


def test_registry_raises_with_helpful_message_for_unknown_name():
    with pytest.raises(KeyError, match="not_a_real_response"):
        get_response("not_a_real_response")


def test_registry_supports_custom_registration():
    def my_response(tau_grid, log_mdot, wavelength, inclination, M_BH, **kwargs):
        return thin_disk_response(tau_grid, log_mdot, wavelength, inclination, M_BH)

    register_response("my_response", my_response)
    try:
        assert get_response("my_response") is my_response
        assert "my_response" in available_responses()
    finally:
        from echofit.responses import _REGISTRY

        del _REGISTRY["my_response"]


def test_thin_disk_response_is_usable_via_the_existing_swap_mechanism():
    """thin_disk_response matches the same (tau_grid, log_mdot, wavelength,
    inclination, M_BH) contract as response_function, so it plugs into the
    existing echofit.model.response_function swap point (CLAUDE.md decision
    #5) with no further wiring -- exactly the point of the registry."""
    import echofit.model as model_mod
    from echofit.synthetic import generate_synthetic_dataset
    from echofit.echofit import EchoFit

    original = model_mod.response_function
    try:
        model_mod.response_function = get_response("thin_disk")

        data = generate_synthetic_dataset(
            M_BH=1.0e8, bands={"g": 4770.0}, n_obs_per_band=15,
            n_freq=6, n_tau=60, tau_max=40.0, noise_level=0.08, seed=0,
        )
        ef = EchoFit(M_BH=1.0e8)
        for name, d in data["bands"].items():
            ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
        ef.build_grid(n_freq=6, n_tau=60)
        ef.fit(rng_seed=0, num_warmup=10, num_samples=10, progress_bar=False)
    finally:
        model_mod.response_function = original


def test_disk_temperature_profile_matches_wien_law_at_the_reference_radius():
    """T(tau_ref) should equal the Wien's-law reference temperature exactly
    -- this is the same anchor point thin_disk_response's own internal
    computation uses to set its absolute temperature scale."""
    M_BH, wavelength, log_mdot = 1e8, 5000.0, 0.0
    tau_ref = lag_scaling(log_mdot, wavelength, M_BH)
    T = float(disk_temperature_profile(tau_ref, log_mdot, wavelength, M_BH))
    assert T == pytest.approx(_WIEN_B_ANGSTROM_KELVIN / wavelength, rel=1e-5)


def test_disk_temperature_profile_vanishes_at_the_isco():
    """No viscous dissipation right at the inner boundary -- the standard
    Shakura-Sunyaev condition, T(r_in) = 0 (down to the function's own
    numerical floor). Checked relative to the temperature well clear of
    the ISCO, rather than an absolute Kelvin value, since the floor itself
    scales with the Wien-law reference temperature (M_BH/wavelength
    dependent)."""
    M_BH, wavelength, log_mdot = 1e8, 5000.0, 0.0
    r_in = 3.0 * _schwarzschild_radius_light_days(M_BH)
    T_at_isco = float(disk_temperature_profile(r_in, log_mdot, wavelength, M_BH))
    T_away = float(disk_temperature_profile(10.0 * r_in, log_mdot, wavelength, M_BH))
    assert T_at_isco < 1e-2 * T_away


def test_disk_temperature_profile_declines_far_from_the_isco():
    """Away from the inner-boundary region (where the profile genuinely
    peaks just outside the ISCO before declining -- not monotonic from
    r_in itself), temperature should fall off smoothly with radius."""
    M_BH, wavelength, log_mdot = 1e8, 5000.0, 0.0
    r_in = 3.0 * _schwarzschild_radius_light_days(M_BH)
    r = np.geomspace(50.0 * r_in, 50.0, 20)  # well clear of the near-ISCO peak
    T = np.asarray(disk_temperature_profile(r, log_mdot, wavelength, M_BH))
    assert np.all(np.diff(T) < 0.0)
