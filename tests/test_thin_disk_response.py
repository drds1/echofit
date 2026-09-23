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

from echofit.forward_model import thin_disk_response
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
