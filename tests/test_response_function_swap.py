"""
Regression test for CLAUDE.md design decision #5 ("response function is
swappable by contract"): swapping ``echofit.model.response_function`` must
be honoured by *both* fitting and ``EchoFit.plot_lightcurve_fits()``.

It previously wasn't -- ``echofit.py`` had its own independent
``from .forward_model import response_function`` import, so a swap on
``model.py`` changed what NUTS fit but not what the plot re-derived psi
from, silently producing a plot inconsistent with the actual fit.
"""

import numpy as np
import jax.numpy as jnp

import echofit.model as model_mod
from echofit.forward_model import lag_scaling
from echofit.synthetic import generate_synthetic_dataset
from echofit.echofit import EchoFit


def _tophat_response(tau_grid, log_mdot, wavelength, inclination, M_BH, width_frac=0.3):
    """A flat-topped boxcar response -- unlike the default skew-normal,
    its values near the peak are (near-)identical, which is what the test
    below checks for."""
    tau_mean = lag_scaling(log_mdot, wavelength, M_BH)
    half_width = jnp.clip(width_frac * tau_mean, 1e-3, None)
    raw = jnp.where(jnp.abs(tau_grid - tau_mean) <= half_width, 1.0, 0.0)
    trapz = jnp.trapezoid if hasattr(jnp, "trapezoid") else jnp.trapz
    area = trapz(raw, tau_grid)
    return raw / jnp.clip(area, 1e-12, None)


def _fit_and_get_psi_median(seed=0):
    data = generate_synthetic_dataset(
        M_BH=1.0e9, bands={"g": 4770.0}, n_obs_per_band=20,
        n_freq=8, n_tau=100, tau_max=50.0, noise_level=0.08, seed=seed,
    )
    ef = EchoFit(M_BH=1.0e9)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=8, n_tau=100)
    ef.fit(rng_seed=0, num_warmup=30, num_samples=30, progress_bar=False)

    fig, axes = ef.plot_lightcurve_fits(n_fine=30, n_pred_samples=10)
    ax_psi = axes[1, 1]  # row 1 = the single band's row; col 1 = psi panel
    (line,) = ax_psi.get_lines()
    return line.get_ydata()


def test_swapping_model_response_function_is_reflected_in_the_plot():
    original = model_mod.response_function
    try:
        default_psi = _fit_and_get_psi_median()

        model_mod.response_function = _tophat_response
        tophat_psi = _fit_and_get_psi_median()
    finally:
        model_mod.response_function = original

    # The default skew-normal has a smoothly-varying peak: the values
    # nearest its maximum still span a real range.
    default_near_peak = default_psi[default_psi > 0.9 * default_psi.max()]
    assert np.ptp(default_near_peak) > 0.05 * default_psi.max()

    # The swapped top-hat is flat across its whole (nonzero) support: the
    # values nearest its maximum are all equal (zero spread). If this test
    # is failing with a nonzero spread here, plot_lightcurve_fits() is
    # silently re-deriving psi from the *original* response_function again
    # instead of the swapped one -- see CLAUDE.md decision #5.
    tophat_near_peak = tophat_psi[tophat_psi > 0.9 * tophat_psi.max()]
    assert np.ptp(tophat_near_peak) < 1e-6
