"""
Tests for lag_mode="free" bands and EchoFit.add_driver_lightcurve(): the
validation guardrails (M_BH required for physical bands, a warning when a
free-lag band has no driver anchor), and a real MCMC recovery check that a
driver-anchored free-lag fit actually recovers the right lag ordering --
which tests/test_shift_degeneracy.py already proved is impossible without
one, exactly.
"""

import warnings

import numpy as np
import pytest

from echofit.echofit import EchoFit
from echofit.synthetic import generate_synthetic_dataset, generate_free_lag_dataset


def test_physical_band_without_M_BH_raises():
    data = generate_synthetic_dataset(
        M_BH=1.0e9, bands={"g": 4770.0}, n_obs_per_band=15, n_freq=6, n_tau=40, seed=0,
    )
    ef = EchoFit(M_BH=None)  # no M_BH given
    d = data["bands"]["g"]
    ef.add_lightcurve("g", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=6, n_tau=40)
    with pytest.raises(ValueError, match="M_BH is required"):
        ef.fit(num_warmup=5, num_samples=5, progress_bar=False)


def test_free_band_without_driver_warns():
    data = generate_free_lag_dataset(
        lines={"line_a": 8.0}, n_obs_per_line=15, n_freq=6, n_tau=40,
        include_driver=False, seed=0,
    )
    ef = EchoFit(M_BH=None)
    d = data["bands"]["line_a"]
    ef.add_lightcurve("line_a", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"], lag_mode="free")
    ef.build_grid(n_freq=6, n_tau=40)
    with pytest.warns(UserWarning, match="not identifiable"):
        ef.fit(num_warmup=5, num_samples=5, progress_bar=False)


def test_free_band_with_driver_does_not_warn():
    data = generate_free_lag_dataset(
        lines={"line_a": 8.0}, n_obs_per_line=15, n_obs_driver=20, n_freq=6, n_tau=40,
        include_driver=True, seed=0,
    )
    ef = EchoFit(M_BH=None)
    d = data["bands"]["line_a"]
    ef.add_lightcurve("line_a", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"], lag_mode="free")
    ef.add_driver_lightcurve(t=data["driver"]["t"], y=data["driver"]["y"], yerr=data["driver"]["yerr"])
    ef.build_grid(n_freq=6, n_tau=40)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        ef.fit(num_warmup=5, num_samples=5, progress_bar=False)


def test_free_lag_recovery_with_driver_anchor():
    """The real payoff: with a driver light curve registered, a multi-line
    free-lag fit should recover the *ordering* of the true lags (each
    line's posterior-median tau in the same order as its true tau) --
    exactly what test_shift_degeneracy.py shows is mathematically
    impossible to guarantee without the driver anchoring the absolute
    scale."""
    data = generate_free_lag_dataset(
        lines={"line_a": 8.0, "line_b": 18.0, "line_c": 30.0},
        sigma_drw_true=0.3, tau_drw_true=25.0, t_span=200.0,
        n_obs_per_line=35, n_obs_driver=70, include_driver=True,
        n_freq=15, n_tau=150, tau_max=60.0, noise_level=0.05, seed=1,
    )
    ef = EchoFit(M_BH=None)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"], lag_mode="free")
    ef.add_driver_lightcurve(t=data["driver"]["t"], y=data["driver"]["y"], yerr=data["driver"]["yerr"])
    # tau_max must match the synthetic generator's (60.0) -- build_grid()'s
    # own default (half the observed baseline) would otherwise give a much
    # wider, weaker Uniform(0, tau_max) prior on each tau_{band} than intended.
    ef.build_grid(n_freq=15, n_tau=150, tau_max=60.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ef.fit(rng_seed=0, num_warmup=400, num_samples=400, progress_bar=False)

    diverging = np.asarray(ef.extra_fields["diverging"])
    assert diverging.mean() < 0.1

    recovered = {name: float(np.median(ef.samples[f"tau_{name}"])) for name in data["bands"]}
    true_order = sorted(data["truth"]["bands"], key=lambda n: data["truth"]["bands"][n]["tau"])
    recovered_order = sorted(recovered, key=lambda n: recovered[n])
    assert recovered_order == true_order, (
        f"recovered lag ordering {recovered_order} doesn't match true ordering {true_order} "
        f"(recovered medians: {recovered}, true: "
        f"{ {n: data['truth']['bands'][n]['tau'] for n in data['bands']} })"
    )
