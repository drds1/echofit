"""
model.py
========

The NumPyro probabilistic model tying together:

* a damped-random-walk (DRW) driver, represented as a truncated Fourier
  series whose sine/cosine amplitudes are given Gaussian priors matching the
  DRW's Lorentzian power spectrum (so ``sigma_drw`` and ``tau_drw`` are
  genuine, interpretable DRW hyperparameters that get inferred);
* a per-band causal response function (see ``forward_model.response_function``)
  that convolves the driver into each band's echo;
* a Gaussian observation likelihood for irregularly sampled multi-band light
  curves.

Only the following are inferred:
    log_mdot, inclination, sigma_drw, tau_drw,
    {S_k, C_k} driver Fourier coefficients,
    {S_band, C_band} per band.

M_BH is a fixed input, never a latent variable.
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .forward_model import response_function, transfer_coeffs, compute_echo


def drw_prior_scale(freqs: jnp.ndarray, sigma_drw, tau_drw) -> jnp.ndarray:
    """Standard deviation of each Fourier coefficient under a DRW prior.

    The DRW has a Lorentzian power spectrum in angular frequency w:

        P(w) = sigma_drw**2 * tau_drw / (1 + (w * tau_drw)**2)

    Discretizing onto the fixed frequency grid ``freqs`` with local spacing
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
    M_BH: float,
    bands: Dict[str, dict],
):
    """NumPyro model for multi-band reverberation-mapped light curves.

    Parameters
    ----------
    freqs : (n_freq,) array
        Fixed grid of driver angular frequencies (rad/day).
    tau_grid : (n_tau,) array
        Fixed grid of lags (days) used to evaluate/normalize psi and its
        Fourier transform.
    M_BH : float
        Fixed black hole mass (solar masses). Not inferred.
    bands : dict
        Mapping ``band_name -> {"t": array, "y": array, "yerr": array,
        "wavelength": float}`` for each observed light curve.
    """
    # -- shared driving-source (DRW) hyperparameters --------------------
    sigma_drw = numpyro.sample("sigma_drw", dist.HalfNormal(2.0))
    tau_drw = numpyro.sample("tau_drw", dist.LogNormal(loc=jnp.log(20.0), scale=1.0))

    prior_scale = drw_prior_scale(freqs, sigma_drw, tau_drw)
    n_freq = freqs.shape[0]

    with numpyro.plate("freq", n_freq):
        S = numpyro.sample("S", dist.Normal(0.0, prior_scale))
        C = numpyro.sample("C", dist.Normal(0.0, prior_scale))

    # -- shared reprocessing parameters ----------------------------------
    log_mdot = numpyro.sample("log_mdot", dist.Normal(0.0, 1.0))
    inclination = numpyro.sample("inclination", dist.Uniform(0.0, 80.0))

    # -- per-band amplitude / offset + likelihood ------------------------
    for band_name, d in bands.items():
        S_band = numpyro.sample(f"S_{band_name}", dist.LogNormal(0.0, 1.0))
        C_band = numpyro.sample(f"C_{band_name}", dist.Normal(0.0, 5.0))

        psi = response_function(
            tau_grid,
            log_mdot=log_mdot,
            wavelength=d["wavelength"],
            inclination=inclination,
            M_BH=M_BH,
        )
        A, B = transfer_coeffs(tau_grid, psi, freqs)
        echo = compute_echo(S, C, freqs, A, B, d["t"])
        y_pred = S_band * echo + C_band

        numpyro.deterministic(f"y_pred_{band_name}", y_pred)
        numpyro.sample(
            f"obs_{band_name}",
            dist.Normal(y_pred, d["yerr"]),
            obs=d["y"],
        )
