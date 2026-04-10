import jax.numpy as jnp
from jax import jit
from jax.scipy.signal import fftconvolve

LAMBDA0 = 5000.0  # reference wavelength (Å or nm — be consistent)
MDOT0 = 1.0
M0 = 1.0e8  # reference mass (solar masses)
TAU0 = 10.0  # days (typical AGN lag scale)


def lag_scaling(log_mdot, wavelength, M_BH):
    mdot = jnp.exp(log_mdot)
    mu = (
        TAU0
        * ((mdot / MDOT0) * (M_BH / M0)) ** (1 / 3)
        * (wavelength / LAMBDA0) ** (4 / 3)
    )
    return mu


def build_response_function(tau, log_mdot, wavelength, inclination, M_BH):
    mu = lag_scaling(log_mdot, wavelength, M_BH)

    # IMPORTANT: decouple width from mu
    sigma = TAU0 * 0.3

    skew = 0.5 * jnp.tanh(inclination)

    return response_function(tau, mu, sigma, skew)


def response_function(tau, mu, sigma, skew):
    # standard Gaussian core
    z = (tau - mu) / (sigma + 1e-8)
    gauss = jnp.exp(-0.5 * z**2)

    # mild skew that DOES NOT affect normalization
    skew_factor = jnp.exp(0.5 * skew * z)

    psi = gauss * skew_factor

    # enforce positivity
    psi = jnp.clip(psi, 1e-12, None)

    # normalize as probability density (stable form)
    psi = psi / (jnp.sum(psi) + 1e-8)

    return psi


@jit
def compute_echo(driver, tau_grid, log_mdot, wavelength, inclination, M_BH):
    psi = build_response_function(
        tau_grid,
        log_mdot,
        wavelength,
        inclination,
        M_BH,
    )

    dtau = tau_grid[1] - tau_grid[0]
    conv = fftconvolve(driver, psi, mode="full") * dtau
    return conv[: driver.shape[0]]


def drw_covariance(t, sigma, tau):
    dt = jnp.abs(t[:, None] - t[None, :])
    return sigma**2 * jnp.exp(-dt / tau)
