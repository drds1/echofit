import jax.numpy as jnp
from .fourier import build_basis, power_spectrum_prior_matrix


def solve_xray_lightcurve(time, flux, sigma, frequencies):
    """
    Deterministic MAP reconstruction of X-ray driving light curve.

    Uses weighted least squares + PSD prior (-2 slope).
    """

    B = build_basis(time, frequencies)

    W = 1.0 / (sigma**2)
    W = W[:, None]

    # design matrix
    A = B.T @ (W * B)

    # PSD prior (log power ~ -2)
    A = A + power_spectrum_prior_matrix(frequencies)

    b = B.T @ (W[:, 0] * flux)

    coeffs = jnp.linalg.solve(A, b)

    return B @ coeffs
