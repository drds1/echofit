import numpy as np
from .interpolation import align_to_observations


def solve_linear_amplitudes(K_basis, flux, sigma, prior_cov):
    """
    Weighted linear solve with heteroscedastic errors.
    """
    W = np.diag(1.0 / sigma**2)

    A = K_basis.T @ W @ K_basis + np.linalg.inv(prior_cov)
    b = K_basis.T @ W @ flux

    return np.linalg.solve(A, b)


def log_likelihood(flux, model, sigma):
    r = flux - model
    return -0.5 * np.sum((r / sigma) ** 2)


def compute_band_likelihood(model_dict, time_dict, flux_dict, sigma_dict):
    logL = 0.0

    for band in flux_dict:

        model = align_to_observations(
            time_dict["model"],
            model_dict[band],
            time_dict[band],
        )

        resid = flux_dict[band] - model
        logL += -0.5 * np.sum((resid / sigma_dict[band]) ** 2)

    return logL
