import numpy as np

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
    return -0.5 * np.sum((r / sigma)**2)
