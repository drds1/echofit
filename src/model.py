import numpy as np
from scipy.signal import fftconvolve

from .fourier import build_basis, power_spectrum_prior_matrix
from .kernel import disk_kernel
from .likelihood import solve_linear_amplitudes

def evaluate_model(time, flux, sigma, params, frequencies):
    M_BH, acc_rate, incl = params

    X_basis = build_basis(time, frequencies)
    K = disk_kernel(time, M_BH, acc_rate, incl)

    K_basis = np.array([
        fftconvolve(X_basis[:, i], K[:, 0], mode='same')
        for i in range(X_basis.shape[1])
    ]).T

    prior_cov = power_spectrum_prior_matrix(frequencies)
    coeffs = solve_linear_amplitudes(K_basis, flux, sigma, prior_cov)

    model_flux = K_basis @ coeffs
    return model_flux, coeffs
