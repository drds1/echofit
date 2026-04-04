import numpy as np
from scipy.signal import fftconvolve

from .fourier import build_basis, power_spectrum_prior_matrix
from .kernel import disk_kernel
from .likelihood import solve_linear_amplitudes


def evaluate_model_multiband(time, flux_dict, sigma_dict, params, frequencies):
    """
    Joint model for multiple light curves.

    flux_dict: {band: flux array}
    sigma_dict: {band: error array}
    """

    M_BH, acc_rate, incl = params

    # Fourier basis (shared driving signal)
    X_basis = build_basis(time, frequencies)

    # Stack all bands into one big system
    K_basis_list = []
    flux_list = []
    sigma_list = []

    for band, flux in flux_dict.items():
        sigma = sigma_dict[band]

        # Band-dependent kernel
        K = disk_kernel(time, M_BH, acc_rate, incl)

        # Convolve each Fourier basis vector
        Kb = np.array([
            fftconvolve(X_basis[:, i], K[:, 0], mode='same')
            for i in range(X_basis.shape[1])
        ]).T

        K_basis_list.append(Kb)
        flux_list.append(flux)
        sigma_list.append(sigma)

    # Stack into one system
    K_basis = np.vstack(K_basis_list)
    flux_all = np.concatenate(flux_list)
    sigma_all = np.concatenate(sigma_list)

    # Prior
    prior_cov = power_spectrum_prior_matrix(frequencies)

    # Solve linear amplitudes
    coeffs = solve_linear_amplitudes(K_basis, flux_all, sigma_all, prior_cov)

    model_all = K_basis @ coeffs

    # Split back per band
    model_dict = {}
    i = 0
    for band in flux_dict:
        n = len(flux_dict[band])
        model_dict[band] = model_all[i:i+n]
        i += n

    return model_dict, coeffs
