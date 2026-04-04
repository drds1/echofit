import numpy as np
from scipy.signal import fftconvolve

from .fourier import build_basis, power_spectrum_prior_matrix
from .kernel import disk_kernel
from .likelihood import solve_linear_amplitudes


def evaluate_echo_model(
    time,
    flux_dict,
    sigma_dict,
    wavelengths,
    params,
    frequencies,
):
    """
    Joint driver + reprocessed multi-wavelength model.

    flux_dict:
        can include:
        - "xray" (optional observed driver)
        - "uv", "optical", etc.
    """

    M_BH, acc_rate, incl = params

    # 1. Fourier driver basis (latent X-ray light curve)
    X_basis = build_basis(time, frequencies)

    K_basis_all = []
    flux_all = []
    sigma_all = []

    # 2. Loop over all light curves (including possible driver)
    for band in flux_dict:

        flux = flux_dict[band]
        sigma = sigma_dict[band]

        # wavelength assignment
        if band == "xray":
            # driver observed directly → identity kernel
            K_band = np.eye(len(time))
        else:
            lam = wavelengths[band]

            K = disk_kernel(
                time,
                M_BH,
                acc_rate,
                incl,
                wavelength=lam,
            )

            # apply kernel to Fourier basis
            K_band = np.array([
                fftconvolve(X_basis[:, i], K[:, 0], mode="same")
                for i in range(X_basis.shape[1])
            ]).T

        K_basis_all.append(K_band)
        flux_all.append(flux)
        sigma_all.append(sigma)

    # stack system
    K_basis = np.vstack(K_basis_all)
    flux_all = np.concatenate(flux_all)
    sigma_all = np.concatenate(sigma_all)

    # prior on Fourier amplitudes (P(f) ∝ f^-2)
    prior_cov = power_spectrum_prior_matrix(frequencies)

    coeffs = solve_linear_amplitudes(
        K_basis,
        flux_all,
        sigma_all,
        prior_cov,
    )

    model_all = K_basis @ coeffs

    # split back
    model_dict = {}
    i = 0
    for band in flux_dict:
        n = len(flux_dict[band])
        model_dict[band] = model_all[i:i+n]
        i += n

    return model_dict, coeffs