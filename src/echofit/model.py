import numpy as np
from scipy.signal import fftconvolve

from .fourier import build_basis, power_spectrum_prior_matrix
from .kernel import disk_kernel
from .likelihood import solve_linear_amplitudes


def evaluate_echo_model(
    time_dict,
    flux_dict,
    sigma_dict,
    wavelengths,
    params,
    frequencies,
    oversample_factor=5,
):
    """
    Multi-band reverberation model with irregular sampling.

    Parameters
    ----------
    time_dict : dict
        {band: time array} (can differ per band)

    flux_dict : dict
        {band: flux array}

    sigma_dict : dict
        {band: error array}

    wavelengths : dict
        {band: wavelength in Angstroms} (not needed for 'xray')

    params : tuple
        (M_BH, acc_rate, incl)

    frequencies : array
        Fourier frequencies

    oversample_factor : int
        Controls resolution of latent time grid
    """

    M_BH, acc_rate, incl = params

    # --------------------------------------------------
    # 1. Build global modelling time grid
    # --------------------------------------------------

    all_times = np.concatenate(list(time_dict.values()))
    t_min, t_max = all_times.min(), all_times.max()

    # estimate smallest cadence across all bands
    dt_min = np.inf
    for t in time_dict.values():
        dt = np.min(np.diff(np.sort(t)))
        if dt > 0:
            dt_min = min(dt_min, dt)

    # define fine grid
    dt_model = dt_min / oversample_factor
    N_model = int((t_max - t_min) / dt_model) + 1
    t_model = np.linspace(t_min, t_max, N_model)

    # --------------------------------------------------
    # 2. Build Fourier basis on latent grid
    # --------------------------------------------------

    X_basis = build_basis(t_model, frequencies)

    # --------------------------------------------------
    # 3. Build joint linear system
    # --------------------------------------------------

    K_basis_list = []
    flux_list = []
    sigma_list = []
    band_slices = {}

    start = 0

    for band in flux_dict:

        t_obs = time_dict[band]
        flux = flux_dict[band]
        sigma = sigma_dict[band]

        # ------------------------------------------
        # 3a. Construct kernel
        # ------------------------------------------

        if band == "xray":
            # identity: direct observation of driver
            K = np.eye(len(t_model))
        else:
            lam = wavelengths[band]

            K = disk_kernel(
                t_model,
                M_BH,
                acc_rate,
                incl,
                wavelength=lam,
            )

        # ------------------------------------------
        # 3b. Convolve Fourier basis with kernel
        # ------------------------------------------

        K_basis_full = np.array([
            fftconvolve(X_basis[:, i], K[:, 0], mode="same")
            for i in range(X_basis.shape[1])
        ]).T

        # ------------------------------------------
        # 3c. Interpolate onto observed times
        # ------------------------------------------

        K_basis_obs = np.array([
            np.interp(t_obs, t_model, K_basis_full[:, i])
            for i in range(K_basis_full.shape[1])
        ]).T

        # store
        K_basis_list.append(K_basis_obs)
        flux_list.append(flux)
        sigma_list.append(sigma)

        band_slices[band] = slice(start, start + len(flux))
        start += len(flux)

    # --------------------------------------------------
    # 4. Stack all bands into one system
    # --------------------------------------------------

    K_basis = np.vstack(K_basis_list)
    flux_all = np.concatenate(flux_list)
    sigma_all = np.concatenate(sigma_list)

    # --------------------------------------------------
    # 5. Solve linear system with prior
    # --------------------------------------------------

    prior_cov = power_spectrum_prior_matrix(frequencies)

    coeffs = solve_linear_amplitudes(
        K_basis,
        flux_all,
        sigma_all,
        prior_cov,
    )

    # --------------------------------------------------
    # 6. Construct model predictions
    # --------------------------------------------------

    model_all = K_basis @ coeffs

    model_dict = {}
    for band, slc in band_slices.items():
        model_dict[band] = model_all[slc]

    return model_dict, coeffs