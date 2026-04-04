import numpy as np

def disk_kernel(time, M_BH, acc_rate, incl, wavelength, lambda0=5000.0):
    """
    Wavelength-dependent thin-disk reverberation kernel.

    τ(λ) ∝ λ^(4/3)
    """

    # base timescale (physical scaling placeholder)
    tau0 = (M_BH / 1e8)**(1/3) * (acc_rate)**(1/3)

    # λ^(4/3) scaling
    tau_lambda = tau0 * (wavelength / lambda0)**(4/3)

    # inclination broadening
    width = 0.3 + 0.7 * np.sin(incl)

    t = time[:, None] - time[None, :]

    kernel = np.exp(-0.5 * ((t - tau_lambda) / width)**2)

    # normalise
    return kernel / kernel.sum(axis=1, keepdims=True)
