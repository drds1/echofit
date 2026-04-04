import numpy as np

np.random.seed(1)

time = np.linspace(0, 100, 600)

# latent driver
driver = np.sin(2 * np.pi * time / 8) + 0.3*np.random.randn(len(time))

def reprocess(lag, smooth):
    shifted = np.interp(time - lag, time, driver, left=0, right=0)
    return np.convolve(shifted, np.ones(smooth)/smooth, mode="same")

# wavelength mapping (Angstroms)
wavelengths = {
    "xray": None,
    "uv": 1500,
    "optical": 5000,
}

flux_dict = {
    "xray": driver + 0.2*np.random.randn(len(time)),
    "uv": reprocess(lag=1.5, smooth=4),
    "optical": reprocess(lag=4.0, smooth=12),
}

sigma_dict = {
    "xray": 0.1*np.ones_like(time),
    "uv": 0.05 + 0.02*np.random.rand(len(time)),
    "optical": 0.08 + 0.03*np.random.rand(len(time)),
}

# add noise
for k in flux_dict:
    flux_dict[k] += sigma_dict[k] * np.random.randn(len(time))

np.savez(
    "data/synthetic_lightcurves.npz",
    time=time,
    **{f"flux_{k}": v for k, v in flux_dict.items()},
    **{f"sigma_{k}": v for k, v in sigma_dict.items()},
)