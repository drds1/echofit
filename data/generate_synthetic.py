import numpy as np

np.random.seed(0)

time = np.linspace(0, 100, 500)

# driving signal
driver = np.sin(2 * np.pi * time / 10)

# create two bands with lag + smoothing
def make_band(lag, smooth):
    shifted = np.interp(time - lag, time, driver, left=0, right=0)
    smoothed = np.convolve(shifted, np.ones(smooth)/smooth, mode="same")
    return smoothed

flux_uv = make_band(lag=1.0, smooth=3)
flux_opt = make_band(lag=3.0, smooth=10)

# heteroscedastic errors
sigma_uv = 0.05 + 0.02*np.random.rand(len(time))
sigma_opt = 0.08 + 0.03*np.random.rand(len(time))

flux_uv += sigma_uv * np.random.randn(len(time))
flux_opt += sigma_opt * np.random.randn(len(time))

np.savez(
    "data/synthetic_lightcurves.npz",
    time=time,
    flux_uv=flux_uv,
    flux_opt=flux_opt,
    sigma_uv=sigma_uv,
    sigma_opt=sigma_opt,
)