import numpy as np
import os

np.random.seed(1)

os.makedirs("data", exist_ok=True)

time = np.linspace(0, 100, 600)

# latent driver
driver = np.sin(2 * np.pi * time / 8) + 0.3 * np.random.randn(len(time))


def reprocess(lag, smooth):
    shifted = np.interp(time - lag, time, driver, left=0, right=0)
    return np.convolve(shifted, np.ones(smooth) / smooth, mode="same")


# generate bands
flux_dict = {
    "xray": driver + 0.2 * np.random.randn(len(time)),
    "uv": reprocess(lag=1.5, smooth=4),
    "optical": reprocess(lag=4.0, smooth=12),
}

sigma_dict = {
    "xray": 0.1 * np.ones_like(time),
    "uv": 0.05 + 0.02 * np.random.rand(len(time)),
    "optical": 0.08 + 0.03 * np.random.rand(len(time)),
}

# add noise
for band in flux_dict:
    flux_dict[band] += sigma_dict[band] * np.random.randn(len(time))


# save each band as CSV
for band in flux_dict:
    data = np.column_stack([time, flux_dict[band], sigma_dict[band]])

    header = "time,flux,sigma"
    filename = f"data/{band}.csv"

    np.savetxt(filename, data, delimiter=",", header=header, comments="")

print("✅ Synthetic CSV light curves written to data/")