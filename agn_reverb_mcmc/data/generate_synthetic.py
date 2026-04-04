import numpy as np

np.random.seed(0)
time = np.linspace(0, 100, 500)
flux = np.sin(2 * np.pi * time / 10)
flux += 0.3 * np.random.randn(len(time))

np.savez("data/synthetic_lightcurves.npz", time=time, flux=flux)
