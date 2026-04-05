import time

import jax
import numpyro
from numpyro.handlers import seed, trace

from echofit.inference import model
from echofit.echofit import EchoFit


# --------------------------------------------------
# LOAD DATA (same as your notebook)
# --------------------------------------------------
fit = EchoFit()

fit.load_csv("xray", "./data/xray.csv")
fit.load_csv("uv", "./data/uv.csv", wavelength=1500)
fit.load_csv("optical", "./data/optical.csv", wavelength=5000)


time_dict = fit.time_dict
flux_dict = fit.flux_dict
sigma_dict = fit.sigma_dict
wavelengths = fit.wavelengths


# --------------------------------------------------
# BUILD MODEL
# --------------------------------------------------
m = model(time_dict, flux_dict, sigma_dict, wavelengths)


# --------------------------------------------------
# SINGLE MODEL EVALUATION (NO MCMC)
# --------------------------------------------------
rng = jax.random.PRNGKey(0)

print("\nRunning single model evaluation...\n")

t0 = time.time()

tr = trace(seed(m, rng)).get_trace()

t1 = time.time()

print(f"⏱️ Time per model evaluation: {t1 - t0:.3f} seconds")


# --------------------------------------------------
# OPTIONAL: REPEAT TO CHECK CONSISTENCY
# --------------------------------------------------
N = 3
times = []

print("\nRepeating evaluation...\n")

for i in range(N):
    t0 = time.time()
    tr = trace(seed(m, rng)).get_trace()
    t1 = time.time()

    dt = t1 - t0
    times.append(dt)
    print(f"Run {i+1}: {dt:.3f} s")

print("\nAverage time:", sum(times) / len(times))


# --------------------------------------------------
# PRINT TRACE KEYS (DEBUG)
# --------------------------------------------------
print("\nModel variables:")
for k in tr.keys():
    print(k)