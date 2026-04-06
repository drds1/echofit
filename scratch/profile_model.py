import time
import jax
import numpyro
from numpyro.handlers import seed, trace

from echofit.inference import model
from echofit.echofit import EchoFit


fit = EchoFit()

fit.load_csv("xray", "./data/xray.csv")
fit.load_csv("uv", "./data/uv.csv", wavelength=1500)
fit.load_csv("optical", "./data/optical.csv", wavelength=5000)

time_dict = fit.time_dict
flux_dict = fit.flux_dict
sigma_dict = fit.sigma_dict
wavelengths = fit.wavelengths


# --------------------------------------------------
# BUILD NUMPYRO MODEL
# --------------------------------------------------
numpyro_model = model(time_dict, flux_dict, sigma_dict, wavelengths)


# --------------------------------------------------
# WARMUP (IMPORTANT: AVOIDS JIT ARTIFACTS)
# --------------------------------------------------
rng = jax.random.PRNGKey(0)
_ = trace(seed(numpyro_model, rng)).get_trace()


# --------------------------------------------------
# TIMED RUNS (PURE COST)
# --------------------------------------------------
def run_once():
    tr = trace(seed(numpyro_model, rng)).get_trace()
    return tr


print("\nRunning benchmark...\n")

times = []

for i in range(5):
    t0 = time.time()
    tr = run_once()
    t1 = time.time()

    dt = t1 - t0
    times.append(dt)
    print(f"Run {i+1}: {dt:.4f} s")


print("\nAverage:", sum(times)/len(times))