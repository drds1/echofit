import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .model import evaluate_model
from .config import frequencies, SIGMA

def numpyro_model(time, flux):
    M_BH = numpyro.sample("M_BH", dist.LogNormal(np.log(1e8), 1.0))
    acc_rate = numpyro.sample("acc_rate", dist.LogNormal(np.log(0.1), 1.0))
    incl = numpyro.sample("incl", dist.Uniform(0, np.pi/2))

    model_flux, _ = evaluate_model(
        time, flux, SIGMA, (M_BH, acc_rate, incl), frequencies
    )

    numpyro.sample("obs", dist.Normal(model_flux, SIGMA), obs=flux)

def run_inference(time, flux):
    kernel = NUTS(numpyro_model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
    mcmc.run(numpyro.random.PRNGKey(0), time=time, flux=flux)
    return mcmc

if __name__ == "__main__":
    data = np.load("data/synthetic_lightcurves.npz")
    mcmc = run_inference(data["time"], data["flux"])
    print(mcmc.summary())
