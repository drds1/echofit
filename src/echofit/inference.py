# src/echofit/inference.py

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist

from .echo_cache import EchoCache
from .fourier_cache import build_fourier_matrices
from .config import frequencies
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from .echo_cache import EchoCache
from .forward import forward_model
from .config import frequencies


# =========================================================
# MODEL
# =========================================================
def model(time_dict, flux_dict, sigma_dict, wavelengths):

    # ---------------------------------------
    # STATIC PRECOMPUTE
    # ---------------------------------------
    t_min = min([t.min() for t in time_dict.values()])
    t_max = max([t.max() for t in time_dict.values()])
    t_model = jnp.linspace(t_min, t_max, 2000)

    X_sin, X_cos = build_fourier_matrices(t_model, frequencies)
    X = jnp.concatenate([X_sin, X_cos], axis=1)

    cache = EchoCache(t_model, wavelengths)

    # ---------------------------------------
    # NUMPYRO MODEL
    # ---------------------------------------
    def _model():

        # -------------------------------
        # 1. Disk parameters
        # -------------------------------
        M_BH = numpyro.sample("M_BH", dist.LogUniform(5, 10))
        acc_rate = numpyro.sample("acc_rate", dist.LogUniform(0.01, 1.0))
        incl = numpyro.sample("incl", dist.Uniform(0, 90))

        params = (M_BH, acc_rate, incl)

        # -------------------------------
        # 2. Random walk prior strength
        # -------------------------------
        sigma_rw = numpyro.sample("sigma_rw", dist.LogNormal(0.0, 1.0))

        # -------------------------------
        # 3. Forward model
        # -------------------------------
        y_model = forward_model(
            cache,
            X,
            t_model,
            time_dict,
            flux_dict,
            sigma_dict,
            params,
            sigma_rw,
        )

        # -------------------------------
        # 4. Likelihood
        # -------------------------------
        y_list = []
        sigma_list = []

        for band in flux_dict:
            if band == "xray":
                continue
            y_list.append(flux_dict[band])
            sigma_list.append(sigma_dict[band])

        y_data = jnp.concatenate(y_list)
        sigma_data = jnp.concatenate(sigma_list)

        resid = (y_data - y_model) / sigma_data
        loglike = -0.5 * jnp.sum(resid ** 2)

        numpyro.factor("loglike", loglike)

    return _model


# =========================================================
# INFERENCE
# =========================================================
def run_inference(
    time_dict,
    flux_dict,
    sigma_dict,
    wavelengths,
    num_warmup=500,
    num_samples=1000
):

    rng_key = jax.random.PRNGKey(0)

    kernel = NUTS(model(time_dict, flux_dict, sigma_dict, wavelengths))

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)

    mcmc.run(rng_key)

    return mcmc


def get_samples(mcmc):
    return mcmc.get_samples()