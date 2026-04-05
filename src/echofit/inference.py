import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .echo_cache import EchoCache
from .fourier_cache import build_fourier_matrices
from .model import evaluate_echo_model
from .config import frequencies


# =========================================================
# BUILD STATIC DESIGN MATRICES (OUTSIDE NUMPYRO MODEL)
# =========================================================
def model(time_dict, flux_dict, sigma_dict, wavelengths):

    t_min = min([t.min() for t in time_dict.values()])
    t_max = max([t.max() for t in time_dict.values()])

    t_model = jnp.linspace(t_min, t_max, 2000)

    # Fourier design matrix (FIXED)
    X_sin, X_cos = build_fourier_matrices(t_model, frequencies)

    # Echo kernel cache (FIXED)
    cache = EchoCache(t_model, wavelengths)

    # =========================================================
    # NUMPYRO MODEL
    # =========================================================
    def _model():

        # -------------------------------
        # 1. DISK PARAMETERS
        # -------------------------------
        M_BH = numpyro.sample("M_BH", dist.LogUniform(5, 10))
        acc_rate = numpyro.sample("acc_rate", dist.LogUniform(0.01, 1.0))
        incl = numpyro.sample("incl", dist.Uniform(0, 90))

        params = (M_BH, acc_rate, incl)

        # =========================================================
        # 2. FOURIER RANDOM WALK PRIOR
        # =========================================================
        K = len(frequencies)

        sigma_rw = numpyro.sample("sigma_rw", dist.LogNormal(0.0, 1.0))

        a0 = numpyro.sample("a0", dist.Normal(0.0, 1.0))
        b0 = numpyro.sample("b0", dist.Normal(0.0, 1.0))

        da = numpyro.sample("da", dist.Normal(0.0, sigma_rw).expand([K - 1]))
        db = numpyro.sample("db", dist.Normal(0.0, sigma_rw).expand([K - 1]))

        a = jnp.concatenate([jnp.array([a0]), a0 + jnp.cumsum(da)])
        b = jnp.concatenate([jnp.array([b0]), b0 + jnp.cumsum(db)])

        # =========================================================
        # 3. FOURIER RECONSTRUCTION
        # =========================================================
        sin_part = jnp.sum(X_sin * a, axis=1)
        cos_part = jnp.sum(X_cos * b, axis=1)

        xray_model = sin_part + cos_part

        # =========================================================
        # 4. X-RAY LIKELIHOOD (OPTIONAL)
        # =========================================================
        if "xray" in time_dict:

            xray_interp = jnp.interp(
                time_dict["xray"],
                t_model,
                xray_model
            )

            numpyro.sample(
                "obs_xray",
                dist.Normal(xray_interp, sigma_dict["xray"]),
                obs=flux_dict["xray"]
            )

        # =========================================================
        # 5. ECHO MODEL
        # =========================================================
        model_dict = evaluate_echo_model(
            cache,
            xray_model,
            params
        )

        # =========================================================
        # 6. LIKELIHOOD
        # =========================================================
        for band in flux_dict.keys():

            if band == "xray":
                continue

            A = numpyro.sample(f"A_{band}", dist.Normal(1.0, 1.0))
            C = numpyro.sample(f"C_{band}", dist.Normal(0.0, 1.0))

            model_interp = jnp.interp(
                time_dict[band],
                t_model,
                model_dict[band]
            )

            numpyro.sample(
                f"obs_{band}",
                dist.Normal(A * model_interp + C, sigma_dict[band]),
                obs=flux_dict[band]
            )

    return _model


# =========================================================
# INFERENCE WRAPPER
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