import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
from .model import evaluate_echo_model
from .config import frequencies


# -----------------------------
# build unified model grid
# -----------------------------
def build_model_grid(time_dict, n=2000):

    all_times = np.concatenate(list(time_dict.values()))
    t_min, t_max = all_times.min(), all_times.max()

    pad = 0.05 * (t_max - t_min)

    return np.linspace(t_min - pad, t_max + pad, n)


# -----------------------------
# numpyro model
# -----------------------------
def model(time_dict, flux_dict, sigma_dict, wavelengths):

    def _model():

        # =========================================================
        # 1. DISK PARAMETERS
        # =========================================================
        M_BH = numpyro.sample("M_BH", dist.LogUniform(5, 10))
        acc_rate = numpyro.sample("acc_rate", dist.LogUniform(0.01, 1.0))
        incl = numpyro.sample("incl", dist.Uniform(0, 90))

        params = (M_BH, acc_rate, incl)

        # =========================================================
        # 2. MODEL GRID
        # =========================================================
        t_model = build_model_grid(time_dict)

        # =========================================================
        # 3. FOURIER DRIVING LIGHT CURVE (LATENT PROCESS)
        # =========================================================

        K = len(frequencies)

        sigma_rw = numpyro.sample("sigma_rw", dist.LogNormal(0.0, 1.0))

        a0 = numpyro.sample("a0", dist.Normal(0.0, 1.0))
        b0 = numpyro.sample("b0", dist.Normal(0.0, 1.0))

        da = numpyro.sample("da", dist.Normal(0.0, sigma_rw).expand([K - 1]))

        db = numpyro.sample("db", dist.Normal(0.0, sigma_rw).expand([K - 1]))

        # build random walk
        a = jnp.concatenate([jnp.array([a0]), a0 + jnp.cumsum(da)])
        b = jnp.concatenate([jnp.array([b0]), b0 + jnp.cumsum(db)])

        phase = 2 * np.pi * jnp.outer(t_model, frequencies)

        xray_model = jnp.sum(a * jnp.sin(phase) + b * jnp.cos(phase), axis=1)

        # =========================================================
        # 4. ECHO MODEL (UV / OPTICAL)
        # =========================================================
        model_dict = evaluate_echo_model(
            t_model,
            xray_model,
            params,
            wavelengths,
        )

        # =========================================================
        # 5. LIKELIHOOD TERMS (with calibration parameters)
        # =========================================================

        # --- X-ray likelihood (OPTIONAL) ---
        if "xray" in time_dict:

            xray_interp = jnp.interp(
                time_dict["xray"],
                t_model,
                xray_model,
            )

            numpyro.sample(
                "obs_xray",
                dist.Normal(
                    xray_interp,
                    sigma_dict["xray"],
                ),
                obs=flux_dict["xray"],
            )

        # --- UV / Optical calibration parameters ---
        band_A = {}
        band_C = {}

        for band in flux_dict.keys():

            if band == "xray":
                continue

            band_A[band] = numpyro.sample(f"A_{band}", dist.LogNormal(0.0, 0.5))

            band_C[band] = numpyro.sample(f"C_{band}", dist.Normal(0.0, 1.0))

        # --- UV / Optical likelihood ---
        for band in flux_dict.keys():

            if band == "xray":
                continue

            model_interp = jnp.interp(
                time_dict[band],
                t_model,
                model_dict[band],
            )

            mu = jnp.mean(model_interp)
            std = jnp.std(model_interp) + 1e-8

            model_norm = (model_interp - mu) / std

            calibrated_model = band_A[band] * model_norm + band_C[band]

            numpyro.sample(
                f"obs_{band}",
                dist.Normal(
                    calibrated_model,
                    sigma_dict[band],
                ),
                obs=flux_dict[band],
            )

    return _model


# -----------------------------
# inference wrapper
# -----------------------------
def run_inference(
    time_dict,
    flux_dict,
    sigma_dict,
    wavelengths,
    num_warmup=500,
    num_samples=1000,
):

    rng_key = jax.random.PRNGKey(0)

    kernel = NUTS(model(time_dict, flux_dict, sigma_dict, wavelengths))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)

    mcmc.run(rng_key)

    return mcmc


def get_samples(mcmc):
    return mcmc.get_samples()
