import numpy as np
import jax.numpy as jnp
from .model import evaluate_echo_model_matrix
from .echo_cache import EchoCache
from .fourier_cache import build_fourier_matrices
from .config import frequencies
import arviz as az


# =========================================================
# RECONSTRUCT LIGHTCURVES FROM POSTERIOR SAMPLES
# =========================================================
def reconstruct_lightcurve_samples(
    time_dict,
    flux_dict,
    sigma_dict,
    wavelengths,
    samples,
    n_draws=100
):

    idx = np.random.choice(len(samples["M_BH"]), n_draws, replace=False)

    # rebuild shared time grid (MUST MATCH MODEL)
    t_min = min([t.min() for t in time_dict.values()])
    t_max = max([t.max() for t in time_dict.values()])
    t_model = jnp.linspace(t_min, t_max, 2000)

    cache = EchoCache(t_model, wavelengths)

    X_sin, X_cos = build_fourier_matrices(t_model, frequencies)

    models = []

    for i in idx:

        # -------------------------------
        # 1. PARAMS
        # -------------------------------
        params = (
            samples["M_BH"][i],
            samples["acc_rate"][i],
            samples["incl"][i],
        )

        # -------------------------------
        # 2. FOURIER RECONSTRUCTION
        # -------------------------------
        K = len(frequencies)

        sigma_rw = samples["sigma_rw"][i]

        a0 = samples["a0"][i]
        b0 = samples["b0"][i]

        da = samples["da"][i]
        db = samples["db"][i]

        a = np.concatenate([[a0], a0 + np.cumsum(da)])
        b = np.concatenate([[b0], b0 + np.cumsum(db)])

        xray = np.sum(X_sin * a, axis=1) + np.sum(X_cos * b, axis=1)

        # -------------------------------
        # 3. ECHO MODEL
        # -------------------------------
        model_dict = evaluate_echo_model_matrix(
            cache,
            xray,
            params
        )

        models.append(model_dict)

    return models


# =========================================================
# SUMMARY STATISTICS
# =========================================================
def summarise_posterior(models):
    """
    Compute median and credible intervals.
    """

    # convert list of dicts → dict of arrays
    bands = models[0].keys()

    summary = {}

    for band in bands:

        stack = np.stack([m[band] for m in models], axis=0)

        summary[band] = {
            "median": np.median(stack, axis=0),
            "lo": np.percentile(stack, 16, axis=0),
            "hi": np.percentile(stack, 84, axis=0),
        }

    return summary


# =========================================================
# ARVIZ WRAPPER
# =========================================================
def to_arviz(mcmc):
    return az.from_numpyro(mcmc)