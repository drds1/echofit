import numpy as np
from .model import evaluate_echo_model
from .config import SIGMA, frequencies
import arviz as az


def reconstruct_lightcurve_samples(
    time_dict, flux_dict, sigma_dict, wavelengths, samples, n_draws=100
):

    idx = np.random.choice(len(samples["M_BH"]), n_draws, replace=False)

    models = []

    for i in idx:

        params = (
            samples["M_BH"][i],
            samples["acc_rate"][i],
            samples["incl"][i],
        )

        model_dict, _ = evaluate_echo_model(
            time_dict,
            flux_dict,
            sigma_dict,
            wavelengths,
            params,
            frequencies,
        )

        models.append(model_dict)

    return models


def summarise_posterior(models):
    """
    Compute median and credible intervals.
    """
    median = np.median(models, axis=0)
    lo = np.percentile(models, 16, axis=0)
    hi = np.percentile(models, 84, axis=0)
    return median, lo, hi


def to_arviz(mcmc):
    return az.from_numpyro(mcmc)
