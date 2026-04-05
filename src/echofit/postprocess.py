import numpy as np
import arviz as az

from .model import evaluate_model
from .config import frequencies, SIGMA


def reconstruct_lightcurve_samples(time, flux, samples, n_draws=100):
    """
    Generate posterior predictive light curves.
    """
    idx = np.random.choice(len(samples["M_BH"]), n_draws, replace=False)

    models = []
    for i in idx:
        params = (
            samples["M_BH"][i],
            samples["acc_rate"][i],
            samples["incl"][i],
        )
        model_flux, _ = evaluate_model(time, flux, SIGMA, params, frequencies)
        models.append(model_flux)

    return np.array(models)


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