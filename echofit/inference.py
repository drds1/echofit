"""
inference.py
============

A small, clean wrapper around NumPyro's NUTS sampler.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
from numpyro.infer import MCMC, NUTS


def run_mcmc(
    model: Callable,
    model_kwargs: dict,
    rng_key: jax.Array,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    target_accept_prob: float = 0.85,
    progress_bar: bool = True,
) -> MCMC:
    """Run NUTS on ``model(**model_kwargs)`` and return the fitted MCMC object.

    Parameters
    ----------
    model : callable
        A NumPyro model function, e.g. ``echofit.model.reverberation_model``.
    model_kwargs : dict
        Keyword arguments passed through to ``model`` on every call
        (frequency grid, tau grid, fixed M_BH, band data, ...).
    rng_key : jax.Array
        A ``jax.random.PRNGKey``.
    num_warmup, num_samples, num_chains : int
        Standard NUTS/MCMC controls.
    target_accept_prob : float
        NUTS target acceptance probability.
    progress_bar : bool
        Whether to show NumPyro's sampling progress bar.

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object; use ``mcmc.get_samples()`` for posterior draws
        and ``mcmc.print_summary()`` / ``arviz`` for diagnostics.
    """
    kernel = NUTS(model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, **model_kwargs)
    return mcmc
