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
    max_tree_depth: Optional[int] = None,
    chain_method: str = "parallel",
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
    max_tree_depth : int, optional
        Caps each NUTS trajectory at ``2**max_tree_depth - 1`` leapfrog
        steps (NumPyro's own default is 10, i.e. up to 1023 steps). If the
        posterior geometry is difficult (e.g. hierarchical funnels), NUTS
        can spend most samples at that ceiling; lowering this bounds the
        worst-case per-sample cost at the expense of exploration efficiency.
        Left as NumPyro's default when not given.
    chain_method : str
        ``"parallel"`` (NumPyro's default, one process/device per chain),
        ``"vectorized"`` (batch chains via ``vmap`` on a single device --
        useful for getting multiple chains for R-hat/ESS diagnostics
        without paying near-linear extra wall time when a run is dominated
        by fixed per-call overhead rather than compute), or ``"sequential"``.
    progress_bar : bool
        Whether to show NumPyro's sampling progress bar.

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object; use ``mcmc.get_samples()`` for posterior draws
        and ``mcmc.print_summary()`` / ``arviz`` for diagnostics.
    """
    nuts_kwargs = dict(target_accept_prob=target_accept_prob)
    if max_tree_depth is not None:
        nuts_kwargs["max_tree_depth"] = max_tree_depth
    kernel = NUTS(model, **nuts_kwargs)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, **model_kwargs)
    return mcmc
