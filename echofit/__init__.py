"""
echofit
=======

Bayesian modelling of AGN reverberation-mapping light curves as a delayed,
smoothed echo of an unobserved driving (lamppost) light curve.

The public entry point is the :class:`EchoFit` class::

    from echofit import EchoFit

    ef = EchoFit(M_BH=1e8)
    ef.add_lightcurve("g", wavelength=4770.0, t=t_g, y=y_g, yerr=yerr_g)
    ef.add_lightcurve("i", wavelength=7625.0, t=t_i, y=y_i, yerr=yerr_i)
    ef.build_grid()
    ef.fit(rng_seed=0)
    ef.plot_lightcurve_fits()
"""

from .echofit import EchoFit
from .forward_model import lag_scaling, response_function, compute_echo
from .synthetic import generate_synthetic_dataset

__all__ = [
    "EchoFit",
    "lag_scaling",
    "response_function",
    "compute_echo",
    "generate_synthetic_dataset",
]

__version__ = "0.1.0"
