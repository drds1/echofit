import numpy as np


def align_to_observations(t_model, model_flux, t_obs):
    """
    Safe NumPy interpolation OUTSIDE JAX model.
    """

    return np.interp(t_obs, t_model, model_flux)
