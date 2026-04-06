import jax.numpy as jnp

from .echo_cache import EchoCache
from .fourier_cache import build_fourier_matrices
from .interp_cache import build_interp_indices
from .config import frequencies


class ModelContext:
    """
    Static container for everything that does NOT depend on parameters.
    Built once per dataset, reused for MAP / inference / MCMC.
    """

    def __init__(self, time_dict, flux_dict, sigma_dict, wavelengths):

        # ---------------------------------------
        # time grid
        # ---------------------------------------
        self.t_min = min([t.min() for t in time_dict.values()])
        self.t_max = max([t.max() for t in time_dict.values()])

        self.t_model = jnp.linspace(self.t_min, self.t_max, 2000)

        # ---------------------------------------
        # Fourier design matrix (STATIC)
        # ---------------------------------------
        X_sin, X_cos = build_fourier_matrices(self.t_model, frequencies)
        self.X = jnp.concatenate([X_sin, X_cos], axis=1)

        # ---------------------------------------
        # echo cache (STATIC)
        # ---------------------------------------
        self.cache = EchoCache(self.t_model, wavelengths)

        # ---------------------------------------
        # interpolation indices (STATIC)
        # ---------------------------------------
        self.interp_idx = build_interp_indices(time_dict, self.t_model)

        # ---------------------------------------
        # FIXED BAND ORDER (IMPORTANT)
        # ---------------------------------------
        self.bands = [b for b in flux_dict if b != "xray"]

        # ---------------------------------------
        # PRESTACKED OBSERVATIONS (CRITICAL FIX)
        # ---------------------------------------
        self.y_data = jnp.concatenate(tuple(flux_dict[b] for b in self.bands))

        self.sigma_data = jnp.concatenate(tuple(sigma_dict[b] for b in self.bands))

        # ---------------------------------------
        # OPTIONAL DEBUG INFO
        # ---------------------------------------
        self.band_sizes = {b: flux_dict[b].shape[0] for b in self.bands}

        # ---------------------------------------
        # raw metadata (keep only if needed)
        # ---------------------------------------
        self.time_dict = time_dict
        self.wavelengths = wavelengths
