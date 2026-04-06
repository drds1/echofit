import jax.numpy as jnp
import numpy as np

from .echo_cache import EchoCache
from .fourier_cache import build_fourier_matrices
from .interp_cache import build_interp_indices


def build_frequencies(t_dict, n_freq=40, min_period=0.5):

    t_min = min([t.min() for t in t_dict.values()])
    t_max = max([t.max() for t in t_dict.values()])

    T_span = t_max - t_min

    f_min = 0.5 / T_span
    f_max = 1.0 / min_period

    return np.logspace(
        np.log10(f_min),
        np.log10(f_max),
        n_freq
    )


class ModelContext:
    def __init__(self, time_dict, flux_dict, sigma_dict, wavelengths):

        # --------------------------
        # BANDS
        # --------------------------
        self.bands = [b for b in flux_dict.keys() if b != "xray"]

        # --------------------------
        # TIME GRID
        # --------------------------
        self.t_min = min([t.min() for t in time_dict.values()])
        self.t_max = max([t.max() for t in time_dict.values()])

        self.t_model = jnp.linspace(self.t_min, self.t_max, 2000)

        # --------------------------
        # FOURIER BASE (NO QR!)
        # --------------------------
        frequencies = build_frequencies(time_dict)

        X_sin, X_cos = build_fourier_matrices(self.t_model, frequencies)
        self.X = jnp.concatenate([X_sin, X_cos], axis=1)

        self.X = self.X / (jnp.std(self.X, axis=0) + 1e-6)

        # --------------------------
        # ECHO CACHE
        # --------------------------
        self.cache = EchoCache(self.t_model, wavelengths)

        # --------------------------
        # INTERPOLATION
        # --------------------------
        self.interp_idx = build_interp_indices(time_dict, self.t_model)

        # --------------------------
        # STACK DATA
        # --------------------------
        y_raw = jnp.concatenate([flux_dict[b] for b in self.bands])
        sigma_raw = jnp.concatenate([sigma_dict[b] for b in self.bands])

        # --------------------------
        # NORMALISATION (CRITICAL)
        # --------------------------
        self.y_mean = jnp.mean(y_raw)
        self.y_std = jnp.std(y_raw) + 1e-6

        self.y_data = (y_raw - self.y_mean) / self.y_std
        self.sigma_data = sigma_raw / self.y_std

        # --------------------------
        # METADATA
        # --------------------------
        self.band_sizes = {b: flux_dict[b].shape[0] for b in self.bands}
        self.band_sizes_list = [self.band_sizes[b] for b in self.bands]

        self.band_cumsum = jnp.cumsum(
            jnp.array([0] + self.band_sizes_list)
        )

        self.t_data = jnp.concatenate([time_dict[b] for b in self.bands])

        self.time_dict = time_dict
        self.wavelengths = wavelengths

        # --------------------------
        # POSTERIOR STORAGE (NEW)
        # --------------------------
        self.beta_mean = None
        self.beta_cov = None