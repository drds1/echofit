import jax.numpy as jnp
from .disk_kernel_cache import precompute_time_differences


class EchoCache:
    def __init__(self, t_model, wavelengths_dict):

        self.t_model = t_model
        self.wavelengths_dict = wavelengths_dict

        # 🔥 COMPUTE ONCE
        self.dt_matrix = precompute_time_differences(t_model)

        # pre-extract wavelengths list (faster loop)
        self.bands = list(wavelengths_dict.keys())
        self.wavelengths = jnp.array(list(wavelengths_dict.values()))