import jax.numpy as jnp
from .disk_kernel_cache import precompute_time_differences


class EchoCache:
    def __init__(self, t_model, wavelengths):

        # JAX array (OK)
        self.t_model = jnp.asarray(t_model)
        self.dt_matrix = precompute_time_differences(self.t_model)

        # 🔥 FORCE PYTHON DICT (VERY IMPORTANT)
        self.wavelengths = dict(wavelengths)