import jax.numpy as jnp


def precompute_time_differences(time):
    """
    Computes (t_i - t_j) once.
    """

    return time[:, None] - time[None, :]
