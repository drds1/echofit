# src/echofit/solve_fourier.py

import jax.numpy as jnp


def build_fourier_design_matrix(t, frequencies):
    """
    Returns design matrix X such that:

        x(t) = X @ theta

    where theta = [a_k, b_k]
    """

    t = jnp.asarray(t)
    omega = 2 * jnp.pi * jnp.asarray(frequencies)

    phase = jnp.outer(t, omega)

    sin_block = jnp.sin(phase)
    cos_block = jnp.cos(phase)

    return jnp.concatenate([sin_block, cos_block], axis=1)


def reconstruct_fourier(t, a, b, frequencies):
    """
    Direct evaluation of Fourier series.
    """
    omega = 2 * jnp.pi * jnp.asarray(frequencies)
    phase = jnp.outer(t, omega)

    return jnp.sum(
        a * jnp.sin(phase) + b * jnp.cos(phase),
        axis=1
    )