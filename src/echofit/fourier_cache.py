import jax.numpy as jnp


def build_fourier_matrices(t_model, frequencies):

    omega = 2 * jnp.pi * jnp.asarray(frequencies)
    phase = jnp.outer(t_model, omega)

    X_sin = jnp.sin(phase)
    X_cos = jnp.cos(phase)

    return X_sin, X_cos