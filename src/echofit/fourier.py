import jax.numpy as jnp


def build_basis(time, frequencies):
    time = jnp.asarray(time)
    frequencies = jnp.asarray(frequencies)

    arg = 2 * jnp.pi * jnp.outer(time, frequencies)

    return jnp.concatenate([jnp.sin(arg), jnp.cos(arg)], axis=1)


def power_spectrum_prior_matrix(frequencies):
    """
    log P(f) ~ f^-2  → variance ~ 1/f^2
    """

    frequencies = jnp.asarray(frequencies)

    var = 1.0 / (frequencies**2)

    return jnp.diag(jnp.concatenate([var, var]))
