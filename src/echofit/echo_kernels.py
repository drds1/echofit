import jax.numpy as jnp


def build_echo_kernels(t_model, wavelengths):
    """
    Builds linear response kernels:

        K[band] @ xray = y_band
    """

    n = len(t_model)
    dt = t_model[1] - t_model[0]

    kernels = {}

    for lam in wavelengths:

        # simple parametric kernel example
        # (replace with your physical model if needed)

        tau = jnp.arange(n) * dt

        # exponential decay response (stable + fast)
        R = jnp.exp(-tau / (0.2 * (1 + lam / 5000)))

        R = R / jnp.sum(R)

        kernels[lam] = R

    return kernels