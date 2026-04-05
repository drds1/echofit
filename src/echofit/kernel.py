import jax.numpy as jnp


def disk_kernel(time, M_BH, acc_rate, incl, wavelength, lambda0=5000.0):
    """
    Wavelength-dependent thin-disk reverberation kernel (JAX-safe).
    τ(λ) ∝ λ^(4/3)
    """

    tau0 = (M_BH / 1e8) ** (1 / 3) * (acc_rate) ** (1 / 3)

    tau_lambda = tau0 * (wavelength / lambda0) ** (4 / 3)

    width = 0.3 + 0.7 * jnp.sin(incl)

    t = time[:, None] - time[None, :]

    kernel = jnp.exp(-0.5 * ((t - tau_lambda) / width) ** 2)

    return kernel / jnp.sum(kernel, axis=1, keepdims=True)
