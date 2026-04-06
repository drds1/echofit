import jax.numpy as jnp


def disk_kernel_from_deltas(
    dt_matrix,
    M_BH,
    acc_rate,
    incl,
    wavelength,
    lambda0=5000.0,
):
    """
    FAST kernel:
    - no recomputation of time[:, None] - time[None, :]
    - no repeated allocation inside MCMC
    """

    tau0 = (M_BH / 1e8) ** (1 / 3) * (acc_rate) ** (1 / 3)

    tau_lambda = tau0 * (wavelength / lambda0) ** (4 / 3)

    width = 0.3 + 0.7 * jnp.sin(incl)

    kernel = jnp.exp(-0.5 * ((dt_matrix - tau_lambda) / width) ** 2)

    # row-normalisation (same as before)
    return kernel / jnp.sum(kernel, axis=1, keepdims=True)
