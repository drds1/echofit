import jax.numpy as jnp


def evaluate_echo_model(
    cache,
    xray,
    params,
):

    M_BH, acc_rate, incl = params

    x = jnp.asarray(xray)

    dt_matrix = cache.dt_matrix

    model_dict = {}

    tau0 = (M_BH / 1e8) ** (1 / 3) * (acc_rate) ** (1 / 3)
    width = 0.3 + 0.7 * jnp.sin(incl)

    for band, wavelength in zip(cache.bands, cache.wavelengths):

        tau_lambda = tau0 * (wavelength / 5000.0) ** (4 / 3)

        kernel = jnp.exp(-0.5 * ((dt_matrix - tau_lambda) / width) ** 2)

        kernel = kernel / jnp.sum(kernel, axis=1, keepdims=True)

        model_dict[band] = jnp.matmul(kernel, x)

    return model_dict