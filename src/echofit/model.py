import jax.numpy as jnp
from .disk_kernel import disk_kernel_from_deltas


def evaluate_echo_model_matrix(cache, X, params):
    """
    X: (T, K) matrix of Fourier basis
    returns dict of (T, K) matrices
    """

    M_BH, acc_rate, incl = params

    dt_matrix = cache.dt_matrix

    model_dict = {}

    wavelengths = cache.wavelengths

    for band in wavelengths:
        wavelength = wavelengths[band]

        Kmat = disk_kernel_from_deltas(
            dt_matrix,
            M_BH,
            acc_rate,
            incl,
            wavelength,
        )

        # 🔥 KEY LINE (batched multiply)
        model_dict[band] = jnp.matmul(Kmat, X)

    return model_dict