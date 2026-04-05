import jax.numpy as jnp
from .kernel import disk_kernel


def evaluate_echo_model(
    t_model,
    xray,
    params,
    wavelengths_dict,
):
    """
    Deterministic forward model.

    NO Fourier fitting here.
    NO MCMC here.
    """

    M_BH, acc_rate, incl = params

    model_dict = {}

    for band, wavelength in wavelengths_dict.items():

        K = disk_kernel(
            t_model,
            M_BH,
            acc_rate,
            incl,
            wavelength,
        )

        model_dict[band] = K @ xray

    return model_dict
