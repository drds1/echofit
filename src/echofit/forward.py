import jax.numpy as jnp

from .model import evaluate_echo_model_matrix
from .forward_core import forward_core_jit


def forward_model(
    cache,
    X,
    t_model,
    time_dict,
    flux_dict,
    sigma_dict,
    params,
    sigma_rw,
):
    """
    Python orchestration:
    builds A matrix, calls JIT core
    """

    M_BH, acc_rate, incl = params

    model_dict = evaluate_echo_model_matrix(cache, X, params)

    y_list = []
    sigma_list = []
    A_blocks = []

    for band in flux_dict:
        if band == "xray":
            continue

        y_list.append(flux_dict[band])
        sigma_list.append(sigma_dict[band])

        Y = model_dict[band]

        interp = jnp.vstack([
            jnp.interp(time_dict[band], t_model, Y[:, k])
            for k in range(Y.shape[1])
        ]).T

        A_blocks.append(interp)

    y_data = jnp.concatenate(y_list)
    sigma_data = jnp.concatenate(sigma_list)
    A = jnp.concatenate(A_blocks, axis=0)

    K = int(X.shape[1])

    y_model = forward_core_jit(
        A,
        y_data,
        sigma_data,
        sigma_rw,
        K=K,
    )

    return y_model