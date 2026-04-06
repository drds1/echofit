import jax.numpy as jnp
from .linear_algebra import build_rw_precision
from .model import evaluate_echo_model_matrix

def predict_lightcurves(
    cache,
    X,
    t_model,
    time_dict,
    flux_dict,
    sigma_dict,
    wavelengths,
    params,
    sigma_rw,
):
    """
    Full forward model:
    θ → β̂ → y_model(t)

    Returns:
        dict: band → model light curve (on t_model grid)
    """

    

    # ---------------------------------------
    # Build echo model matrices
    # ---------------------------------------
    model_dict = evaluate_echo_model_matrix(cache, X, params)

    # ---------------------------------------
    # Stack observed data
    # ---------------------------------------
    y_list = []
    sigma_list = []
    A_blocks = []

    for band in flux_dict:
        if band == "xray":
            continue

        y_list.append(flux_dict[band])
        sigma_list.append(sigma_dict[band])

        Y = model_dict[band]  # (T, K)

        interp = jnp.vstack([
            jnp.interp(time_dict[band], t_model, Y[:, k])
            for k in range(Y.shape[1])
        ]).T

        A_blocks.append(interp)

    y_data = jnp.concatenate(y_list)
    sigma_data = jnp.concatenate(sigma_list)
    A = jnp.concatenate(A_blocks, axis=0)

    # ---------------------------------------
    # Solve for beta_hat
    # ---------------------------------------
    Sigma_inv = jnp.diag(1.0 / (sigma_data ** 2))

    K = int(X.shape[1])
    Q = build_rw_precision(K, sigma_rw)

    At_Sinv = A.T @ Sigma_inv
    precision = At_Sinv @ A + Q
    rhs = At_Sinv @ y_data

    beta_hat = jnp.linalg.solve(precision, rhs)

    # ---------------------------------------
    # Build final model curves
    # ---------------------------------------
    out = {}

    for band in flux_dict:
        if band == "xray":
            continue

        out[band] = model_dict[band] @ beta_hat

    return out