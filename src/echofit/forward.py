import jax.numpy as jnp

from .model import evaluate_echo_model_matrix
from .forward_core import forward_core_jit


def forward_model(
    cache,
    X,
    t_model,
    interp_idx,
    ctx,
    params,
    sigma_rw,
    C,
    S,
):
    """
    Full forward model INCLUDING per-band offset (C) and scaling (S).
    """

    # ---------------------------------------
    # 1. ECHO MODEL (depends on physics params)
    # ---------------------------------------
    model_dict = evaluate_echo_model_matrix(cache, X, params)

    # ---------------------------------------
    # 2. STACK MODELS (bands, time, K)
    # ---------------------------------------
    Y_stack = jnp.stack([model_dict[b] for b in ctx.bands], axis=0)  # (B, T, K)

    # ---------------------------------------
    # 3. STACK INTERPOLATION INDICES
    # ---------------------------------------
    idx_stack = jnp.stack([ctx.interp_idx[b] for b in ctx.bands], axis=0)  # (B, N_b)

    # ---------------------------------------
    # 4. GATHER DESIGN MATRIX
    # ---------------------------------------
    A = jnp.take_along_axis(Y_stack, idx_stack[..., None], axis=1)

    A = A.reshape(-1, A.shape[-1])  # (sum N_b, K)

    # ---------------------------------------
    # 5. SOLVE FOR LIGHT CURVE
    # ---------------------------------------
    y_echo = forward_core_jit(
        A,
        ctx.y_data,
        ctx.sigma_data,
        sigma_rw,
        K=X.shape[1],
    )

    # ---------------------------------------
    # 6. APPLY PER-BAND C + S
    # ---------------------------------------
    y_list = []

    offset = 0

    for i, band in enumerate(ctx.bands):

        n = len(ctx.time_dict[band])

        y_band = y_echo[offset : offset + n]

        y_band = C[i] + S[i] * y_band

        y_list.append(y_band)

        offset += n

    return jnp.concatenate(y_list)
