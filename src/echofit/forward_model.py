import jax.numpy as jnp
from jax import jit

from .model import evaluate_echo_model_matrix


# =========================================================
# FORWARD MODEL (DATA SPACE)
# =========================================================

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
    Evaluate model at DATA LOCATIONS.

    Assumes:
    - ctx.beta_mean already computed during inference
    - ctx.y_mean / ctx.y_std define data scaling
    """

    assert ctx.beta_mean is not None, "beta_mean was never set in ctx!"

    model_dict = evaluate_echo_model_matrix(cache, X, params)

    # -----------------------------------------------------
    # Build design matrix at observed times
    # -----------------------------------------------------
    A_blocks = []

    for b in ctx.bands:
        A_b = model_dict[b][interp_idx[b], :]
        A_blocks.append(A_b)

    A = jnp.concatenate(A_blocks, axis=0)

    # -----------------------------------------------------
    # Fixed posterior coefficients (NO SOLVE HERE)
    # -----------------------------------------------------
    beta = ctx.beta_mean

    # -----------------------------------------------------
    # Linear reconstruction
    # -----------------------------------------------------
    y_base = A @ beta

    # -----------------------------------------------------
    # Apply per-band amplitude/offset
    # -----------------------------------------------------
    y_out = []
    offset = 0

    for i, b in enumerate(ctx.bands):
        n = ctx.band_sizes[b]

        y_b = y_base[offset:offset + n]
        y_b = C[i] + S[i] * y_b

        y_out.append(y_b)
        offset += n

    # -----------------------------------------------------
    # Return in DATA SPACE
    # -----------------------------------------------------
    return jnp.concatenate(y_out)


forward_model_jit = jit(
    forward_model,
    static_argnames=("cache", "interp_idx", "ctx")
)


# =========================================================
# FORWARD MODEL (GRID / PLOTTING)
# =========================================================

def forward_model_grid(
    cache,
    X,
    ctx,
    params,
    sigma_rw,
    C,
    S,
):
    """
    Evaluate model on FULL MODEL GRID (t_model).

    IMPORTANT:
    - NO solving
    - NO rescaling hacks
    - SAME beta used as in data space
    """

    model_dict = evaluate_echo_model_matrix(cache, X, params)

    beta = ctx.beta_mean

    grid_dict = {}

    for i, b in enumerate(ctx.bands):

        # full design matrix (time grid)
        A_full = model_dict[b]

        # -------------------------------------------------
        # linear reconstruction
        # -------------------------------------------------
        y_full = A_full @ beta

        # -------------------------------------------------
        # apply band calibration
        # -------------------------------------------------
        y_full = C[i] + S[i] * y_full

        # -------------------------------------------------
        # return to physical units
        # -------------------------------------------------
        y_full = y_full * ctx.y_std + ctx.y_mean

        grid_dict[b] = y_full

    return grid_dict