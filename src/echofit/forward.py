# src/echofit/forward.py

import jax.numpy as jnp
from .model import evaluate_echo_model_matrix


def build_design_matrix(cache, X, ctx, params):

    model_dict = evaluate_echo_model_matrix(cache, X, params)

    A_blocks = []

    for b in ctx.bands:
        A_b = model_dict[b][ctx.interp_idx[b], :]
        A_blocks.append(A_b)

    A = jnp.concatenate(A_blocks, axis=0)

    # KEEP physics scaling
    A = A / (jnp.std(A, axis=0) + 1e-6)

    return A