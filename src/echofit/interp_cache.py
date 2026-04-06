import jax.numpy as jnp


def build_interp_indices(time_dict, t_model):
    """
    For each band, precompute nearest indices in t_model.
    """

    idx_dict = {}

    for band, t in time_dict.items():
        if band == "xray":
            continue

        # nearest index (fast approximation of interp)
        idx = jnp.searchsorted(t_model, t)

        # clamp to valid range
        idx = jnp.clip(idx, 0, len(t_model) - 1)

        idx_dict[band] = idx

    return idx_dict