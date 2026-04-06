import jax.numpy as jnp


def build_rw_precision(K, sigma):
    """
    First-order random walk precision matrix
    """

    D = jnp.eye(K) - jnp.eye(K, k=-1)
    D = D[1:]  # remove first row

    Q = (D.T @ D) / (sigma ** 2)

    # small regularisation
    Q += 1e-6 * jnp.eye(K)

    return Q