import numpy as np

def build_basis(time, frequencies):
    sin_part = np.sin(2 * np.pi * np.outer(time, frequencies))
    cos_part = np.cos(2 * np.pi * np.outer(time, frequencies))
    return np.hstack([sin_part, cos_part])

def power_spectrum_prior_matrix(frequencies):
    var = 1.0 / (frequencies**2)
    return np.diag(np.concatenate([var, var]))
