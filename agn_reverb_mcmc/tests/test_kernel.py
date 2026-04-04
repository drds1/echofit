import numpy as np
from src.kernel import disk_kernel

def test_kernel_normalisation():
    t = np.linspace(0, 10, 50)
    K = disk_kernel(t, 1e8, 0.1, 0.5)
    row_sums = K.sum(axis=1)
    assert np.allclose(row_sums, 1.0)
