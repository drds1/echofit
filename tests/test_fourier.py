import numpy as np
from src.fourier import build_basis

def test_basis_shape():
    t = np.linspace(0, 10, 100)
    f = np.linspace(0.01, 1, 10)
    B = build_basis(t, f)
    assert B.shape == (100, 20)
