import numpy as np

N_FREQ = 60
F_MIN = 1 / 100.0
F_MAX = 1 / 0.5

frequencies = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQ)
SIGMA = 0.05
