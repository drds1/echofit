import numpy as np

def disk_kernel(time, M_BH, acc_rate, incl):
    tau0 = (M_BH / 1e8)**(1/3) * (acc_rate)**(1/3)
    width = 0.5 * (1 + np.sin(incl))
    t = time[:, None] - time[None, :]
    K = np.exp(-0.5 * ((t - tau0) / width)**2)
    return K / K.sum(axis=1, keepdims=True)
