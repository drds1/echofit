import numpy as np
import os

np.random.seed(1)
os.makedirs("data", exist_ok=True)

# -------------------------
# time grid
# -------------------------
time = np.linspace(0, 100, 100)
dt = time[1] - time[0]

N = len(time)

# ============================================================
# 1. DRW / OU PROCESS DRIVER
# ============================================================

def simulate_drw(time, tau_drw, sigma, mu=0.0):
    """
    Exact discrete OU (DRW) simulation.

    X_{t+dt} = X_t * exp(-dt/tau) + random kick
    """

    x = np.zeros_like(time)
    x[0] = mu

    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]

        phi = np.exp(-dt / tau_drw)
        var = sigma**2 * (1 - np.exp(-2 * dt / tau_drw))

        x[i] = phi * x[i - 1] + np.sqrt(var) * np.random.randn()

    return x


# -------------------------
# true DRW parameters (ground truth)
# -------------------------
TRUE_tau_drw = 30.0   # days
TRUE_sigma = 1.0

driver = simulate_drw(time, TRUE_tau_drw, TRUE_sigma)

# normalize to match inference convention (optional but recommended)
driver = (driver - np.mean(driver)) / np.std(driver)


# ============================================================
# 2. RESPONSE FUNCTION ψ (matches inference model)
# ============================================================

def response_function(tau, tau0, width):
    psi = np.exp(-0.5 * ((tau - tau0) / width) ** 2)
    psi[tau < 0] = 0.0

    # normalize as continuous kernel
    psi = psi / (np.sum(psi) * dt + 1e-12)
    return psi


def convolve_driver(driver, psi):
    return np.convolve(driver, psi, mode="same") * dt


# lag grid
tau_grid = np.arange(0, 50, dt)


# ============================================================
# 3. BAND DEFINITIONS (ground truth)
# ============================================================

bands = {
    "xray": {
        "S": 1.0,
        "C": 0.0,
        "tau0": 0.0,
        "width": 0.3,
        "noise": 0.1,
    },
    "uv": {
        "S": 1.5,
        "C": 0.2,
        "tau0": 2.0,
        "width": 1.0,
        "noise": 0.05,
    },
    "optical": {
        "S": 2.0,
        "C": -0.1,
        "tau0": 5.0,
        "width": 2.0,
        "noise": 0.08,
    },
}


flux_dict = {}
sigma_dict = {}

# ============================================================
# 4. GENERATE LIGHT CURVES
# ============================================================

for band, p in bands.items():

    psi = response_function(tau_grid, p["tau0"], p["width"])

    echo = convolve_driver(driver, psi)

    flux = p["S"] * echo + p["C"]

    sigma = p["noise"] * np.ones_like(time)
    flux = flux + sigma * np.random.randn(len(time))

    flux_dict[band] = flux
    sigma_dict[band] = sigma


# ============================================================
# 5. SAVE CSV
# ============================================================

for band in flux_dict:
    fb = flux_dict[band]
    if band == 'uv':
        fb += 1.5  # add small offset to test C parameter recovery
    data = np.column_stack([time, fb, sigma_dict[band]])

    header = "time,flux,sigma"
    filename = f"data/{band}.csv"

    np.savetxt(filename, data, delimiter=",", header=header, comments="")

print("✅ DRW-based synthetic dataset generated (fully consistent with inference model)")