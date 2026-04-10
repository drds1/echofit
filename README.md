# EchoFit

**EchoFit** is a lightweight framework for modelling multi-band reverberation mapping data using a physically motivated response function and a stochastic driving light curve model.

The package is designed to infer:

* Accretion disc response function parameters
* Time delays between bands
* A latent stochastic driving process (Damped Random Walk; DRW)

---

# 🚀 Features

* Multi-band light curve fitting
* Physically motivated response functions
* DRW (Gaussian Process) latent driver model
* MCMC inference (NumPyro / HMC)
* Flexible parameter fixing for controlled experiments
* Diagnostic plotting:

  * Light curve fits
  * Response functions
  * MCMC traces
  * Cost function evolution
  * Power spectrum
  * Triangle (corner) plots

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/echofit.git
cd echofit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

You may also need:

```bash
pip install numpy jax jaxlib matplotlib corner numpyro
```

---

# 📊 Data Format

EchoFit expects light curves in CSV format:

```
time,flux,sigma
```

Example:

```
0.0, 1.23, 0.05
0.2, 1.10, 0.05
...
```

Each band is loaded separately.

---

# 🧪 Generating Synthetic Data

You can generate test data using:

```bash
python generate_synthetic.py
```

This creates:

* A DRW driving light curve
* Multiple echo light curves via convolution with response functions
* Realistic noise

Output files are saved in:

```
data/
```

---

# ⚙️ Basic Usage

```python
from echofit import EchoFit

fit = EchoFit()

fit.add_lightcurve_csv("data/xray.csv", wavelength=1.0)
fit.add_lightcurve_csv("data/uv.csv", wavelength=2000.0)
fit.add_lightcurve_csv("data/optical.csv", wavelength=5000.0)

fit.build_grid()

fit.fit(
    num_warmup=500,
    num_samples=1000,
)
```

---

# 🔒 Fixing Parameters

You can fix parameters during inference:

```python
fit.fit(
    num_warmup=500,
    num_samples=1000,
    fixed_params={
        "inclination": 0.0,
        "log_sigma": 0.0,
    },
)
```

This removes them from sampling while still using them in the model.

---

# 📈 Plotting

## Light curve fits

```python
fit.plot_lightcurve_fits()
```

Shows:

* Reconstructed driving light curve (posterior mean)
* Echo light curves
* Response functions

---

## MCMC diagnostics

```python
fit.plot_mcmc_diagnostics()
```

Trace plots for all parameters.

---

## Extended diagnostics

```python
fit.plot_extended_diagnostics()
```

Includes:

* Cost function vs iteration
* Driver power spectrum
* DRW timescale marker

---

## Triangle plot

```python
fit.plot_triangle()
```

Shows:

* Posterior distributions
* Parameter degeneracies
* Confidence intervals

---

# 🧠 Model Overview

## Driving Light Curve

The latent driver is modeled as a **Damped Random Walk (DRW)**:

* Parameters:

  * `log_tau_drw` — characteristic timescale
  * `log_sigma` — variability amplitude

This defines a Gaussian Process prior over all possible driving light curves.

---

## Response Function

Each band has a response function:

```
ψ(τ; log_mdot, wavelength, inclination, M_BH)
```

This describes how the disc reprocesses the driving signal.

---

## Observed Light Curves

Each band is modeled as:

```
y(t) = S * (driver ⊗ ψ)(t) + C + noise
```

Where:

* `S` = scaling
* `C` = offset
* `⊗` = convolution

---

# ⚠️ Important Notes

## 1. The driver is NOT directly sampled

The model does **not** explicitly sample the driving light curve.

Instead:

* It marginalizes over all possible DRW realizations
* The likelihood is computed analytically via covariance matrices

---

## 2. Driver reconstruction is approximate

When plotting, the driver is reconstructed as:

* Posterior mean of the GP
* Conditioned on observed data

This is:

* Useful for visualization
* Not a unique solution

---

## 3. Degeneracies

Common degeneracies include:

* `S` vs driver amplitude
* `C` vs long-timescale driver trends
* Response width vs DRW timescale

Fixing parameters can help:

```python
fixed_params = {
    "log_sigma": 0.0
}
```

---

# 🔍 Convergence Tips

Typical good settings:

```python
num_warmup = 1000
num_samples = 2000
```

Check convergence using:

* Trace plots
* Cost function stabilization
* Triangle plots

Good mixing = chains:

* Move freely
* Show no trends
* Have stable distributions

---

# 🧪 Development Notes

* Grid spacing is controlled globally via:

```python
config["dt"]
```

* Recommended:

```python
dt = 0.2  # days
```

* Avoid dataset-dependent grids for consistency

---

# 🚧 Future Improvements

* Full GP conditioning using response operator (exact reconstruction)
* Faster linear algebra (JAX acceleration)
* Variational inference option
* Multi-object fitting
* Improved response function physics

---

# 📜 License

MIT License

---

# 🙌 Acknowledgements

This project is inspired by reverberation mapping techniques in AGN physics and Gaussian Process time series modelling.

---

# 💬 Final Note

If something looks wrong in the plots, it usually is.

The best debugging tools are:

* Triangle plots
* Synthetic data recovery
* Visual inspection of response functions

---
