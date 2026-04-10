# EchoFit

EchoFit is a Bayesian inference framework for modeling reverberation mapping / echo tomography light curves using a damped random walk (DRW) driving process and parameterized response functions.

The model jointly infers:
- The latent driving light curve (as a Gaussian Process / DRW)
- Time-delay response functions for multiple wavelength bands
- Scaling and offset parameters per band
- Physical parameters such as mass accretion rate and geometry-dependent quantities

---

# Model Overview

EchoFit assumes:

## 1. Driving process
The latent driver is modeled as a DRW (Ornstein–Uhlenbeck process), parameterized by:
- `log_tau_drw` → variability timescale
- `log_sigma` → amplitude of stochastic variability

## 2. Response model
Each photometric band is generated via convolution:

F_band(t) = S * (ψ(τ) * F_driver) + C

Where:
- ψ(τ) is the response function
- S is a scaling parameter
- C is a constant offset

## 3. Physical parameters
Depending on configuration, the model may include:
- `log_mdot` (mass accretion rate proxy)
- `inclination`
- response shape parameters (e.g. wavelength-dependent lag structure)

---

# Project Structure

echofit/
├── src/echofit/
│   ├── echofit.py              # main model class
│   ├── plotting.py             # diagnostics + visualization
│   ├── forward_model.py        # convolution + response functions
│   └── inference.py            # inference engine
│
├── data/
│   ├── generate_synthetic.py   # synthetic DRW + echo data generator
│
├── pyproject.toml
└── README.md

---

# Installation

This project uses Poetry.

Install dependencies:

poetry install

Activate environment:

poetry shell

---

# Generating synthetic data

Synthetic datasets are generated using a DRW driver and convolution-based echo model.

Run:

poetry run python ./data/generate_synthetic.py

This will output CSV files:

data/xray.csv
data/uv.csv
data/optical.csv

Each file contains:

time, flux, sigma

---

# Running inference

Typical workflow:

from echofit.echofit import EchoFit

fit = EchoFit(config)
fit.add_lightcurve_csv("data/xray.csv", band="xray")
fit.add_lightcurve_csv("data/uv.csv", band="uv")
fit.add_lightcurve_csv("data/optical.csv", band="optical")

fit.build_model()
fit.fit(num_warmup=200, num_samples=1000)

---

# Visualisation

Light curve + response function fits:

fit.plot_lightcurve_fits()

MCMC diagnostics:

fit.plot_extended_diagnostics()
fit.plot_mcmc_diagnostics()

Triangle / posterior structure plot:

fit.plot_triangle()

---

# Key Diagnostics

EchoFit includes built-in checks for:

## MCMC behaviour
- Trace plots of all parameters
- Log-likelihood convergence
- Mixing diagnostics

## Model validation
- Echo reconstruction vs observed light curves
- Response function uncertainty bands
- Power spectrum of inferred driver
- DRW timescale comparison

---

# Important Notes

## 1. Driver is latent
The driving light curve is not directly observed. It is inferred via a DRW Gaussian Process conditioned on all observed bands.

## 2. Fixed parameters
Parameters can be held fixed during inference:

fit.fit(num_warmup=100, num_samples=500, fixed_params={
    "inclination": 0.0,
    "log_sigma": 0.0
})

Fixed parameters:
- are excluded from sampling
- are treated as constants in forward model evaluation

## 3. Degeneracies
Some parameters (especially S, C, and driver amplitude) may exhibit degeneracies depending on normalization choices.

---

# Model assumptions

- Linear convolution between driver and response
- DRW (Ornstein–Uhlenbeck) stochastic process for driver
- Gaussian observational noise
- Stationary response kernels per band

---

# Scientific interpretation

The model performs probabilistic deconvolution of multi-band light curves into:
- a shared stochastic driving process
- wavelength-dependent transfer functions

---

# Future improvements

- Full HMC inference (replacing current MCMC sampler)
- Improved driver reconstruction (posterior sampling instead of mean-field estimate)
- Flexible non-parametric response functions
- Fourier-based driver modeling alternative
- Hierarchical population inference

---

# License

Internal research code (update as needed).

---

# Author Notes

Designed for reverberation mapping and time-domain inference with explicit physical interpretability and full posterior sampling of latent driving processes.