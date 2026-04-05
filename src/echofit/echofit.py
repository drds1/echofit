from importlib.resources import path

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from .inference import run_inference, get_samples
from .model import evaluate_echo_model
from .postprocess import to_arviz
from .config import frequencies, SIGMA


class EchoFit:
    def __init__(self):
        self.time_dict = {}
        self.flux_dict = {}
        self.sigma_dict = {}
        self.wavelengths = {}

        self.mcmc = None
        self.samples = None
        self.idata = None

    # ----------------------
    # DATA HANDLING
    # ----------------------

    def add_lightcurve(self, name, time, flux, sigma, wavelength=None):
        """
        Add a light curve.

        name: 'xray', 'uv', 'optical', etc.
        wavelength: required unless xray
        """
        self.time_dict[name] = time
        self.flux_dict[name] = flux
        self.sigma_dict[name] = sigma

        if name != "xray":
            if wavelength is None:
                raise ValueError(f"Wavelength required for {name}")
            self.wavelengths[name] = wavelength

    def load_npz(self, path):
        data = np.load(path)

        self.time = data["time"]

        for key in data:
            if key.startswith("flux_"):
                band = key.replace("flux_", "")
                self.flux_dict[band] = data[key]
                self.sigma_dict[band] = data[f"sigma_{band}"]

    def load_csv(self, name, path, wavelength=None):
        data = np.genfromtxt(path, delimiter=",", names=True)
        self.add_lightcurve(
        name,
        data["time"],
        data["flux"],
        data["sigma"],
        wavelength=wavelength,
        )

    # ----------------------
    # INFERENCE
    # ----------------------

    def run_mcmc(self):
        self.mcmc = run_inference(
            self.time,
            self.flux_dict,
            self.sigma_dict,
            self.wavelengths,
        )
        self.samples = get_samples(self.mcmc)
        self.idata = to_arviz(self.mcmc)

    # ----------------------
    # DIAGNOSTICS
    # ----------------------

    def plot_trace(self):
        az.plot_trace(self.idata)
        plt.show()

    def summary(self):
        return az.summary(self.idata)

    def plot_posteriors(self):
        az.plot_posterior(self.idata)
        plt.show()

    # ----------------------
    # POSTERIOR PREDICTIVE
    # ----------------------

    def posterior_predictive(self, n_draws=100):
        idx = np.random.choice(len(self.samples["M_BH"]), n_draws)

        models = {band: [] for band in self.flux_dict}

        for i in idx:
            params = (
                self.samples["M_BH"][i],
                self.samples["acc_rate"][i],
                self.samples["incl"][i],
            )

            model_dict, _ = evaluate_echo_model(
                self.time_dict,
                self.flux_dict,
                self.sigma_dict,
                self.wavelengths,
                params,
                frequencies,
            )

            for band in models:
                models[band].append(model_dict[band])

        for band in models:
            models[band] = np.array(models[band])

        return models

    # ----------------------
    # PLOTTING
    # ----------------------

    def plot_lightcurves(self, n_draws=200):
        models = self.posterior_predictive(n_draws)

        for band in self.flux_dict:
            median = np.median(models[band], axis=0)
            lo = np.percentile(models[band], 16, axis=0)
            hi = np.percentile(models[band], 84, axis=0)

            plt.figure()
            plt.errorbar(
                self.time,
                self.flux_dict[band],
                yerr=self.sigma_dict[band],
                fmt=".",
                label="data",
            )
            plt.plot(self.time, median, label="model")
            plt.fill_between(self.time, lo, hi, alpha=0.3)

            plt.title(band)
            plt.xlabel("Time")
            plt.ylabel("Flux")
            plt.legend()
            plt.show()