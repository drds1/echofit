from functools import cached_property

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from .inference import run_inference, get_samples
from .model import evaluate_echo_model
from .postprocess import to_arviz
from .config import frequencies


class EchoFit:

    def __init__(
        self, time_dict=None, flux_dict=None, sigma_dict=None, wavelengths=None
    ):
        self.time_dict = time_dict or {}
        self.flux_dict = flux_dict or {}
        self.sigma_dict = sigma_dict or {}
        self.wavelengths = wavelengths or {}

        self.mcmc = None
        self.samples = None
        self.idata = None

    # ----------------------
    # MODEL GRID (single source of truth)
    # ----------------------
    @cached_property
    def t_model(self):

        all_times = np.concatenate(list(self.time_dict.values()))

        t_min = np.min(all_times)
        t_max = np.max(all_times)

        pad = 0.05 * (t_max - t_min)

        return np.linspace(t_min - pad, t_max + pad, 2000)

    # ----------------------
    # VALIDATION
    # ----------------------
    def _validate(self):

        if len(self.flux_dict) == 0:
            raise ValueError("No data loaded.")

        if "xray" not in self.flux_dict:
            raise ValueError("X-ray driving light curve is required.")

        keys = self.flux_dict.keys()

        for k in keys:
            if k not in self.time_dict:
                raise ValueError(f"Missing time array for band: {k}")
            if k not in self.sigma_dict:
                raise ValueError(f"Missing sigma for band: {k}")

    # ----------------------
    # DATA LOADING
    # ----------------------
    def add_lightcurve(self, name, time, flux, sigma, wavelength=None):

        self.time_dict[name] = time
        self.flux_dict[name] = flux
        self.sigma_dict[name] = sigma

        if name != "xray":
            if wavelength is None:
                raise ValueError(f"Wavelength required for {name}")
            self.wavelengths[name] = wavelength

    def load_npz(self, path):

        data = np.load(path, allow_pickle=True)

        for key in data:

            if key.startswith("time_"):
                band = key.replace("time_", "")
                self.time_dict[band] = data[key]

            elif key.startswith("flux_"):
                band = key.replace("flux_", "")
                self.flux_dict[band] = data[key]

            elif key.startswith("sigma_"):
                band = key.replace("sigma_", "")
                self.sigma_dict[band] = data[key]

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

        self._validate()

        self.mcmc = run_inference(
            self.time_dict,
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

        n = len(self.samples["M_BH"])
        n_draws = min(n_draws, n)

        idx = np.random.choice(n, n_draws, replace=False)

        models = {band: [] for band in self.flux_dict}

        for i in idx:

            params = (
                self.samples["M_BH"][i],
                self.samples["acc_rate"][i],
                self.samples["incl"][i],
            )

            model_dict, _ = evaluate_echo_model(
                self.t_model,
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
    def _band_color(self, band):

        if band == "xray":
            return "black"

        # wavelength-based coloring (simple heuristic)
        wl = self.wavelengths.get(band, None)

        if wl is None:
            return "gray"

        # UV range
        if wl <= 2000:
            return "purple"

        # optical range
        if wl <= 8000:
            return "tab:orange"

        return "tab:red"

    def plot_raw_lightcurve_data(
        self, normalize=False, errorbars=True, title="Observed light curves"
    ):
        """
        Plot all loaded light curves before inference.
        """

        if self.time_dict is None or self.flux_dict is None:
            raise ValueError("No data loaded. Use load_csv() first.")

        # -------------------------------------------------
        # sort bands by wavelength (low → high)
        # -------------------------------------------------
        bands = list(self.time_dict.keys())

        def wl_key(b):
            # xray gets highest priority (top panel or bottom depending preference)
            if b == "xray":
                return -1
            return self.wavelengths.get(b, 0)

        bands = sorted(bands, key=wl_key, reverse=False)

        # -------------------------------------------------
        # create stacked panels
        # -------------------------------------------------
        fig, axes = plt.subplots(
            len(bands), 1, figsize=(10, 2.5 * len(bands)), sharex=True
        )

        if len(bands) == 1:
            axes = [axes]

        for ax, band in zip(axes, bands):

            t = self.time_dict[band]
            f = self.flux_dict[band]
            color = self._band_color(band)

            if normalize:
                f = (f - f.mean()) / f.std()

            if errorbars and band in self.sigma_dict:
                ax.errorbar(
                    t,
                    f,
                    yerr=self.sigma_dict[band],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    alpha=0.6,
                )
            else:
                ax.scatter(t, f, s=10)

            label = band
            if band in self.wavelengths:
                label += f" ({self.wavelengths[band]} Å)"

            ax.set_ylabel(label)
            ax.grid(alpha=0.2)

        axes[-1].set_xlabel("Time")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_lightcurves(self, n_draws=200):

        models = self.posterior_predictive(n_draws)

        for band in self.flux_dict:

            median = np.median(models[band], axis=0)
            lo = np.percentile(models[band], 16, axis=0)
            hi = np.percentile(models[band], 84, axis=0)

            plt.figure()

            plt.errorbar(
                self.time_dict[band],
                self.flux_dict[band],
                yerr=self.sigma_dict[band],
                fmt=".",
                label="data",
            )

            # IMPORTANT FIX: align model to data grid
            t_data = self.time_dict[band]
            median_i = np.interp(t_data, self.t_model, median)
            lo_i = np.interp(t_data, self.t_model, lo)
            hi_i = np.interp(t_data, self.t_model, hi)

            plt.plot(t_data, median_i, label="model")
            plt.fill_between(t_data, lo_i, hi_i, alpha=0.3)

            plt.title(band)
            plt.xlabel("Time")
            plt.ylabel("Flux")
            plt.legend()
            plt.show()
