import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import jax

from numpyro.infer import MCMC, NUTS

from .context import ModelContext
from .inference import model, run_inference
from .postprocess import to_arviz


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
        self.ctx = None

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------
    def _validate(self):

        if len(self.flux_dict) == 0:
            raise ValueError("No data loaded.")

        if "xray" not in self.flux_dict:
            raise ValueError("X-ray driving light curve is required.")

        for k in self.flux_dict:
            if k not in self.time_dict:
                raise ValueError(f"Missing time array for band: {k}")
            if k not in self.sigma_dict:
                raise ValueError(f"Missing sigma for band: {k}")

    # --------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------
    def add_lightcurve(self, name, time, flux, sigma, wavelength=None):

        self.time_dict[name] = time
        self.flux_dict[name] = flux
        self.sigma_dict[name] = sigma

        if name != "xray":
            if wavelength is None:
                raise ValueError(f"Wavelength required for {name}")
            self.wavelengths[name] = wavelength

    def load_csv(self, name, path, wavelength=None):

        data = np.genfromtxt(path, delimiter=",", names=True)

        flux = data["flux"]

        # ---------------------------------------
        # 🔥 NORMALISE (center only, NOT scale)
        # ---------------------------------------
        flux = flux - np.mean(flux)

        self.add_lightcurve(
            name,
            data["time"],
            flux,
            data["sigma"],
            wavelength=wavelength,
        )

    # --------------------------------------------------
    # CONTEXT BUILDING (IMPORTANT)
    # --------------------------------------------------
    def build_context(self):

        self.ctx = ModelContext(
            time_dict=self.time_dict,
            flux_dict=self.flux_dict,
            sigma_dict=self.sigma_dict,
            wavelengths=self.wavelengths,
        )

    # --------------------------------------------------
    # INFERENCE (NUTS)
    # --------------------------------------------------
    def run_mcmc(self, num_warmup=200, num_samples=1000):

        self._validate()

        if self.ctx is None:
            self.build_context()

        self.mcmc = run_inference(
            self.ctx,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )

        self.samples = self.mcmc.get_samples()
        self.idata = to_arviz(self.mcmc)

    # --------------------------------------------------
    # LIVE MCMC
    # --------------------------------------------------
    def run_mcmc_live(
        self,
        num_warmup=300,
        num_steps=50,
        num_rounds=20,
        n_plot_draws=20,
    ):
        """
        Run MCMC in chunks and visualize fitted light curves live.
        Uses forward_model(ctx, ...) as single source of truth.
        """

        self._validate()

        if self.ctx is None:
            self.build_context()

        ctx = self.ctx

        from .forward import forward_model

        # ---------------------------------------
        # Precompute band slicing (flattened y_model → per-band)
        # ---------------------------------------
        bands = [b for b in ctx.flux_dict.keys() if b != "xray"]

        offsets = {}
        start = 0
        for b in bands:
            n = len(ctx.flux_dict[b])
            offsets[b] = (start, start + n)
            start += n

        # ---------------------------------------
        # MCMC setup
        # ---------------------------------------
        rng_key = jax.random.PRNGKey(0)

        kernel = NUTS(model(ctx))
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_steps)

        state = None

        # ---------------------------------------
        # Plot setup
        # ---------------------------------------
        plt.ion()

        all_bands = list(self.flux_dict.keys())

        fig, axes = plt.subplots(
            len(all_bands), 1, figsize=(10, 3 * len(all_bands)), sharex=True
        )

        if len(all_bands) == 1:
            axes = [axes]

        model_store = {band: [] for band in bands}

        # ---------------------------------------
        # MCMC loop
        # ---------------------------------------
        for i in range(num_rounds):

            rng_key, subkey = jax.random.split(rng_key)

            mcmc.run(subkey, init_params=state)
            state = mcmc.post_warmup_state

            samples = mcmc.get_samples()

            n_samples = len(samples["M_BH"])
            idx = np.arange(max(0, n_samples - n_plot_draws), n_samples)

            # ---------------------------------------
            # Evaluate forward model
            # ---------------------------------------
            for j in idx:

                params = (
                    samples["M_BH"][j],
                    samples["acc_rate"][j],
                    samples["incl"][j],
                )

                sigma_rw = samples["sigma_rw"][j]

                y_model = forward_model(
                    ctx.cache,
                    ctx.X,
                    ctx.t_model,
                    ctx.interp_idx,
                    ctx,
                    params,
                    sigma_rw,
                )

                y_model_np = np.array(y_model)

                for band in bands:
                    i0, i1 = offsets[band]
                    model_store[band].append(y_model_np[i0:i1])

            # ---------------------------------------
            # Plot update
            # ---------------------------------------
            for ax, band in zip(axes, all_bands):

                ax.clear()

                t = self.time_dict[band]
                y = self.flux_dict[band]

                ax.errorbar(
                    t,
                    y,
                    yerr=self.sigma_dict[band],
                    fmt=".",
                    alpha=0.4,
                )

                if band != "xray" and len(model_store[band]) > 5:

                    models = np.array(model_store[band])

                    median = np.median(models, axis=0)
                    lo = np.percentile(models, 16, axis=0)
                    hi = np.percentile(models, 84, axis=0)

                    ax.plot(t, median, label="model")
                    ax.fill_between(t, lo, hi, alpha=0.3)

                ax.set_title(f"{band} (round {i+1}/{num_rounds})")
                ax.grid(alpha=0.2)

            plt.pause(0.1)

        plt.ioff()
        plt.show()

        # ---------------------------------------
        # Save results
        # ---------------------------------------
        self.mcmc = mcmc
        self.samples = mcmc.get_samples()

    # --------------------------------------------------
    # POSTERIOR PLOTS
    # --------------------------------------------------
    def _band_color(self, band):
        """
        Assign consistent colors to bands for plotting.
        Purely cosmetic helper.
        """

        if band == "xray":
            return "black"

        wl = self.wavelengths.get(band, None)

        if wl is None:
            return "gray"

        # UV
        if wl <= 2000:
            return "purple"

        # optical
        if wl <= 8000:
            return "tab:orange"

        # infrared / redder
        return "tab:red"

    def plot_trace(self):
        az.plot_trace(self.idata)
        plt.show()

    def summary(self):
        return az.summary(self.idata)

    def plot_posteriors(self):
        az.plot_posterior(self.idata)
        plt.show()

    def plot_raw_lightcurve_data(
        self, normalize=False, errorbars=True, title="Observed light curves"
    ):
        """
        Plot all loaded light curves before inference.
        """

        if len(self.time_dict) == 0 or len(self.flux_dict) == 0:
            raise ValueError("No data loaded. Use load_csv() first.")

        bands = list(self.time_dict.keys())

        # sort by wavelength (xray first)
        def wl_key(b):
            if b == "xray":
                return -1
            return self.wavelengths.get(b, 0)

        bands = sorted(bands, key=wl_key)

        fig, axes = plt.subplots(
            len(bands), 1, figsize=(10, 2.5 * len(bands)), sharex=True
        )

        if len(bands) == 1:
            axes = [axes]

        for ax, band in zip(axes, bands):

            t = self.time_dict[band]
            f = self.flux_dict[band]
            color = self._band_color(band) if hasattr(self, "_band_color") else "C0"

            if normalize:
                f = (f - np.mean(f)) / np.std(f)

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
                ax.scatter(t, f, s=10, color=color)

            label = band
            if band in self.wavelengths:
                label += f" ({self.wavelengths[band]} Å)"

            ax.set_ylabel(label)
            ax.grid(alpha=0.2)

        axes[-1].set_xlabel("Time")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
