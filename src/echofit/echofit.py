import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from .plotting import plot_mcmc_diagnostics
from .inference import run_mcmc, model
from .plotting import plot_lightcurve_fits


class EchoFit:
    def __init__(self, M_BH):
        self.M_BH = M_BH
        self.bands = []

    def add_lightcurve(self, t, y, yerr, wavelength):
        self.bands.append(
            dict(
                t=jnp.array(t),
                y=jnp.array(y),
                yerr=jnp.array(yerr),
                wavelength=wavelength,
            )
        )

    def build_grid(self):
        all_t = jnp.concatenate([b["t"] for b in self.bands])
        tmin, tmax = all_t.min(), all_t.max()

        grid_t = jnp.linspace(tmin, tmax, 500)

        # simple synthetic driver (fixed)
        driver = jnp.sin(grid_t / 10.0)

        tau_grid = jnp.linspace(0, 50, 200)

        self.data = dict(
            grid_t=grid_t,
            driver=driver,
            tau_grid=tau_grid,
            bands=self.bands,
            M_BH=self.M_BH,
        )

    def fit(self, num_warmup=500, num_samples=1000):
        self.build_grid()
        rng_key = jax.random.PRNGKey(0)
        self.mcmc = run_mcmc(
            model, self.data, rng_key, num_warmup=num_warmup, num_samples=num_samples
        )

    def plot_lightcurve_fits(self):
        plot_lightcurve_fits(self.mcmc.get_samples(), self.data)

    def _wavelength_to_color(self, wavelengths):
        """
        Map wavelengths to colors using a perceptual colormap.
        Short λ → blue, long λ → red.
        """
        wavelengths = np.array(wavelengths)

        # normalize to [0, 1]
        wmin, wmax = wavelengths.min(), wavelengths.max()
        norm = (wavelengths - wmin) / (wmax - wmin + 1e-8)

        cmap = plt.get_cmap("turbo")  # nice blue→red gradient
        return cmap(norm)

    def plot_raw_lightcurve_data(self):
        """
        Plot raw light curves in stacked panels ordered by wavelength.
        """

        if len(self.bands) == 0:
            raise ValueError("No light curves added.")

        # sort bands by wavelength
        bands_sorted = sorted(self.bands, key=lambda b: b["wavelength"])
        wavelengths = [b["wavelength"] for b in bands_sorted]

        colors = self._wavelength_to_color(wavelengths)

        n = len(bands_sorted)

        fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n), sharex=True)

        if n == 1:
            axes = [axes]

        for i, (band, color) in enumerate(zip(bands_sorted, colors)):
            ax = axes[i]

            ax.errorbar(
                band["t"],
                band["y"],
                yerr=band["yerr"],
                fmt="o",
                color=color,
                ecolor=color,
                alpha=0.9,
                markersize=4,
                capsize=2,
            )

            ax.set_ylabel("Flux")
            ax.set_title(f"λ = {band['wavelength']:.0f}", loc="left")

            # cleaner look
            ax.grid(alpha=0.2)

        axes[-1].set_xlabel("Time")

        plt.tight_layout()
        plt.show()

    def plot_mcmc_diagnostics(self):
        plot_mcmc_diagnostics(self.mcmc)
