import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import jax

from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp

from .context import ModelContext
from .inference import model, run_inference
from .postprocess import to_arviz
import jax
from .forward_model import forward_model, forward_model_grid
from .model import evaluate_echo_model_matrix


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

    def compute_beta_posterior(self):

        ctx = self.ctx
        samples = self.samples

        i = 0  # just use MAP sample for stability (first fix)

        params = (
            float(samples["M_BH"][i]),
            float(samples["acc_rate"][i]),
            float(samples["incl"][i]),
        )

        sigma_rw = float(samples["sigma_rw"][i])

        # rebuild A
        model_dict = evaluate_echo_model_matrix(
            ctx.cache,
            ctx.X,
            params
        )

        A_blocks = []
        for b in ctx.bands:
            A_blocks.append(model_dict[b][ctx.interp_idx[b], :])

        A = jnp.concatenate(A_blocks, axis=0)

        # SAME PRIOR AS BEFORE
        K = A.shape[1]
        D = jnp.eye(K) - jnp.eye(K, k=-1)
        D = D[1:]

        Q_prior = (D.T @ D) / (sigma_rw ** 2) + 1e-6 * jnp.eye(K)

        Sigma = jnp.diag(ctx.sigma_data ** 2)

        Sigma_y = A @ jnp.linalg.inv(Q_prior) @ A.T + Sigma
        Sigma_y_inv = jnp.linalg.inv(Sigma_y)

        y = ctx.y_data

        beta_mean = jnp.linalg.inv(Q_prior) @ A.T @ Sigma_y_inv @ y

        ctx.beta_mean = beta_mean

    # --------------------------------------------------
    # INFERENCE (NUTS)
    # --------------------------------------------------
    def run_mcmc(self, num_warmup=200, num_samples=1000):

        self._validate()

        if self.ctx is None:
            self.build_context()

        self.mcmc = run_inference(
        self.ctx.X,
        self.ctx.y_data,
        self.ctx.sigma_data,
        self.ctx.cache,
        self.ctx.interp_idx,
        self.ctx.bands,
        self.ctx.band_sizes,
        self.ctx.t_model,
        num_warmup,
        num_samples,
        )

        self.samples = self.mcmc.get_samples()

        self.ctx.beta_mean = np.mean(self.samples["beta"], axis=0)
        self.ctx.beta_cov = np.cov(self.samples["beta"], rowvar=False)
        print("=== BETA DEBUG ===")
        print("beta_mean:", self.ctx.beta_mean)
        print("norm:", jnp.linalg.norm(self.ctx.beta_mean))
        print("max:", jnp.max(jnp.abs(self.ctx.beta_mean)))
        print("=== BETA DEBUG ===")
        print("beta_mean:", self.ctx.beta_mean)
        print("norm:", jnp.linalg.norm(self.ctx.beta_mean))
        print("max:", jnp.max(jnp.abs(self.ctx.beta_mean)))
        self.compute_beta_posterior()
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

        idata = self.idata

        # force full numpy conversion (safety net)
        for var in idata.posterior.data_vars:
            idata.posterior[var].values = np.array(idata.posterior[var].values)

        az.plot_trace(idata)
        plt.show()

    def summary(self):
        return az.summary(self.idata)

    def plot_posteriors(self):

        idata = self.idata

        # force clean numeric arrays
        for var in idata.posterior.data_vars:
            vals = np.array(idata.posterior[var].values)

            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                continue

            idata.posterior[var].values = np.nan_to_num(
                idata.posterior[var].values,
                nan=np.median(vals),
                posinf=np.percentile(vals, 99),
                neginf=np.percentile(vals, 1),
            )

        az.plot_posterior(
            idata,
            round_to=3,
            point_estimate="mean",
            skipna=True,
        )

    def plot_lightcurves(self, num_samples=50):

        ctx = self.ctx
        samples = self.samples
        nsamples = len(samples["M_BH"])

        idx = np.random.choice(
            nsamples,
            size=min(num_samples, nsamples),
            replace=False
        )

        grid_models = {b: [] for b in ctx.bands}

        # ---------------------------------------
        # posterior sampling
        # ---------------------------------------
        for i in idx:

            params = (
                float(samples["M_BH"][i]),
                float(samples["acc_rate"][i]),
                float(samples["incl"][i]),
            )

            sigma_rw = float(samples["sigma_rw"][i])

            C = np.array(samples["C"][i])
            S = np.exp(np.array(samples["logS"][i]))

            model_grid = forward_model_grid(
                ctx.cache,
                ctx.X,
                ctx,
                params,
                sigma_rw,
                C,
                S,
            )

            for b in ctx.bands:
                grid_models[b].append(np.array(model_grid[b]))

        # ---------------------------------------
        # plotting
        # ---------------------------------------
        fig, axes = plt.subplots(
            len(ctx.bands),
            1,
            figsize=(10, 3 * len(ctx.bands))
        )

        if len(ctx.bands) == 1:
            axes = [axes]

        t_model = np.array(ctx.t_model)

        for ax, b in zip(axes, ctx.bands):

            models = np.array(grid_models[b])

            median = np.median(models, axis=0)
            lo = np.percentile(models, 16, axis=0)
            hi = np.percentile(models, 84, axis=0)

            # ---------------------------------------
            # 🔥 DENORMALISE MODEL
            # ---------------------------------------
            median = median * ctx.y_std + ctx.y_mean
            lo = lo * ctx.y_std + ctx.y_mean
            hi = hi * ctx.y_std + ctx.y_mean

            # MODEL
            ax.plot(t_model, median, lw=2, label="model")
            ax.fill_between(t_model, lo, hi, alpha=0.3)

            # ---------------------------------------
            # DATA (DENORMALISED)
            # ---------------------------------------
            t_data = np.array(self.time_dict[b])
            y_data = np.array(self.flux_dict[b])
            y_err = np.array(self.sigma_dict[b])

            ax.errorbar(
                t_data,
                y_data,
                yerr=y_err,
                fmt=".",
                alpha=0.6,
                label="data"
            )

            ax.set_title(f"{b}")
            ax.legend()
            ax.grid(alpha=0.2)

        plt.tight_layout()
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
