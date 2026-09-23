"""
run_manager.py
===============

Filesystem layout and serialization helpers behind ``EchoFit(title=...)``'s
on-disk run outputs::

    <output_root>/<title>/run_<YYYYMMDD_HHMMSS>/
        manifest.json        run config + fit progress (for resuming/inspection)
        data.npz              the registered light curve data (t, y, yerr per band)
        checkpoint/
            state.pkl          last NUTS sampler state (for resuming mid-sampling)
            samples.npz        posterior samples accumulated so far
            extra_fields.npz   diagnostics (e.g. "diverging") accumulated so far
        chains.nc              final posterior as an ArviZ InferenceData (netCDF)
        report.html + *.png    the same visual smoke-test-style report

Output root resolution (highest to lowest priority):
    1. the ``output_dir`` argument passed explicitly to ``EchoFit``
    2. the ``ECHOFIT_OUTPUT_DIR`` environment variable
    3. ``./outputs`` (relative to the current working directory)

Resuming only covers the sampling phase, not warmup: if a run is
interrupted before its first checkpoint (i.e. during warmup or before
``checkpoint_every`` samples have been collected), resuming restarts the
fit from scratch, reusing the same run directory rather than creating a
new one.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ENV_VAR = "ECHOFIT_OUTPUT_DIR"
_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def resolve_output_root(output_dir: Optional[str] = None) -> Path:
    """Resolve the output root directory: explicit arg > env var > ./outputs."""
    if output_dir is not None:
        return Path(output_dir)
    if ENV_VAR in os.environ:
        return Path(os.environ[ENV_VAR])
    return Path("outputs")


def new_run_dir(output_root: Path, title: str) -> Path:
    """Create and return a fresh ``<output_root>/<title>/run_<timestamp>/``."""
    timestamp = datetime.now().strftime(_TIMESTAMP_FMT)
    run_dir = Path(output_root) / title / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoint").mkdir(exist_ok=True)
    return run_dir


def find_run_dir(output_root: Path, title: str, run_id: str = "latest") -> Path:
    """Locate an existing run directory for ``title`` -- ``run_id="latest"``
    picks the most recently *named* ``run_*`` subdirectory (timestamped names
    sort chronologically), or pass an exact ``run_<timestamp>`` string.
    """
    title_dir = Path(output_root) / title
    if not title_dir.is_dir():
        raise FileNotFoundError(f"No runs found for title {title!r} under {output_root}")
    if run_id == "latest":
        candidates = sorted(p for p in title_dir.glob("run_*") if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No run_* directories under {title_dir}")
        return candidates[-1]
    run_dir = title_dir / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run {run_id!r} not found under {title_dir}")
    return run_dir


# -- plain JSON manifest ------------------------------------------------

def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def save_json(path, obj: dict):
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def load_json(path) -> dict:
    return json.loads(Path(path).read_text())


# -- light curve data -----------------------------------------------------

def save_bands_npz(path, bands: Dict[str, dict]):
    names = list(bands.keys())
    flat = {}
    for name, d in bands.items():
        flat[f"{name}__t"] = d["t"]
        flat[f"{name}__y"] = d["y"]
        flat[f"{name}__yerr"] = d["yerr"]
        flat[f"{name}__wavelength"] = np.asarray(d["wavelength"])
    np.savez(path, names=np.array(names), **flat)


def load_bands_npz(path) -> Dict[str, dict]:
    z = np.load(path)
    names = [str(n) for n in z["names"]]
    return {
        name: dict(
            t=z[f"{name}__t"], y=z[f"{name}__y"], yerr=z[f"{name}__yerr"],
            wavelength=float(z[f"{name}__wavelength"]),
        )
        for name in names
    }


# -- NUTS sampler state (for resuming mid-sampling) ------------------------

def save_state(path, state):
    """Pickle a NumPyro HMCState, converting jax arrays to numpy first so the
    file doesn't depend on jax's device/backend at save time."""
    import jax

    numpy_state = jax.tree_util.tree_map(np.asarray, state)
    with open(path, "wb") as f:
        pickle.dump(numpy_state, f)


def load_state(path):
    """Load a pickled HMCState and convert its arrays back to jax arrays."""
    import jax.numpy as jnp
    import jax

    with open(path, "rb") as f:
        numpy_state = pickle.load(f)
    return jax.tree_util.tree_map(jnp.asarray, numpy_state)


# -- posterior samples / extra fields (checkpoint accumulation) -----------

def save_samples_npz(path, samples: Dict[str, np.ndarray]):
    np.savez(path, **samples)


def load_samples_npz(path) -> Dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}
