#!/usr/bin/env bash
# run_ngc5548_fit.sh
# ===================
#
# A worked, runnable example of echofit applied to a *real* AGN light
# curve dataset: NGC 5548's AGN STORM continuum monitoring campaign
# (Fausnaugh, Denney, Barth, et al. 2016, ApJ 821, 56 -- D. Starkey is a
# co-author), downloaded directly from VizieR (catalog J/ApJ/821/56). See
# README.md's "Real-data worked example" section for the full story.
#
# Real data is slower than the synthetic examples elsewhere in this repo:
# 13 bands, ~2200 data points total, real (irregular, gappy) cadence, from
# 16 different observatories. Step 2 below is a fast 4-band look to check
# the pipeline end to end before committing to the full 13-band run in
# Step 3, which can take a while (tens of minutes, hardware-dependent).
#
# Usage:
#   ./scripts/run_ngc5548_fit.sh
#
# Run from the repository root. If you're not using `poetry shell`, prefix
# every `python` call below with `poetry run` instead (already done here).

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_PY="poetry run python"

# ---------------------------------------------------------------------
# Step 1: download the real light curve data (VizieR J/ApJ/821/56). See
# scripts/download_ngc5548_storm_data.py's own docstring for exactly what
# this fetches and how the per-band files are built.
# ---------------------------------------------------------------------
echo "== Step 1: downloading NGC 5548 AGN STORM light curves =="
$RUN_PY scripts/download_ngc5548_storm_data.py --outdir ngc5548_storm_data
ls -l ngc5548_storm_data/

# ---------------------------------------------------------------------
# Step 2: a quick look first -- 4 bands spanning the wavelength range
# (UV, u, g, z), a purely in-memory fit, no checkpointing.
#
#   --m-bh        NGC 5548's black hole mass is genuinely uncertain across
#                 the literature (the campaign caught it in an unusual
#                 "BLR holiday" state) -- ~5e7 Msun here is a commonly
#                 cited classic value, not a recommendation; check the
#                 current literature and adjust if you have a preferred
#                 estimate.
#   --dense-mass  worth it for a model this size (13 bands' worth of
#                 S_band/C_band, plus the driver's own Fourier
#                 coefficients) -- see CLAUDE.md decision #17.
# ---------------------------------------------------------------------
echo
echo "== Step 2: quick 4-band look (no checkpointing) =="
$RUN_PY scripts/fit_lightcurves.py \
    --m-bh 5e7 \
    --band uv1158 1157.5 ngc5548_storm_data/uv1158_band.txt \
    --band u 3472 ngc5548_storm_data/u_band.txt \
    --band g 4776 ngc5548_storm_data/g_band.txt \
    --band z 9157 ngc5548_storm_data/z_band.txt \
    --num-warmup 400 \
    --num-samples 400 \
    --dense-mass \
    --output-dir ngc5548_quick_look

echo
echo "Report written to ngc5548_quick_look/report.html"

# ---------------------------------------------------------------------
# Step 3: the full 13-band managed/resumable run -- the real analysis.
# Checkpointed, so it can be safely interrupted and resumed (Step 4).
# ---------------------------------------------------------------------
echo
echo "== Step 3: full 13-band managed fit (this is the slow one) =="
$RUN_PY scripts/fit_lightcurves.py \
    --title ngc5548_storm \
    --m-bh 5e7 \
    --band uv1158 1157.5 ngc5548_storm_data/uv1158_band.txt \
    --band uv1367 1367.0 ngc5548_storm_data/uv1367_band.txt \
    --band uv1479 1478.5 ngc5548_storm_data/uv1479_band.txt \
    --band uv1746 1746.0 ngc5548_storm_data/uv1746_band.txt \
    --band u 3472 ngc5548_storm_data/u_band.txt \
    --band B 4369 ngc5548_storm_data/B_band.txt \
    --band g 4776 ngc5548_storm_data/g_band.txt \
    --band V 5404 ngc5548_storm_data/V_band.txt \
    --band r 6176 ngc5548_storm_data/r_band.txt \
    --band R 6440 ngc5548_storm_data/R_band.txt \
    --band i 7648 ngc5548_storm_data/i_band.txt \
    --band I 8561 ngc5548_storm_data/I_band.txt \
    --band z 9157 ngc5548_storm_data/z_band.txt \
    --num-warmup 800 \
    --num-samples 1000 \
    --dense-mass \
    --checkpoint-every 200 \
    --report-every 400 \
    --output-dir outputs

echo
echo "Report written under outputs/ngc5548_storm/ -- open the newest"
echo "run_*/report.html in a browser. Compare the fitted lag-vs-wavelength"
echo "trend against Fausnaugh et al. 2016's own headline result: lags"
echo "follow the lambda^(4/3) disk-reprocessing scaling, but imply a disk"
echo "~3x larger than standard thin-disk theory predicts."

# ---------------------------------------------------------------------
# Step 4: resume it (a no-op here since Step 3 already ran to completion --
# this is what you'd run again after a crash/interruption instead).
# ---------------------------------------------------------------------
echo
echo "== Step 4: resume (no-op, already complete) =="
$RUN_PY scripts/fit_lightcurves.py --title ngc5548_storm --output-dir outputs --resume
