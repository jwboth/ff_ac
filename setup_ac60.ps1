# =============================================================================
# setup_ac60.ps1  -  Set up + calibrate AC60 (the seed/template for all groups).
# Run from the ff_ac repo root AFTER:  uv sync ;  .\.venv\Scripts\activate
# Flags confirmed against this ff_ac's DarSIA:
#   setup.py       : --all --depth --segmentation --facies --protocol --rig --force --show
#   calibration.py : --color-embedding --mass --default-mass --volume --reset --delete --show
#   Both take  --config <common.toml> <run.toml>  (multiple files merged).
# =============================================================================
param(
  [string]$Albus   = "Z:\Albus\Raw data",          # SERVER (protocol gen reads no images; pure protocol)
  [string]$Florida = "C:\Users\olav_\Documents\Claude\Projects\FF Albus\Florida*.xlsx",
  [switch]$SkipData,                       # skip protocol/pressure generation
  [switch]$InteractiveMass                 # use --mass (sliders) instead of --default-mass
)
$ErrorActionPreference = "Stop"
$cfg = "config/common.toml config/run_ac/ac60.toml"
function Run($c){ Write-Host "`n>>> $c" -ForegroundColor Cyan; iex $c }

# ---- 0. AC60 data (protocols + pressure; config already generated) ----------
if (-not $SkipData) {
  Run "python scripts/generate_protocols.py --albus-root `"$Albus`" --out protocols --experiments AC60"
  Run "python scripts/generate_pressure.py   --protocols-root protocols --experiments ac60 --pressure-xlsx $Florida --altitude-diff-m 40 --window-hours 36"
}

# ---- 1. SETUP (rig): depth, labels/segmentation, facies, protocols, rig ------
Run "python scripts/setup.py --config $cfg --all"

# ---- 2. CALIBRATION step 1: colour embedding (colour paths per facies) -------
#   Uses calibration images calibration1/2 (10 min-48 h) + baseline DSC19883.
#   With [color.path.*] calibration_mode = "manual" an interactive editor MAY
#   open; set it to "auto" in common.toml for a fully non-interactive run.
Run "python scripts/calibration.py --config $cfg --color-embedding"

# ---- 3. CALIBRATION step 2: colour-to-mass model ----------------------------
#   --default-mass = non-interactive baseline (the model Optuna loads + optimises).
#   --mass         = interactive sliders to overlay injected vs total mass
#                    (exactly what the Optuna auto-calibration replaces).
if ($InteractiveMass) {
  Run "python scripts/calibration.py --config $cfg --mass"
} else {
  Run "python scripts/calibration.py --config $cfg --default-mass"
}

Write-Host "`nAC60 set up + calibrated. Artifacts in Z:\Albus\Results\ac60 -> ready to seed the groups." -ForegroundColor Green
