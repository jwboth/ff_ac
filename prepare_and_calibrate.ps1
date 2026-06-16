# =============================================================================
# prepare_and_calibrate.ps1
# One-shot local preparation + Optuna auto-calibration for the Albus (AC) series.
# Run from the ff_ac repo root AFTER:  uv sync ;  .\.venv\Scripts\activate
#
# Steps 1-2 are pure data prep (no DarSIA science). Step 3 is a ONE-TIME manual
# prerequisite (AC60 colour-embedding) - skip if AC60 is already calibrated from
# the earlier ff_ac test. Steps 4-6 run the grouped Optuna calibration.
# =============================================================================
param(
  [string]$Albus   = "Z:\Albus\Raw data",          # SERVER: protocol gen is pure-protocol (no image reads); group clustering reads first images
  [string]$DataRoot = "Z:\Albus\Raw data",                          # SERVER: path written into configs (where distributed workers read)
  [string]$Results = "Z:\Albus\Results",
  [string]$Florida = "C:\Users\olav_\Documents\Claude\Projects\FF Albus\Florida*.xlsx",          # Bergen station files (hPa)
  [string]$Queue   = "Z:\Albus\Queue\Kalibrering",  # SHARED path all machines reach (multi-machine batch)
  [int]   $Workers = 12,
  [switch]$SkipPrep,                               # skip steps 1-2 if already done
  [switch]$RunAC60Calibration                      # run step 3 (interactive colour-embedding)
)
$ErrorActionPreference = "Stop"
$common = "config/common.toml"
function Run($cmd) { Write-Host "`n>>> $cmd" -ForegroundColor Cyan; iex $cmd }

# ---- 1. Protocols + pressure (per-experiment data) --------------------------
if (-not $SkipPrep) {
  Run "python scripts/generate_protocols.py --albus-root `"$Albus`" --out protocols --all"
  Run "python scripts/generate_pressure.py   --protocols-root protocols --all --pressure-xlsx $Florida --altitude-diff-m 40 --window-hours 36"

# ---- 2. Configs + Wasserstein config + calibration groups -------------------
  Run "python scripts/generate_configs.py --albus-root `"$Albus`" --out config/run_ac --data-root `"$DataRoot`" --results-root `"$Results`" --all"
  Run "python scripts/generate_wasserstein_config.py --out config/wasserstein_ac.toml --results-root `"$Results`" --resize 0.10"
  Run "python scripts/group_calibration.py --albus-root `"$Albus`" --out config/calibration_groups --groups 10"
}

# ---- 3. Rig setup (shared by all AC experiments; run once) ------------------
Run "python scripts/setup.py --config $common config/run_ac/ac60.toml --all"

# ---- 3b. AC60 baseline calibration = structural template + Optuna start point
#   The auto-calibration LOADS an existing calibration and perturbs its values,
#   so AC60 must already have a colour-embedding + (default) mass calibration in
#   $Results\ac60. If it exists from the earlier ff_ac test, leave -RunAC60Calibration off.
if ($RunAC60Calibration) {
  Write-Host "`n[manual] AC60 colour-embedding is interactive - follow the GUI prompts." -ForegroundColor Yellow
  Run "python scripts/calibration.py --config $common config/run_ac/ac60.toml --color-embedding"
  Run "python scripts/calibration.py --config $common config/run_ac/ac60.toml --default-mass"
}

# ---- 4. Seed all 10 group representatives from AC60 -------------------------
Run "python scripts/apply_calibration_groups.py --groups-file config/calibration_groups/groups.json --results-root `"$Results`" --seed-from ac60"

# ---- 5. Optuna auto-calibration of the 10 representatives -------------------
Run "python scripts/distributed_auto_calibration_queue.py master --queue `"$Queue`" --groups-file config/calibration_groups/groups.json --baseline-trial true --max-iters 500 --warmup-iters 0 --param-ranges `"value1=0,2;value2=0,2;value3=0,2;value4=0,2;value5=0,2;value6=0,2`" --param-levels `"value1=8;value2=8;value3=8;value4=8;value5=8;value6=8`" --reset-queue"
Run "python scripts/distributed_auto_calibration_queue.py watchdog --queue `"$Queue`" --workers $Workers --control-dir `"$Queue\worker_limits`" --stop-when-drained"
Run "python scripts/distributed_auto_calibration_queue.py best --queue `"$Queue`""

# ---- 6. Propagate optimised representatives to their group members ----------
Run "python scripts/apply_calibration_groups.py --groups-file config/calibration_groups/groups.json --results-root `"$Results`""

Write-Host "`nDone. Best per-representative calibration saved; membe