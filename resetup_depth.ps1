# =============================================================================
# resetup_depth.ps1  -  Re-bake depth.npz + geometry with the VARYING depth map
# for the AC60 seed + the 10 calibration-group representatives.
#
# Run this AFTER common.toml [depth] measurements was switched to
# depth_measurements.csv (already done). Only --depth + --rig are re-run;
# segmentation / facies / color-paths are depth-independent and untouched.
# Originaldata (Raw data) berores ALDRI - kun resultatmappene (depth.npz, rig).
#
#   .\resetup_depth.ps1
# Deretter:
#   .\run_autocalibration.ps1            # re-seeder reps fra ac60 + skriver master/watchdog
#   (kjor master + watchdog i hver sin terminal/maskin; flere --workers = raskere)
#   .\run_autocalibration.ps1 -Propagate # etterpa: push optimaliserte reps til gruppemedlemmer
# =============================================================================
param(
  [string]  $Python = ".\.venv\Scripts\python.exe",
  [string]  $Common = "config/common.toml",
  [string[]]$Runs   = @("ac60","ac31","ac26","ac42","ac22","ac27","ac48","ac51","ac50","ac53","ac58")
)
$ErrorActionPreference = "Stop"
$i = 0
foreach ($r in $Runs) {
  $i++
  Write-Host "`n>>> [$i/$($Runs.Count)] re-setup depth+rig: $r" -ForegroundColor Cyan
  & $Python scripts/setup.py --config $Common "config/run_ac/$r.toml" --depth --rig
  if ($LASTEXITCODE -ne 0) { throw "setup feilet for $r (exit $LASTEXITCODE)" }
}
Write-Host "`nFerdig: depth.npz + geometri regenerert med dybdekartet for alle $($Runs.Count) konfigurasjoner." -ForegroundColor Green
Write-Host "Neste: .\run_autocalibration.ps1  (re-seeder + skriver master/watchdog-kommandoene)." -ForegroundColor Green
