# resetup_seg6.ps1 - Re-bake 6-segment color paths + seed mass model for ac60 + the
# 10 reps, IN PLACE (Z:\Albus\Results\<rep>). Depth/segmentation/facies/rig are reused
# (already set up), so only --color-embedding + --default-mass are re-run. Originaldata berores ALDRI.
param(
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$Common = "config_seg6/common.toml",
  [string[]]$Runs = @("ac60","ac31","ac26","ac42","ac22","ac27","ac48","ac51","ac50","ac53","ac58")
)
$ErrorActionPreference = "Stop"
$i=0
foreach ($r in $Runs) {
  $i++
  $cfg = "config_seg6/run_ac/$r.toml"
  Write-Host "`n>>> [$i/$($Runs.Count)] seg6 color-embedding + seed: $r" -ForegroundColor Cyan
  & $Python scripts/calibration.py --config $Common $cfg --color-embedding
  if ($LASTEXITCODE -ne 0) { throw "color-embedding feilet for $r" }
  & $Python scripts/calibration.py --config $Common $cfg --default-mass
  if ($LASTEXITCODE -ne 0) { throw "default-mass feilet for $r" }
}
Write-Host "`nFerdig: 6-segment fargestier + seed for alle $($Runs.Count) konfigurasjoner." -ForegroundColor Green
