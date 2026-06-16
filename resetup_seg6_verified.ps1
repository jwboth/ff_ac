# resetup_seg6_verified.ps1
# Regenererer 6-segment fargestier + masse-seed for ac60 + de 10 reps, IN PLACE,
# og VERIFISERER at den lastede signal-modellen faktisk får 7 noder (value0..6)
# FØR du relanserer flåten. Stopper hvis en seed ikke blir seg6.
# Originaldata (rabilder) berores ALDRI - kun --color-embedding + --default-mass.
param(
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$Common = "config_seg6/common.toml",
  [string]$ConfigDir = "config_seg6/run_ac",
  [int]$Expect = 7,
  [string[]]$Runs = @("ac60","ac31","ac26","ac42","ac22","ac27","ac48","ac51","ac50","ac53","ac58")
)
$ErrorActionPreference = "Stop"
$fail = @()
$i = 0
foreach ($r in $Runs) {
  $i++
  $cfg = "$ConfigDir/$r.toml"
  Write-Host "`n>>> [$i/$($Runs.Count)] seg6 regenerering: $r" -ForegroundColor Cyan

  & $Python scripts/calibration.py --config $Common $cfg --color-embedding
  if ($LASTEXITCODE -ne 0) { throw "color-embedding feilet for $r" }
  # --reset TVINGER ombygging av color_to_mass-interpretasjonen fra de nye seg6-stiene.
  # UTEN --reset laster den den gamle cachede seg3-modellen (calibration_color_to_mass_analysis.py:307).
  & $Python scripts/calibration.py --config $Common $cfg --default-mass --reset
  if ($LASTEXITCODE -ne 0) { throw "default-mass feilet for $r" }

  Write-Host "    verifiserer nodeantall..." -ForegroundColor DarkGray
  & $Python scripts/verify_seg_nodes.py $r --config-dir $ConfigDir --expect $Expect
  if ($LASTEXITCODE -ne 0) {
    Write-Host "    !! $r fikk IKKE $Expect noder - seed ble ikke regenerert riktig" -ForegroundColor Red
    $fail += $r
  } else {
    Write-Host "    OK: $r er seg$($Expect-1)" -ForegroundColor Green
  }
}

Write-Host ""
if ($fail.Count -gt 0) {
  Write-Host "MISLYKKET for: $($fail -join ', '). IKKE relanser flaten for disse er fikset." -ForegroundColor Red
  exit 1
} else {
  Write-Host "Alle $($Runs.Count) konfigurasjoner verifisert som seg$($Expect-1). Klar for relansering." -ForegroundColor Green
}
