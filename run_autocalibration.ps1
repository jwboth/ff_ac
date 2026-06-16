# =============================================================================
# run_autocalibration.ps1  -  Grouped Optuna auto-calibration (ported from ff_um)
# Run from the ff_ac repo root. Uses .venv python directly (no activation).
#
#   .\run_autocalibration.ps1 -SmokeTest          # 1) validate objective on AC60 (5 iters), then stop
#   .\run_autocalibration.ps1 -SeedOnly           # 2) seed the 10 representatives from AC60
#   .\run_autocalibration.ps1                      # 3) seed (unless -SkipSeed) + print the master/watchdog commands
#   .\run_autocalibration.ps1 -Propagate          # 4) after the run: push optimised reps to their group members
#
# master + watchdog must run CONCURRENTLY (two terminals / machines): the master
# drives the Optuna study and enqueues trials; watchdog workers evaluate them.
# =============================================================================
param(
  [string]  $Python  = ".\.venv\Scripts\python.exe",
  [string]  $Queue   = "C:\Users\olav_\Documents\Darsia_Queue\Kalibrering",
  [string]  $Results = "Z:\Albus\Results",
  [string]  $Groups  = "config/calibration_groups/groups.json",
  [string[]]$Reps    = @("ac31","ac26","ac42","ac22","ac27","ac48","ac51","ac50","ac53","ac58"),
  [int]     $Workers = 10,
  [int]     $MaxIters= 500,
  [switch]  $SmokeTest,
  [switch]  $SeedOnly,
  [switch]  $SkipSeed,
  [switch]  $Propagate
)
function Run($c){ Write-Host "`n>>> $c" -ForegroundColor Cyan; iex $c }

if ($SmokeTest) {
  Run "$Python scripts/auto_calibrate_color_to_mass.py --runs ac60 --mode per-run --max-iters 5 --use-facies --per-label --config-dir config/run_ac --logs-dir logs"
  return
}
if ($Propagate) {
  Run "$Python scripts/apply_calibration_groups.py --groups-file $Groups --results-root `"$Results`""
  return
}
if (-not $SkipSeed) {
  Run "$Python scripts/apply_calibration_groups.py --groups-file $Groups --results-root `"$Results`" --seed-from ac60"
}
if ($SeedOnly) { return }

$repsArg = $Reps -join ' '
Write-Host "`n=== Kjor disse to i HVER SIN terminal (samtidig) ===" -ForegroundColor Green
Write-Host "TERMINAL 1 (master):" -ForegroundColor Yellow
Write-Host "$Python scripts/distributed_auto_calibration_queue.py master --queue `"$Queue`" --runs $repsArg --config-dir config/run_ac --max-iters $MaxIters --warmup-iters 100 --warmup-levels 1.0,0.75,0.5 --run-mode parallel --max-in-flight-per-run 3 --use-facies --per-label"
Write-Host "`nTERMINAL 2 (watchdog, kan ogsa kjores pa flere maskiner):" -ForegroundColor Yellow
Write-Host "$Python scripts/distributed_auto_calibration_queue.py watchdog --queue `"$Queue`" --workers $Workers"
Write-Host "`nNaar master er ferdig (beste kalibrering lagres per rep), kjor:  .\run_autocalibration.ps1 -Propagate" -ForegroundColor Green
