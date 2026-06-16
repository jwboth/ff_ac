# =============================================================================
# setup_all.ps1 - DarSIA setup (depth/labels/facies/rig) for ALL AC experiments
# except those without a protocol (AC38). Run from the ff_ac repo root AFTER:
#   uv sync ; .\.venv\Scripts\activate   (or use .\.venv\Scripts\python.exe)
# NB: does NOT run --protocol, so the pre-generated protocol CSVs are untouched.
# =============================================================================
param(
  [string]  $Common = "config/common.toml",
  [string]  $RunDir = "config/run_ac",
  [string[]]$Skip   = @("ac38"),          # no protocol xlsx -> cannot set up
  [switch]  $ContinueOnError              # keep going if one experiment fails
)
$python = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$configs = Get-ChildItem (Join-Path $RunDir "ac*.toml") |
           Sort-Object { [int]($_.BaseName -replace '\D','') }
$done = 0; $failed = @()
foreach ($c in $configs) {
  if ($Skip -contains $c.BaseName) {
    Write-Host "-- skipping $($c.BaseName) (in -Skip list)" -ForegroundColor Yellow; continue
  }
  Write-Host "`n========== SETUP $($c.BaseName) ==========" -ForegroundColor Cyan
  & $python scripts/setup.py --config $Common $c.FullName --all
  if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: $($c.BaseName) (exit $LASTEXITCODE)" -ForegroundColor Red
    $failed += $c.BaseName
    if (-not $ContinueOnError) { Write-Host "Stopping (use -ContinueOnError to skip failures)."; break }
  } else { $done++ }
}
Write-Host "`nDone: $done set up. Failed: $(if($failed){$failed -join ', '}else{'none'})." -ForegroundColor Green
