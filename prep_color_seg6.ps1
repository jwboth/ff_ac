# prep_color_seg6.ps1
# Prepares the per-rep calibration cache for ONE colour-correction state (on/off) and
# writes the color-state stamp that build_context auto-follows. Toggling colour = re-run
# this with the other -Color (rebuilds rig + embedding + seed for that state, as agreed).
#
# Noisy darsia output goes to per-rep log files (logs\prep\<rep>_<color>.log); the terminal
# shows one concise line per rep, and a CSV summary is written incrementally to
# logs\prep_color_<color>_<timestamp>.csv so you can review what happened at a glance.
#
# Originaldata (rabilder) berores ALDRI - kun rigg-cache + embedding + seed under
# results-stien per rep (Z:\Albus\Results\<rep>\...).
#
#   .\prep_color_seg6.ps1 -Color on            # hele flaten WITH colour
#   .\prep_color_seg6.ps1 -Color off           # hele flaten WITHOUT colour
#   .\prep_color_seg6.ps1 -Color on -Runs ac53 # ett utvalg
#
param(
  [Parameter(Mandatory = $true)][ValidateSet("on", "off")][string]$Color,
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$Common = "config_seg6/common.toml",
  [string]$ConfigDir = "config_seg6/run_ac",
  [string]$ColorOverlay = "config_seg6/coloron.toml",
  [string[]]$Runs = @("ac60","ac31","ac26","ac42","ac22","ac27","ac48","ac51","ac50","ac53","ac58")
)
$ErrorActionPreference = "Continue"   # keep going across reps; we record per-rep status

$stampDir = Join-Path $ConfigDir ".color_state"
New-Item -ItemType Directory -Force -Path $stampDir | Out-Null
$logDir = "logs/prep"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$csv = "logs/prep_color_${Color}_${stamp}.csv"

function Run-Step([string[]]$cmdArgs, [string]$log) {
  # run python with the given args, append all output to the log, return exit code
  & $Python @cmdArgs *>> $log
  return $LASTEXITCODE
}

$i = 0
foreach ($r in $Runs) {
  $i++
  $cfg = "$ConfigDir/$r.toml"
  $cfgs = @($Common, $cfg)
  if ($Color -eq "on") { $cfgs += $ColorOverlay }
  $log = Join-Path $logDir "${r}_${Color}.log"
  Remove-Item $log -ErrorAction SilentlyContinue
  "=== prep $r color=$Color $(Get-Date -Format o) ===" | Out-File $log

  Write-Host ("[{0}/{1}] {2} color={3} ... " -f $i, $Runs.Count, $r, $Color) -ForegroundColor Cyan -NoNewline

  $row = [ordered]@{
    run = $r; color = $Color; rig = ""; checker_active = ""; checker_residual = "";
    checker_verdict = ""; embedding = ""; mass = ""; stamp = ""; log = $log
  }

  # 1) rig (builds rig + runs find_colorchecker)
  $rc = Run-Step (@("scripts/setup.py", "--config") + $cfgs + @("--rig")) $log
  $row.rig = if ($rc -eq 0) { "ok" } else { "FAIL($rc)" }

  # 2) checker verification (colour=on only)
  if ($Color -eq "on" -and $rc -eq 0) {
    $out = & $Python scripts/check_colorchecker.py $r 2>&1 | Out-String
    $out | Out-File $log -Append
    $row.checker_active = if ($out -match "active=True") { "yes" } elseif ($out -match "active=False") { "no" } else { "?" }
    if ($out -match "last_residual\s*=\s*([\d.eE+\-]+)") { $row.checker_residual = "{0:N4}" -f [double]$matches[1] }
    $row.checker_verdict = if ($out -match "VERDICT: correction looks GOOD") { "GOOD" } elseif ($out -match "VERDICT: correction BAD") { "BAD" } else { "?" }
  }

  # 3) color-embedding
  if ($rc -eq 0) {
    $rc2 = Run-Step (@("scripts/calibration.py", "--config") + $cfgs + @("--color-embedding")) $log
    $row.embedding = if ($rc2 -eq 0) { "ok" } else { "FAIL($rc2)" }
  } else { $row.embedding = "skip" }

  # 4) default-mass --reset
  if ($row.embedding -eq "ok") {
    $rc3 = Run-Step (@("scripts/calibration.py", "--config") + $cfgs + @("--default-mass", "--reset")) $log
    $row.mass = if ($rc3 -eq 0) { "ok" } else { "FAIL($rc3)" }
  } else { $row.mass = "skip" }

  # 5) stamp (only if the whole chain succeeded)
  if ($row.rig -eq "ok" -and $row.embedding -eq "ok" -and $row.mass -eq "ok") {
    Set-Content -Path (Join-Path $stampDir "$r.txt") -Value $Color -NoNewline
    $row.stamp = $Color
  } else { $row.stamp = "NOT-WRITTEN" }

  # incremental CSV append (so a crash still leaves a partial summary)
  [pscustomobject]$row | Export-Csv -Path $csv -NoTypeInformation -Append -Encoding UTF8

  $col = if ($row.stamp -eq "NOT-WRITTEN") { "Red" } elseif ($row.checker_verdict -eq "BAD" -or $row.checker_active -eq "no") { "Yellow" } else { "Green" }
  Write-Host ("rig={0} checker={1}/{2} res={3} embed={4} mass={5} -> stamp={6}" -f `
      $row.rig, $row.checker_active, $row.checker_verdict, $row.checker_residual, $row.embedding, $row.mass, $row.stamp) -ForegroundColor $col
}

Write-Host "`nSummary CSV: $csv" -ForegroundColor Cyan
Write-Host "Per-rep logs: $logDir\<rep>_${Color}.log" -ForegroundColor DarkGray
if (Test-Path $csv) { Import-Csv $csv | Format-Table -AutoSize }
