[CmdletBinding()]
param(
    [ValidateSet("cuda", "opencl")]
    [string]$Backend = "cuda",

    [switch]$Stop
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$queue = "\\Moderskipet\Darsia_Queue\Kalibrering_AC_phase_screen_20260727_phase_residualgas_balanced"
$logsDir = "Z:\Albus\Autokalibrering_log\phase_screen_20260727\phase_residualgas_balanced\facies1_perlabel1_warmup150_optuna800_parallel_20260727_2133"
$seedParams = "Z:\Albus\Autokalibrering_log\phase_screen_20260727\phase_screen_seed_params.json"
$controlDir = Join-Path $queue "control"
$candidateRuns = @("ac20", "ac26", "ac27", "ac24", "ac32")

foreach ($required in @($python, $queue, $logsDir, $seedParams)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is unavailable: $required"
    }
}

New-Item -ItemType Directory -Path $controlDir -Force | Out-Null
foreach ($hostName in @("Moderskipet", "Olav")) {
    Set-Content `
        -LiteralPath (Join-Path $controlDir "$hostName.txt") `
        -Value "0" `
        -Encoding ascii
}

$allProcesses = @(Get-CimInstance Win32_Process)
$active = @(
    $allProcesses |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match "Kalibrering_AC_phase_screen_20260727_phase_residualgas_balanced" -and
            $_.CommandLine -match "distributed_auto_calibration_queue\.py\s+(master|watchdog|worker)"
        }
)
if ($Stop) {
    $childrenByParent = @{}
    foreach ($processInfo in $allProcesses) {
        $parent = [int]$processInfo.ParentProcessId
        if (-not $childrenByParent.ContainsKey($parent)) {
            $childrenByParent[$parent] = [System.Collections.Generic.List[int]]::new()
        }
        $childrenByParent[$parent].Add([int]$processInfo.ProcessId)
    }

    $orderedIds = [System.Collections.Generic.List[int]]::new()
    $visited = [System.Collections.Generic.HashSet[int]]::new()
    function Add-ProcessTreePostOrder {
        param([int]$ProcessId)

        if (-not $visited.Add($ProcessId)) {
            return
        }
        if ($childrenByParent.ContainsKey($ProcessId)) {
            foreach ($childId in $childrenByParent[$ProcessId]) {
                Add-ProcessTreePostOrder -ProcessId $childId
            }
        }
        $orderedIds.Add($ProcessId)
    }

    foreach ($rootProcess in $active) {
        Add-ProcessTreePostOrder -ProcessId ([int]$rootProcess.ProcessId)
    }
    foreach ($processId in $orderedIds) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Stopped $($orderedIds.Count) residual campaign process(es); worker limits remain 0."
    exit 0
}
if ($active.Count -gt 0) {
    $ids = ($active | Select-Object -ExpandProperty ProcessId) -join ", "
    throw "Residual campaign still has active processes: $ids"
}

$inProgress = @(Get-ChildItem -LiteralPath (Join-Path $queue "in_progress") -File)
$results = @(Get-ChildItem -LiteralPath (Join-Path $queue "results") -File)
if ($inProgress.Count -gt 0 -or $results.Count -gt 0) {
    throw "Queue is not at a clean boundary: in_progress=$($inProgress.Count), results=$($results.Count)"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$pending = @(Get-ChildItem -LiteralPath (Join-Path $queue "pending") -File)
if ($pending.Count -gt 0) {
    $archive = Join-Path $queue "archive\pre_local_gpu_$timestamp\pending"
    New-Item -ItemType Directory -Path $archive -Force | Out-Null
    foreach ($path in $pending) {
        Move-Item -LiteralPath $path.FullName -Destination $archive
    }
}

$unfinishedRuns = @(
    foreach ($run in $candidateRuns) {
        $completeMarker = Join-Path $queue "run_complete\$run.json"
        $doneCount = @(
            Get-ChildItem `
                -LiteralPath (Join-Path $queue "done") `
                -File `
                -Filter "${run}_optuna_*.json"
        ).Count
        if (-not (Test-Path -LiteralPath $completeMarker) -and $doneCount -lt 800) {
            $run
        }
    }
)
if ($unfinishedRuns.Count -eq 0) {
    Write-Host "All residual-gas runs already have 800 Optuna trials."
    exit 0
}

$optunaDir = Join-Path $env:LOCALAPPDATA "ff_ac\optuna\phase_residualgas_20260727"
$processLogDir = Join-Path $repoRoot "logs\phase_residual_local_gpu"
New-Item -ItemType Directory -Path $optunaDir, $processLogDir -Force | Out-Null

$env:FFAC_TITRATION_FLASH = "on"
$env:FFAC_TEMPLATE_REGISTRATION = "ac14_template"
$env:FFAC_TEMPLATE_REGISTRATION_MODE = "partial_affine"
$env:FFAC_TEMPLATE_REGISTRATION_STRICT = "on"
$env:FFAC_COLOR_PATH_ANCHOR = "ac60"
$env:FFAC_COLOR_PATH_ANCHOR_WEIGHT = "0.75"
$env:FFAC_COLOR_PATH_ANCHOR_STRICT = "on"
$env:FFAC_PHASE_SEPARATION = "residual-gas"
$env:FFAC_MASTER_LIGHT_CONTEXT = "on"
Remove-Item Env:\FFAC_STATIC_LIGHT_CORRECTION -ErrorAction SilentlyContinue
Remove-Item Env:\FFAC_COUPLE_AQ_GAS -ErrorAction SilentlyContinue
Remove-Item Env:\FFAC_SIGNAL_PARAMETERIZATION -ErrorAction SilentlyContinue

$stdout = Join-Path $processLogDir "master_${Backend}_${timestamp}.stdout.log"
$stderr = Join-Path $processLogDir "master_${Backend}_${timestamp}.stderr.log"
$arguments = @(
    "scripts/distributed_auto_calibration_queue.py",
    "master",
    "--queue", $queue,
    "--no-clear-queue",
    "--runs"
) + $unfinishedRuns + @(
    "--config-dir", "config_seg6/run_ac",
    "--logs-dir", $logsDir,
    "--exact-logs-dir",
    "--use-facies", "true",
    "--per-label", "true",
    "--objective-integral", "window-balanced",
    "--bounds-file", "config/bounds_seg6_titration.json",
    "--no-save-calibration",
    "--seed-params-file", $seedParams,
    "--optuna-seed", "17",
    "--optuna-storage-dir", $optunaDir,
    "--max-iters", "800",
    "--warmup-iters", "150",
    "--run-mode", "parallel",
    "--max-active-runs", "1",
    "--max-in-flight-per-run", "1",
    "--control-dir", $controlDir,
    "--sanity-every", "0",
    "--sanity-scale", "1.00",
    "--quality-dtype", "float32",
    "--local-eval-backend", $Backend,
    "--local-run", $unfinishedRuns[0]
)

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

$manifest = [ordered]@{
    schema = 1
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    pid = $process.Id
    host = [Environment]::MachineName
    backend = $Backend
    optuna_storage = "memory"
    runs = $unfinishedRuns
    queue = $queue
    logs_dir = $logsDir
    optuna_dir = $optunaDir
    stdout = $stdout
    stderr = $stderr
    archived_pending = $pending.Count
}
$manifestPath = Join-Path $processLogDir "manifest_${Backend}_${timestamp}.json"
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding utf8

Start-Sleep -Seconds 2
if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
    throw "Local GPU master exited during startup. Inspect $stderr"
}

$manifest | ConvertTo-Json -Depth 4
Write-Host "Manifest: $manifestPath"
