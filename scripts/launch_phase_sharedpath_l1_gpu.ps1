[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cuda", "opencl")]
    [string]$Backend,

    [Parameter(Mandatory = $true)]
    [string[]]$Runs,

    [ValidateRange(1, 3)]
    [int]$ParallelRunners = 3,

    [string]$CampaignId = "phase_sharedpath_l1_20260729",

    [string]$LogsRoot = "Z:\Albus\Autokalibrering_log\phase_sharedpath_l1_20260729",

    [string]$QueueRoot = "\\Moderskipet\Darsia_Queue\Kalibrering_AC_phase_sharedpath_l1_20260729",

    [ValidateRange(0, 120)]
    [int]$StaggerSeconds = 15,

    [switch]$Stop
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$seedParams = "Z:\Albus\Autokalibrering_log\phase_screen_20260727\phase_screen_seed_params.json"
$expectedDepthSha = "375be487d0bb598964404432a386316a82496afbc3477aaa3fcf7b81c98fcd21"
$processLogDir = Join-Path $repoRoot "logs\phase_sharedpath_l1_gpu"

if ($Runs.Count -eq 0) {
    throw "At least one run is required."
}
if ($Runs.Count -gt $ParallelRunners) {
    throw "Received $($Runs.Count) runs, but ParallelRunners=$ParallelRunners."
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment is unavailable: $python"
}
if (-not (Test-Path -LiteralPath $seedParams)) {
    throw "Phase-screen seed parameters are unavailable: $seedParams"
}

$processPattern = [regex]::Escape($CampaignId)
$allProcesses = @(Get-CimInstance Win32_Process)
$active = @(
    $allProcesses |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match $processPattern -and
            $_.CommandLine -match "distributed_auto_calibration_queue\.py\s+master"
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
    Write-Host "Stopped $($orderedIds.Count) process(es) for $CampaignId."
    exit 0
}

if ($active.Count -gt 0) {
    $ids = ($active | Select-Object -ExpandProperty ProcessId) -join ", "
    throw "Campaign already has active master process(es): $ids"
}

$variantLogs = Join-Path $LogsRoot "phase_sharedpath_l1"
New-Item -ItemType Directory -Path $variantLogs, $processLogDir -Force | Out-Null
$depthReport = Join-Path $LogsRoot (
    "depth_preflight_{0}_{1}.json" -f [Environment]::MachineName, $Backend
)
& $python (Join-Path $repoRoot "scripts\verify_ac_depth_maps.py") `
    --runs $Runs `
    --measurements (Join-Path $repoRoot "data\depth_measurements.csv") `
    --output $depthReport
if ($LASTEXITCODE -ne 0) {
    throw "Depth-map preflight failed with exit code $LASTEXITCODE."
}

$env:FFAC_TITRATION_FLASH = "on"
$env:FFAC_TEMPLATE_REGISTRATION = "ac14_template"
$env:FFAC_TEMPLATE_REGISTRATION_MODE = "partial_affine"
$env:FFAC_TEMPLATE_REGISTRATION_STRICT = "on"
$env:FFAC_COLOR_PATH_ANCHOR = "ac60"
$env:FFAC_COLOR_PATH_ANCHOR_WEIGHT = "0.75"
$env:FFAC_COLOR_PATH_ANCHOR_STRICT = "on"
$env:FFAC_MASTER_LIGHT_CONTEXT = "on"
$env:FFAC_REQUIRE_VARYING_DEPTH = "on"
$env:FFAC_EXPECTED_DEPTH_SHA256 = $expectedDepthSha
Remove-Item Env:\FFAC_STATIC_LIGHT_CORRECTION -ErrorAction SilentlyContinue
Remove-Item Env:\FFAC_COUPLE_AQ_GAS -ErrorAction SilentlyContinue
Remove-Item Env:\FFAC_SIGNAL_PARAMETERIZATION -ErrorAction SilentlyContinue
Remove-Item Env:\FFAC_PHASE_SEPARATION -ErrorAction SilentlyContinue

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$launched = @()
foreach ($runValue in $Runs) {
    $run = $runValue.ToLowerInvariant()
    $queue = "${QueueRoot}_${run}"
    if (
        (Test-Path -LiteralPath $queue) -and
        @(Get-ChildItem -LiteralPath $queue -Force -ErrorAction SilentlyContinue).Count -gt 0
    ) {
        throw "Fresh campaign queue is not empty: $queue"
    }

    $runLogs = Join-Path $variantLogs $run
    $controlDir = Join-Path $queue "control"
    $optunaDir = Join-Path $env:LOCALAPPDATA (
        "ff_ac\optuna\{0}\{1}_{2}" -f $CampaignId, $run, $Backend
    )
    New-Item -ItemType Directory -Path $runLogs, $optunaDir -Force | Out-Null

    $stdout = Join-Path $processLogDir (
        "{0}_{1}_{2}_{3}.stdout.log" -f $CampaignId, $run, $Backend, $timestamp
    )
    $stderr = Join-Path $processLogDir (
        "{0}_{1}_{2}_{3}.stderr.log" -f $CampaignId, $run, $Backend, $timestamp
    )
    $arguments = @(
        "scripts/distributed_auto_calibration_queue.py",
        "master",
        "--queue", $queue,
        "--runs", $run,
        "--config-dir", "config_seg6/run_ac",
        "--logs-dir", $runLogs,
        "--exact-logs-dir",
        "--use-facies", "true",
        "--per-label", "true",
        "--objective-integral", "off",
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
        "--local-run", $run
    )

    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru

    $launched += [ordered]@{
        run = $run
        pid = $process.Id
        host = [Environment]::MachineName
        backend = $Backend
        queue = $queue
        logs_dir = $runLogs
        optuna_dir = $optunaDir
        stdout = $stdout
        stderr = $stderr
        depth_sha256 = $expectedDepthSha
    }
    Start-Sleep -Seconds 2
    if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
        throw "$run local $Backend master exited during startup. Inspect $stderr"
    }
    if ($StaggerSeconds -gt 0 -and $run -ne $Runs[-1].ToLowerInvariant()) {
        Start-Sleep -Seconds $StaggerSeconds
    }
}

$manifest = [ordered]@{
    schema = 1
    campaign_id = $CampaignId
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    host = [Environment]::MachineName
    backend = $Backend
    parallel_runners = $launched.Count
    depth_report = $depthReport
    processes = $launched
}
$manifestPath = Join-Path $processLogDir (
    "manifest_{0}_{1}_{2}_{3}.json" -f
        $CampaignId,
        [Environment]::MachineName,
        $Backend,
        $timestamp
)
$manifest | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath $manifestPath -Encoding utf8

$manifest | ConvertTo-Json -Depth 6
Write-Host "Manifest: $manifestPath"
