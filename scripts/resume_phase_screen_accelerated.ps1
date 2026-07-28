[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Main", "Worker")]
    [string]$Role,

    [switch]$Stop
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

$hostName = [Environment]::MachineName
$queuePrefix = "Kalibrering_AC_phase_screen_20260727_"
$variants = @(
    "phase_control_titration",
    "phase_sharedpath_balanced",
    "phase_residualgas_balanced"
)
$queues = $variants | ForEach-Object {
    "\\Moderskipet\Darsia_Queue\$queuePrefix$_"
}

function Set-LocalWorkerLimit {
    param([int]$Limit)

    foreach ($queue in $queues) {
        $controlDir = Join-Path $queue "control"
        New-Item -ItemType Directory -Path $controlDir -Force | Out-Null
        Set-Content `
            -LiteralPath (Join-Path $controlDir "$hostName.txt") `
            -Value ([string]$Limit) `
            -Encoding ascii
    }
}

function Stop-LocalCampaignProcesses {
    $allProcesses = @(Get-CimInstance Win32_Process)
    $roots = @(
        $allProcesses | Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match [regex]::Escape($queuePrefix) -and
            $_.CommandLine -match "distributed_auto_calibration_queue\.py\s+(watchdog|worker)"
        }
    )
    if ($roots.Count -eq 0) {
        return 0
    }

    $childrenByParent = @{}
    foreach ($process in $allProcesses) {
        $parent = [int]$process.ParentProcessId
        if (-not $childrenByParent.ContainsKey($parent)) {
            $childrenByParent[$parent] = [System.Collections.Generic.List[int]]::new()
        }
        $childrenByParent[$parent].Add([int]$process.ProcessId)
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

    foreach ($root in $roots) {
        Add-ProcessTreePostOrder -ProcessId ([int]$root.ProcessId)
    }
    foreach ($processId in $orderedIds) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    return $orderedIds.Count
}

Set-LocalWorkerLimit -Limit 0
$stopped = Stop-LocalCampaignProcesses
if ($Stop) {
    Write-Host "Stopped $stopped local watchdog/worker process(es); worker limits are 0."
    exit 0
}

$isMain = $Role -eq "Main"
$workersPerQueue = if ($isMain) { 2 } else { 4 }
$evaluationBackend = if ($isMain) { "auto" } else { "prepared" }
$cudaWorkers = if ($isMain) { 1 } else { 0 }

$env:FFAC_TITRATION_FLASH = "on"
$env:FFAC_TEMPLATE_REGISTRATION = "ac14_template"
$env:FFAC_TEMPLATE_REGISTRATION_MODE = "partial_affine"
$env:FFAC_TEMPLATE_REGISTRATION_STRICT = "on"

$logDir = Join-Path $repoRoot "logs\phase_screen_accelerated"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$started = @()

for ($index = 0; $index -lt $queues.Count; $index++) {
    $queue = $queues[$index]
    $variant = $variants[$index]
    $controlDir = Join-Path $queue "control"
    $stdout = Join-Path $logDir "${variant}_${hostName}_${timestamp}.stdout.log"
    $stderr = Join-Path $logDir "${variant}_${hostName}_${timestamp}.stderr.log"
    $arguments = @(
        "scripts/distributed_auto_calibration_queue.py",
        "watchdog",
        "--queue", $queue,
        "--config-dir", "config_seg6/run_ac",
        "--use-facies", "true",
        "--per-label", "true",
        "--bounds-file", "config/bounds_seg6_titration.json",
        "--control-dir", $controlDir,
        "--workers", [string]$workersPerQueue,
        "--worker-stall-seconds", "600",
        "--idle-exit-seconds", "900",
        "--stickiness-wait-seconds", "20",
        "--stagger-seconds", "10",
        "--threads-per-worker", "1",
        "--max-tasks-per-worker", "0",
        "--eval-backend", $evaluationBackend,
        "--cuda-workers", [string]$cudaWorkers
    )

    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
    $started += [pscustomobject]@{
        variant = $variant
        pid = $process.Id
        role = $Role
        workers = $workersPerQueue
        backend = $evaluationBackend
        cuda_workers = $cudaWorkers
        stdout = $stdout
        stderr = $stderr
    }
}

$manifest = Join-Path $logDir "watchdogs_${hostName}_${timestamp}.json"
$started | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifest -Encoding utf8
Set-LocalWorkerLimit -Limit $workersPerQueue
Start-Sleep -Seconds 3

$started | Select-Object variant, pid, workers, backend, cuda_workers | Format-Table -AutoSize
Write-Host "Manifest: $manifest"
Write-Host "Stopped old process count: $stopped"
