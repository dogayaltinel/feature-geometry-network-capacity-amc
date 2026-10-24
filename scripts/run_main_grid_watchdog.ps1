param(
    [int]$PollSeconds = 30,
    [int]$StaleSeconds = 180
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) { $Python = Join-Path $Root '.venv\Scripts\python.exe' }
$GridRoot = Join-Path $Root 'main_grid_v1'
$StatusPath = Join-Path $GridRoot 'main_grid_status.json'
$WatchdogLog = Join-Path $GridRoot 'watchdog.log'
$RunnerOut = Join-Path $GridRoot 'watchdog_runner_stdout.log'
$RunnerErr = Join-Path $GridRoot 'watchdog_runner_stderr.log'

New-Item -ItemType Directory -Force -Path $GridRoot | Out-Null
$mutex = New-Object System.Threading.Mutex($false, 'FeatureGeometry_MainGrid_Watchdog')
if (-not $mutex.WaitOne(0)) {
    exit 0
}

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $WatchdogLog -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message)
}

function Get-ActiveRun($Status) {
    @($Status.runs | Where-Object { $_.status -eq 'running' }) | Select-Object -First 1
}

function Get-RunAgeSeconds($Run) {
    $runDir = Join-Path $Root ('records\validation\' + $Run.run_id)
    $epochLog = Join-Path $runDir 'epoch_log.csv'
    if (Test-Path $epochLog) {
        return ((Get-Date) - (Get-Item $epochLog).LastWriteTime).TotalSeconds
    }
    if ($Run.started_utc) {
        try { return ((Get-Date) - ([DateTime]::Parse($Run.started_utc))).TotalSeconds } catch { }
    }
    return [double]::PositiveInfinity
}

$runner = $null
try {
    Write-Log 'Watchdog started.'
    while ($true) {
        if (-not (Test-Path $StatusPath)) {
            Write-Log 'Status file not found; waiting.'
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        $status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
        $remaining = @($status.runs | Where-Object { $_.status -ne 'completed' })
        if ($remaining.Count -eq 0) {
            Write-Log 'All main-grid runs completed.'
            break
        }

        if ($runner -and -not $runner.HasExited) {
            $active = Get-ActiveRun $status
            if ($active -and ((Get-RunAgeSeconds $active) -gt $StaleSeconds)) {
                Write-Log ("Stale active run detected: {0}; stopping runner tree." -f $active.run_id)
                & taskkill.exe /PID $runner.Id /T /F | Out-Null
                $runner = $null
            } else {
                Start-Sleep -Seconds $PollSeconds
                continue
            }
        }

        # A runner may have been started outside this watchdog. Do not create
        # a second runner while its active epoch log is still fresh.
        $externalActive = Get-ActiveRun $status
        if (-not $runner -and $externalActive -and ((Get-RunAgeSeconds $externalActive) -le $StaleSeconds)) {
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        $runner = Start-Process -FilePath $Python `
            -ArgumentList @('scripts\run_main_grid.py', '--max-retries', '1') `
            -WorkingDirectory $Root `
            -RedirectStandardOutput $RunnerOut `
            -RedirectStandardError $RunnerErr `
            -WindowStyle Hidden `
            -PassThru
        Write-Log ("Runner started with PID {0}." -f $runner.Id)
        Start-Sleep -Seconds $PollSeconds
    }
} finally {
    if ($mutex) { $mutex.ReleaseMutex() | Out-Null; $mutex.Dispose() }
    Write-Log 'Watchdog stopped.'
}
