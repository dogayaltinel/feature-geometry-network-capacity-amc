param(
    [int]$PollSeconds = 30,
    [int]$StaleSeconds = 900
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) { $Python = Join-Path $Root '.venv\Scripts\python.exe' }
$QueueRoot = Join-Path $Root 'confirmatory_v1'
$StatusPath = Join-Path $QueueRoot 'confirmatory_status.json'
$WatchdogLog = Join-Path $QueueRoot 'watchdog.log'
$RunnerOut = Join-Path $QueueRoot 'watchdog_runner_stdout.log'
$RunnerErr = Join-Path $QueueRoot 'watchdog_runner_stderr.log'

New-Item -ItemType Directory -Force -Path $QueueRoot | Out-Null
$mutex = New-Object System.Threading.Mutex($false, 'FeatureGeometry_Confirmatory_Watchdog')
if (-not $mutex.WaitOne(0)) { exit 0 }

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $WatchdogLog -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message)
}

function Get-ActiveRun($Status) {
    @($Status.runs | Where-Object { $_.status -eq 'running' }) | Select-Object -First 1
}

function Get-RunAgeSeconds($Run) {
    $runDir = Join-Path $Root ('records\validation\' + $Run.run_id)
    $epochLog = Join-Path $runDir 'epoch_log.csv'
    if (Test-Path $epochLog) { return ((Get-Date) - (Get-Item $epochLog).LastWriteTime).TotalSeconds }
    if ($Run.started_utc) {
        try { return ((Get-Date) - ([DateTime]::Parse($Run.started_utc))).TotalSeconds } catch { }
    }
    return [double]::PositiveInfinity
}

function Get-ActiveStatusAgeSeconds($Run) {
    if ($Run.started_utc) {
        try { return ((Get-Date) - ([DateTime]::Parse($Run.started_utc))).TotalSeconds } catch { }
    }
    return [double]::PositiveInfinity
}

$runner = $null
$runnerStartedAt = $null
try {
    Write-Log 'Confirmatory watchdog started.'
    while ($true) {
        if (-not (Test-Path $StatusPath)) { Start-Sleep -Seconds $PollSeconds; continue }
        $status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
        $remaining = @($status.runs | Where-Object { $_.status -in @('queued', 'running') })
        if ($remaining.Count -eq 0) { Write-Log 'All confirmatory runs completed.'; break }

        if ($runner -and -not $runner.HasExited) {
            $active = Get-ActiveRun $status
            if ($active -and ((Get-RunAgeSeconds $active) -gt $StaleSeconds)) {
                # A newly started runner may resume from an old epoch log and
                # need time to restore the checkpoint before writing a new row.
                # Never kill it during that startup grace period.
                $runnerAge = if ($runnerStartedAt) { ((Get-Date) - $runnerStartedAt).TotalSeconds } else { 0 }
                if ($runnerAge -gt $StaleSeconds) {
                    Write-Log ("Stale active run detected: {0}; stopping runner tree." -f $active.run_id)
                    & taskkill.exe /PID $runner.Id /T /F | Out-Null
                    $runner = $null
                    $runnerStartedAt = $null
                } else { Start-Sleep -Seconds $PollSeconds; continue }
            } else { Start-Sleep -Seconds $PollSeconds; continue }
        }

        $externalActive = Get-ActiveRun $status
        if (-not $runner -and $externalActive) {
            # The runner can disappear from this PowerShell session while its
            # child process is still restoring a checkpoint.  Respect the
            # status timestamp during that startup window; the runner lock is
            # the final guard against a second queue process.
            $logAge = Get-RunAgeSeconds $externalActive
            $statusAge = Get-ActiveStatusAgeSeconds $externalActive
            if (($logAge -le $StaleSeconds) -or ($statusAge -le $StaleSeconds)) {
                Start-Sleep -Seconds $PollSeconds
                continue
            }
        }

        $runner = Start-Process -FilePath $Python `
            -ArgumentList @('scripts\run_confirmatory_subset.py', '--max-retries', '5') `
            -WorkingDirectory $Root `
            -RedirectStandardOutput $RunnerOut `
            -RedirectStandardError $RunnerErr `
            -WindowStyle Hidden `
            -PassThru
        $runnerStartedAt = Get-Date
        Write-Log ("Runner started with PID {0}." -f $runner.Id)
        Start-Sleep -Seconds $PollSeconds
    }
} finally {
    if ($mutex) { $mutex.ReleaseMutex() | Out-Null; $mutex.Dispose() }
    Write-Log 'Confirmatory watchdog stopped.'
}
