# Supervisor for the 436 GB corpus pull.
#
# The link throttles down to well under 1 MB/s for long stretches, and a block
# that fails all its retries marks the file as error. Re-running 01 picks up
# from the per-file checkpoints, so this loop simply keeps going until every
# manifest row is ok.
#
# Exit codes from 01:
#   0  every queued file was handled
#   3  the stall watchdog fired: no byte arrived for --stall-timeout seconds
# A stall means the sockets are dead, not that the link is busy, so the pass is
# restarted after a short pause. A pass that made no progress for any other
# reason backs off for five minutes instead of hammering a dead link.
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$target = 2890

while ($true) {
    $before = [int](python scripts\manifest_ok.py)
    if ($before -ge $target) {
        Write-Output ("[{0}] all {1} recordings downloaded" -f (Get-Date -Format 'HH:mm:ss'), $target)
        break
    }
    Write-Output ("[{0}] pass starting at {1}/{2}" -f (Get-Date -Format 'HH:mm:ss'), $before, $target)
    python -u scripts\01_download_dataset.py --workers 12 --chunk-threads 1
    $rc = $LASTEXITCODE
    $after = [int](python scripts\manifest_ok.py)
    Write-Output ("[{0}] pass finished: {1} -> {2}/{3} (exit {4})" -f (Get-Date -Format 'HH:mm:ss'), $before, $after, $target, $rc)
    if ($rc -eq 3) {
        Write-Output ("[{0}] stalled sockets, restarting in 30s" -f (Get-Date -Format 'HH:mm:ss'))
        Start-Sleep 30
    } elseif ($after -le $before) {
        Write-Output ("[{0}] no progress, backing off 300s" -f (Get-Date -Format 'HH:mm:ss'))
        Start-Sleep 300
    }
}
