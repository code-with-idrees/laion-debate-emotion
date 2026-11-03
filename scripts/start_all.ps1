# start_all.ps1 - launch the three long-running jobs, skipping any already up.
#
#   powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1
#
# Jobs, in order of importance on a shared link:
#   1. corpus download supervisor  -> logs/dl_dataset.log   (436 GB, days)
#   2. live pipeline               -> logs/live_pipeline.log (diarize/ASR-free
#                                     turns/snippets for whatever has landed)
#   3. model weights               -> logs/dl_weights.log    (one stream only,
#                                     so it does not compete with the corpus)
#   4. scoring supervisor          -> logs/run_scores.log    (GPU: voicenet,
#                                     empathic, voiceclap, then assemble)
#
# Safe to re-run: a job whose process is already alive is reported, not started
# twice. Two downloaders on the same .part file would corrupt it.
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)

function Get-Running([string]$pattern) {
    # Match the *invocation*, not any process whose command line merely mentions
    # the script: a shell command that greps for "run_scores" used to count as a
    # running job and silently skipped the start.
    $self = $PID
    $parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$PID").ParentProcessId
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe'" |
      Where-Object {
          $_.ProcessId -ne $self -and $_.ProcessId -ne $parent -and
          $_.CommandLine -like $pattern
      }).Count
}

function Start-Logged([string]$what, [string]$file, [string[]]$argList,
                      [string]$outLog, [string]$errLog) {
    Start-Process -FilePath $file -ArgumentList $argList `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
        -WindowStyle Hidden
    Write-Output ("  started {0} -> {1}" -f $what, $outLog)
}

Write-Output ("[{0}] root: {1}" -f (Get-Date -Format 'HH:mm:ss'), (Get-Location))

if ((Get-Running '*-File scripts\run_corpus.ps1*') -or (Get-Running '*scripts\01_download_dataset.py*')) {
    Write-Output '  corpus download    already running'
} else {
    Start-Logged 'corpus download' 'powershell' @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'scripts\run_corpus.ps1'
    ) 'logs\dl_dataset.log' 'logs\dl_dataset.err.log'
}

if (Get-Running '*scripts\live_pipeline.py*') {
    Write-Output '  live pipeline      already running'
} else {
    Start-Logged 'live pipeline' 'python' @(
        '-u', 'scripts\live_pipeline.py', '--limit', '20', '--sleep', '180',
        '--jobs', '4'
    ) 'logs\live_pipeline.log' 'logs\live_pipeline.err.log'
}

if (Get-Running '*scripts\00_download_models.py*') {
    Write-Output '  model weights      already running'
} else {
    Start-Logged 'model weights' 'python' @(
        '-u', 'scripts\00_download_models.py', '--only', 'voiceclap', 'empathic',
        '--workers', '1', '--chunk-threads', '1'
    ) 'logs\dl_weights.log' 'logs\dl_weights.err.log'
}

if (Get-Running '*-File scripts\run_scores.ps1*') {
    Write-Output '  scoring            already running'
} else {
    Start-Logged 'scoring (3 models)' 'powershell' @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'scripts\run_scores.ps1'
    ) 'logs\run_scores.log' 'logs\run_scores.err.log'
}

Start-Sleep 5
Write-Output ''
python scripts\manifest_ok.py --verbose
Write-Output ''
Write-Output 'Watch it (Ctrl-C stops the display, not the download):'
Write-Output ('  cd ' + (Get-Location).Path)
Write-Output '  python scripts\progress.py'
Write-Output 'or double-click scripts\progress.cmd'
Write-Output 'Logs: logs\dl_dataset.log, logs\live_pipeline.log, logs\dl_weights.log'

