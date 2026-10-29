# install_autostart.ps1 - make the pipeline restart itself at every logon.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
#
# A per-user Startup shortcut, no administrator rights needed (a Scheduled Task
# would need elevation, which this account does not have). At logon it runs
# start_all.ps1 minimized and hidden; start_all skips any job that is already
# alive, so it is safe to run repeatedly.
#
# Remove it again with:
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Remove
param([switch]$Remove)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$startup = [Environment]::GetFolderPath('Startup')
$link = Join-Path $startup 'LAION Debate Corpus.lnk'

if ($Remove) {
    if (Test-Path $link) { Remove-Item $link -Force; Write-Output "removed $link" }
    else { Write-Output "nothing to remove" }
    return
}

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($link)
$sc.TargetPath = 'powershell.exe'
$sc.Arguments = ('-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass ' +
                 '-File "{0}\scripts\start_all.ps1"' -f $root)
$sc.WorkingDirectory = $root
$sc.WindowStyle = 7          # minimized
$sc.Description = 'Resume the sleeping-debate download and pipeline at logon'
$sc.Save()

Write-Output "installed $link"
Write-Output ("  -> {0}" -f $sc.Arguments)
Write-Output 'At every logon start_all.ps1 runs hidden and resumes from the block checkpoints.'
Write-Output 'Undo with: powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Remove'
