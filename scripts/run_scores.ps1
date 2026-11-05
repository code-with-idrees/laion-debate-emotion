# Supervisor for the three LAION scoring stages.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_scores.ps1
#
# Scoring is GPU-bound while the corpus pull is network-bound, so the two run
# side by side. The stages share one 24 GB GPU, so they are chained rather than
# started at once: each stage loads its model, drains every snippet that exists
# right now, exits, and the next one starts. A loop pass therefore costs one
# model load per stage, which is the price of not blowing VRAM.
#
# Every stage skips files whose output already parses and covers all segments,
# so re-running the loop as the pipeline cuts new snippets is safe and cheap.
#
# When every snippet has all three scores the loop assembles the Parquet and
# stops.
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)

function Pending {
    # scripts\pending.py prints "voicenet empathic voiceclap" pending counts.
    # It is a file, not an inline `python -c`: Windows PowerShell 5.1 drops the
    # double quotes inside a multi-line here-string passed to a native command,
    # so the inline version failed and returned nothing.
    return (python scripts\pending.py)
}

while ($true) {
    $ts = Get-Date -Format 'HH:mm:ss'
    $p = (Pending) -split '\s+'
    Write-Output ("[{0}] pending voicenet={1} empathic={2} voiceclap={3}" -f $ts, $p[0], $p[1], $p[2])

    if ("$p" -match '^0\s+0\s+0$') {
        Write-Output ("[{0}] all snippets scored; assembling" -f (Get-Date -Format 'HH:mm:ss'))
        python -u scripts\09_assemble_dataset.py
        break
    }

    # Only run a stage that still has work. A finished stage would otherwise
    # re-scan all 2,890 manifests (and reload its model) on every pass, which is
    # pure overhead once VoiceCLAP is the only stage left.
    if ($p[0] -ne '0') { python -u scripts\07_score_voicenet.py --batch-size 16 }
    if ($p[1] -ne '0') { python -u scripts\06_score_empathic.py --batch-size 16 }
    if ($p[2] -ne '0') { python -u scripts\08_score_voiceclap.py --batch-size 8 }

    Start-Sleep 120
}

