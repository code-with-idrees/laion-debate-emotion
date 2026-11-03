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
    # scored files per stage, from the same counter status.py uses
    $out = python -c @"
import sys
sys.path.insert(0, 'scripts')
import config as C
idx = {p.name[:-len('.index.jsonl')] for p in C.SNIPPETS.glob('*.index.jsonl')}
def done(suffix):
    return {p.name[:-len(suffix)] for p in C.SCORES.glob('*' + suffix)} & idx
print(len(idx) - len(done('.voicenet.jsonl')),
      len(idx) - len(done('.empathic.jsonl')),
      len(idx) - len(done('.voiceclap.jsonl')))
"@
    return $out
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

    python -u scripts\07_score_voicenet.py --batch-size 16
    python -u scripts\06_score_empathic.py --batch-size 16
    python -u scripts\08_score_voiceclap.py --batch-size 2

    Start-Sleep 120
}
