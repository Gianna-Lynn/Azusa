$logDir = Join-Path $PSScriptRoot "target\\logs"
$logPath = Join-Path $logDir "sift1m_recall_experiment.out"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (Test-Path -LiteralPath $logPath) {
    Remove-Item -LiteralPath $logPath -Force
}

cargo test --release test_hnsw_in_sift1m -- --nocapture 2>&1 |
    Tee-Object -FilePath $logPath

exit $LASTEXITCODE
