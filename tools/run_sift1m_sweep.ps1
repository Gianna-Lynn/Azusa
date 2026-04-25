param(
    [string]$RemoteRepo = "/mnt/sevenT/zeliny/Repositories/Azusa",
    [string]$SweepName = "sift1m_hnsw_sweep_1m_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
)

$ErrorActionPreference = "Stop"

$sweepRootRemote = "target/benchmark/sweeps/$SweepName"
$launcherLogRemote = "$RemoteRepo/$sweepRootRemote/launcher.log"

ssh E5_server "cd $RemoteRepo && mkdir -p '$sweepRootRemote'"

$remoteScript = @'
#!/usr/bin/env bash
set -euo pipefail
cd "__REMOTE_REPO__"
git pull --ff-only

sweep_root="target/benchmark/sweeps/__SWEEP_NAME__"
mkdir -p "$sweep_root"

manifest="$sweep_root/manifest.csv"
printf 'label,m,ef_construction,base_limit,query_limit,ks,ef_search\n' > "$manifest"

echo "sweep_root=$sweep_root"
echo "manifest=$manifest"

run_one() {
  m="$1"
  efc="$2"
  label="m${m}_efc${efc}"
  run_dir="$sweep_root/$label"
  mkdir -p "$run_dir"
  echo "=== start $label ===" | tee -a "$sweep_root/sweep.log"
  cargo run --release --bin sift1m_benchmark -- \
    --mode hnsw \
    --base-limits 1000000 \
    --ks 1,5,10 \
    --ef-search 60,100,300 \
    --query-limit 100 \
    --m "$m" \
    --ef-construction "$efc" \
    > "$run_dir/run.log" 2>&1
  cp target/benchmark/sift1m.csv "$run_dir/sift1m.csv"
  cp target/benchmark/sift1m.md "$run_dir/sift1m.md"
  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$label" "$m" "$efc" "1000000" "100" "1,5,10" "60,100,300" \
    >> "$manifest"
  echo "=== done $label ===" | tee -a "$sweep_root/sweep.log"
}

run_one 8 100
run_one 16 100
run_one 16 200
run_one 32 200
'@

$remoteScript = $remoteScript.Replace("__REMOTE_REPO__", $RemoteRepo).Replace("__SWEEP_NAME__", $SweepName)
$remoteScript = $remoteScript -replace "`r`n", "`n"

$localScriptPath = Join-Path $env:TEMP "$SweepName.sh"
[System.IO.File]::WriteAllText(
    $localScriptPath,
    $remoteScript,
    (New-Object System.Text.UTF8Encoding($false))
)

Get-Content $localScriptPath -Raw | ssh E5_server "cat > /tmp/$SweepName.sh"
Remove-Item $localScriptPath

ssh E5_server "chmod +x /tmp/$SweepName.sh && nohup /tmp/$SweepName.sh > '$launcherLogRemote' 2>&1 & echo `$!"
