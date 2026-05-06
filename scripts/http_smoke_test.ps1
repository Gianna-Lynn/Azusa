# Azusa HTTP demo commands.
#
# How to run:
# 1. Start the server in another terminal:
#      cargo run --bin azusa_server
# 2. Run this script from the repository root:
#      powershell -ExecutionPolicy Bypass -File .\scripts\http_smoke_test.ps1
#
# This script is intentionally simple: it is just the four curl.exe commands
# for health -> insert -> search -> get record, plus one 404 check.

$BaseUrl = "http://127.0.0.1:3000"

$InsertJson = @'
{"id":1,"vector":[1.0,0.0,0.0],"metadata":{"doc_id":"demo","chunk_id":"c1","source":"manual","page":1}}
'@

$SearchJson = @'
{"vector":[1.0,0.0,0.0],"k":1,"ef_search":16}
'@

$InsertJsonPath = Join-Path $env:TEMP "azusa_insert.json"
$SearchJsonPath = Join-Path $env:TEMP "azusa_search.json"
$InsertJson | Set-Content -Path $InsertJsonPath -NoNewline -Encoding ascii
$SearchJson | Set-Content -Path $SearchJsonPath -NoNewline -Encoding ascii

Write-Host ""
Write-Host "1. GET /health"
Write-Host "Expected: JSON with status=ok and current record_count."
curl.exe "$BaseUrl/health"
Write-Host ""

Write-Host ""
Write-Host "2. POST /insert"
Write-Host "Expected: HTTP 201 Created. Re-running with the same id returns 409 Conflict."
curl.exe -i -X POST "$BaseUrl/insert" `
  -H "Content-Type: application/json" `
  --data-binary "@$InsertJsonPath"
Write-Host ""

Write-Host ""
Write-Host "3. POST /search"
Write-Host "Expected: JSON with results containing id=1."
curl.exe -X POST "$BaseUrl/search" `
  -H "Content-Type: application/json" `
  --data-binary "@$SearchJsonPath"
Write-Host ""

Write-Host ""
Write-Host "4. GET /record/1"
Write-Host "Expected: JSON for record id=1, including metadata."
curl.exe "$BaseUrl/record/1"
Write-Host ""

Write-Host ""
Write-Host "5. GET /record/999"
Write-Host "Expected: HTTP 404 Not Found."
curl.exe -i "$BaseUrl/record/999"
Write-Host ""

Write-Host ""
Write-Host "Done."
