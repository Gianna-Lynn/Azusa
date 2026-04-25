param(
    [string]$InputCsv = "target/benchmark/server_sift1m.csv",
    [string]$OutputHtml = "target/benchmark/sift1m_viz.html"
)

$ErrorActionPreference = "Stop"
$culture = [System.Globalization.CultureInfo]::InvariantCulture

function Convert-ToMicroseconds {
    param([string]$Raw)

    if ([string]::IsNullOrWhiteSpace($Raw) -or $Raw -eq "-") {
        return $null
    }

    $value = $Raw.Trim()
    if ($value.EndsWith("us")) {
        return [double]::Parse($value.Substring(0, $value.Length - 2), $culture)
    }
    if ($value.EndsWith("µs")) {
        return [double]::Parse($value.Substring(0, $value.Length - 2), $culture)
    }
    if ($value.EndsWith("ms")) {
        return [double]::Parse($value.Substring(0, $value.Length - 2), $culture) * 1000.0
    }
    if ($value.EndsWith("s")) {
        return [double]::Parse($value.Substring(0, $value.Length - 1), $culture) * 1000000.0
    }

    return [double]::Parse($value, $culture)
}

function Format-Us {
    param([double]$Value)

    if ($Value -ge 1000000.0) {
        return ("{0:N2} s" -f ($Value / 1000000.0))
    }
    if ($Value -ge 1000.0) {
        return ("{0:N2} ms" -f ($Value / 1000.0))
    }
    return ("{0:N1} us" -f $Value)
}

function Html-Escape {
    param([string]$Text)

    return [System.Net.WebUtility]::HtmlEncode($Text)
}

function Map-X {
    param(
        [double]$Value,
        [double]$MinLog,
        [double]$MaxLog,
        [double]$Left,
        [double]$Width
    )

    $logValue = [Math]::Log10($Value)
    if ($MaxLog -le $MinLog) {
        return $Left + $Width / 2.0
    }
    return $Left + (($logValue - $MinLog) / ($MaxLog - $MinLog)) * $Width
}

function Map-Y {
    param(
        [double]$Value,
        [double]$Min,
        [double]$Max,
        [double]$Top,
        [double]$Height
    )

    if ($Max -le $Min) {
        return $Top + $Height / 2.0
    }
    return $Top + ($Max - $Value) / ($Max - $Min) * $Height
}

$rows = Import-Csv $InputCsv | ForEach-Object {
    [pscustomobject]@{
        algorithm      = $_.algorithm
        base_limit     = [int]$_.base_limit
        query_limit    = [int]$_.query_limit
        k              = [int]$_.k
        ef_search      = if ($_.ef_search -eq "-" -or [string]::IsNullOrWhiteSpace($_.ef_search)) { $null } else { [int]$_.ef_search }
        avg_search_us  = Convert-ToMicroseconds $_.avg_search_time
        total_search_us = Convert-ToMicroseconds $_.total_search_time
        recall         = [double]$_.recall
        exact_count    = [int]$_.exact_count
        build_us       = Convert-ToMicroseconds $_.build_time
    }
}

$baseLimits = $rows.base_limit | Sort-Object -Unique
$colors = @{
    1  = "#2563eb"
    5  = "#f97316"
    10 = "#16a34a"
}

function Build-TradeoffChart {
    param(
        [int]$BaseLimit,
        [object[]]$PanelRows
    )

    $hnswRows = $PanelRows | Where-Object { $_.algorithm -eq "hnsw" } | Sort-Object k, ef_search
    $flatRows = $PanelRows | Where-Object { $_.algorithm -eq "flat" }
    $ks = ($PanelRows | Select-Object -ExpandProperty k -Unique | Sort-Object)

    $points = New-Object System.Collections.Generic.List[object]
    foreach ($row in $PanelRows) {
        if ($null -ne $row.avg_search_us) {
            $points.Add($row)
        }
    }

    $xValues = @($points | ForEach-Object { $_.avg_search_us })
    $yValues = @($points | ForEach-Object { $_.recall })

    $minX = [Math]::Max((($xValues | Measure-Object -Minimum).Minimum), 1.0)
    $maxX = ($xValues | Measure-Object -Maximum).Maximum
    $minY = ($yValues | Measure-Object -Minimum).Minimum
    $maxY = ($yValues | Measure-Object -Maximum).Maximum

    if ($minY -gt 0.05) {
        $minY = [Math]::Max(0.0, $minY - 0.03)
    }
    $maxY = [Math]::Min(1.0, $maxY + 0.01)

    $left = 90.0
    $top = 35.0
    $plotWidth = 720.0
    $plotHeight = 300.0
    $right = $left + $plotWidth
    $bottom = $top + $plotHeight

    $minLog = [Math]::Log10($minX)
    $maxLog = [Math]::Log10($maxX)

    $svg = New-Object System.Text.StringBuilder
    [void]$svg.AppendLine("<section class=""panel"">")
    [void]$svg.AppendLine("  <h2>Base limit $BaseLimit</h2>")
    [void]$svg.AppendLine("  <svg viewBox=""0 0 920 430"" role=""img"" aria-label=""Recall latency tradeoff chart for base limit $BaseLimit"">")
    [void]$svg.AppendLine("    <rect x=""0"" y=""0"" width=""920"" height=""430"" rx=""18"" class=""panel-bg"" />")
    [void]$svg.AppendLine("    <text x=""24"" y=""28"" class=""svg-title"">Recall vs latency tradeoff</text>")
    [void]$svg.AppendLine("    <text x=""24"" y=""50"" class=""svg-subtitle"">x = average search time (log scale, us), y = recall@k</text>")

    for ($i = 0; $i -le 5; $i++) {
        $yValue = $minY + (($maxY - $minY) * $i / 5.0)
        $y = Map-Y -Value $yValue -Min $minY -Max $maxY -Top $top -Height $plotHeight
        [void]$svg.AppendLine(("    <line x1=""{0:F1}"" y1=""{1:F1}"" x2=""{2:F1}"" y2=""{1:F1}"" class=""grid"" />" -f $left, $y, $right))
        [void]$svg.AppendLine(("    <text x=""{0:F1}"" y=""{1:F1}"" class=""axis-label y-label"">{2:P0}</text>" -f 66.0, ($y + 4.0), $yValue))
    }

    for ($i = 0; $i -le 5; $i++) {
        $logValue = $minLog + (($maxLog - $minLog) * $i / 5.0)
        $xValue = [Math]::Pow(10.0, $logValue)
        $x = Map-X -Value $xValue -MinLog $minLog -MaxLog $maxLog -Left $left -Width $plotWidth
        [void]$svg.AppendLine(("    <line x1=""{0:F1}"" y1=""{1:F1}"" x2=""{0:F1}"" y2=""{2:F1}"" class=""grid"" />" -f $x, $top, $bottom))
        [void]$svg.AppendLine(("    <text x=""{0:F1}"" y=""{1:F1}"" class=""axis-label x-label"">{2}</text>" -f ($x - 12.0), ($bottom + 24.0), (Format-Us $xValue)))
    }

    [void]$svg.AppendLine(("    <line x1=""{0:F1}"" y1=""{1:F1}"" x2=""{0:F1}"" y2=""{2:F1}"" class=""axis"" />" -f $left, $top, $bottom))
    [void]$svg.AppendLine(("    <line x1=""{0:F1}"" y1=""{1:F1}"" x2=""{2:F1}"" y2=""{1:F1}"" class=""axis"" />" -f $left, $bottom, $right))

    $legendX = 760.0
    $legendY = 56.0
    [void]$svg.AppendLine(("    <rect x=""{0}"" y=""{1}"" width=""132"" height=""110"" rx=""12"" class=""legend-bg"" />" -f $legendX, $legendY))
        [void]$svg.AppendLine(("    <text x=""{0}"" y=""{1}"" class=""legend-title"">Legend</text>" -f ($legendX + 14), ($legendY + 22)))

    $legendItems = @(
        @{ label = "Flat baseline"; color = "#111827"; style = "flat" }
    )
    foreach ($k in $ks) {
        $legendItems += @{ label = "HNSW k=$k"; color = $colors[$k]; style = "line" }
    }

    $legendIndex = 0
    foreach ($item in $legendItems) {
        $yy = $legendY + 42 + ($legendIndex * 20)
        if ($item.style -eq "flat") {
            [void]$svg.AppendLine(("    <rect x=""{0}"" y=""{1}"" width=""11"" height=""11"" transform=""rotate(45 {0} {1})"" fill=""{2}"" stroke=""{2}"" />" -f ($legendX + 16), ($yy - 3), $item.color))
        } else {
            [void]$svg.AppendLine(("    <line x1=""{0}"" y1=""{1}"" x2=""{2}"" y2=""{1}"" stroke=""{3}"" stroke-width=""3"" />" -f ($legendX + 10), ($yy + 2), ($legendX + 28), $item.color))
            [void]$svg.AppendLine(("    <circle cx=""{0}"" cy=""{1}"" r=""4"" fill=""white"" stroke=""{2}"" stroke-width=""2"" />" -f ($legendX + 19), ($yy + 2), $item.color))
        }
            [void]$svg.AppendLine(("    <text x=""{0}"" y=""{1}"" class=""legend-item"">{2}</text>" -f ($legendX + 36), ($yy + 6), (Html-Escape $item.label)))
        $legendIndex++
    }

    foreach ($k in $ks) {
        $series = $hnswRows | Where-Object { $_.k -eq $k } | Sort-Object ef_search
        if ($series.Count -eq 0) { continue }

        $path = New-Object System.Text.StringBuilder
        $first = $true
        foreach ($row in $series) {
            $x = Map-X -Value $row.avg_search_us -MinLog $minLog -MaxLog $maxLog -Left $left -Width $plotWidth
            $y = Map-Y -Value $row.recall -Min $minY -Max $maxY -Top $top -Height $plotHeight
            if ($first) {
                [void]$path.Append(("M {0:F1},{1:F1}" -f $x, $y))
                $first = $false
            } else {
                [void]$path.Append((" L {0:F1},{1:F1}" -f $x, $y))
            }
        }

        $stroke = $colors[$k]
        [void]$svg.AppendLine(("    <path d=""{0}"" fill=""none"" stroke=""{1}"" stroke-width=""3"" />" -f $path.ToString(), $stroke))

        foreach ($row in $series) {
            $x = Map-X -Value $row.avg_search_us -MinLog $minLog -MaxLog $maxLog -Left $left -Width $plotWidth
            $y = Map-Y -Value $row.recall -Min $minY -Max $maxY -Top $top -Height $plotHeight
            $tooltip = "base=$BaseLimit, k=$($row.k), ef=$($row.ef_search), recall=$([math]::Round($row.recall * 100, 2))%, avg=$((Format-Us $row.avg_search_us))"
            [void]$svg.AppendLine(("    <circle cx=""{0:F1}"" cy=""{1:F1}"" r=""5"" fill=""white"" stroke=""{2}"" stroke-width=""3""><title>{3}</title></circle>" -f $x, $y, $stroke, (Html-Escape $tooltip)))
        }
    }

    foreach ($row in $flatRows) {
        $x = Map-X -Value $row.avg_search_us -MinLog $minLog -MaxLog $maxLog -Left $left -Width $plotWidth
        $y = Map-Y -Value $row.recall -Min $minY -Max $maxY -Top $top -Height $plotHeight
        $tooltip = "base=$BaseLimit, flat, k=$($row.k), recall=$([math]::Round($row.recall * 100, 2))%, avg=$((Format-Us $row.avg_search_us))"
        [void]$svg.AppendLine(("    <rect x=""{0:F1}"" y=""{1:F1}"" width=""10"" height=""10"" transform=""rotate(45 {2:F1} {3:F1})"" fill=""#111827"" stroke=""white"" stroke-width=""1.5""><title>{4}</title></rect>" -f ($x - 5.0), ($y - 5.0), $x, $y, (Html-Escape $tooltip)))
    }

    [void]$svg.AppendLine("  </svg>")

    $summaryRows = $ks | ForEach-Object {
        $k = $_
        $flat = $flatRows | Where-Object { $_.k -eq $k } | Select-Object -First 1
        $hnsw60 = $hnswRows | Where-Object { $_.k -eq $k -and $_.ef_search -eq 60 } | Select-Object -First 1
        $hnsw100 = $hnswRows | Where-Object { $_.k -eq $k -and $_.ef_search -eq 100 } | Select-Object -First 1
        $hnsw300 = $hnswRows | Where-Object { $_.k -eq $k -and $_.ef_search -eq 300 } | Select-Object -First 1

        [pscustomobject]@{
            k = $k
            flat_time = if ($flat) { Format-Us $flat.avg_search_us } else { "-" }
            flat_recall = if ($flat) { "{0:P1}" -f $flat.recall } else { "-" }
            hnsw60_time = if ($hnsw60) { Format-Us $hnsw60.avg_search_us } else { "-" }
            hnsw60_recall = if ($hnsw60) { "{0:P1}" -f $hnsw60.recall } else { "-" }
            hnsw100_time = if ($hnsw100) { Format-Us $hnsw100.avg_search_us } else { "-" }
            hnsw100_recall = if ($hnsw100) { "{0:P1}" -f $hnsw100.recall } else { "-" }
            hnsw300_time = if ($hnsw300) { Format-Us $hnsw300.avg_search_us } else { "-" }
            hnsw300_recall = if ($hnsw300) { "{0:P1}" -f $hnsw300.recall } else { "-" }
        }
    }

    $table = New-Object System.Text.StringBuilder
    [void]$table.AppendLine("  <table class=""summary"">")
    [void]$table.AppendLine("    <thead><tr><th>k</th><th>flat time</th><th>flat recall</th><th>HNSW 60 time</th><th>HNSW 60 recall</th><th>HNSW 100 time</th><th>HNSW 100 recall</th><th>HNSW 300 time</th><th>HNSW 300 recall</th></tr></thead>")
    [void]$table.AppendLine("    <tbody>")
    foreach ($row in $summaryRows) {
        [void]$table.AppendLine(("      <tr><td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td><td>{4}</td><td>{5}</td><td>{6}</td><td>{7}</td><td>{8}</td></tr>" -f $row.k, $row.flat_time, $row.flat_recall, $row.hnsw60_time, $row.hnsw60_recall, $row.hnsw100_time, $row.hnsw100_recall, $row.hnsw300_time, $row.hnsw300_recall))
    }
    [void]$table.AppendLine("    </tbody>")
    [void]$table.AppendLine("  </table>")

    return $svg.ToString() + $table.ToString() + "</section>"
}

$panels = foreach ($baseLimit in $baseLimits) {
    Build-TradeoffChart -BaseLimit $baseLimit -PanelRows ($rows | Where-Object { $_.base_limit -eq $baseLimit })
}

$generatedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$html = @"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SIFT1M benchmark visualization</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #11182e;
      --panel2: #161f3b;
      --text: #e5eefc;
      --muted: #9fb0d0;
      --grid: rgba(159, 176, 208, 0.18);
      --axis: rgba(229, 238, 252, 0.65);
      --border: rgba(255, 255, 255, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #182348 0%, #0b1020 45%, #070b16 100%);
      color: var(--text);
      padding: 28px;
    }
    .wrap { max-width: 1300px; margin: 0 auto; }
    h1 { margin: 0 0 8px 0; font-size: 2rem; }
    .sub { color: var(--muted); margin-bottom: 20px; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 22px;
    }
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px 18px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    }
    .card .label { color: var(--muted); font-size: 0.88rem; }
    .card .value { font-size: 1.5rem; font-weight: 700; margin-top: 6px; }
    .grid { display: grid; gap: 18px; }
    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 0 0 16px 0;
      overflow: hidden;
      box-shadow: 0 18px 50px rgba(0,0,0,0.28);
    }
    .panel h2 { margin: 0; padding: 18px 22px 0; font-size: 1.2rem; }
    svg { width: 100%; height: auto; display: block; }
    .panel-bg { fill: rgba(10, 15, 30, 0.85); stroke: rgba(255,255,255,0.08); }
    .grid { stroke: var(--grid); stroke-width: 1; }
    .axis { stroke: var(--axis); stroke-width: 1.5; }
    .svg-title { fill: var(--text); font-size: 18px; font-weight: 700; }
    .svg-subtitle { fill: var(--muted); font-size: 12px; }
    .axis-label { fill: var(--muted); font-size: 11px; }
    .legend-bg { fill: rgba(16, 23, 44, 0.92); stroke: rgba(255,255,255,0.10); }
    .legend-title { fill: var(--text); font-size: 12px; font-weight: 700; }
    .legend-item { fill: var(--muted); font-size: 12px; }
    .summary {
      width: calc(100% - 32px);
      margin: 0 16px 10px;
      border-collapse: collapse;
      background: rgba(10, 15, 30, 0.65);
      overflow: hidden;
      border-radius: 14px;
    }
    .summary th, .summary td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      text-align: right;
      white-space: nowrap;
    }
    .summary th:first-child, .summary td:first-child { text-align: center; }
    .summary th {
      color: var(--text);
      font-size: 0.86rem;
      background: rgba(255,255,255,0.04);
    }
    .summary td { color: var(--muted); font-size: 0.92rem; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>SIFT1M benchmark visualization</h1>
    <div class="sub">Generated at $generatedAt from <code>$InputCsv</code></div>
    <div class="cards">
      <div class="card"><div class="label">Base limits</div><div class="value">$($baseLimits -join ", ")</div></div>
      <div class="card"><div class="label">Query limit</div><div class="value">$($rows[0].query_limit)</div></div>
      <div class="card"><div class="label">Algorithms</div><div class="value">flat + hnsw</div></div>
      <div class="card"><div class="label">HNSW ef_search</div><div class="value">60 / 100 / 300</div></div>
    </div>
    <div class="grid">
$($panels -join "`n")
    </div>
  </div>
</body>
</html>
"@

$outDir = Split-Path -Parent $OutputHtml
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

Set-Content -Path $OutputHtml -Value $html -Encoding UTF8
Write-Host "wrote $OutputHtml"
