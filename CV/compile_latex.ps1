# Compile a LaTeX file to PDF with bundled Tectonic.
#
# Usage from this CV directory:
#   .\compile_latex.ps1 .\CV_v2.1.tex
#
# If PowerShell blocks script execution, use:
#   powershell -ExecutionPolicy Bypass -File .\compile_latex.ps1 -TexPath .\CV_v2.1.tex
#
# Optional:
#   .\compile_latex.ps1 .\CV_v2.1.tex -OutDir .\build

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$TexPath,

    [Parameter(Mandatory = $false)]
    [string]$OutDir
)

$ErrorActionPreference = "Stop"

function Resolve-ExistingFile {
    param([string]$Path)

    $resolved = Resolve-Path -LiteralPath $Path -ErrorAction Stop
    if ($resolved.Count -ne 1) {
        throw "Expected exactly one input file, got $($resolved.Count): $Path"
    }
    return $resolved.ProviderPath
}

function Find-Tectonic {
    $latexPluginRoot = Join-Path $env:USERPROFILE ".codex\plugins\cache\openai-bundled\latex"
    if (Test-Path -LiteralPath $latexPluginRoot) {
        $bundled = Get-ChildItem -LiteralPath $latexPluginRoot -Directory |
            ForEach-Object { Join-Path $_.FullName "bin\tectonic.exe" } |
            Where-Object { Test-Path -LiteralPath $_ } |
            Sort-Object -Descending |
            Select-Object -First 1

        if ($null -ne $bundled) {
            return $bundled
        }
    }

    $fromPath = Get-Command tectonic.exe -ErrorAction SilentlyContinue
    if ($null -ne $fromPath) {
        return $fromPath.Source
    }

    throw "tectonic.exe not found. Install/use the Codex LaTeX plugin or add Tectonic to PATH."
}

$inputFile = Resolve-ExistingFile $TexPath
if ([System.IO.Path]::GetExtension($inputFile).ToLowerInvariant() -ne ".tex") {
    throw "Input must be a .tex file: $inputFile"
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $outputDir = Split-Path -Parent $inputFile
} else {
    $outputDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutDir)
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

$tectonic = Find-Tectonic
$pdfPath = Join-Path $outputDir ([System.IO.Path]::GetFileNameWithoutExtension($inputFile) + ".pdf")

Write-Host "Tectonic: $tectonic"
Write-Host "Input:    $inputFile"
Write-Host "OutDir:   $outputDir"
Write-Host ""

$env:TECTONIC_UNTRUSTED_MODE = "1"

& $tectonic -X compile `
    --outdir $outputDir `
    --outfmt pdf `
    --print `
    --keep-logs `
    --untrusted `
    $inputFile

$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host ""
    Write-Host "LaTeX compile failed with exit code $exitCode." -ForegroundColor Red
    Write-Host "If Tectonic reports a missing .sty file, allow network access once so it can populate its bundle cache."
    exit $exitCode
}

if (-not (Test-Path -LiteralPath $pdfPath)) {
    throw "Compile command succeeded, but PDF was not found: $pdfPath"
}

Write-Host ""
Write-Host "PDF written:" -ForegroundColor Green
Write-Host $pdfPath
