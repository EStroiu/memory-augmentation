param(
    [string]$PythonBin = "",
    [string]$DataDir = "",
    [string]$OutDir = "",
    [int]$Runs = 2000,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not $PythonBin) {
    $PythonBin = Join-Path $rootDir ".venv/Scripts/python.exe"
}
if (-not (Test-Path $PythonBin)) {
    throw "Python interpreter not found: $PythonBin"
}

if (-not $DataDir) {
    $candidateA = Join-Path $rootDir "dataset"
    $candidateB = Join-Path $rootDir "avalon-nlu/dataset"
    if (Test-Path $candidateA) {
        $DataDir = $candidateA
    }
    elseif (Test-Path $candidateB) {
        $DataDir = $candidateB
    }
    else {
        throw "Could not find dataset dir. Checked: '$candidateA' and '$candidateB'"
    }
}

if (-not $OutDir) {
    $OutDir = Join-Path $rootDir "outputs/analysis"
}
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

$twoServOut = Join-Path $OutDir "random_baseline_two_servant.json"
$oneServOut = Join-Path $OutDir "random_baseline_one_servant.json"

Write-Host "Running random baseline (two-servant)..."
& $PythonBin "$rootDir/scripts/random_baseline.py" `
    --data_dir "$DataDir" `
    --runs "$Runs" `
    --seed "$Seed" `
    --out "$twoServOut"

Write-Host "Running random baseline (one-servant)..."
& $PythonBin "$rootDir/scripts/random_baseline.py" `
    --data_dir "$DataDir" `
    --runs "$Runs" `
    --seed "$Seed" `
    --one-servant `
    --out "$oneServOut"

Write-Host "Done."
Write-Host "Two-servant: $twoServOut"
Write-Host "One-servant: $oneServOut"
