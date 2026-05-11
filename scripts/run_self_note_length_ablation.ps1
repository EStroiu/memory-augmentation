param(
    [string]$PythonBin = "",
    [string]$DataDir = "",
    [string]$OutRoot = "",
    [string]$LlmModel = "ollama:llama2:13b",
    [int]$NumRuns = 5,
    [int]$Seed = 42,
    [switch]$NoAvalonMentioned,
    [switch]$OneServant,
    [switch]$SaveLlmIo
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

if (-not $OutRoot) {
    $OutRoot = Join-Path $rootDir "outputs/eval_self_note_length"
}
if (-not (Test-Path $OutRoot)) {
    New-Item -ItemType Directory -Path $OutRoot -Force | Out-Null
}

$settings = @(
    @{ Selection = "newest"; K = "1" },
    @{ Selection = "newest"; K = "2" },
    @{ Selection = "newest"; K = "3" },
    @{ Selection = "newest"; K = "5" },
    @{ Selection = "oldest"; K = "1" },
    @{ Selection = "oldest"; K = "2" },
    @{ Selection = "oldest"; K = "3" },
    @{ Selection = "oldest"; K = "5" }
)

foreach ($s in $settings) {
    $selection = $s.Selection
    $k = $s.K
    $settingOut = Join-Path $OutRoot ("{0}_k-{1}" -f $selection, $k)
    if (-not (Test-Path $settingOut)) {
        New-Item -ItemType Directory -Path $settingOut -Force | Out-Null
    }

    Write-Host "Running self-note ablation: selection=$selection k=$k runs=$NumRuns model=$LlmModel"

    $cmdArgs = @(
        "$rootDir/scripts/evaluate_pipeline.py",
        "--data_dir", "$DataDir",
        "--outdir", "$settingOut",
        "--exp", "custom",
        "--baseline_prompt", "none",
        "--current_prompt", "llm_self_note",
        "--llm",
        "--llm_model", "$LlmModel",
        "--llm_fixer_current", "typechat_role_note",
        "--memory_format", "template",
        "--num_runs", "$NumRuns",
        "--seed", "$Seed",
        "--self_note_selection", "$selection",
        "--self_note_k", "$k"
    )

    if ($NoAvalonMentioned) {
        $cmdArgs += "--no-avalon-mentioned"
    }
    if ($OneServant) {
        $cmdArgs += "--one-servant"
    }
    if ($SaveLlmIo) {
        $cmdArgs += "--save_llm_io"
    }

    & $PythonBin @cmdArgs
}

Write-Host "Self-note length ablation complete."
Write-Host "Results root: $OutRoot"
