param(
    [string]$PythonBin = "",
    [string]$DataDir = "",
    [string]$OutDir = "",
    [string]$LlmModel = "ollama:llama2:13b",
    [int]$NumRuns = 10,
    [int]$Seed = 42,
    [switch]$NoAvalonMentioned,
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

if (-not $OutDir) {
    $OutDir = Join-Path $rootDir "outputs/eval_grouped_role_task_10runs"
}
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

function Run-Eval {
    param(
        [string]$Label,
        [string]$BaselinePrompt = "none",
        [string]$CurrentPrompt = "none",
        [string]$FixerBaseline = "",
        [string]$FixerCurrent = ""
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running: $Label"
    Write-Host "============================================================"

    $args = @(
        "$rootDir/scripts/evaluate_pipeline.py",
        "--exp", "custom",
        "--data_dir", "$DataDir",
        "--memory_format", "template",
        "--outdir", "$OutDir",
        "--llm",
        "--llm_model", "$LlmModel",
        "--num_runs", "$NumRuns",
        "--seed", "$Seed",
        "--grouped-role-task",
        "--baseline_prompt", "$BaselinePrompt",
        "--current_prompt", "$CurrentPrompt"
    )

    if ($FixerBaseline) {
        $args += @("--llm_fixer_baseline", "$FixerBaseline")
    }

    if ($FixerCurrent) {
        $args += @("--llm_fixer_current", "$FixerCurrent")
    }
    if ($NoAvalonMentioned) {
        $args += "--no-avalon-mentioned"
    }
    if ($SaveLlmIo) {
        $args += "--save_llm_io"
    }

    & $PythonBin @args
}

# 1) Baseline only
Run-Eval -Label "baseline_full_transcript (baseline-only, grouped task)" `
    -BaselinePrompt "baseline_full_transcript" `
    -CurrentPrompt "none" `
    -FixerBaseline "typechat_role"

# 2) Vector memory (standalone, no baseline arm)
Run-Eval -Label "vector_memory (standalone, grouped task)" `
    -BaselinePrompt "none" `
    -CurrentPrompt "vector_memory" `
    -FixerCurrent "typechat_role"

# 3) Belief vector (standalone, no baseline arm)
Run-Eval -Label "belief_vector (standalone, grouped task)" `
    -BaselinePrompt "none" `
    -CurrentPrompt "belief_vector" `
    -FixerCurrent "typechat_beliefs"

# 4) Belief vector + social (standalone, no baseline arm)
Run-Eval -Label "belief_vector+social (standalone, grouped task)" `
    -BaselinePrompt "none" `
    -CurrentPrompt "belief_vector+social" `
    -FixerCurrent "typechat_beliefs"

# 5) Self note (standalone, no baseline arm)
Run-Eval -Label "llm_self_note (standalone, grouped task)" `
    -BaselinePrompt "none" `
    -CurrentPrompt "llm_self_note" `
    -FixerCurrent "typechat_role_note"

Write-Host ""
Write-Host "Grouped-role 10-run suite complete. Outputs saved under: $OutDir"
