param(
    [string]$PythonBin = "",
    [string]$DataDir = "",
    [string]$OutDir = "",
    [string]$LlmModel = "ollama:llama2:13b",
    [int]$NumRuns = 3
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
    $OutDir = Join-Path $rootDir "outputs/eval_with_avalon_one_serv"
}

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

function Run-Eval {
    param(
        [string]$Label,
        [string]$BaselinePrompt,
        [string]$CurrentPrompt,
        [string]$FixerBaseline,
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
        "--one-servant",
        "--baseline_prompt", "$BaselinePrompt",
        "--current_prompt", "$CurrentPrompt",
        "--llm_fixer_baseline", "$FixerBaseline"
    )

    if ($FixerCurrent) {
        $args += @("--llm_fixer_current", "$FixerCurrent")
    }

    & $PythonBin @args
}

# 1) Baseline only
Run-Eval -Label "baseline_full_transcript (baseline-only)" `
    -BaselinePrompt "baseline_full_transcript" `
    -CurrentPrompt "none" `
    -FixerBaseline "typechat_role"

# 2) Memory vector
Run-Eval -Label "baseline_full_transcript vs vector_memory" `
    -BaselinePrompt "baseline_full_transcript" `
    -CurrentPrompt "vector_memory" `
    -FixerBaseline "typechat_role" `
    -FixerCurrent "typechat_role"

# 3) Belief vector
Run-Eval -Label "baseline_full_transcript vs belief_vector" `
    -BaselinePrompt "baseline_full_transcript" `
    -CurrentPrompt "belief_vector" `
    -FixerBaseline "typechat_role" `
    -FixerCurrent "typechat_beliefs"

# 4) Belief vector + social
Run-Eval -Label "baseline_full_transcript vs belief_vector+social" `
    -BaselinePrompt "baseline_full_transcript" `
    -CurrentPrompt "belief_vector+social" `
    -FixerBaseline "typechat_role" `
    -FixerCurrent "typechat_beliefs"

# 5) Self note
Run-Eval -Label "baseline_full_transcript vs llm_self_note" `
    -BaselinePrompt "baseline_full_transcript" `
    -CurrentPrompt "llm_self_note" `
    -FixerBaseline "typechat_role" `
    -FixerCurrent "typechat_role_note"

Write-Host ""
Write-Host "Suite complete. Outputs saved under: $OutDir"
