# Memory Augmentation for Avalon NLU
This project explores simple memory-augmentation techniques over the Avalon NLU dataset.
It builds sentence-embedding indexes of game rounds (quests), retrieves similar rounds as
context, and assembles prompts you can feed to an LLM. It also includes an evaluation
pipeline that compares retrieval policies and memory formats.

## Prerequisites

- Python 3.9 or newer is recommended.

## Creating a virtual environment

```bash
# From the repository root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data layout

- Default data lives under `avalon-nlu/dataset/` with multiple `*.json` games.
- Some scripts also look for `avalon-nlu/streamlit/sample.json` as a tiny example. If it's missing, the scripts will fall back to the first file in `dataset/`.


## Evaluate the pipeline over the whole dataset

Use `scripts/evaluate_pipeline.py` to run retrieval and prompting over all `dataset/*.json` games.
It builds a sentence-embedding index, retrieves top-k similar rounds as memory, and compares
two prompting strategies: a baseline (no retrieved memory in the prompt) vs a memory-augmented
prompt. The script reports per-role precision/recall/F1, micro-F1, confusion matrices, and
summarizes prompt sizes.

Common invocations (run from repo root):

```bash
# Baseline vs current prompts using an LLM via Ollama
python scripts/evaluate_pipeline.py \
	--data_dir avalon-nlu/dataset \
	--k 3 \
	--memory_format template \
	--outdir outputs/eval \
	--llm --llm_model ollama:llama2:13b

# Try a different memory format (template + heuristic summary)
python scripts/evaluate_pipeline.py \
	--data_dir avalon-nlu/dataset \
	--k 3 \
	--memory_format template+summary \
	--outdir outputs/eval \
	--llm --llm_model ollama:llama2:13b

# Limit to a few games for a quick smoke test
python scripts/evaluate_pipeline.py \
	--data_dir avalon-nlu/dataset \
	--k 3 \
	--memory_format template \
	--max_games 5 \
	--outdir outputs/eval \
	--llm --llm_model ollama:llama2:13b

# Run without an LLM (uses retrieval-only top-1 proposer role as a simple baseline)
python scripts/evaluate_pipeline.py \
	--data_dir avalon-nlu/dataset \
	--k 3 \
	--memory_format template \
	--outdir outputs/eval
```

Outputs are saved under `outputs/eval/<timestamp>_k-..._mem-*_llm/` and include:
- `config.json`: captured CLI args
- Per-run JSONs: `runs/run_*/results_baseline.json`, `results_current.json`
- Per-run aggregates: `runs/run_*/aggregate_baseline.json`, `aggregate_current.json`
- Run-mean aggregates: `aggregate_baseline_mean.json`, `aggregate_current_mean.json`,
	`aggregate_combined_mean.json`
- F1 summary tables: `role_f1_table.csv` and `role_f1_table.txt` (and per-run F1s if `--num_runs > 1`)
- Visuals (if Plotly available): `prompt_lengths.html`, `confusion_baseline.html`, `confusion_current.html`


## LLM evaluation with memory augmentation (TypeChat-like JSON)

The evaluation script can compare LLM performance with vs without memory augmentation.
Prompts enforce a TypeChat-like contract: the model must return a one-line JSON object
`{ "role": "<ROLE>" }`. The returned role is validated against the dataset’s allowed
roles, and the script computes per-role metrics and micro-F1.

Local (Ollama) example:

```bash
# Ensure Ollama is installed and running, and you have the model pulled, e.g.:
#   ollama pull ollama:llama2:13b
python scripts/evaluate_pipeline.py \
	--data_dir avalon-nlu/dataset \
	--k 3 \
	--memory_format template \
	--outdir outputs/eval \
	--llm --llm_model ollama:llama2:13b
```

OpenAI example (stubbed path):

```bash
export OPENAI_API_KEY=...   # or pass via --openai-api-key
python scripts/evaluate_pipeline.py \
	--data_dir avalon-nlu/dataset \
	--k 3 \
	--memory_format template \
	--outdir outputs/eval \
	--llm --llm_model openai:gpt-4o-mini
```