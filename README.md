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
It computes Recall@k and MRR, and summarizes prompt sizes.

Common invocations (run from repo root):

```bash
# Baseline vs current (nearest neighbor + template by default)
python scripts/evaluate_pipeline.py --data_dir avalon-nlu/dataset --k 3 --policy nn --memory_format template

# Try temporally-weighted re-ranking
python scripts/evaluate_pipeline.py --data_dir avalon-nlu/dataset --k 3 --policy temporal --alpha 0.5

# Use template + heuristic summary as memory format
python scripts/evaluate_pipeline.py --data_dir avalon-nlu/dataset --k 3 --memory_format template+summary

# Limit to a few games for a quick smoke test
python scripts/evaluate_pipeline.py --data_dir avalon-nlu/dataset --k 3 --max_games 5
```

Outputs are saved under `outputs/eval/<timestamp>_k-.../` and include:
- `config.json`: captured CLI args
- `results_baseline.json` / `aggregate_baseline.json`
- `results_current.json` / `aggregate_current.json`
- `aggregate_combined.json`: baseline vs current summary
- Visuals (if Plotly available): `recall_bar.html`, `mrr_bar.html`, `prompt_lengths.html`


## WIP: call an LLM (stubbed by default)

Currenlty the evaluation script includes a placeholder for an OpenAI call for downstream role prediction.
By default it does nothing. To enable the stub path, you can pass `--llm` and supply an API key;
however, the call is still a no‑op for safety. Maybe later it will also be used for local models.

```bash
export OPENAI_API_KEY=...   # or pass via --openai-api-key
python scripts/evaluate_pipeline.py --data_dir dataset --llm --openai-model gpt-4o-mini
```