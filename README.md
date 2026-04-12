# 🔯 DEEP DECISION OPTIMIZATION 📈

Tiantian(Crystal) ZHANG @ Columbia University Undegrad 
contact: t.zhang8@columbia.edu (paper idea dicsuiion) 

& Jierui(Jerry) Zuo@ UW incoming Phd,Tsinghua undegrad 
contact: zuojr22@gmail.com (repository reprodction issue discussion

🔥This is the repo implementation first prosposed at https://arxiv.org/abs/2509.18138 (short for RIPLM rank induced Plucket Luce Mirror deceny), and the full paper about DDO generalizing RIPLM is soon coming out. Here is a short tutorial about these two meyhods.


# RIPLM vs General DFL Benchmarks

This repository contains a reproducible benchmark project comparing RIPLM against standard decision-focused learning (DFL) baselines on:

- a ranking-control task close to RIPLM's intended setting; and
- exact small-scale variants of canonical DFL benchmark families (`ShortestPath`, `Matching`, and `Knapsack`).

The repository keeps the generated benchmark artifacts that support the paper-style report, so it is ready both for inspection and for rerunning the experiments locally.

## Repository contents

- `main.tex`: LaTeX report.
- `scripts/run_benchmark_comparison.py`: end-to-end benchmark pipeline.
- `data/`: committed experiment outputs, summaries, and hyperparameter selections.
- `figures/`: committed figures used by the report.
- `tables/`: generated LaTeX tables included by `main.tex`.
- `requirements.txt`: Python dependencies for the benchmark script.
- `Makefile`: convenience targets for rerunning experiments and compiling the report.

## Reproducing the benchmark

### 1. Create a Python environment

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Run the benchmark pipeline

```bash
python scripts/run_benchmark_comparison.py
```

This regenerates:

- CSV outputs in `data/`
- figure assets in `figures/`
- LaTeX tables in `tables/`

The committed results already reflect a completed run. Re-running the script should only be done if you intentionally want to regenerate artifacts.

## Compiling the LaTeX report

### Local compilation

Install a LaTeX distribution that provides `pdflatex` first:

- Windows: MiKTeX or TeX Live
- macOS: MacTeX
- Linux: TeX Live

Then compile with either:

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

or:

```bash
make tex
```

The report expects the committed `figures/` and `tables/` directories to be present, which they are in this repository.

### Overleaf

Upload the full repository contents and compile `main.tex` with pdfLaTeX.

## Repository layout

```text
.
|-- data/
|-- figures/
|-- experiments/
|   `-- riplm_dfl_benchmark/
|-- scripts/
|-- tables/
|-- .gitignore
|-- Makefile
|-- README.md
|-- main.tex
`-- requirements.txt
```

## Notes

- The structured tasks are exact small-scale variants of benchmark families from the general DFL benchmarking literature.
- They are intentionally small enough that feasible decision sets can be enumerated exactly.
- That makes the RIPLM adaptation precise in this project, but it is not a claim of full-scale benchmark parity.

## Additional experiment

An additional collaborator-style shortest-path benchmark is available under `experiments/riplm_dfl_benchmark/`.

- What it is: a small enumerated-path RIPLM/DFL comparison on a layered shortest-path task.
- Main script: `experiments/riplm_dfl_benchmark/scripts/run_benchmark_comparison.py`
- Run it:

```bash
cd experiments/riplm_dfl_benchmark
python scripts/run_benchmark_comparison.py --out_dir .
```
