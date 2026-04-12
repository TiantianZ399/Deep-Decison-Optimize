# 🔯 DEEP DECISION OPTIMIZATION in LLM📈

Tiantian(Crystal) ZHANG @ Columbia University Undegrad 
contact: t.zhang8@columbia.edu (mailing author) 

& Jierui(Jerry) Zuo@ UW incoming Phd,Tsinghua undegrad 
contact: zuojr22@gmail.com 

Happy to contact and discuss ideas through email.

🔥This is the repo implementation first prosposed at https://arxiv.org/abs/2509.18138 (short for RIPLM rank induced Plucket Luce Mirror descent).

# RIPLM vs General DFL Benchmarks

This repository contains a reproducible benchmark project comparing RIPLM against standard decision-focused learning (DFL) baselines on:

- a ranking-control task close to RIPLM's intended setting; and
- exact small-scale variants of canonical DFL benchmark families (`ShortestPath`, `Matching`, and `Knapsack`).

The repository keeps the generated benchmark artifacts that support the paper-style report, so it is ready both for inspection and for rerunning the experiments locally.

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

RIPLM / DDO-MD shortest-path benchmark

This repository contains a small but fully reproducible decision-focused learning benchmark that mirrors the enumerate-all-feasible-paths adaptation discussed in the conversation.

What is in the repo?
	•	scripts/run_benchmark_comparison.py: main reproduction script
	•	data/: graph specification and task configuration
	•	figures/: generated plots
	•	tables/: raw CSVs and LaTeX tables
	•	summary_zh.md: Chinese experiment summary
	•	main.tex: short LaTeX report that includes the generated tables/figures

Methods compared
	•	mse: edge-cost regression baseline
	•	spo: heuristic straight-through SPO baseline
	•	spo+: exact SPO+ surrogate over the enumerated path set
	•	ddo-md: exact path-simplex mirror descent / softmax-over-paths surrogate

Reproduce

From the experiment directory (experiments/riplm_dfl_benchmark/ inside the main repository):

python scripts/run_benchmark_comparison.py --out_dir .

Or, from the top-level repository root:

python experiments/riplm_dfl_benchmark/scripts/run_benchmark_comparison.py --out_dir experiments/riplm_dfl_benchmark

This will regenerate all CSV tables, TeX tables, figures, and summary_zh.md in this experiment folder.

Notes
	•	The graph is intentionally small enough that all feasible s-t paths can be enumerated exactly.
	•	This makes the comparison close in spirit to the collaborator-style benchmark: the methods are compared on the same explicit path simplex.
	•	spo is included only as a heuristic baseline; it uses a regret-scaled straight-through direction rather than a literal gradient of the discontinuous SPO loss.

DDO-MD vs. SPO / SPO+ / MSE: Summary of the Enumerated-Path Benchmark

Experimental setup
	•	Task: Fully enumerate all feasible paths in a small layered shortest-path graph, and compare mse, spo, spo+, and ddo-md on the same path set.
	•	Fairness: All methods share the same graph, the same synthetic teacher, the same linear student, the same train/validation/test split rule, the same Adam optimizer, and the same hyperparameter budget.
	•	Graph size: 33 edges and 81 feasible paths; all paths have the same length.
	•	Data: train=96, val=128, test=512, feature_dim=16. The synthetic teacher contains both a linear term and a sinusoidal nonlinear term, while the linear student is intentionally misspecified.
	•	Model selection: Hyperparameters are selected on the validation set using standard path regret (equivalently, standard SPO loss). Two seeds are used for tuning, and final results are reported over five seeds.
	•	Training budget: 8 epochs per hyperparameter configuration during tuning; 12 epochs for the final five-seed summary runs.

Selected hyperparameters
	•	ddo-md: lr=0.1, tau=0.05
	•	mse: lr=0.03, tau=0.0
	•	spo: lr=0.001, tau=0.0
	•	spo+: lr=0.03, tau=0.0

Five-seed summary

Method	Standard SPO loss (= path regret)	Path accuracy	Edge overlap	Runtime (s)	Regret rank
ddo-md	0.5704 ± 0.0270	0.0742 ± 0.0154	0.3524 ± 0.0217	0.0291 ± 0.0015	1
mse	0.6255 ± 0.0322	0.0734 ± 0.0088	0.3266 ± 0.0122	0.0243 ± 0.0011	2
spo+	0.6393 ± 0.0289	0.0586 ± 0.0135	0.3181 ± 0.0119	0.0306 ± 0.0018	3
spo	1.0593 ± 0.0499	0.0090 ± 0.0033	0.1908 ± 0.0159	0.0290 ± 0.0009	4

Main takeaways
	•	The best mean standard SPO loss / path regret is achieved by ddo-md = 0.5704, outperforming the second-best method, mse = 0.6255.
	•	On this collaborator-style enumerated-path benchmark, ddo-md achieves lower mean path regret than spo (0.5704 vs. 1.0593), spo+ (0.5704 vs. 0.6393), and mse (0.5704 vs. 0.6255).
	•	In terms of path accuracy, ddo-md reaches 0.0742, compared with 0.0734 for mse. In this experiment, the regret improvement is more pronounced than the exact path-match improvement.
	•	In terms of edge overlap, ddo-md reaches 0.3524, suggesting that even when exact path accuracy is not dramatically separated, it still more consistently places more correct edges into the final predicted path.
	•	The direct spo row should be interpreted only as a heuristic baseline: it is not an exact gradient method, but rather a regret-scaled straight-through direction.

File note

This benchmark is intended as a fully reproducible, explicit-path comparison where all methods are evaluated on exactly the same feasible path set.

