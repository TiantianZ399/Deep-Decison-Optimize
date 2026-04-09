# RIPLM / DDO-MD shortest-path benchmark

This repository contains a small but fully reproducible decision-focused learning benchmark that mirrors the *enumerate-all-feasible-paths* adaptation discussed in the conversation.

## What is in the repo?

- `scripts/run_benchmark_comparison.py`: main reproduction script
- `data/`: graph specification and task configuration
- `figures/`: generated plots
- `tables/`: raw CSVs and LaTeX tables
- `summary_zh.md`: Chinese experiment summary
- `main.tex`: short LaTeX report that includes the generated tables/figures

## Methods compared

- `mse`: edge-cost regression baseline
- `spo`: heuristic straight-through SPO baseline
- `spo+`: exact SPO+ surrogate over the enumerated path set
- `ddo-md`: exact path-simplex mirror descent / softmax-over-paths surrogate

## Reproduce

From the experiment directory (`experiments/riplm_dfl_benchmark/` inside the main repository):

```bash
python scripts/run_benchmark_comparison.py --out_dir .
```

Or, from the top-level repository root:

```bash
python experiments/riplm_dfl_benchmark/scripts/run_benchmark_comparison.py --out_dir experiments/riplm_dfl_benchmark
```

This will regenerate all CSV tables, TeX tables, figures, and `summary_zh.md` in this experiment folder.

## Notes

- The graph is intentionally small enough that **all feasible s-t paths can be enumerated exactly**.
- This makes the comparison close in spirit to the collaborator-style benchmark: the methods are compared on the *same explicit path simplex*.
- `spo` is included only as a heuristic baseline; it uses a regret-scaled straight-through direction rather than a literal gradient of the discontinuous SPO loss.
