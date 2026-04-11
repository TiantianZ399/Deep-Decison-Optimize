# 🔯 DEEP DECISION OPTIMIZATION 📈

Tiantian(Crystal) ZHANG Columbia University Undegrad  & Jierui(Jerry) Zuo UW incoming Phd, 
contact: t.zhang8@columbia.edu (paper idea dicsuiion) &  zuojr22@gmail.com (repository reprodction issue

This is the repo implementation first prosposed at https://arxiv.org/abs/2509.18138, and the full paper is soon coming out.

# DDO Multilayer Repo

This local repo revision reframes the project around **Deep Decision Optimization (DDO)** rather than treating RIPLM as a generic decision-focused baseline.

## What is in this repo

- `theory/spo_geometry.md` — standard SPO, a direct-SPO heuristic baseline, SPO+, entropic smoothing, and the bridge to RIPLM.
- `theory/notes_after_spo_geometry.md` — the corrected framing: DDO as the umbrella, RIPLM as the simplex specification, and the move to multi-layer structured decisions.
- `scripts/run_multilayer_path_ddo_benchmark.py` — exact synthetic benchmark on a layered shortest-path problem.
- `reports/multilayer_path_ddo_report.md` — summary of the run.

## Benchmark idea

The structured setting is a **multi-layer path decision**:

- context features generate edge costs on a layered graph,
- the downstream decision is the shortest path,
- evaluation uses the **standard SPO loss** (decision regret under the true costs), exact discrete path regret, and exact path accuracy,
- and the DDO method is an **entropic mirror-descent path layer** using exact forward-backward path marginals.

For this shortest-path benchmark, the reported `path_regret` is exactly the empirical **SPO loss** whenever the shortest path under the prediction is unique. Under continuous random costs, ties occur with probability zero, so in practice `path_regret` and `spo_loss` coincide.

## Methods compared

- `mse`: two-stage regression on edge costs.
- `spo`: direct-SPO heuristic baseline. It trains against the exact predicted path using a regret-scaled straight-through direction. This is a heuristic, not a true gradient method for the discontinuous SPO loss.
- `spo+`: exact path-level SPO+ surrogate.
- `ddo-md`: the structured DDO prototype in this repo, obtained by replacing the hard augmented SPO oracle with an entropically smoothed path layer.

## Important distinction

This repo includes both:

- **standard SPO explicitly as the main evaluation loss**, and
- a separate **direct-SPO heuristic training baseline** labeled `spo`.

The training baseline is intentionally labeled heuristic because the original SPO loss is nonconvex and discontinuous; the code therefore uses a straight-through update direction rather than claiming to optimize exact SPO directly.

## Why this repo exists

The earlier benchmark direction overreached by framing RIPLM as a general combinatorial DFL baseline. This repo fixes that by:

1. keeping the evaluation exact and discrete,
2. making the geometry explicit,
3. using a structured setting where entropic mirror descent has a clean interpretation.

## Run

```bash
python scripts/run_multilayer_path_ddo_benchmark.py
```

The script writes raw CSVs, summary tables, figures, and a markdown report to `data/`, `figures/`, and `reports/`.


