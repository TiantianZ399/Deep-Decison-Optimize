import itertools
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
TABLE_DIR = PROJECT_ROOT / "tables"
for p in [DATA_DIR, FIG_DIR, TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def softmax(x: np.ndarray, tau: float = 1.0, axis: int = -1) -> np.ndarray:
    z = x / tau
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


class Adam:
    def __init__(self, shape_w, shape_b, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.mw = np.zeros(shape_w)
        self.vw = np.zeros(shape_w)
        self.mb = np.zeros(shape_b)
        self.vb = np.zeros(shape_b)

    def step(self, w: np.ndarray, b: np.ndarray, gw: np.ndarray, gb: np.ndarray):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        self.mw = b1 * self.mw + (1 - b1) * gw
        self.vw = b2 * self.vw + (1 - b2) * (gw * gw)
        self.mb = b1 * self.mb + (1 - b1) * gb
        self.vb = b2 * self.vb + (1 - b2) * (gb * gb)
        mwh = self.mw / (1 - b1**self.t)
        vwh = self.vw / (1 - b2**self.t)
        mbh = self.mb / (1 - b1**self.t)
        vbh = self.vb / (1 - b2**self.t)
        w = w - self.lr * mwh / (np.sqrt(vwh) + self.eps)
        b = b - self.lr * mbh / (np.sqrt(vbh) + self.eps)
        return w, b


# --------------------------
# Ranking control benchmark
# --------------------------

def generate_rank_dataset(n_samples=2100, n_items=50, d=8, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(scale=1.0, size=(n_items, d))
    b_true = rng.normal(scale=0.2, size=(n_items,))
    x = rng.normal(size=(n_samples, d))
    latent = x @ w_true.T + b_true + rng.normal(scale=noise, size=(n_samples, n_items))
    costs = 1.0 / (1.0 + np.exp(latent))
    return x.astype(np.float64), costs.astype(np.float64)


def decision_cost_weighted(costs: np.ndarray, pred_scores: np.ndarray, k: int = 5, weights=None) -> float:
    if weights is None:
        weights = 1 / np.log2(np.arange(2, k + 2))
    idx = np.argsort(-pred_scores, axis=1)[:, :k]
    pred = np.take_along_axis(costs, idx, axis=1)
    return float(np.mean((pred * weights).sum(axis=1) / weights.sum()))


def optimal_cost_weighted(costs: np.ndarray, k: int = 5, weights=None) -> float:
    if weights is None:
        weights = 1 / np.log2(np.arange(2, k + 2))
    idx = np.argsort(costs, axis=1)[:, :k]
    vals = np.take_along_axis(costs, idx, axis=1)
    return float(np.mean((vals * weights).sum(axis=1) / weights.sum()))


def regret_weighted_rank(costs: np.ndarray, pred_scores: np.ndarray, k: int = 5, weights=None) -> float:
    return decision_cost_weighted(costs, pred_scores, k, weights) - optimal_cost_weighted(costs, k, weights)


def ndcg_at_k(costs: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    gains = 1 - costs
    order = np.argsort(-scores, axis=1)[:, :k]
    pred_g = np.take_along_axis(gains, order, axis=1)
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (pred_g * disc).sum(axis=1)
    ideal = np.take_along_axis(gains, np.argsort(-gains, axis=1)[:, :k], axis=1)
    idcg = (ideal * disc).sum(axis=1)
    return float(np.mean(dcg / np.maximum(idcg, 1e-12)))


def train_rank_method(x_train, c_train, x_val, c_val, method, lr=0.1, tau=0.2, k=5, epochs=80, batch_size=128, seed=0, l2=1e-4):
    rng = np.random.default_rng(seed)
    n, d = x_train.shape
    n_items = c_train.shape[1]
    w = rng.normal(scale=0.01, size=(n_items, d))
    b = np.zeros(n_items)
    best = None
    best_metric = np.inf
    for _ in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            xb = x_train[batch]
            cb = c_train[batch]
            if method == "mse":
                preds = xb @ w.T + b
                err = preds - cb
                gw = err.T @ xb / len(batch) + l2 * w
                gb = err.mean(axis=0)
            elif method == "spo+":
                preds = xb @ w.T + b
                x_true = np.zeros_like(cb)
                x_aug = np.zeros_like(cb)
                rows = np.arange(len(batch))[:, None]
                true_idx = np.argsort(cb, axis=1)[:, :k]
                aug_idx = np.argsort(2 * preds - cb, axis=1)[:, :k]
                x_true[rows, true_idx] = 1.0 / k
                x_aug[rows, aug_idx] = 1.0 / k
                grad = 2 * (x_true - x_aug)
                gw = grad.T @ xb / len(batch) + l2 * w
                gb = grad.mean(axis=0)
            elif method in ("plfo", "riplm"):
                scores = xb @ w.T + b
                p = softmax(scores, tau=tau, axis=1)
                mean_c = (p * cb).sum(axis=1, keepdims=True)
                delta = cb - mean_c if method == "riplm" else (p / tau) * (cb - mean_c)
                gw = delta.T @ xb / len(batch) + l2 * w
                gb = delta.mean(axis=0)
            else:
                raise ValueError(method)
            w -= lr * gw
            b -= lr * gb
        if method in ("mse", "spo+"):
            val_scores = -(x_val @ w.T + b)
        else:
            val_scores = x_val @ w.T + b
        val_metric = regret_weighted_rank(c_val, val_scores, k=k)
        if val_metric < best_metric:
            best_metric = val_metric
            best = (w.copy(), b.copy())
    return best, float(best_metric)


def tune_rank_hyperparams():
    x, c = generate_rank_dataset(seed=101)
    x_train, c_train = x[:100], c[:100]
    x_val, c_val = x[100:1100], c[100:1100]
    grids = {
        "mse": {"lr": [0.01, 0.03, 0.1, 0.3], "tau": [0.2]},
        "spo+": {"lr": [0.001, 0.003, 0.01, 0.03, 0.1], "tau": [0.2]},
        "plfo": {"lr": [0.001, 0.003, 0.01, 0.03, 0.1], "tau": [0.03, 0.05, 0.1, 0.2, 0.5, 1.0]},
        "riplm": {"lr": [0.01, 0.03, 0.1, 0.3, 1.0], "tau": [0.03, 0.05, 0.1, 0.2, 0.5, 1.0]},
    }
    rows = []
    best_hps = {}
    for method, grid in grids.items():
        best = (np.inf, None, None)
        for tau in grid["tau"]:
            for lr in grid["lr"]:
                _, val = train_rank_method(x_train, c_train, x_val, c_val, method, lr=lr, tau=tau, seed=0)
                rows.append({"suite": "ranking", "task": "RankingControl", "method": method, "lr": lr, "tau": tau, "val_regret": val})
                if val < best[0]:
                    best = (val, lr, tau)
        best_hps[method] = {"lr": best[1], "tau": best[2]}
    pd.DataFrame(rows).to_csv(DATA_DIR / "ranking_tuning_grid.csv", index=False)
    return best_hps


def run_ranking_eval(best_hps, seeds=(1, 2, 3, 4, 5)):
    rows = []
    for seed in seeds:
        x, c = generate_rank_dataset(seed=seed)
        x_train, c_train = x[:100], c[:100]
        x_val, c_val = x[100:1100], c[100:1100]
        x_test, c_test = x[1100:], c[1100:]
        for method, hp in best_hps.items():
            t0 = time.perf_counter()
            model, _ = train_rank_method(x_train, c_train, x_val, c_val, method, lr=hp["lr"], tau=hp["tau"], seed=seed)
            runtime = time.perf_counter() - t0
            w, b = model
            scores = -(x_test @ w.T + b) if method in ("mse", "spo+") else (x_test @ w.T + b)
            rows.append(
                {
                    "suite": "ranking",
                    "task": "RankingControl",
                    "seed": seed,
                    "method": method,
                    "lr": hp["lr"],
                    "tau": hp["tau"],
                    "regret": regret_weighted_rank(c_test, scores, k=5),
                    "ndcg": ndcg_at_k(c_test, scores, k=5),
                    "runtime_s": runtime,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "ranking_raw.csv", index=False)
    summary = (
        df.groupby("method")
        .agg(regret_mean=("regret", "mean"), regret_std=("regret", "std"), ndcg_mean=("ndcg", "mean"), ndcg_std=("ndcg", "std"), runtime_mean=("runtime_s", "mean"), runtime_std=("runtime_s", "std"))
        .reset_index()
    )
    summary.to_csv(DATA_DIR / "ranking_summary.csv", index=False)
    return df, summary


# -------------------------------------------
# Structured benchmark families (exact small)
# -------------------------------------------

def enumerate_grid_paths(n_rows=4, n_cols=4):
    edge_list = []
    edge_to_idx = {}
    for r in range(n_rows):
        for c in range(n_cols):
            if c + 1 < n_cols:
                edge_to_idx[((r, c), (r, c + 1))] = len(edge_list)
                edge_list.append(((r, c), (r, c + 1)))
            if r + 1 < n_rows:
                edge_to_idx[((r, c), (r + 1, c))] = len(edge_list)
                edge_list.append(((r, c), (r + 1, c)))
    paths = []

    def dfs(r, c, vec):
        if (r, c) == (n_rows - 1, n_cols - 1):
            paths.append(vec.copy())
            return
        if c + 1 < n_cols:
            i = edge_to_idx[((r, c), (r, c + 1))]
            vec[i] = 1
            dfs(r, c + 1, vec)
            vec[i] = 0
        if r + 1 < n_rows:
            i = edge_to_idx[((r, c), (r + 1, c))]
            vec[i] = 1
            dfs(r + 1, c, vec)
            vec[i] = 0

    dfs(0, 0, np.zeros(len(edge_list), dtype=np.float64))
    return np.stack(paths)


def enumerate_matchings(n=4):
    perms = list(itertools.permutations(range(n)))
    decisions = []
    for p in perms:
        x = np.zeros((n, n), dtype=np.float64)
        for i, j in enumerate(p):
            x[i, j] = 1.0
        decisions.append(x.reshape(-1))
    return np.stack(decisions)


def enumerate_knapsack(weights, capacity):
    sols = []
    n = len(weights)
    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.float64)
        if np.dot(weights, x) <= capacity:
            sols.append(x)
    return np.stack(sols)


STRUCTURED_TASKS = {
    "ShortestPath": {"D": enumerate_grid_paths(), "kind": "positive", "m": 24},
    "Matching": {"D": enumerate_matchings(4), "kind": "positive", "m": 16},
    "Knapsack": {"D": enumerate_knapsack(np.array([2, 3, 4, 5, 1, 2, 3, 4], dtype=np.float64), 10), "kind": "negative_reward", "m": 8},
}


def make_true_map(m, d, hidden=16, seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(scale=0.8, size=(d, hidden))
    b1 = rng.normal(scale=0.2, size=(hidden,))
    w2 = rng.normal(scale=0.6, size=(hidden, m))
    b2 = rng.normal(scale=0.2, size=(m,))
    return w1, b1, w2, b2


def nonlinear_costs(z, params, kind="positive", noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    w1, b1, w2, b2 = params
    h = np.tanh(z @ w1 + b1)
    raw = h @ w2 + b2 + rng.normal(scale=noise, size=(len(z), w2.shape[1]))
    if kind == "positive":
        return 0.1 + np.log1p(np.exp(raw))
    if kind == "negative_reward":
        rewards = 0.1 + np.log1p(np.exp(raw))
        return -rewards
    raise ValueError(kind)


def make_atomic_dataset(m_atomic, kind, n_train=120, n_val=200, n_test=800, d=10, seed=0, noise=0.2):
    rng = np.random.default_rng(seed)
    total = n_train + n_val + n_test
    z = rng.normal(size=(total, d))
    params = make_true_map(m_atomic, d, hidden=16, seed=seed + 11)
    c = nonlinear_costs(z, params, kind=kind, noise=noise, seed=seed + 17)
    return z[:n_train], c[:n_train], z[n_train : n_train + n_val], c[n_train : n_train + n_val], z[n_train + n_val :], c[n_train + n_val :]


def regret_from_atomic(c_true, c_pred, dmat):
    pred_idx = np.argmin(c_pred @ dmat.T, axis=1)
    pred_cost = np.sum(c_true * dmat[pred_idx], axis=1)
    opt_cost = np.min(c_true @ dmat.T, axis=1)
    regret = float(np.mean(pred_cost - opt_cost))
    opt_acc = float(np.mean(pred_idx == np.argmin(c_true @ dmat.T, axis=1)))
    return regret, opt_acc, float(np.mean(pred_cost)), float(np.mean(opt_cost))


def train_atomic_method(z_train, c_train, z_val, c_val, dmat, method, lr=1e-2, tau=0.2, epochs=80, batch_size=64, seed=0, l2=1e-5):
    rng = np.random.default_rng(seed)
    n, d = z_train.shape
    m = c_train.shape[1]
    w = rng.normal(scale=0.05, size=(m, d))
    b = np.zeros(m)
    opt = Adam(w.shape, b.shape, lr=lr)
    best = None
    best_reg = np.inf
    val_opt = np.min(c_val @ dmat.T, axis=1)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            zb = z_train[idx]
            cb = c_train[idx]
            pred = zb @ w.T + b
            if method == "mse":
                grad_c = pred - cb
            elif method == "spo+":
                idx_true = np.argmin(cb @ dmat.T, axis=1)
                idx_aug = np.argmin((2 * pred - cb) @ dmat.T, axis=1)
                x_true = dmat[idx_true]
                x_aug = dmat[idx_aug]
                grad_c = 2 * (x_true - x_aug)
            elif method in ("plfo", "riplm"):
                c_sol = cb @ dmat.T
                scores = -(pred @ dmat.T)
                p = softmax(scores, tau=tau, axis=1)
                mean_c = np.sum(p * c_sol, axis=1, keepdims=True)
                if method == "plfo":
                    gscore = p * (c_sol - mean_c) / tau
                else:
                    gscore = c_sol - mean_c
                grad_c = -(gscore @ dmat)
            else:
                raise ValueError(method)
            gw = grad_c.T @ zb / len(idx) + l2 * w
            gb = grad_c.mean(axis=0)
            w, b = opt.step(w, b, gw, gb)
        pred_val = z_val @ w.T + b
        pred_idx = np.argmin(pred_val @ dmat.T, axis=1)
        pred_cost = np.sum(c_val * dmat[pred_idx], axis=1)
        reg = float(np.mean(pred_cost - val_opt))
        if reg < best_reg:
            best_reg = reg
            best = (w.copy(), b.copy())
    return best, best_reg


def tune_structured_hyperparams():
    grids = {
        "mse": {"lr": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1], "tau": [0.2]},
        "spo+": {"lr": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "tau": [0.2]},
        "plfo": {"lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "tau": [0.05, 0.1, 0.2, 0.5]},
        "riplm": {"lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], "tau": [0.05, 0.1, 0.2, 0.5]},
    }
    all_rows = []
    best_hps = {}
    pilot_seeds = {"ShortestPath": 201, "Matching": 202, "Knapsack": 203}
    for task, meta in STRUCTURED_TASKS.items():
        dmat = meta["D"]
        z_train, c_train, z_val, c_val, _, _ = make_atomic_dataset(meta["m"], meta["kind"], seed=pilot_seeds[task])
        best_hps[task] = {}
        for method, grid in grids.items():
            best = (np.inf, None, None)
            for tau in grid["tau"]:
                for lr in grid["lr"]:
                    _, val = train_atomic_method(z_train, c_train, z_val, c_val, dmat, method, lr=lr, tau=tau, seed=0)
                    all_rows.append({"suite": "structured", "task": task, "method": method, "lr": lr, "tau": tau, "val_regret": val})
                    if val < best[0]:
                        best = (val, lr, tau)
            best_hps[task][method] = {"lr": best[1], "tau": best[2]}
    pd.DataFrame(all_rows).to_csv(DATA_DIR / "structured_tuning_grid.csv", index=False)
    return best_hps


def run_structured_eval(best_hps, seeds=(1, 2, 3, 4, 5)):
    rows = []
    for task, meta in STRUCTURED_TASKS.items():
        dmat = meta["D"]
        for seed in seeds:
            z_train, c_train, z_val, c_val, z_test, c_test = make_atomic_dataset(meta["m"], meta["kind"], seed=seed)
            for method, hp in best_hps[task].items():
                t0 = time.perf_counter()
                model, _ = train_atomic_method(z_train, c_train, z_val, c_val, dmat, method, lr=hp["lr"], tau=hp["tau"], seed=seed)
                runtime = time.perf_counter() - t0
                w, b = model
                pred = z_test @ w.T + b
                regret, opt_acc, pred_cost, opt_cost = regret_from_atomic(c_test, pred, dmat)
                rows.append(
                    {
                        "suite": "structured",
                        "task": task,
                        "seed": seed,
                        "method": method,
                        "lr": hp["lr"],
                        "tau": hp["tau"],
                        "regret": regret,
                        "opt_acc": opt_acc,
                        "decision_cost": pred_cost,
                        "opt_cost": opt_cost,
                        "runtime_s": runtime,
                        "n_decisions": len(dmat),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "structured_raw.csv", index=False)
    summary = (
        df.groupby(["task", "method"])
        .agg(regret_mean=("regret", "mean"), regret_std=("regret", "std"), opt_acc_mean=("opt_acc", "mean"), opt_acc_std=("opt_acc", "std"), runtime_mean=("runtime_s", "mean"), runtime_std=("runtime_s", "std"), n_decisions=("n_decisions", "first"))
        .reset_index()
    )
    summary.to_csv(DATA_DIR / "structured_summary.csv", index=False)
    return df, summary


# ---------------------
# Reporting and figures
# ---------------------

def latex_escape(s):
    return s.replace("+", "+")


def format_pm(mean, std, digits=3):
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def save_hyperparams(rank_hps, struct_hps):
    rows = []
    for method, hp in rank_hps.items():
        rows.append({"suite": "ranking", "task": "RankingControl", "method": method, **hp})
    for task, d in struct_hps.items():
        for method, hp in d.items():
            rows.append({"suite": "structured", "task": task, "method": method, **hp})
    hp_df = pd.DataFrame(rows)
    hp_df.to_csv(DATA_DIR / "selected_hyperparams.csv", index=False)
    return hp_df


def make_ranking_table(summary: pd.DataFrame):
    best_regret = summary["regret_mean"].min()
    best_ndcg = summary["ndcg_mean"].max()
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Regret$@5$ $\\downarrow$ & NDCG$@5$ $\\uparrow$ & Train time (s) \\\\",
        "\\midrule",
    ]
    order = ["mse", "spo+", "plfo", "riplm"]
    names = {"mse": "Two-stage MSE", "spo+": "SPO+", "plfo": "PL-FO", "riplm": "RIPLM"}
    for m in order:
        row = summary[summary["method"] == m].iloc[0]
        reg = format_pm(row.regret_mean, row.regret_std, 4)
        ndcg = format_pm(row.ndcg_mean, row.ndcg_std, 4)
        rt = format_pm(row.runtime_mean, row.runtime_std, 3)
        if abs(row.regret_mean - best_regret) < 1e-12:
            reg = "\\textbf{" + reg + "}"
        if abs(row.ndcg_mean - best_ndcg) < 1e-12:
            ndcg = "\\textbf{" + ndcg + "}"
        lines.append(f"{names[m]} & {reg} & {ndcg} & {rt} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLE_DIR / "ranking_table.tex").write_text("\n".join(lines), encoding="utf-8")



def make_structured_table(summary: pd.DataFrame):
    lines = [
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Task & Method & Regret $\\downarrow$ & Opt. acc. $\\uparrow$ \\\\",
        "\\midrule",
    ]
    task_order = ["ShortestPath", "Matching", "Knapsack"]
    method_order = ["mse", "spo+", "plfo", "riplm"]
    method_names = {"mse": "Two-stage MSE", "spo+": "SPO+", "plfo": "PL-FO", "riplm": "RIPLM"}
    for task in task_order:
        sdf = summary[summary["task"] == task].copy()
        best_regret = sdf["regret_mean"].min()
        best_acc = sdf["opt_acc_mean"].max()
        first = True
        for m in method_order:
            row = sdf[sdf["method"] == m].iloc[0]
            reg = format_pm(row.regret_mean, row.regret_std, 3)
            acc = format_pm(row.opt_acc_mean, row.opt_acc_std, 3)
            if abs(row.regret_mean - best_regret) < 1e-12:
                reg = "\\textbf{" + reg + "}"
            if abs(row.opt_acc_mean - best_acc) < 1e-12:
                acc = "\\textbf{" + acc + "}"
            task_cell = f"\\multirow{{4}}{{*}}{{{task}}}" if first else ""
            lines.append(f"{task_cell} & {method_names[m]} & {reg} & {acc} \\\\")
            first = False
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    (TABLE_DIR / "structured_table.tex").write_text("\n".join(lines), encoding="utf-8")



def make_hyperparam_table(hp_df: pd.DataFrame):
    lines = [
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Task & Method & Learning rate & Temperature $\\tau$ \\\\",
        "\\midrule",
    ]
    order = [
        ("RankingControl", "mse"),
        ("RankingControl", "spo+"),
        ("RankingControl", "plfo"),
        ("RankingControl", "riplm"),
        ("ShortestPath", "mse"),
        ("ShortestPath", "spo+"),
        ("ShortestPath", "plfo"),
        ("ShortestPath", "riplm"),
        ("Matching", "mse"),
        ("Matching", "spo+"),
        ("Matching", "plfo"),
        ("Matching", "riplm"),
        ("Knapsack", "mse"),
        ("Knapsack", "spo+"),
        ("Knapsack", "plfo"),
        ("Knapsack", "riplm"),
    ]
    names = {"mse": "Two-stage MSE", "spo+": "SPO+", "plfo": "PL-FO", "riplm": "RIPLM"}
    for task, method in order:
        row = hp_df[(hp_df["task"] == task) & (hp_df["method"] == method)].iloc[0]
        lines.append(f"{task} & {names[method]} & {row.lr:g} & {row.tau:g} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLE_DIR / "hyperparams_table.tex").write_text("\n".join(lines), encoding="utf-8")



def make_average_rank_table(struct_df: pd.DataFrame):
    rank_rows = []
    for task, sdf in struct_df.groupby("task"):
        means = sdf.groupby("method")["regret"].mean().sort_values()
        for rank, (method, val) in enumerate(means.items(), start=1):
            rank_rows.append({"task": task, "method": method, "rank": rank, "regret_mean": val})
    ranks = pd.DataFrame(rank_rows)
    avg = ranks.groupby("method")["rank"].mean().reset_index().sort_values("rank")
    avg.to_csv(DATA_DIR / "structured_average_ranks.csv", index=False)
    lines = [
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Method & Average regret rank on structured tasks \\\\",
        "\\midrule",
    ]
    names = {"mse": "Two-stage MSE", "spo+": "SPO+", "plfo": "PL-FO", "riplm": "RIPLM"}
    best = avg["rank"].min()
    for _, row in avg.iterrows():
        val = f"{row['rank']:.2f}"
        if abs(row["rank"] - best) < 1e-12:
            val = "\\textbf{" + val + "}"
        lines.append(f"{names[row['method']]} & {val} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLE_DIR / "average_rank_table.tex").write_text("\n".join(lines), encoding="utf-8")


def make_figures(rank_summary: pd.DataFrame, struct_summary: pd.DataFrame, hp_df: pd.DataFrame):
    plt.rcParams.update({"figure.dpi": 180})

    # Ranking bar plot
    order = ["mse", "spo+", "plfo", "riplm"]
    names = ["Two-stage MSE", "SPO+", "PL-FO", "RIPLM"]
    rs = rank_summary.set_index("method").loc[order]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(names, rs["regret_mean"], yerr=rs["regret_std"], capsize=4)
    ax.set_ylabel("Regret@5")
    ax.set_title("Ranking control benchmark")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ranking_regret.png", bbox_inches="tight")
    plt.close(fig)

    # Structured grouped bars
    task_order = ["ShortestPath", "Matching", "Knapsack"]
    method_order = ["mse", "spo+", "plfo", "riplm"]
    method_names = ["Two-stage MSE", "SPO+", "PL-FO", "RIPLM"]
    width = 0.18
    x = np.arange(len(task_order))
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for j, method in enumerate(method_order):
        sub = struct_summary[struct_summary["method"] == method].set_index("task").loc[task_order]
        ax.bar(x + (j - 1.5) * width, sub["regret_mean"], width, yerr=sub["regret_std"], capsize=3, label=method_names[j])
    ax.set_xticks(x)
    ax.set_xticklabels(task_order)
    ax.set_ylabel("Decision regret")
    ax.set_title("Exact small-scale DFL benchmark families")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "structured_regret.png", bbox_inches="tight")
    plt.close(fig)

    # Runtime heatmap
    runtime_pivot = struct_summary.pivot(index="method", columns="task", values="runtime_mean").loc[method_order, task_order]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    im = ax.imshow(runtime_pivot.values, aspect="auto")
    ax.set_xticks(np.arange(len(task_order)))
    ax.set_xticklabels(task_order)
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(["Two-stage MSE", "SPO+", "PL-FO", "RIPLM"])
    for i in range(runtime_pivot.shape[0]):
        for j in range(runtime_pivot.shape[1]):
            ax.text(j, i, f"{runtime_pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("Mean training time per run (s)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "runtime_heatmap.png", bbox_inches="tight")
    plt.close(fig)


# ----------
# Main entry
# ----------

def main():
    rank_hps = tune_rank_hyperparams()
    struct_hps = tune_structured_hyperparams()
    hp_df = save_hyperparams(rank_hps, struct_hps)

    rank_df, rank_summary = run_ranking_eval(rank_hps)
    struct_df, struct_summary = run_structured_eval(struct_hps)

    make_ranking_table(rank_summary)
    make_structured_table(struct_summary)
    make_hyperparam_table(hp_df)
    make_average_rank_table(struct_df)
    make_figures(rank_summary, struct_summary, hp_df)

    # concise JSON-like summary for easy inspection
    summary_text = {
        "ranking_best": rank_summary.sort_values("regret_mean").iloc[0].to_dict(),
        "structured_avg_rank": pd.read_csv(DATA_DIR / "structured_average_ranks.csv").to_dict(orient="records"),
    }
    (DATA_DIR / "summary_snapshot.json").write_text(pd.Series(summary_text).to_json(), encoding="utf-8")
    print("Done. Artifacts written to:")
    print(DATA_DIR)
    print(FIG_DIR)
    print(TABLE_DIR)


if __name__ == "__main__":
    main()
