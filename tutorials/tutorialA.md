# Deep Decision Optimization (DDO)

### Part A — Learning to Decide, Not Just Predict

> **Based on:** *Second-Order Smart Predict+Optimize via Rank-Induced Plackett–Luce Mirror Descent* (Zhang, 2026)

---

## What Is This?

Most machine learning models are trained to **predict** something well — a score, a label, a ranking. But in the real world, predictions are only valuable insofar as they lead to good **decisions**.

**Decision-focused learning (DFL)** is a research direction that trains models to optimize the quality of their *decisions* directly, rather than just minimizing prediction error.

This repository implements **RIPLM** (Rank-Induced Plackett–Luce Mirror Descent), a clean algorithm for decision-focused learning specifically designed for **ranking problems** — situations where a model must order N items and the quality of that ranking is what matters.

---

## The Core Problem (In Plain English)

Imagine you're a system that must rank N items (documents, products, options) given some input. After ranking, the real world reveals the actual cost or quality of each item. The question is: **how do you train your model to produce better rankings over time?**

The naive approach trains the model to predict costs accurately, then ranks by those predictions. The problem: accurate predictions don't always lead to good decisions. The model is optimized for the wrong objective.

The DFL approach: train the model so that the **ranking decisions themselves** improve — using the real cost of the ranking as the training signal.

The central difficulty: rankings are discrete (you can't take a gradient through a hard sort). This is where the math becomes interesting.

---

## The Key Idea: Soften the Ranking

Instead of committing to a hard ranking, RIPLM represents the model's decision as a **soft probability distribution over items** using the **Plackett–Luce (PL) model**.

Concretely, given a score vector `s` (one number per item), the probability assigned to each item is:

```
p_i = exp(s_i / τ) / sum_j exp(s_j / τ)
```

This is a **softmax with temperature τ**. When τ is small, the distribution concentrates on the highest-scoring item. When τ is large, probability spreads more evenly.

The surrogate loss minimized is the **expected cost** under this distribution:

```
Loss(s, ℓ) = sum_i [ p_i × actual_cost_i ]  =  <p, ℓ>
```

This is smooth and differentiable — so we can compute exact gradients and update the model properly.

---

## The Algorithm: RIPLM

### What gets updated

RIPLM updates the **score vector `s`** directly, using a second-order method called **mirror descent** with the PL log-partition function as the mirror map.

The log-partition potential is:

```
Ψ_τ(s) = τ × log( sum_i exp(s_i / τ) )
```

Its gradient is exactly the softmax: `∇Ψ_τ(s) = p`. This is the bridge between scores and probabilities.

### The update rule (step by step)

At each training step `t`:

1. Compute item probabilities: `p_i = softmax(s_i / τ)`
2. Observe the cost vector `ℓ` (the actual cost of each item this round)
3. Compute the **average cost**: `ℓ̄ = <p, ℓ> = sum_i p_i × ℓ_i`
4. Compute the **centered cost residual**: `r_i = ℓ_i - ℓ̄`
5. Compute the gradient: `g_i = (p_i / τ) × r_i`  *(note: sum of g_i equals zero)*
6. Apply the RIPLM update: `s_i ← s_i - η × τ × (g_i / p_i)`

Substituting step 5 into step 6, this reduces to:

```
s_i  ←  s_i  -  η × (ℓ_i - ℓ̄)
```

**In plain English:** push up scores for items that cost less than average this round; push down scores for items that cost more than average. No complex solver needed.

### Why the gradient sums to zero

The gradient `g` has the property that `sum_i g_i = 0` — it is **centered**. This is not a coincidence: it follows directly from the softmax normalization. And this zero-sum property is what makes the fast second-order update possible.

---

## Why This Update Is Theoretically Powerful

### 1. The Hessian has special structure (Theorem 4.1)

The Hessian of the PL potential — the matrix describing how the loss curves — has the form:

```
H = (1/τ) × ( Diag(p) - p pᵀ )
```

This is a **diagonal matrix minus a rank-one matrix** (called diagonal-plus-rank-one structure). This looks dense, but it is actually very cheap to work with.

### 2. Inverting the Hessian is free (Lemma 4.2)

Normally, second-order optimization requires inverting a large matrix — an expensive O(N²) or O(N³) operation. But because the gradient `g` is centered (sums to zero), the Sherman–Morrison formula collapses the rank-one correction entirely, giving:

```
H⁻¹ g  =  τ × (g / p)     [element-wise division]
```

This is a **closed-form inverse** computable in **O(N) time** — linear in the number of items, not quadratic.

> **Plain English:** Second-order methods are usually slow because inverting curvature information is expensive. Here, the PL geometry makes inversion free — it reduces to dividing each gradient entry by the corresponding probability. This is why the method scales well.

### 3. It is a natural gradient step (Theorem 5.1)

The RIPLM update is provably equivalent to the **natural gradient** on the Plackett–Luce statistical manifold. The Fisher information matrix of the PL family is exactly the Hessian `H`, so `H⁻¹ g` is by definition the natural gradient step.

> **Plain English:** Regular gradient descent moves in a direction that ignores the shape of the probability space. Natural gradient descent moves in the direction that accounts for the geometry of how probabilities curve — it's a smarter step. RIPLM gets this for free.

---

## Comparison to Existing Approaches

The paper is explicit about where RIPLM fits relative to prior work:

| Approach                                   | Strategy                                                     | Trade-off                                                    |
| ------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **SPO+** (Elmachtoub & Grigas, 2022)       | Convex surrogate loss tailored to predict-and-optimize       | Problem-specific; doesn't use ranking geometry               |
| **KKT-based layers** (OptNet, CVXPYLayers) | Differentiate through optimizer via implicit KKT differentiation | Very general, but backward pass requires solving a dense linear system |
| **RIPLM (this repo)**                      | PL relaxation + closed-form second-order update              | Specialized to ranking; O(N) backward pass; no dense linear solve |

**The paper's own caution:** RIPLM does not claim to dominate SPO+ or KKT methods in general. The paper states explicitly: *for ranking-style decision layers that are well modeled by a PL relaxation, RIPLM provides an analytically tractable alternative to generic KKT-based differentiation and a geometry-aware alternative to first-order surrogate updates.* A full empirical comparison is left for future work.

---

## Theoretical Guarantee: Variance-Adaptive Regret (Theorem 5.2)

The paper proves a **regret bound** — a guarantee on how well RIPLM performs over T rounds compared to the best fixed decision in hindsight.

Define the **total variance** across rounds:

```
V_T = sum over t of: sum_i p_i × (ℓ_i - ℓ̄)²
```

This measures how much costs fluctuate around their expected value each round.

The regret bound is:

```
Total regret  ≤  C × sqrt( V_T × log N )
```

for a universal constant `C`. This is called **variance-adaptive** because:

- If costs are stable round to round (low variance, small `V_T`), regret is small
- If costs are chaotic (high variance), regret is larger but still controlled
- The bound grows with the log of the number of items N, not N itself

**The paper's own caveat:** This bound applies to the smooth PL surrogate. Converting it back to the discrete ranking benchmark introduces a small additional approximation term that depends on the temperature τ (specifically, it is exponentially small in 1/τ).

---

## What This Repo Does NOT Cover

The paper is deliberately narrow and honest about its scope. This repository reflects the same:

- **No empirical comparison yet.** The paper explicitly defers benchmark comparisons with SPO+, OptNet, and CVXPYLayers to future work.
- **No claim of universal dominance.** RIPLM is designed for the ranking regime under a PL relaxation — it does not replace general-purpose solvers.
- **No multi-step or sequential settings.** This is a one-step decision layer. Sequential agent settings are in Part B.

---

## Repository Structure

```
/riplm            # Core RIPLM update rule and PL decision layer
/models           # Score model implementations (maps input → score vector s)
/experiments      # Synthetic and ranking benchmarks
/theory           # Derivations, proof sketches, and extended notes
```

---

## Glossary

| Term                      | Plain-English Meaning                                        |
| ------------------------- | ------------------------------------------------------------ |
| Decision-focused learning | Training on decision quality, not just prediction accuracy   |
| Plackett–Luce model       | A smooth probability distribution over rankings              |
| Mirror descent            | Optimization using the geometry of a chosen potential function |
| Natural gradient          | A gradient step that accounts for the curvature of probability space |
| Hessian                   | A matrix describing how a function curves (second-order information) |
| Diagonal-plus-rank-one    | A matrix structure that is cheap to invert                   |
| O(N)                      | Running time proportional to the number of items — linear, fast |
| Variance-adaptive regret  | A performance guarantee that tightens when costs are stable  |

---

## References

- Zhang (2026) — *Second-Order Smart Predict+Optimize via Rank-Induced Plackett–Luce Mirror Descent* ← **this paper**
- Elmachtoub & Grigas (2022) — Smart Predict-and-Optimize (SPO+)
- Amos & Kolter (2017) — OptNet: Differentiable optimization as a neural network layer
- Agrawal et al. (2019) — CVXPYLayers: Differentiable convex optimization layers
- Gaillard, Stoltz & Van Erven (2014) — Second-order bounds with excess losses
- Bubeck (2015) — Convex Optimization: Algorithms and Complexity

---

## Part B

For the extension to **LLM-based agents and sequential decision-making**, see [README_PartB.md](./README_PartB.md).
