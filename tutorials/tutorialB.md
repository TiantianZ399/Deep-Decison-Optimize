# Deep Decision Optimization (DDO)

### Part B — Differentiable Decision-Making for LLM Agents

> **Extends:** *Second-Order Smart Predict+Optimize via Rank-Induced Plackett–Luce Mirror Descent* (Zhang, 2026)
> **Start here:** Read [README_PartA.md](./README_PartA.md) first for the theoretical foundation.

---

## What Is This?

Language model agents — systems that use an LLM to choose sequences of actions like selecting tools, ranking retrieved documents, or picking the next step — typically rely on either:

- **Reinforcement learning (RL):** trial and error with reward signals, which is powerful but noisy and unstable
- **Heuristics:** manually designed action selection, which doesn't improve from data

This repository explores a third path: applying the **RIPLM framework from Part A** to the action selection decisions inside an agent loop.

The core idea: many agent decisions (which tool to call, which document to retrieve, which action to take next) are **ranking-style problems** — the agent must order or select from a set of candidates. RIPLM is designed exactly for this. If each action-selection step can be treated as a differentiable PL distribution over candidates, we can compute structured gradients through it directly — without sampling noise.

---

## The Core Problem (In Plain English)

An LLM agent operates in a loop:

1. Observe the current state (prompt, task history, environment feedback)
2. Generate a set of candidate actions (tools to call, documents to retrieve, next steps)
3. Score each candidate using the model
4. Select an action (or a weighted combination of actions)
5. Observe the outcome — how good or bad was that action?
6. Update the model to make better choices next time

The hard part is step 6: **how do you train the model to improve its action choices?**

Standard RL answers: sample one action, observe a reward, run REINFORCE. This works, but requires many samples and suffers from high variance — especially when the reward is sparse or delayed.

RIPLM answers: treat the action distribution at each step as a **Plackett–Luce distribution over the candidate set**, then apply the same closed-form second-order update from Part A. This gives structured, low-variance gradients at each step — without needing to sample.

---

## How the Part A Theory Applies Here

In Part A, the algorithm works on a score vector over N ranked items. In Part B, the same structure applies — but the "items" are now **candidate actions**:

| Part A concept      | Part B equivalent                 |
| ------------------- | --------------------------------- |
| N items to rank     | K candidate actions at step t     |
| Score vector `s`    | Action scores from the LLM        |
| Cost vector `ℓ`     | Outcome quality of each action    |
| PL distribution `p` | Action selection distribution     |
| RIPLM update        | Score update for action selection |

The update rule is identical to Part A:

```
action_score_i  ←  action_score_i  -  η × (outcome_cost_i - average_outcome_cost)
```

Push up scores for actions that worked better than average; push down scores for actions that worked worse.

And because the Part A guarantees hold — O(N) update, natural gradient interpretation, variance-adaptive regret — they carry over to each step of the agent loop.

---

## Why This Is Better Than REINFORCE (For Step-Wise Decisions)

REINFORCE, the standard policy gradient algorithm, works like this at each step:

> Sample one action. Observe a reward. Scale the log-probability of that action by the reward. Update.

Problems with this:

- You only learn from **one action per step** — everything else is ignored
- The gradient is **high variance** — a single sample gives a noisy signal
- Many samples are needed before the signal is reliable

RIPLM at each step:

- Evaluates (or scores) **all candidate actions**
- Uses the **cost difference** from the average directly
- The gradient is centered and structured — **low variance by design**

| Property              | REINFORCE              | RIPLM (per step)                  |
| --------------------- | ---------------------- | --------------------------------- |
| Gradient source       | One sampled action     | All scored candidates             |
| Variance              | High                   | Low (centered gradient)           |
| Backward pass cost    | Standard               | O(K) in number of candidates      |
| Second-order info     | No                     | Yes, via PL Hessian               |
| Theoretical guarantee | Asymptotic convergence | Variance-adaptive regret per step |

> **Important:** this comparison holds for the **step-wise action selection** problem. RIPLM does not solve long-horizon credit assignment — that remains a separate challenge (see scope section below).

---

## Where This Applies in LLM Agents

The most natural applications are decisions where a **fixed candidate set** exists at each step:

**Tool selection**
The agent must choose from a set of available APIs or tools. Each tool is a candidate; RIPLM updates scores based on which tools led to better task outcomes.

**Retrieval ranking**
The agent must rank or select from retrieved documents before generating a response. RIPLM treats the document set as the candidate set and updates ranking scores based on downstream answer quality.

**Action selection in structured environments**
When the action space is finite and can be enumerated at each step (e.g. navigation, code selection, plan steps), RIPLM applies directly.

---

## What This Repo Does NOT Model

This is the most important section to read carefully. Part B extends Part A to sequential settings, but it does **not** solve every problem in agent training:

**No long-horizon credit assignment.** RIPLM updates scores based on the outcome of the *current* action. Connecting an early action to a reward that only appears many steps later (the classic RL credit assignment problem) is not addressed here.

**No value functions.** There are no V(s) or Q(s, a) estimates. RIPLM is not a replacement for actor-critic or Q-learning methods in full MDP settings.

**No full MDP formalism.** We treat each step as an approximately independent ranking decision. This is a reasonable approximation when step-wise feedback is available, but breaks down in highly correlated sequential settings.

**No empirical validation yet.** As with Part A, full benchmark comparisons are deferred to future work.

The goal of Part B is to handle the **local, step-wise action selection decision** well — and show that the RIPLM update gives a structured, principled alternative to REINFORCE sampling for this part of the problem.

---

## Relationship to Part A

Part A proves the core properties of RIPLM:

- O(N) closed-form second-order update (Lemma 4.2)
- Equivalence to natural gradient on the PL manifold (Theorem 5.1)
- Variance-adaptive regret bound (Theorem 5.2)

Part B applies these properties at each step of an agent loop. The theory from Part A carries over directly to each individual decision step. The new challenge in Part B is the **sequential structure** — states evolve, action sets change, and outcomes at one step affect the context at the next.

---

## Benchmarks (Planned)

**Tool-use tasks**

- ToolBench-style evaluation: selecting from a catalog of APIs given a task
- Metric: task success rate, tool selection accuracy

**Retrieval and ranking**

- BEIR-style: ranking documents given a query, evaluating retrieval quality
- RAG pipeline evaluation: measuring answer quality as a function of retrieval decisions
- Metric: NDCG, recall@k, downstream answer quality

**Lightweight sequential environments**

- Simplified multi-step tasks with per-step action feedback
- Metric: decision accuracy per step, training stability across steps

---

## Repository Structure

```
/riplm            # Core RIPLM update rule (shared with Part A)
/agent            # Agent loop and state management
/actions          # Candidate action generation and scoring
/llm_interface    # Model wrappers for scoring and prompting
/benchmarks       # Tool-use, retrieval, and sequential task evaluations
```

---

## Summary

Part B takes the theoretically grounded RIPLM algorithm from Part A and asks: **can the same structured, geometry-aware update improve action selection in LLM agents?**

For step-wise decisions over a finite candidate set, the answer is yes — and the O(N) update, natural gradient interpretation, and variance-adaptive guarantee from Part A all carry over directly. For the harder problems of long-horizon credit assignment and full sequential optimization, more work is needed.

---

## References

- Zhang (2026) — *Second-Order Smart Predict+Optimize via Rank-Induced Plackett–Luce Mirror Descent* ← **the paper this builds on**
- Elmachtoub & Grigas (2022) — SPO+
- Amos & Kolter (2017) — OptNet
- Gaillard, Stoltz & Van Erven (2014) — Second-order bounds with excess losses
- Wilder, Dilkina & Tambe (2019) — Multi-level decision-focused learning

---

## Part A

For the foundational theory, algorithm derivation, and regret guarantees, see [README_PartA.md](./README_PartA.md).
