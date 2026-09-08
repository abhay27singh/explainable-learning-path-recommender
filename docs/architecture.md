# System Architecture — Explainable AI Personalized Learning Path Recommender

**Date:** 2026-08-09
**Status:** Proposed, awaiting approval
**Companion documents:** [`docs/superpowers/specs/2026-08-09-xai-learning-path-design.md`](superpowers/specs/2026-08-09-xai-learning-path-design.md), [`docs/paper-corrections.md`](paper-corrections.md)

---

## 0. What the paper obliges us to build

Configuration values the paper already states, which the implementation honors exactly so that Sections III–IV become true descriptions of this code:

| Component | Paper's stated value | Honored |
|---|---|---|
| KT encoder | Transformer, 2 layers, hidden 128, 4 heads, max seq 200 | Yes |
| Graph encoder | GCN, 3 conv layers, LayerNorm + ReLU, 64-dim output | Yes |
| RL agent | DQN, 3-layer FC 256-128-64, ReLU, replay 100,000 | Yes |
| Optimizer | Adam, lr 1e-3, cosine annealing, L2 1e-4 | Yes |
| Batch size | 64 | Yes |
| Sessionization | 30 minutes of inactivity | Yes |
| Split | 80/10/10 stratified, 5-fold CV | Yes |
| Significance | paired t-tests, p < 0.05 | Yes |
| KT metrics | AUC-ROC, RMSE | Yes |
| Path metrics | completion rate, NDCG, avg knowledge gain | Yes |
| Baselines | DKT, RL-DKT, GNN-based, collaborative filtering | Yes |
| Ablations | −knowledge graph, −temporal attention, −explainability | Yes |
| User study | n=30, 5-point Likert | Yes (run separately) |
| Explanation | priority backtracking over knowledge graph | Yes |

Values the paper states that the implementation **must diverge from**, requiring paper edits:

| Item | Paper | Reality |
|---|---|---|
| Concept graph | 1,247 nodes / 3,891 prereq / 1,023 coreq | Derived from OULAD metadata; expect ~250 nodes, ~400–700 edges |
| Hardware | i9-12900K, RTX 3080, Ubuntu 22.04 | Apple Silicon, MPS backend |
| Stack | Python 3.9, PyTorch 1.12, PyG 2.2, SB3 1.7, RDFLib 6.2 | Python 3.11, PyTorch 2.4+, no PyG, no SB3, no RDFLib (see §2) |
| Replay buffer | "(99)" in III-D vs 100,000 in IV-C | 100,000 |
| GCN layers | "(2, 4)" in III-D vs 3 in IV-C | Search range {2,3,4}, selected 3 |

---

## 1. Architecture overview

Five layers. Each writes artifacts to disk; each downstream layer loads artifacts rather than recomputing.

```
┌────────────────────────────────────────────────────────────────────────┐
│ L1  DATA LAYER          OULAD CSV → Parquet → interaction sequences    │
├────────────────────────────────────────────────────────────────────────┤
│ L2  GRAPH LAYER         concept DAG (derived + curated) → dense adj    │
├────────────────────────────────────────────────────────────────────────┤
│ L3  KNOWLEDGE TRACING   KG-DKT: GCN concept embeddings                 │
│                         + time-aware Transformer → mastery vector      │
├────────────────────────────────────────────────────────────────────────┤
│ L4  PLANNING            GreedyPlanner (always works)                   │
│                         DQNPlanner (trained in DKT-simulated env)      │
├────────────────────────────────────────────────────────────────────────┤
│ L5  EXPLANATION         priority backtracking + reward decomposition   │
│                         + SHAP + counterfactual → templated text       │
└────────────────────────────────────────────────────────────────────────┘
        ↓ consumed by ↓
   EVAL HARNESS  (tables, ablations, t-tests)      STREAMLIT DEMO
```

---

## 2. Software requirements

### 2.1 Runtime

| Package | Version | Purpose | Notes |
|---|---|---|---|
| Python | 3.11 | Runtime | 3.12 works; 3.11 is the safest wheel target |
| PyTorch | ≥ 2.4 | All neural components | MPS backend on Apple Silicon |
| NumPy | ≥ 1.26 | Numerics | |
| **DuckDB** | ≥ 1.0 | **Primary ETL and analytics engine — all preprocessing is SQL** | Embedded, no server, reads/writes Parquet natively, faster than pandas at 10.6M rows |
| pandas | ≥ 2.2 | Thin boundary layer between DuckDB results and PyTorch tensors | Demoted; not the ETL engine |
| pyarrow | ≥ 15 | Parquet I/O | Keeps 10.6M clickstream rows fast to reload |
| SQLite | stdlib | Runtime application store — recommendation logs, user-study responses, session state | Separate from the analytical store |
| mlxtend | ≥ 0.23 | FP-Growth association rule mining | Data mining component (§4.4) |
| networkx | ≥ 3.2 | DAG validation, ancestor traversal | Pure Python, trivial at ~250 nodes |
| scikit-learn | ≥ 1.4 | AUC/RMSE/NDCG, CF baseline | |
| scipy | ≥ 1.12 | paired t-tests | |
| gymnasium | ≥ 0.29 | Env interface for the RL environment | Standard interface; no SB3 needed |
| shap | ≥ 0.45 | Feature attribution over student features | captum is an acceptable substitute |
| PyYAML | ≥ 6 | Configs and graph files | |
| matplotlib | ≥ 3.8 | Figures for the paper | |
| **FastAPI** | ≥ 0.115 | Inference API — model held in memory, JSON responses | Replaces Streamlit; also delivers the "RESTful API for LMS integration" the paper lists as future work |
| uvicorn | ≥ 0.30 | ASGI server | |
| pydantic | ≥ 2.8 | Request/response schemas | |
| **React + Vite + TypeScript** | React 18, Vite 5 | Demo frontend | Real SPA; no full-script re-execution |
| Cytoscape.js | ≥ 3.30 | Concept-graph visualization with mastery colouring | |
| pytest | ≥ 8 | Tests | |

### 2.2 Dependencies deliberately dropped from the paper's stack

- **PyTorch Geometric** — the graph is ~250 nodes. A 3-layer GCN over a **dense normalized adjacency** is a 20-line module, avoids a heavy dependency, and sidesteps sparse-op gaps in the MPS backend. Dense 250×250 is 62,500 floats — free.
- **Stable-Baselines3** — we need **action masking** over the concept space (illegal actions = prerequisites unmet). A custom Double-DQN with masked Q-values is ~150 lines and gives exact control. SB3's DQN does not support masking without contrib workarounds.
- **RDFLib** — the paper lists it for "semantic relationship management" but describes no RDF/OWL/SPARQL work anywhere. Unused; removed.
- **Streamlit** — its execution model re-runs the entire script on every widget interaction, so a demo that holds a Transformer plus a DQN in memory becomes visibly sluggish in front of an audience. Replaced by a FastAPI inference service with a React SPA: the model loads once at process start, and each interaction is a sub-100 ms JSON call.

All removals are paper corrections, tracked in `docs/paper-corrections.md` (T3-9, T3-10).

### 2.2.1 Why DuckDB rather than pandas for ETL

The project's required skill set is **Python, Machine Learning, Data Mining, SQL**. SQL is load-bearing here rather than decorative:

- The entire preprocessing pipeline — sessionization, concept aggregation, feature engineering, split generation, cohort statistics — is written as **SQL queries against DuckDB**, stored as reviewable `.sql` files in `sql/`.
- DuckDB is columnar and vectorized, so a 10.6M-row aggregation runs faster than the pandas equivalent while staying embedded (no server to install or run).
- Every transformation becomes inspectable and auditable as SQL, which is materially better for a paper that must defend its preprocessing.
- Prerequisite-edge mining (§4.4) is expressed as SQL window functions and self-joins — genuine data mining in SQL, not Python loops.

Split of responsibilities:

| Store | Engine | Holds |
|---|---|---|
| Analytical | DuckDB (`data/oulad.duckdb`) | Raw tables, derived events, sequences, mined patterns, splits |
| Runtime | SQLite (`data/app.db`) | Recommendation logs, explanation records, user-study responses |

### 2.3 Hardware and compute profile

Target: Apple Silicon, MPS backend, CPU fallback.

- KG-DKT: ~1.5M parameters. Batch 64, seq 200, d=128. Estimated **minutes per epoch**, well under an hour for a full 5-fold run.
- DQN: ~200k parameters, but every environment step requires a KG-DKT forward pass. This is the compute bottleneck. Mitigations: incremental decoding with a KV cache, environment vectorization (32 parallel episodes), and mastery-vector caching.
- Dataset in memory: ~500 MB as Parquet-backed DataFrames.

No GPU is required. If a CUDA machine appears, only `device` changes.

---

## 3. Layer 1 — Data

### 3.1 OULAD source tables

| File | Rows (approx.) | Key columns used |
|---|---|---|
| `studentInfo.csv` | 32,593 | id_student, code_module, code_presentation, gender, region, highest_education, imd_band, age_band, num_of_prev_attempts, studied_credits, disability, final_result |
| `studentVle.csv` | ~10.6M | id_student, id_site, date, sum_click |
| `studentAssessment.csv` | ~174k | id_assessment, id_student, date_submitted, score, is_banked |
| `assessments.csv` | 206 | id_assessment, code_module, code_presentation, assessment_type, date, weight |
| `vle.csv` | ~6.4k | id_site, code_module, code_presentation, activity_type, week_from, week_to |
| `courses.csv` | 22 | code_module, code_presentation, module_presentation_length |
| `studentRegistration.csv` | 32,593 | date_registration, date_unregistration |

Exact counts are verified and emitted to `results/dataset_stats.json` in Stage 0 rather than asserted.

### 3.2 The central modelling problem, and the decision

Assessment events average roughly **5 per student**. A Transformer with sequence length 200 cannot be trained on sequences of length 5, and knowledge tracing on 5 events is meaningless.

VLE events are plentiful (~325 per student on average) but carry **no correctness signal** — only click counts.

**Decision: dual-signal sequences with masked supervision.**

A student's sequence is the chronological merge of two event kinds:

- **VLE events** — `(concept, day, sum_click)`. These are *context* tokens. They update the model's hidden state but **produce no loss term**.
- **Assessment events** — `(concept, day, score)`, labelled `correct = score ≥ 40` (the OULAD pass threshold). These are *supervised* tokens and are the only ones contributing to the loss and to AUC/RMSE.

Consequences:

- Sequences become long (hundreds of events), so `max_seq_len = 200` is genuinely appropriate — the paper's stated value is justified rather than arbitrary.
- Supervision remains genuine — no invented labels.
- AUC-ROC and RMSE are computed over assessment predictions only, which keeps them comparable to published knowledge-tracing results.
- Mastery estimates for concepts that never carry an assessment are **uncalibrated**, and reach their values through GCN propagation from prerequisite-linked concepts. This must be stated as a limitation in the paper — and it is precisely why the knowledge-graph ablation is meaningful rather than decorative.

### 3.3 Preprocessing pipeline — SQL-first

Every step below is a numbered SQL file in `sql/`, executed against DuckDB by a thin Python runner. Python handles only tensor construction.

| File | Step |
|---|---|
| `sql/00_load_raw.sql` | Ingest the 7 OULAD CSVs into DuckDB tables |
| `sql/01_dataset_stats.sql` | Real row counts, events per student, assessments per student, class balance |
| `sql/02_sessionize.sql` | Session boundaries at **30 minutes of inactivity**. OULAD timestamps are day-granular, so one student-day resolves to one session. **This must be stated in the paper** — the current draft implies precision the dataset lacks |
| `sql/03_concepts.sql` | Derive `(module, week-block)` concepts from `vle.week_from`/`week_to` |
| `sql/04_site_to_concept.sql` | Map every `id_site` to exactly one concept |
| `sql/05_events.sql` | Aggregate VLE to `(student, concept, day, clicks)`; union with assessment events binarized at `score ≥ 40`; median-impute missing scores |
| `sql/06_edges_structural.sql` | Week-precedence and assessment-coverage edges |
| `sql/07_edges_mined.sql` | Sequential-pattern and association-rule edges (§4.4) |
| `sql/08_features.sql` | One-hot gender, region, highest_education, imd_band, age_band, disability; numeric prev_attempts, studied_credits |
| `sql/09_sequences.sql` | Per-student chronological merge, windowed to 200 events with stride |
| `sql/10_splits.sql` | 80/10/10 stratified on `final_result`, **grouped by student**; 5 CV folds |
| `sql/11_cohorts.sql` | Subgroup cohort definitions for the fairness analysis |

**Artifacts:** `data/oulad.duckdb`, `data/processed/{events,sequences}.parquet`, `data/processed/splits.json`

---

## 4. Layer 2 — Concept graph

The paper's claimed graph cannot come from OULAD. This is how a real, reproducible, auditable graph is built instead.

### 4.1 Node derivation (algorithmic, not invented)

`vle.csv` gives each material a `code_module`, `activity_type`, and a `week_from`/`week_to` window. Module presentations run ~240–270 days ≈ 38 weeks.

**Concept = (module, week-block)**, where a week-block groups VLE sites whose week windows overlap. This yields roughly 30–40 concepts per module × 7 modules ≈ **210–280 concepts** — derived directly from OULAD metadata, with no invention.

Human-readable labels are assigned in `graph/concept_labels.yaml`. OULAD ships no material titles, so labels are curated; the mapping is version-controlled and inspectable.

### 4.2 Edge derivation

| Edge type | Rule | Source |
|---|---|---|
| `prerequisite` | week-block *w* → *w+1* within a module | Temporal precedence in OULAD |
| `prerequisite` | all blocks with `week_to ≤ assessment.date` → the assessed block | `assessments.csv` |
| `corequisite` | blocks sharing a week window, differing in activity type | `vle.csv` |
| `prerequisite` (mined) | directional sequential patterns above confidence and lift thresholds | §4.4, `sql/07_edges_mined.sql` |
| `corequisite` (mined) | FP-Growth association rules over per-student concept sets | §4.4 |
| `prerequisite` (cross-module) | curated overlay for shared foundational topics | `graph/cross_module_edges.yaml`, documented |

Expected scale: **~400–700 edges**. Emitted to `results/graph_stats.json` — the paper quotes that file, never a remembered number.

### 4.4 Data mining component

The required skill set names **Data Mining** explicitly, and mining is what makes the concept graph defensible rather than hand-asserted. Week-order alone is weak evidence of a prerequisite; observed learner behavior is stronger.

**A. Sequential pattern mining for prerequisite edges** (`sql/07_edges_mined.sql`)

For every ordered concept pair `(a, b)` where some student engaged with both:

```sql
precedence(a,b) = COUNT(students where first_touch(a) < first_touch(b))
                / COUNT(students who touched both)

lift(a,b)       = P(pass_b | touched_a_first) / P(pass_b)
```

An edge `a → b` is proposed when `precedence(a,b) ≥ 0.75`, `lift(a,b) > 1.1`, and support exceeds a minimum student count. High directional asymmetry in *when* concepts are engaged, combined with a measurable effect on downstream success, is exactly what a prerequisite looks like in behavioral data. Pure SQL: self-join plus window functions.

**B. Association rule mining for corequisite edges**

FP-Growth (mlxtend) over per-student concept-engagement itemsets. Rules with high confidence in both directions and no temporal asymmetry become corequisite edges.

**C. Learner profile clustering**

K-means (with silhouette-based *k* selection) over behavioral features — session count, click volume, regularity, assessment timeliness, forum participation. Produces 4–6 learner profiles used three ways: as a model feature, as a cold-start prior for students with no history, and inside explanations ("learners with your engagement profile typically…"). This also fills the paper's unaddressed cold-start gap.

**D. Cycle breaking**

Mined edges can produce cycles. Enforce the DAG by removing the lowest-confidence edge in each detected cycle, and log every removal to `results/graph_stats.json`.

**Ablation value:** the graph can be rebuilt from structural edges only, versus structural + mined. This turns the paper's `−knowledge graph` ablation into two distinct, more informative experiments.

### 4.3 Representation

- `networkx.DiGraph` for traversal, ancestor queries, and DAG validation
- Dense symmetric normalized adjacency `Â = D^-1/2 (A + I) D^-1/2` as a `torch.Tensor[C, C]` for the GCN
- Separate adjacency per edge type; the GCN sums typed messages

**Enforced by test:** the prerequisite graph is acyclic; every VLE `id_site` maps to exactly one concept; no orphan concepts.

**Artifacts:** `artifacts/graph/{concepts.yaml, edges.parquet, adjacency.pt, graph_stats.json}`

---

## 5. Layer 3 — KG-DKT knowledge tracing

### 5.1 Graph encoder

```
E ∈ R^{C×64}                     learned concept embedding table
for l in 1..3:                   (paper: 3 graph convolution layers)
    H = ReLU(LayerNorm(Â · H · W_l))
    H = dropout(H)
Z ∈ R^{C×64}                     prerequisite-aware concept embeddings
```

`Z` is recomputed each forward pass, so gradients flow into the graph structure. This is what makes the `−knowledge graph` ablation (replace `Z` with a free embedding table) a real experiment.

### 5.2 Interaction embedding

For event *t* with concept `c_t`, response `r_t ∈ {correct, incorrect, unlabelled}`, kind `k_t ∈ {vle, assessment}`:

```
x_t = W_c · Z[c_t] + W_r · emb(r_t) + W_k · emb(k_t) + W_f · f_t        → R^128
f_t = [ log(1 + clicks), Δt_prev, Δt_since_same_concept, day/course_length ]
```

Static student features are projected to `R^128` and prepended as a context token, so demographics condition the whole sequence.

### 5.3 Time-aware Transformer

2 layers, 4 heads, `d_model = 128`, causal mask, max length 200 — the paper's configuration exactly.

**Forgetting mechanism** — additive attention bias:

```
bias[h, t, s] = −γ_h · log(1 + Δdays(t, s))          γ_h ≥ 0, learned per head
attn_logits = QKᵀ/√d_k + bias + causal_mask
```

Older interactions are attenuated, with a learned per-head decay rate. This is the concrete realization of the paper's claimed "time-aware attention" and "forgetting effects" — currently claimed but never specified. The `−temporal` ablation sets `γ = 0` and drops the `Δt` features.

### 5.4 Heads

```
predict_next(h_t, c) = σ( MLP( [h_t ; Z[c]] ) )       → P(correct on concept c)
mastery(h_t)          = [ predict_next(h_t, c) for c in 1..C ]   → R^C
```

The mastery readout is one batched matrix operation over all `C ≈ 250` concepts — cheap enough to call every RL environment step.

**Loss:** binary cross-entropy over assessment-labelled positions only. Optimizer Adam, lr 1e-3, cosine annealing, weight decay 1e-4, batch 64 — the paper's values.

**Metrics:** AUC-ROC, RMSE on held-out students, mean ± std over 5 folds.

### 5.4.1 Probability calibration

Mastery values are not internal scores here — they drive the RL reward, they gate the prerequisite action mask, and they are **shown to users inside explanations** ("Stack Frames is 0.41"). An uncalibrated 0.41 shown to a student is a false statement, and the paper's central claim is trust.

- Fit **temperature scaling** on the validation fold; keep isotonic regression as a fallback if reliability is non-monotonic
- Report **Expected Calibration Error** and a reliability diagram alongside AUC/RMSE
- Calibrate before the mastery vector reaches the planner or the explainer

This is roughly half a day of work and converts "trustworthy explanations" from an assertion into something measured.

### 5.4.2 Cold start

Students with little or no history get a **learner-profile prior** from the Stage 0 clustering (§4.4C) instead of an uninformative default. Evaluation reports AUC at 0, 1, 5, 10, and 20 prior interactions as a curve.

The paper's related-work section criticizes cold start in prior systems and never addresses its own. This closes that gap with a figure.

### 5.5 Baselines

| Baseline | Definition | Purpose |
|---|---|---|
| **DKT** | Piech et al. LSTM, one-hot `2C` input, hidden 128 | Paper's Table I row 1 |
| **RL-DKT** | DKT + DQN planner, no graph, no time bias | Paper's Table I row 2 |
| **GNN-based** | GCN concept embeddings + LSTM, no time bias | Paper's Table I row 3 |
| **Collaborative filtering** | Item-item CF over the student × concept engagement matrix | Fixes the abstract's unfulfilled promise (T2-2) |
| **SAKT** | Self-Attentive Knowledge Tracing (Pandey & Karypis, 2019) | **Added.** A knowledge-tracing reviewer will immediately ask why a Transformer KT model is compared only against LSTM-DKT. Without an attention-based KT baseline, the comparison looks selected to flatter |
| **Majority-class / no-skill** | Predicts the base rate | **Added.** Proves AUC is not an artifact of class imbalance |
| **BKT** *(optional)* | Bayesian Knowledge Tracing | Classic reference point; include if time allows |

All baselines get the same tuning budget, recorded in `results/tuning_budget.json`, because reviewers assume unfair tuning by default.

**Artifacts:** `artifacts/models/kgdkt_fold{k}.pt`, `results/table1.json`

---

## 6. Layer 4 — Planning

### 6.1 Environment (`LearningPathEnv`, gymnasium.Env)

The paper says "the recommendation challenge is formulated as an MDP" and then never formalizes it. Here is the formalization the paper is missing.

**State** `s_t ∈ R^{2C+k+1}`:

```
[ mastery vector m_t ∈ R^C
| coverage mask ∈ {0,1}^C          concepts already recommended this episode
| student static features ∈ R^k
| remaining step budget ∈ R ]
```

**Action** `a_t ∈ {1..C}` — the next concept to study.

**Action mask** — `a` is legal only if `m_t[a] < τ_mastered` and every prerequisite `p` of `a` satisfies `m_t[p] ≥ τ_prereq`. Masking encodes the prerequisite constraint structurally instead of hoping the agent learns it, and it shrinks the effective action space enormously.

**Transition** — append a simulated interaction with concept `a_t`, sampling the response from `Bernoulli(m_t[a_t])`, re-run KG-DKT, read the new mastery vector. **The frozen KG-DKT model is the student simulator.** No offline alternative exists; the paper must state this plainly.

**Reward:**

```
r_t = α · Σ_c max(0, m_{t+1}[c] − m_t[c])     total mastery gain, including transfer
    + β · 1[prerequisites of a_t satisfied]    pedagogical soundness
    − δ · 1[m_t[a_t] ≥ τ_mastered]             redundancy penalty
    − η                                        step cost, favours shorter paths
```

Defaults `α=1.0, β=0.2, δ=0.5, η=0.05`, all in `configs/rl.yaml` and reported in the paper.

**Termination** — target concept set all above `τ_mastered`, or step budget exhausted.

**Episode initialization** — sample a real student prefix from the training split, so simulated rollouts start from genuine learner states.

### 6.2 GreedyPlanner — the fallback that guarantees a demo

```
recommend(s, k) = top-k legal actions by one-step predicted total mastery gain
```

One-step lookahead through the frozen KG-DKT. A beam variant searches depth 3.

Three roles: the demo's guaranteed backend, a legitimate non-RL baseline row in Table II, and the bar the DQN must clear to justify itself.

### 6.3 DQNPlanner

- Network: `state_dim → 256 → 128 → 64 → C`, ReLU — the paper's stated shape
- Double DQN, target network synced every 1,000 steps
- Replay buffer 100,000 — the paper's stated value
- ε-greedy, 1.0 → 0.05 over 50,000 steps
- γ = 0.95
- Illegal actions masked to `−inf` before the argmax and before the target's max

**Shared interface.** `GreedyPlanner` and `DQNPlanner` both implement `Planner.recommend(state, k) -> list[int]`. The demo and the evaluation harness are agnostic to which is loaded. This is the structural guarantee that DQN failure cannot break the demo.

### 6.4 Off-policy evaluation — the answer to the circularity objection

The sharpest reviewer objection available against this paper is: *the DKT model trains the agent and then also scores it, so the reported path quality measures the simulator's agreement with itself.*

Simulator-only evaluation cannot answer this. **Off-policy evaluation on logged OULAD trajectories can.**

Treat each student's observed concept sequence as a logged trajectory generated by an unknown behavior policy. Estimate the learned policy's value without ever running it:

| Estimator | Role |
|---|---|
| **Inverse Propensity Scoring (IPS)** | Unbiased under a modelled behavior policy; high variance |
| **Self-normalized IPS (SNIPS)** | Variance-reduced, the practical default |
| **Doubly Robust (DR)** | Combines IPS with the model's value estimate; robust when either component is right |

The behavior policy is estimated from data (a simple next-concept classifier over the observed sequences). Propensity clipping bounds the variance; effective sample size is reported alongside every estimate.

Reporting IPS/SNIPS/DR **next to** the simulated completion rate turns the weakest part of the evaluation into a strength: the two families of estimate are independent, and agreement between them is genuine evidence.

### 6.5 Reward-term ablation

The reward has four named terms, so each is ablated individually (α, β, δ, η set to zero in turn). This shows the reward design is principled rather than tuned until the numbers looked good — a question reviewers ask about every hand-designed reward.

**Artifacts:** `artifacts/models/dqn.pt`, `results/table2.json`, `results/ope.json`, `results/reward_ablation.json`

---

## 7. Layer 5 — Explanation

Runs **after** the planner selects — post-hoc, as the abstract says. The methodology section's claim to the contrary must be corrected (T2-1).

### 7.1 Priority backtracking interpreter

For recommended concept `c*`:

1. Traverse `ancestors(c*)` in the prerequisite DAG (depth-first).
2. For each ancestor `a`, compute deficit `max(0, τ_prereq − m[a])`.
3. Rank ancestors by deficit × path weight; retain the top-k.
4. Emit satisfied prerequisites as supporting evidence and unmet ones as gaps.

### 7.2 Reward decomposition

The reward is a sum of named terms by construction, so each term's contribution to the chosen action's value is reported **exactly** — no approximation needed, unlike SHAP over Q-values.

### 7.3 SHAP over student features

`shap` (or captum) attributions over the static student feature vector for the mastery prediction on `c*`. Answers "which of my characteristics drove this?" — and doubles as the input to the fairness analysis the paper needs (T1-8).

### 7.4 Counterfactual

Recompute the forward pass with `m[a] := 1.0` for the top unmet prerequisite:

> "If Stack Frames were mastered, predicted gain on Recursion rises from 0.11 to 0.34."

Cheap, and the single most convincing thing in a live demo.

### 7.5 Renderer

```
Explanation(
  concept, prereq_satisfied[], prereq_gaps[],
  reward_terms{}, shap_values{}, counterfactual{}, text
)
```

Rendered to templated pedagogical prose:

> Recommended **Recursion** because *Loops* mastery is 0.82 (met) but *Stack Frames* is 0.41 (gap). Closing this gap predicts +0.23 mastery gain toward your Data Structures goal.

### 7.6 Objective explainability metrics

The paper evaluates its headline contribution by Likert survey alone (T2-4). These three make it quantitative:

| Metric | Definition |
|---|---|
| **Fidelity** | Correlation between cited prerequisite deficits and actual counterfactual gain when those deficits are removed |
| **Sufficiency** | Fraction of the recommendation's Q-advantage recovered by the top-k cited factors alone |
| **Stability** | Jaccard overlap of cited factors under small perturbations of the mastery vector |

**Artifacts:** `results/explainability.json`

---

## 8. Data flow

```
OULAD CSVs
    │
    ▼  01_prepare_data.py   →  runs sql/00–05, sql/08–11
data/oulad.duckdb  +  data/processed/events.parquet ─┐
    │                                                │
    ▼  02_build_graph.py    →  runs sql/06–07         │
artifacts/graph/{concepts.yaml,adjacency.pt}          │
results/graph_stats.json, results/mined_rules.json    │
    │                                                 │
    ├──────────────────┬──────────────────────────────┘
    ▼                  ▼  03_build_sequences.py  →  sql/09
    │          data/processed/sequences.parquet + splits.json
    │                  │
    └────────┬─────────┘
             ▼  04_train_kt.py   (5 folds × {proposed, DKT, RL-DKT, GNN, CF})
    artifacts/models/kgdkt_fold{k}.pt
    results/table1.json                    ← AUC-ROC, RMSE, mean ± std, p-values
             │
             ▼  05_train_rl.py   (frozen KG-DKT = simulator)
    artifacts/models/dqn.pt
    results/table2.json                    ← completion, NDCG, knowledge gain
             │
             ▼  06_eval_explain.py
    results/explainability.json            ← fidelity, sufficiency, stability
             │
             ▼  07_ablations.py            (−KG, −temporal, −explain)
    results/ablations.json
             │
             ▼  08_make_figures.py
    results/figures/*.png
    results/tables/*.tex                   ← pasted directly into the paper
             │
             ▼
    api/  (FastAPI, loads artifacts/ once at startup)  ←──  web/  (React SPA)
```

**Runtime request path in the demo:**

```
browser: GET /api/students?q=…                → student list (SQL over DuckDB)
browser: GET /api/students/{id}/mastery       → mastery vector m ∈ R^C   (cached)
browser: POST /api/recommend {id, k, planner} → concepts + explanations
              server-side:
                load sequence  → KG-DKT forward → m
                Planner.recommend(state, k)     → concepts
                Explainer.explain(state, c)     → Explanation per concept
                log to SQLite app.db
browser: POST /api/counterfactual {id, concept, assume_mastered}
                                              → recomputed gain
```

The model is loaded once into the FastAPI process and mastery vectors are cached per student, so interactions are sub-100 ms. Nothing re-executes a script per widget change.

**API endpoints** (these also constitute the "RESTful API for LMS integration" the paper lists as future work):

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/students` | Search and page students |
| GET | `/api/students/{id}` | Profile, demographics, learner cluster |
| GET | `/api/students/{id}/mastery` | Per-concept mastery vector |
| GET | `/api/graph` | Concept graph nodes and edges for visualization |
| POST | `/api/recommend` | Top-k recommendations with explanations |
| POST | `/api/counterfactual` | "What if this prerequisite were mastered?" |
| GET | `/api/metrics` | Live results from `results/*.json` |

---

## 9. Module interfaces

Contracts fixed up front so the layers stay independently testable.

```python
class ConceptGraph:
    n_concepts: int
    def prereqs(self, c: int) -> list[int]: ...
    def ancestors(self, c: int) -> list[int]: ...
    def adjacency(self, edge_type: str) -> torch.Tensor: ...   # [C, C] dense normalized
    def is_dag(self) -> bool: ...
    def label(self, c: int) -> str: ...

class KnowledgeTracer(Protocol):
    def mastery(self, seq: Sequence) -> np.ndarray: ...        # [C]
    def predict_next(self, seq: Sequence, c: int) -> float: ...
    def step(self, seq: Sequence, c: int, correct: bool) -> Sequence: ...

class Planner(Protocol):
    def recommend(self, state: State, k: int = 1) -> list[int]: ...

class Explainer:
    def explain(self, state: State, action: int) -> Explanation: ...

class LearningPathEnv(gymnasium.Env):
    def reset(self, *, student_id: int | None = None) -> tuple[State, dict]: ...
    def step(self, action: int) -> tuple[State, float, bool, bool, dict]: ...
    def action_mask(self) -> np.ndarray: ...                   # [C] bool
```

`GreedyPlanner` and `DQNPlanner` both satisfy `Planner`. Nothing downstream knows which is loaded.

---

## 10. Repository layout

```
elpr/
  db/         duckdb_runner.py  app_store.py        # SQL execution, SQLite runtime store
  data/       oulad.py  sequences.py  features.py   # thin wrappers over sql/
  mining/     sequential.py  association.py  clustering.py
  graph/      build.py  concept_graph.py  gcn.py
  models/     kgdkt.py  baselines.py  layers.py
  planner/    env.py  greedy.py  dqn.py  state.py
  explain/    backtrack.py  attribution.py  counterfactual.py  render.py  metrics.py
  eval/       metrics.py  ablations.py  stats.py  fairness.py  emit.py
sql/          00_load_raw.sql … 11_cohorts.sql      # the ETL, reviewable as SQL
configs/      data.yaml  model.yaml  rl.yaml  mining.yaml  eval.yaml
graph/        concept_labels.yaml  cross_module_edges.yaml
scripts/      01_prepare_data.py … 08_make_figures.py
api/          main.py  routes.py  schemas.py  service.py
web/          src/  index.html  vite.config.ts  package.json
tests/        test_sql.py  test_graph.py  test_mining.py  test_sequences.py
              test_kgdkt.py  test_env.py  test_explain.py  test_metrics.py
data/         raw/  processed/  oulad.duckdb  app.db
artifacts/    graph/  models/
results/      figures/  tables/  *.json
docs/         architecture.md  build-plan.md  paper-corrections.md  superpowers/specs/
Makefile
```

`make all` regenerates every table and figure. `make api` starts the inference service. `make web` starts the Vite dev server. `make demo` runs both.

---

## 11. Build order

| Stage | Scope | Ends with |
|---|---|---|
| 0 | L1 + L2 | Dataset and graph statistics, DAG validated by test |
| 1 | L3 + baselines | Table I, real, with mean ± std and p-values |
| 2 | L5 + GreedyPlanner | Working recommendations with explanations — demoable |
| 3 | L4 DQN | Table II, DQN vs greedy |
| 4 | Demo + eval harness | `make all`, `make demo`, all figures |
| 5 | User study | n=30 Likert, real data for Section V-C |

Stopping after Stage 2 still yields a working demo and a real Table I.

---

## 12. Risks

| Risk | Signal | Mitigation |
|---|---|---|
| DQN degenerates to a constant policy | Action entropy collapses | GreedyPlanner backs the demo; report DQN failure as an ablation finding |
| Knowledge tracing AUC near chance | Fold AUC < 0.6 | Dual-signal sequences (§3.2); if still weak, report honestly and analyze |
| Mastery uncalibrated on assessment-free concepts | Mastery saturates near a constant | Known consequence of §3.2; stated as a limitation; the KG ablation quantifies it |
| Simulator is circular — DKT trains the agent and scores it | — | Also evaluate NDCG against real successful-student trajectories, which the simulator cannot fake |
| MPS op gaps | Runtime fallback warnings | Dense adjacency, no sparse ops, CPU fallback path |
| Curated labels look arbitrary | Reviewer objection | Nodes are algorithmically derived; only labels are curated, and both are version-controlled |

**The circularity risk is the one a sharp reviewer will find.** The defense is that Table I (AUC/RMSE) is measured against real held-out student outcomes, and NDCG is measured against real successful-student trajectories. Only the simulated completion rate and knowledge gain depend on the simulator, and both must be labelled *simulated* in the paper.
