# Master Build Plan

**Date:** 2026-08-09 (revision 2 — SQL/data-mining stack, FastAPI+React demo, evaluation hardening)
**Companion:** [`docs/architecture.md`](architecture.md), [`docs/paper-corrections.md`](paper-corrections.md)

---

## 0. Shape of the system

```
┌──────────────────────────────────────────────────────────────────────────┐
│  L0  STORAGE          DuckDB (analytical)  ·  SQLite (runtime)           │
├──────────────────────────────────────────────────────────────────────────┤
│  L1  SQL ETL          sql/00–11 · sessionize · concepts · features       │
│                       · splits · fairness cohorts                        │
├──────────────────────────────────────────────────────────────────────────┤
│  L2  DATA MINING      sequential prerequisite mining · FP-Growth         │
│                       corequisites · k-means learner profiles            │
│                       · cycle breaking → concept DAG                     │
├──────────────────────────────────────────────────────────────────────────┤
│  L3  KNOWLEDGE        KG-DKT: 3-layer GCN → concept embeddings           │
│      TRACING          → 2-layer/4-head Transformer + time-aware bias     │
│                       → mastery head → temperature calibration           │
├──────────────────────────────────────────────────────────────────────────┤
│  L4  PLANNING         GreedyPlanner (always works)                       │
│                       DQNPlanner (masked Double-DQN, DKT-simulated env)  │
├──────────────────────────────────────────────────────────────────────────┤
│  L5  EXPLANATION      priority backtracking · reward decomposition       │
│                       · SHAP · counterfactual · templated prose          │
├──────────────────────────────────────────────────────────────────────────┤
│  L6  EVALUATION       5-fold CV · Holm-corrected t-tests · effect sizes  │
│                       · calibration · cold-start · off-policy evaluation │
│                       · fairness · sensitivity · efficiency              │
├──────────────────────────────────────────────────────────────────────────┤
│  L7  SERVING          FastAPI (model in memory) → React/Vite SPA         │
└──────────────────────────────────────────────────────────────────────────┘
```

| Stage | Name | Effort | Depends on | Fills |
|---|---|---|---|---|
| 0 | Data foundation, SQL ETL, mined graph | 3–4 d | — | III-B, IV-B |
| 1 | Knowledge tracing, baselines, calibration | 5–7 d | 0 | **Table I** |
| 2 | Explainability + greedy planner | 3–4 d | 1 | III-C, Fig. 4, V-C |
| 3 | RL + off-policy evaluation | 6–9 d | 1, 2 | **Table II**, Fig. 3 |
| 4 | Demo, harness, hardening | 4–5 d | 1–3 | Figs 1–6, V-D/E |
| 5 | User study + expert edge validation | 1 d work / 1 wk elapsed | 2 | IV-D, V-C, ethics |
| 6 | Paper integration | 4–6 d | all | whole paper |

Total: roughly **5–7 weeks part-time**.

---

## What changed in revision 2

**Stack changes (from your requirements):**

1. **DuckDB + SQL** is the ETL engine; the whole pipeline is reviewable `.sql` files
2. **SQLite** as the runtime application store
3. **Data mining** promoted to a first-class layer — mined prerequisites, association rules, learner clustering
4. **Streamlit removed**; FastAPI inference service + React/Vite/TypeScript SPA

**Evaluation hardening (new — each preempts a specific reviewer objection):**

| # | Addition | Objection it preempts | Effort |
|---|---|---|---|
| 5 | **Probability calibration** (temperature scaling, ECE, reliability diagram) | "You show students a mastery of 0.41 — is that number meaningful?" | 0.5 d |
| 6 | **Off-policy evaluation** (IPS / SNIPS / Doubly Robust) | "Your DKT trains the agent *and* scores it. This is circular." | 1 d |
| 7 | **SAKT baseline** + no-skill baseline | "Why is your Transformer compared only to an LSTM?" | 1 d |
| 8 | **Holm-Bonferroni correction + Cohen's d** | "Four baselines × several metrics, uncorrected. And is the gain meaningful or just significant?" | 2 h |
| 9 | **Cold-start curve** (AUC at 0/1/5/10/20 interactions) | "You criticize cold start in related work and never address your own." | 0.5 d |
| 10 | **Reward-term ablation** (α, β, δ, η each zeroed) | "Your reward has four hand-picked terms. Did you tune until it looked good?" | 0.5 d |
| 11 | **Threshold sensitivity** (τ_mastered, τ_prereq, mining thresholds) | "Why 0.75? Why 1.1? How fragile is this?" | 0.5 d |
| 12 | **Efficiency benchmark** (p50/p95 latency, params, explanation overhead) | Your paper claims an interpretability/efficiency tradeoff with two unsourced numbers. | 0.5 d |
| 13 | **Expert validation of mined edges** (50 edges, 3 raters, precision + agreement) | "Your knowledge graph is the weakest link. Who says these prerequisites are real?" | folds into Stage 5 |
| 14 | **Advisor override in the demo** | Your abstract promises tools for academic advisers; nothing in the system serves an adviser. | 0.5 d |
| 15 | **Provenance manifest + Docker + lockfile** | "Data and code availability?" | 0.5 d |

Net addition: roughly **6 days** for items that otherwise cost you the paper.

---

## Stage 0 — Data foundation, SQL ETL, mined concept graph

**Goal:** every downstream stage loads clean artifacts and never touches a raw CSV. The whole ETL is SQL, reviewable file by file.

### Work

1. Scaffold — `pyproject.toml`, `Makefile`, `configs/*.yaml`, package skeleton, pytest, ruff
2. Acquire OULAD; verify integrity by checksum and row count rather than trusting documentation
3. Load all 7 tables into **DuckDB** (`data/oulad.duckdb`); Parquet exports for tensor construction
4. `elpr/db/duckdb_runner.py` — thin runner executing numbered `.sql` files in order, with schema assertions
5. **`sql/00`–`sql/05`** — raw load, dataset statistics, sessionization, concept derivation, site→concept mapping, unified event table
6. Emit **real** statistics to `results/dataset_stats.json`. This verifies or corrects every dataset number quoted in the paper
7. Sessionize at 30 minutes of inactivity. **OULAD timestamps are day-granular**, so one student-day resolves to one session — record the limitation
8. Concepts = `(module, week-block)` from `vle.week_from`/`week_to`
9. **`sql/06`** — structural edges: week precedence, assessment coverage, same-week corequisites
10. **`sql/07`** — mined edges, in SQL:
    - **Sequential prerequisite mining** — `precedence(a,b)` from first-touch ordering, `lift(a,b)` on downstream pass rate; propose `a → b` at precedence ≥ 0.75, lift > 1.1, support above a floor
    - **FP-Growth** (mlxtend) over per-student concept itemsets → corequisite edges
11. **Cycle breaking** — drop the lowest-confidence edge per detected cycle; log every removal
12. **Learner-profile clustering** — k-means over session count, click volume, regularity, assessment timeliness, forum participation; *k* by silhouette. Feeds the model, the cold-start prior, and explanations
13. `graph/concept_labels.yaml`, `graph/cross_module_edges.yaml`
14. `ConceptGraph` — networkx DiGraph, ancestor queries, dense normalized adjacency per edge type
15. **`sql/08`–`sql/11`** — features, sequence assembly (VLE context + assessment supervised, windowed to 200 with stride), stratified grouped splits with 5 folds, fairness cohorts
16. SQLite runtime schema (`data/app.db`)
17. Concept-graph figure

### Tests

- DAG acyclic after cycle breaking · every `id_site` maps to exactly one concept · no orphans
- Adjacency symmetric and correctly normalized
- No student ID in two splits · sequence chronology monotonic · supervised positions labelled, context positions not
- Every SQL file's output schema and row-count invariants
- Mined edges satisfy their stated confidence, lift, and support thresholds
- Clustering stable across seeds (adjusted Rand index above threshold)

### Artifacts

`data/oulad.duckdb`, `data/app.db`, `data/processed/{events,sequences}.parquet`, `data/processed/splits.json`,
`artifacts/graph/{concepts.yaml,edges.parquet,adjacency.pt}`,
`results/{dataset_stats,graph_stats,mined_rules,learner_profiles}.json`, `results/figures/concept_graph.png`

### Exit

Tests green. Real statistics emitted. Can print any student's full labelled sequence.

---

## Stage 1 — Knowledge tracing, baselines, calibration

**Goal:** Table I becomes real. Highest paper value per unit of risk — supervised learning either converges or visibly fails within minutes.

### Work

1. `layers.py` — GCN block, time-aware attention bias, interaction embedding
2. `kgdkt.py` — 3-layer GCN → concept embeddings → 2-layer/4-head Transformer with time bias → mastery head
3. Training loop — Adam, lr 1e-3, cosine annealing, weight decay 1e-4, batch 64, early stopping, seeded
4. **Loss masking** — BCE over assessment positions only; VLE positions contribute nothing
5. **Calibration** — temperature scaling fitted on validation, isotonic fallback; report ECE and a reliability diagram; calibrate before mastery reaches the planner or explainer
6. **Cold-start evaluation** — AUC at 0/1/5/10/20 prior interactions, using the learner-profile prior at 0
7. Baselines: **DKT** (Piech LSTM) · **SAKT** (self-attentive KT) · **GNN-based** (GCN+LSTM) · **collaborative filtering** · **no-skill** majority baseline · *BKT optional*
8. *RL-DKT's row is deferred to Stage 3* — its tracer trains jointly with the TD loss
9. 5-fold CV with mean ± std for every metric
10. **Statistics** — paired t-tests, **Holm-Bonferroni correction** across the comparison family, **Cohen's d** effect sizes
11. Hyperparameter search — GCN layers {2,3,4}, heads {2,4,8}, dropout, lr; **identical budget for baselines**, recorded to `results/tuning_budget.json`
12. Per-module AUC breakdown alongside the pooled number
13. Emit `results/table1.json` + `results/tables/table1.tex`

### Tests

- Shapes for every module · causal mask blocks all future positions
- Loss ignores unlabelled positions (gradient check)
- Time bias monotonically decreasing in Δt
- Calibrated probabilities improve ECE on held-out data
- Fixed seed reproduces identical loss

### Exit

Table I real for 5 of 6 rows, with mean ± std, corrected p-values, and effect sizes. Calibration and cold-start figures emitted.

---

## Stage 2 — Explainability + greedy planner

**Goal:** a working, explainable recommender **before** any RL exists. Demo insurance, and the paper's actual novelty.

### Work

1. `state.py` — `State`: mastery vector, coverage mask, student features, learner cluster, budget
2. `greedy.py` — top-k legal actions by one-step predicted total mastery gain; depth-3 beam variant
3. `backtrack.py` — priority backtracking: DFS over `ancestors(c*)`, deficit `max(0, τ − m[a])`, ranked by deficit × path weight
4. `counterfactual.py` — recompute with `m[a] := 1.0` for the top unmet prerequisite
5. `attribution.py` — SHAP over static student features; also feeds the Stage 4 fairness analysis
6. `render.py` — `Explanation` → templated pedagogical prose, including the learner-profile clause
7. `explain/metrics.py` — the objective measures the paper entirely lacks:
   - **Fidelity** — correlation between cited deficits and actual counterfactual gain
   - **Sufficiency** — fraction of advantage recovered by the top-k cited factors
   - **Stability** — Jaccard overlap of cited factors under mastery perturbation
8. **Failure-case gallery** — sample recommendations the model gets wrong, saved for the paper's limitations section and for the demo
9. `scripts/06_eval_explain.py` → `results/explainability.json`

### Tests

- Backtracking returns only genuine ancestors · deficit ranking correct
- Counterfactual gain monotonic in the removed deficit · stability bounded in [0,1]
- Greedy never returns an action violating the prerequisite mask

### Exit

Any student → 5 ranked recommendations with gaps, counterfactuals, and readable prose. **Fully demoable with zero RL.**

---

## Stage 3 — Reinforcement learning + off-policy evaluation

**Goal:** Table II. Highest-risk stage, deliberately last, greedy already banked as fallback and baseline.

### Work

1. `env.py` — `LearningPathEnv(gymnasium.Env)`: state assembly, action masking, transition via frozen calibrated KG-DKT, four-term reward, dual termination
2. Episodes initialized from real student prefixes in the training split
3. **Performance work — the real bottleneck.** Every env step is a KT forward pass: KV-cache incremental decoding, 32 vectorized envs, mastery caching
4. `dqn.py` — Double DQN, `state_dim → 256 → 128 → 64 → C`, replay 100k, ε 1.0→0.05 over 50k steps, target sync every 1k, γ=0.95, illegal actions masked to `−inf` before every argmax **and** every target max
5. **RL-DKT baseline** — LSTM tracer trained jointly with the TD loss; fills its deferred Table I row and its Table II row
6. Path metrics — simulated completion rate (labelled *simulated*), **NDCG against real successful-student next-concepts**, defined and normalized knowledge gain
7. **Off-policy evaluation** — behavior policy estimated from logged sequences; **IPS, SNIPS, Doubly Robust** estimates with propensity clipping and effective sample size reported. This is the answer to the circularity objection
8. **Reward-term ablation** — α, β, δ, η zeroed in turn
9. Diagnostics — action entropy, Q-value distribution, mode-collapse detector, reward-term breakdown over training

### Tests

- Masked actions never selected, in any state
- Each reward term's sign and trigger condition
- Both termination paths reachable
- Seeded rollouts deterministic
- OPE estimators recover known values on a synthetic logged dataset

### Exit

Table II real, with simulated **and** off-policy estimates side by side. **Either outcome acceptable:** DQN beats greedy → contribution confirmed; DQN does not → honest ablation finding, demo unaffected.

---

## Stage 4 — Demo, harness, hardening

**Goal:** one command regenerates the paper; one command launches the presentation.

### Work

1. **FastAPI service** (`api/`) — model loaded once at startup, mastery cached per student, seven endpoints, sub-100 ms responses. Doubles as the "RESTful API for LMS integration" the paper lists as future work
2. **React + Vite + TypeScript SPA** (`web/`) — student picker; Cytoscape.js concept graph coloured by mastery; recommended path; explanation cards with gaps and counterfactuals; live greedy ↔ DQN toggle
3. **Advisor override** — mark a concept as already known and watch the path rebuild live. This is the only artifact that supports the abstract's claim to serve academic advisers
4. Path export to PDF from the demo
5. Recommendation and explanation logging to SQLite; also supplies the Stage 5 stimulus
6. `ablations.py` — −knowledge graph (free embeddings) · structural-only vs structural+mined graph · −temporal attention (γ=0, no Δt) · −calibration · −explainability (trust delta from Stage 5)
7. **Fairness analysis** — subgroup AUC and recommendation quality by gender, disability, IMD band, age band, over the `sql/11` cohorts
8. **Threshold sensitivity** — sweep τ_mastered, τ_prereq, mining precedence and lift; one figure
9. **Efficiency benchmark** — p50/p95 latency, throughput, parameter counts, and the explanation module's overhead measured separately. Replaces the paper's two unsourced latency numbers
10. `make_figures.py` — all six figures · LaTeX table emitter
11. `Makefile` — `all`, `api`, `web`, `demo`, `test`
12. **Provenance** — every results file stamped with git SHA, config hash, seed; Dockerfile and a pinned lockfile for the data/code availability statement

### Exit

`make all` from a clean checkout reproduces every number. `make demo` launches API + SPA from checkpoints. Every paper number traces to a file.

---

## Stage 5 — User study + expert edge validation

**Goal:** convert the paper's softest table into collected data, and validate the graph — its weakest link.

### Work

1. Protocol, informed consent, department/IRB approval — **Tier-1 missing content (T1-5)**
2. Questionnaire — genuine 5-point Likert on transparency, trust, pedagogical utility, plus free text. The stray `/7` value must not recur
3. Stimulus — real explanations from the demo, two conditions: with and without explanation
4. Recruit 30 CS/engineering students; record demographics
5. Analyze — means, std, paired test between conditions, effect size
6. **Expert validation of mined edges** — sample 50 mined prerequisite edges, 3 raters, report precision and inter-rater agreement (Cohen's κ). Cheap, and it is the only direct evidence that the mined graph is real
7. Emit `results/user_study.json`, `results/edge_validation.json`

### Exit

Real Likert values with std and a significance test. Mined-edge precision with agreement statistics. Ethics statement drafted.

---

## Stage 6 — Paper integration

**Goal:** no number in the paper without provenance in `results/`.

### Work

1. Replace every number in Tables I and II, ablations, user study, and conclusion from `results/`
2. Add the missing mathematics — MDP tuple, reward function, DKT formulation, GCN propagation, time-aware attention, combined loss with lambda (T1-4)
3. Algorithm blocks — training loop, priority backtracking interpreter
4. Work `paper-corrections.md` top to bottom: T1 → T2 → T3
5. Add missing sections — contributions list, formal problem statement, complexity analysis, limitations (with the failure-case gallery), threats to validity, ethics statement, data/code availability, fairness results, cold-start results
6. **New results sections earned by revision 2** — calibration, off-policy evaluation, sensitivity, efficiency, mined-edge validation
7. Rebuild references — authors for every entry, DOIs/arXiv IDs, correct OULAD citation, deduplicate [1]/[11], repair every citation-to-claim mismatch
8. Fix the template — letter-spacing, section ordering, table titles, figure placement
9. Final consistency pass — every number cross-checked against its results file

### Exit

Every quantitative claim traceable. Zero T1 and T2 items outstanding.

---

## Critical path

```
0 ──► 1 ──► 2 ──► 3 ──► 4 ──► 6
            └──► 5 ──────────┘
```

Stage 5 runs parallel to Stages 3–4 once Stage 2 produces real explanations.

## Fallback checkpoints

| Stop after | You still have |
|---|---|
| Stage 1 | Real Table I with calibration, cold-start, corrected statistics — a defensible short paper |
| Stage 2 | Plus a working explainable recommender demo |
| Stage 3 | Plus Table II, off-policy evaluation, and the full claimed architecture |
| Stage 4 | Plus reproducibility, fairness, sensitivity, and efficiency |
