# Explainable AI-Based Personalized Learning Path Recommendation System — Design Spec

**Date:** 2026-08-09
**Status:** Awaiting user review
**Paper:** `Explainable_AI_Based_Personalized_Learning_Path_Recommendation_System_for_University_Students_updated.pdf`

---

## 1. Goal

Build a working implementation of the system described in the paper, such that:

1. Every number in the paper's results section is produced by code in this repository, not asserted.
2. The system can be demonstrated live to a supervisor and classmates (pick a student, see traced mastery, see a recommended learning path, see the pedagogical explanation for each step).
3. The paper's Methodology and Implementation sections become literally true descriptions of this code.

Non-goal: deployment, LMS integration, multi-user auth, REST API. The paper lists these as future work; they stay future work.

---

## 2. Dataset reality (drives everything else)

**Dataset:** Open University Learning Analytics Dataset (OULAD), as the paper states.

What OULAD actually contains:

- ~32,593 students, 7 modules, 22 module-presentations
- ~10.6M VLE clickstream rows across ~20 activity types (`oucontent`, `quiz`, `forumng`, `resource`, `subpage`, `url`, `glossary`, …)
- ~174k student-assessment rows — roughly **5 assessment interactions per student on average**
- Demographics: gender, region, highest education, IMD band, age band, prior attempts, studied credits, disability
- Final result: Pass / Fail / Withdrawn / Distinction

What OULAD does **not** contain, and the paper assumes it does:

- Concept or knowledge-component annotations
- Prerequisite relationships between concepts
- Learning-outcome mappings
- Any curriculum graph

### 2.1 Interaction unit — key design decision

Assessment-level sequences average ~5 events per student. That is far too short for a Transformer with sequence length 200, and too sparse for knowledge tracing.

**Decision:** model interactions at **VLE-activity level**, projected onto concepts.

- Each VLE activity (`site_id`) is mapped to one concept via a curated mapping file.
- An interaction is `(student, concept, timestamp, engagement_signal)`.
- The **supervision target** is a binary correctness/success label derived from the assessment outcomes that follow, plus an engagement-based proxy for activities with no assessment attached.
- This yields sequences of hundreds of interactions per student, making the paper's stated `max_seq_len = 200` genuinely appropriate.

This must be documented explicitly in the paper. It is a defensible modelling choice, but it is a choice, and the current draft does not mention it.

### 2.2 Knowledge graph construction

Since OULAD ships no prerequisite structure, the concept graph is **constructed and checked into the repository** as auditable YAML:

- `graph/concepts.yaml` — concept nodes with names and descriptions
- `graph/prerequisites.yaml` — directed prerequisite edges (must form a DAG; enforced by test)
- `graph/vle_mapping.yaml` — OULAD `site_id` / activity-type → concept mapping

Target scale: **120–250 concept nodes**, on the order of 200–500 prerequisite edges. Derived from the actual OULAD module syllabi (Social Science and STEM modules) rather than invented.

The paper's claim of 1,247 concepts and 3,891 prerequisite edges is not reachable from OULAD and must be corrected to the real graph statistics, which the build emits to `results/graph_stats.json`.

---

## 3. Architecture

Four modules, in a pipeline. Each writes artifacts to disk so downstream stages load checkpoints instead of retraining.

```
elpr/
  data/      OULAD loading, sessionization, sequence building
  graph/     concept DAG loading, validation, GCN encoder
  models/    KG-DKT (Transformer + time-aware attention), baselines
  planner/   greedy/beam planner, DQN agent, simulator environment
  explain/   priority backtracking interpreter, SHAP, text generation
  eval/      metrics, ablations, statistical tests, table/figure emitters
scripts/     01_prepare_data.py … 06_run_ablations.py
app/         Streamlit demo
results/     metrics JSON, figures, LaTeX tables (all generated)
docs/        this spec, implementation plan
tests/       per-module tests
```

### 3.1 Data + graph layer

- Load OULAD CSVs into Parquet for fast repeated access.
- Sessionize clickstream at **30 minutes of inactivity** (as the paper states).
- One-hot encode categoricals; median-impute missing assessment scores (as the paper states).
- Temporal features: session length, inter-activity interval, per-concept mastery curve over 7-day windows (as the paper states).
- Split 80/10/10, stratified by final result, with 5-fold CV for reported metrics (as the paper states).

### 3.2 KG-DKT — knowledge tracing

- Transformer encoder: **2 layers, hidden dim 128, 4 attention heads, max sequence length 200** — exactly the paper's stated configuration.
- Concept embeddings produced by a **GCN over the prerequisite DAG: 3 convolution layers, layer norm + ReLU, 64-dimensional output** — exactly the paper's stated configuration.
- **Time-aware attention bias** encoding elapsed time since the previous interaction with the same concept. This is the forgetting mechanism the paper claims; without it there is no forgetting model.
- Output: per-concept mastery probability at each timestep.
- Loss: binary cross-entropy on next-interaction correctness.

**Metrics:** AUC-ROC, RMSE.

### 3.3 Planner — RL path recommendation

MDP formulation:

- **State:** DKT mastery vector over concepts + student demographic/behavioral features
- **Action:** select the next concept to study
- **Reward:** predicted mastery gain + prerequisite-satisfaction bonus − redundancy penalty
- **Environment:** the **frozen KG-DKT model acting as a student simulator**. No offline alternative exists — real students cannot be assigned recommendations retrospectively. This must be stated plainly in the paper.
- **Agent:** DQN, 3-layer fully-connected 256-128-64, ReLU, replay buffer 100,000 — exactly the paper's stated configuration.

**Mandatory fallback:** a **greedy/beam planner** is implemented first and always available — select the highest predicted-mastery-gain concept among those whose prerequisites are satisfied. It serves three purposes:

1. Guaranteed working backend for the demo, independent of DQN convergence
2. A legitimate non-RL baseline row in the path-quality table
3. The bar the DQN must clear to justify its inclusion

If DQN fails to beat the greedy planner, that is reported honestly as an ablation finding. The demo still works either way.

### 3.4 Explainability

**Priority backtracking interpreter:** for a recommended concept, traverse its prerequisite ancestry in the DAG, collect unmet prerequisites ranked by mastery deficit, and combine with the DQN Q-value decomposition and SHAP attributions over student features.

Emits templated pedagogical text, e.g.:

> Recommended **Recursion** because *Loops* mastery is 0.82 (met) but *Stack Frames* is 0.41 (gap). Closing this gap predicts +0.23 mastery gain toward your Data Structures goal.

This is **post-hoc** explanation — it runs after the policy chooses. The paper's abstract says post-hoc; the methodology section claims it is not post-hoc. The abstract is correct and the methodology overclaims. See §6.

---

## 4. Build stages

Ordered by paper value divided by risk. Each stage ends in something demonstrable.

| Stage | Deliverable | Paper section it fills |
|---|---|---|
| 0. Data + graph | Parquet sequences, validated DAG, dataset/graph stats | III-B, IV-B |
| 1. KG-DKT + baselines | Trained model, 5-fold CV, paired t-tests | Table I, IV-C, IV-D |
| 2. Explainability + greedy planner | Working recommendations with explanations | III-C, Fig. 4 |
| 3. RL (DQN) | Trained agent vs. greedy baseline | Table II, III-C |
| 4. Demo UI + results generation | Streamlit app, `make all` regenerating every table/figure | V, demo |

**Baselines to implement for comparison:** LSTM-DKT, RL-DKT, GNN-based recommender, collaborative filtering.

**Ablations:** remove knowledge graph encoding; remove temporal attention; remove explainability module.

**User study:** run after Stage 2, once real explanations exist. 30 participants, 5-point Likert on transparency, trust, and pedagogical utility, as the paper specifies. This is cheap and genuinely collectable from classmates, and converts the paper's softest table into real data.

---

## 5. Metric definitions

The paper reports metrics without defining them. Each must be defined precisely in code and in the paper:

- **AUC-ROC / RMSE** — on next-interaction binary correctness prediction, held-out students (not held-out interactions from seen students).
- **NDCG** — ranking of recommended next concepts against the concepts that successful students (Pass/Distinction) actually engaged with next. Ground-truth relevance must be stated.
- **Completion rate** — **simulated**: fraction of simulated learners reaching a target concept set within a step budget, under the frozen-DKT environment. Must be labelled "simulated" in the paper; it is not observed student behaviour.
- **Average knowledge gain** — mean predicted mastery increase per recommended step. Currently a unitless quantity in the paper; the normalization must be stated.
- **Inference latency** — measured on the actual hardware used, reported with that hardware named.

---

## 6. Paper corrections required

### 6.1 Claims that cannot be achieved as written

| Claim | Location | Problem | Fix |
|---|---|---|---|
| 1,247 concept nodes, 3,891 prerequisite edges, 1,023 co-requisite edges | IV-B | OULAD has no concept or prerequisite annotations at any scale | Report the real curated graph statistics from `results/graph_stats.json` (~120–250 nodes) |
| Knowledge graph "generated from course prerequisite relationship links and learning outcomes mapping" in OULAD | III-B | OULAD ships neither | Describe the curated-and-published graph construction honestly |
| AUC-ROC 0.892 | Table I | Optimistic for knowledge tracing on OULAD's sparse assessment signal | Report measured value; expect roughly 0.75–0.87 |
| Hardware: i9-12900K, 64GB RAM, RTX 3080, Ubuntu 22.04 | IV-A | Not the hardware being used | State actual hardware |
| Inference 187 ms vs 112 ms | V-E | Measured on hardware not in use | Re-measure and report |

### 6.2 Numbers that are asserted and must be regenerated by real runs

All of these are currently unsupported by any code and must be replaced with measured output:

- Table I: DKT 0.841/0.189, RL-DKT 0.857/0.176, GNN 0.863/0.168, Proposed 0.892/0.143
- Table II: completion 0.651/0.703/0.718/0.784; NDCG 0.742/0.821/0.809/0.856; knowledge gain 0.156/0.189/0.194/0.234
- Ablations: AUC 0.861 without knowledge graph; completion 0.723 without temporal modelling; AUC 0.889 without explainability; trust 4.08 → 3.5
- User study: transparency 4.21/5, trust 4.08/5, pedagogical utility 4.32/5, n=30
- "3.4% relative gain in AUC-ROC" — recompute from measured values

Baseline rows in particular cannot be cited from prior work, because they are not published figures for this dataset and task; they must be produced by running the baselines here.

### 6.3 Internal contradictions to fix regardless of code

1. **Post-hoc contradiction.** Abstract: "embedding post-hoc explanation generation". Section III-A/III-C: explainability integrated "rather than in isolation as current techniques do" and "remove the post-hoc explainability constraints". Pick one. The implementation is post-hoc; the abstract is correct.
2. **Collaborative filtering baseline.** The abstract promises comparison against collaborative filtering; no such baseline appears in Table I. Either implement it (planned) or drop the claim.
3. **GCN layer count.** Section III-D: "numbers of graph convolution layers (2, 4)". Section IV-C: "three graph convolution layers". Reconcile — the tuning range and the chosen value should be distinguished.
4. **Graph attention vs GCN.** Section III-C alternates between "graph attention mechanisms" and "Graph Convolutional Networks". Pick one and use it consistently.
5. **Likert scale.** Section V-C reports "usefulness to the classroom (08/7)" while the method specifies a 5-point Likert scale. A `/7` value cannot come from a 5-point scale.
6. **Citation errors.** References [1] and [11] are the same paper duplicated. Reference [5] is cited in Section II as the origin of DKT with recurrent networks, but points to Badran & Preisach 2025; DKT originates with Piech et al., 2015. Bloom's taxonomy is cited to [12], a graph-neural-network paper.

### 6.4 Garbled text (PDF corruption or typos) to repair

These appear mangled in the current draft and must be rewritten regardless of results:

- "The initial learning rate is set to 0 … We will use 001" → learning rate 0.001
- "experience replay buffer size (99)" → 100,000 (contradicts Section IV-C already)
- "dropout 0(0." and "L2 weight decay and dropout 0(0." → state the actual dropout value
- "During training, the batch size is 64 and the number of3." → incomplete sentence
- "paired t-tests (p less than 0. 05)" / "using 05)" → p < 0.05
- Section V-B and V-D sentences with numbers spliced mid-clause ("The percentage of students who finish all sections is the highest (0). … (784)") — these need full rewriting once real numbers exist
- "trust (mean = 4)." followed by "The mean scores for pedagogical utility (08/5)" — split decimals

---

## 7. Reproducibility requirements

- Fixed seeds; seed recorded in every results file
- `make all` regenerates every table and figure in `results/` from scratch
- `make demo` launches the Streamlit app from saved checkpoints
- Every number in the paper traceable to a file in `results/`
- Tests: DAG acyclicity, mapping coverage, sequence-builder correctness, metric implementations against known-answer cases

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| DQN fails to converge or degenerates to a constant policy | Greedy planner is the demo backend and a reported baseline; DQN failure is a publishable ablation result |
| Knowledge tracing signal too weak on OULAD | VLE-level interaction unit (§2.1); if AUC stays near chance, report it and analyze why |
| Curated graph seen as arbitrary | Graph is checked in, versioned, documented, and derived from real module syllabi; reviewers can inspect it |
| Timeline overrun | Stages are ordered so that stopping after Stage 2 still yields a working demo plus Table I |

---

## 9. Open questions for the user

1. Hardware and deadline — the design assumes local Apple Silicon over a few weeks. Confirm or correct.
2. Which OULAD modules to build the concept graph for — all 7, or the STEM modules only (smaller, more coherent graph)?
3. Is the user study feasible for you (access to ~30 classmates)?
