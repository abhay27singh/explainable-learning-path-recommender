# What the Paper Claims vs What We Measured

**Date:** 2026-08-11
**Purpose:** every quantitative claim in the draft, set against the value produced by code.

Nothing in the "measured" column is quoted from memory. Each traces to a file in
`results/`, regenerable with `make all`.

---

## 1. Headline accuracy

| Claim in draft | Measured | Verdict |
|---|---|---|
| AUC-ROC **0.892** | **0.9538 ± 0.0019** | Higher, and real |
| RMSE **0.143** | **0.2621 ± 0.0026** | Draft figure unreachable |
| No standard deviations | ± reported for all 5 folds | Added |
| No p-values despite claiming t-tests | Holm-corrected p for every comparison | Added |
| No calibration | **ECE 0.0054** | Added |

The RMSE deserves care. On a binary target with a 68% base rate, a constant predictor
scores ≈0.466. Reaching 0.143 would require near-perfect prediction of individual
student outcomes. It is not attainable and should not be defended.

## 2. Table I — comparative accuracy

| Model | Draft | Measured | p (Holm) |
|---|---|---|---|
| Majority / no-skill | — | 0.5000 ± 0.0000 | < 0.0001 |
| SAKT | — | 0.9233 ± 0.0018 | < 0.0001 |
| DKT | 0.841 | **0.9435 ± 0.0017** | 0.0009 |
| RL-DKT | 0.857 | not implemented | — |
| GNN-based | 0.863 | **0.9502 ± 0.0014** | 0.0220 |
| Proposed | 0.892 | **0.9538 ± 0.0019** | — |

The draft's baseline figures are not published values for this dataset and task, so
they cannot be cited. Every row above was produced by running that baseline here, on
the same splits, with the same tuning budget.

Two rows the draft does not have, both of which a reviewer would ask for: **SAKT**
(attention without the graph — proves the gain is not merely "we used a transformer")
and **majority** (proves the AUC is not an artefact of class imbalance).

## 3. Ablations

| Ablation | Measured | p (Holm) | Effect |
|---|---|---|---|
| − knowledge graph | 0.9539 ± 0.0021 | 0.7704 | none |
| − temporal attention | 0.9532 ± 0.0014 | 0.4507 | none |

**Neither component changes standard accuracy.** The draft asserts both are essential.

## 4. The cold-concept experiment — the paper's real result

Standard AUC is computed only at assessment positions, and assessments exist for
**89 of 237 concepts**, each carrying thousands of labelled examples. The knowledge
graph's purpose is the other **148 concepts**, which never enter that measurement. The
standard ablation is structurally incapable of detecting what it ablates.

Withholding every assessment for 27 of the 89 assessed concepts, and scoring only
those:

| | AUC |
|---|---|
| With knowledge graph | **0.9004 ± 0.0059** |
| Without | **0.8679 ± 0.0042** |
| Difference | **+0.0325**, p = 0.0003, Cohen's d = 5.06 |

The graph wins on every fold. This yields a precise claim in place of a vague one:

> Knowledge-graph propagation contributes nothing where direct supervision is
> plentiful, and contributes substantially where it is absent.

## 5. Table II — path quality

| Planner | Draft | Measured NDCG@5 | Hit@5 | Prereq violations |
|---|---|---|---|---|
| curriculum (syllabus order) | — | 0.0612 | 0.1832 | 0.0000 |
| popularity | — | 0.1069 | 0.2936 | 0.0000 |
| random (legal moves) | — | 0.1300 | 0.3455 | 0.0000 |
| weakest-first | — | 0.1344 | 0.3642 | 0.0000 |
| **greedy (proposed)** | NDCG 0.856 | **0.1532** | 0.3687 | 0.0000 |

Measured over 906 decisions from 400 held-out successful learners, against what those
learners actually studied next.

**Read the effect sizes, not only the p-values.** All comparisons are significant, but:

| vs | Difference | p | Cohen's d |
|---|---|---|---|
| curriculum | +0.0921 | < 0.0001 | +0.36 (small) |
| popularity | +0.0464 | < 0.0001 | +0.17 (negligible) |
| random | +0.0232 | 0.0125 | +0.09 (negligible) |
| weakest-first | +0.0189 | 0.0167 | +0.08 (negligible) |

With 906 paired decisions, trivial differences reach significance. The honest reading:
**the prerequisite mask does most of the work; the learned ranking adds a small,
consistent improvement.**

Claims the draft cannot support: completion rate 0.784 and average knowledge gain
0.234 were never measured, and completion rate can only ever be simulated — no student
in OULAD followed a recommendation from this system.

One claim that is unambiguous: **prerequisite violations are 0.0000** across all 906
decisions and every planner. The pedagogical guarantee is verified, not assumed.

## 6. Explainability

The draft evaluates its headline contribution with a Likert survey alone. A survey
measures whether people liked an explanation, not whether it is true of the model.

| Measure | Value | Meaning |
|---|---|---|
| Fidelity | 0.5318 | cited deficits do predict measured counterfactual improvement |
| Sufficiency | 0.0020 | only 0.2% of the gain lands on the concept named |
| Stability | 0.5310 | about half the cited gaps survive ±0.02 perturbation |

Sufficiency is the uncomfortable one and the reason the metric was worth building: the
explanation was **true but radically incomplete**, since almost all the benefit came
from transfer it never mentioned. The renderer now names the beneficiaries. No survey
would have surfaced this.

## 7. Dataset and graph

| Claim in draft | Measured |
|---|---|
| 1,247 concept nodes | **237** |
| 3,891 prerequisite edges | **724** (285 mined, 252 assessment, 230 sequence) |
| 1,023 corequisite edges | **302** |
| Graph from OULAD "prerequisite relationship links" | No such data exists; derived and mined |
| 30-minute inactivity sessionization | OULAD is day-granular; one student-day = one session |
| — | New: placement proxy validated at **ρ = 0.975** |
| — | New: **0 cycles**; every removal would be logged |
| — | New: only **89 of 237** concepts carry assessments |

Additional measured facts the draft lacks: 27,904 sequences over 25,101 students,
median 169 events each, 242,670 supervised tokens, label balance 67.9% / 32.1%, three
learner profiles (silhouette 0.418, stability ARI 0.990).

## 8. Implementation

| Claim | Reality |
|---|---|
| i9-12900K, RTX 3080, Ubuntu 22.04 | Intel Mac (no GPU) for development; free Colab T4 for training |
| PyTorch Geometric 2.2.0 | Not used — dense adjacency at 237 nodes |
| Stable-Baselines3 1.7.0 | Not used — no RL implemented |
| RDFLib 6.2.0 | Not used — no RDF/OWL work exists |
| Python 3.9, PyTorch 1.12 | Python 3.11, PyTorch 2.2.2 locally / 2.11 on Colab |
| Inference 187 ms vs 112 ms | **~160 ms** end-to-end per recommendation, measured on this hardware |
| Replay buffer "(99)" / 100,000 | No replay buffer — no RL |
| 2 layers, 128 hidden, 4 heads, seq 200 | **Implemented exactly as stated** |
| 3 GCN layers, 64-dim embeddings | **Implemented exactly as stated** |
| Adam, lr 1e-3, cosine annealing, L2 1e-4, batch 64 | **Implemented exactly as stated** |
| 80/10/10 stratified, 5-fold CV, paired t-tests | **Implemented**, plus grouping by student and Holm correction |

Model size: **466,569 parameters** — 85% transformer, 6% graph encoder.

## 9. Claims with no implementation at all

| Draft claim | Status |
|---|---|
| DQN reinforcement-learning agent | Not implemented. Greedy planner beats random by only d = 0.09, leaving little headroom |
| RL-DKT baseline | Not implemented (depends on the above) |
| Collaborative filtering baseline | Promised in abstract, not implemented |
| User study, n = 30, Likert 4.21 / 4.08 / 4.32 | Not conducted |
| Completion rate 0.784 | Not measured; can only be simulated |
| Average knowledge gain 0.234 | Not measured; now redefined in log-odds |

## 10. What the project delivers that the draft does not claim

- A **REST API** — the draft lists this as future work; it exists and runs
- A **working demo** over 25,101 learners at ~160 ms per recommendation
- **Adviser override**, the only artefact serving the "tools for academic advisers" claim
- **Calibration**, cold-start analysis, objective explainability metrics
- **Learner profiling** (k = 3) usable as a cold-start prior
- **32 automated tests** guarding the invariants
- Full reproducibility: raw CSV to results in one command

---

## Summary

The draft contains **six numbers that are unreachable**, **eleven that were never
measured**, and **four components that were never built**. It also understates what the
work actually achieves: the accuracy is higher than claimed, the calibration is
excellent, and the cold-concept experiment produces a sharper and more interesting
finding than the one originally asserted.

The rewrite is not damage control. It is replacing a weaker fabricated story with a
stronger true one.
