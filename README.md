# Explainable Learning Path Recommendation

Knowledge tracing and prerequisite-aware learning-path recommendation over the
[Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
(OULAD), with post-hoc explanations grounded in a mined prerequisite graph.

Every number below was produced by code in this repository and is written to
`results/`. Nothing is asserted.

---

## Results

| Model | AUC-ROC | RMSE | ECE | p (Holm) |
|---|---|---|---|---|
| Majority (no-skill) | 0.5000 ± 0.0000 | 0.4669 | 0.0037 | < 0.0001 |
| SAKT | 0.9233 ± 0.0018 | 0.2992 | 0.0081 | < 0.0001 |
| DKT (Piech et al.) | 0.9435 ± 0.0017 | 0.2751 | 0.0102 | 0.0009 |
| GNN-based | 0.9502 ± 0.0014 | 0.2665 | 0.0064 | 0.0220 |
| **Proposed (KG-DKT)** | **0.9538 ± 0.0019** | **0.2621** | **0.0054** | — |

5-fold cross-validation, folds grouped by student. Paired *t*-tests against the
proposed model, Holm-corrected.

### When does a knowledge graph actually help?

Removing the prerequisite graph changes standard accuracy by nothing at all
(0.9539 vs 0.9538). That result is misleading, and the reason matters: accuracy is
measured only at assessment positions, and **only 89 of 237 concepts carry
assessments** — each with thousands of labelled examples. The graph's purpose is the
other 148 concepts, which never enter the measurement.

Withholding *every* assessment for 27 of the 89 assessed concepts, then scoring only
those:

| | AUC |
|---|---|
| With knowledge graph | **0.9004 ± 0.0059** |
| Without | **0.8679 ± 0.0042** |
| Difference | **+0.0325**  ·  p = 0.0003  ·  Cohen's *d* = 5.06 |

The graph wins on every fold.

> **Knowledge-graph propagation contributes nothing where direct supervision is
> plentiful, and substantially where it is absent — and standard ablation protocols
> are structurally unable to detect this.**

### Learning path quality

906 recommendation decisions across 400 held-out successful learners. Ground truth is
the concepts each learner actually studied next — real behaviour, not simulation.

| Planner | NDCG@5 | Hit@5 | Prereq violations |
|---|---|---|---|
| **greedy (proposed)** | **0.1532** | **0.3687** | **0.0000** |
| weakest-first | 0.1344 | 0.3642 | 0.0000 |
| random (legal moves) | 0.1300 | 0.3455 | 0.0000 |
| popularity | 0.1069 | 0.2936 | 0.0000 |
| curriculum (syllabus order) | 0.0612 | 0.1832 | 0.0000 |

All differences significant after correction, but the effect sizes are the honest
signal: *d* = 0.09 against random, *d* = 0.36 against syllabus order. **The
prerequisite mask does most of the work; the learned ranking adds a small consistent
improvement.** Zero prerequisite violations across all 906 decisions is a verified
guarantee rather than an assumption.

---

## Architecture

```
OULAD  →  DuckDB SQL ETL  →  mined concept graph  →  KG-DKT  →  planner  →  explainer
          sql/00–12           237 concepts           2-layer     prerequisite  priority
                              724 prereq edges       transformer  action mask   backtracking
                              0 cycles               + 3-layer                  + counterfactuals
                                                     GCN
```

**KG-DKT** — a 3-layer graph convolution over the prerequisite graph produces concept
embeddings; a 2-layer, 4-head transformer with time-aware attention turns a learner's
history into a mastery estimate for every concept. Older interactions are discounted by
a learned per-head decay:

```
bias[h, i, j] = −softplus(γ_h) · log(1 + Δdays(i, j))
```

466,569 parameters.

**Planner** — a concept is eligible only if not already mastered and every prerequisite
is at or above the readiness threshold. Pedagogy is enforced structurally, so an
unprepared recommendation is impossible rather than unlikely. Candidates are ranked by
simulated mastery gain against an *idle week* baseline, in log-odds.

**Explainer** — priority backtracking over the prerequisite ancestry, ranking gaps by
deficit decayed with graph distance, plus counterfactuals computed by querying the
model. Every clause of the generated text traces to a number; no language model is
involved, which is what makes it auditable.

---

## The dataset, and what it does not contain

OULAD provides 32,593 enrolments, 10.6M clickstream events, 173,912 assessment records
across 7 modules. It does **not** provide:

- **Concepts or prerequisites.** Derived here as *(module, week-of-study)* → 237
  concepts; edges are structural plus mined from behaviour, requiring both consistent
  ordering and a measurable effect on downstream success.
- **Titles for learning materials.** Concepts therefore cannot honestly be named by
  topic. Labels describe position and activity composition instead.
- **Sub-day timestamps.** Only integer day offsets, so a 30-minute sessionization
  threshold is not computable.

Only 17.6% of materials carry scheduling metadata; the observed-access fallback is
validated at **ρ = 0.975** against those that do.

**Label construction.** OULAD records only *submitted* assessments, giving a 95.6% pass
rate — 46% of expected submissions never happened. Treating non-submission as failure
restores a 67.9 / 32.1 balance and changes the target to *successful completion*.
Consequently much of the signal is engagement, not knowledge. That is the right target
for an intervention system, but it is not the same claim as measuring understanding.

---

## Quick start

```bash
make install                                    # Python 3.11 venv

mkdir -p data/raw && cd data/raw                # 487 MB
curl -L -o oulad.zip https://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip
unzip oulad.zip && cd ../..
# sha256: 90dda45037939953f979072fa70a809ebe07e90ec783c408762af7698a3825ec

.venv/bin/python scripts/01_prepare_data.py     # ~18s
.venv/bin/python scripts/02_build_graph.py      # ~2.5 min
.venv/bin/python scripts/03_build_sequences.py
.venv/bin/python scripts/07_concept_labels.py

make test                                       # 32 tests
make api                                        # http://localhost:8420
```

Training runs on a free Colab T4 — see [`RUNBOOK.md`](RUNBOOK.md). Checkpoints are not
committed; the demo states plainly when it is running untrained.

## Web application

FastAPI service and a single-page app with scrypt-hashed accounts and role-based
access. **Students** track their own learning, record activity, and receive explained
recommendations that update live. **Advisers** search all 25,101 dataset learners plus
registered students, inspect any record, and override concepts a learner already knows.
A results dashboard reads `results/*.json` directly, so the site cannot drift from the
paper.

*Demonstration-grade authentication: no TLS, no rate limiting, no account recovery. Do
not reuse a real password.*

---

## Documentation

| Document | Contents |
|---|---|
| [`CONTEXT.md`](CONTEXT.md) | Start here — full project state and next steps |
| [`docs/architecture.md`](docs/architecture.md) | System design and rationale |
| [`docs/what-we-found.md`](docs/what-we-found.md) | Four data problems and their resolutions |
| [`docs/claimed-vs-measured.md`](docs/claimed-vs-measured.md) | Every claim against its measured value |
| [`docs/paper-corrections.md`](docs/paper-corrections.md) | Defect list by severity |
| [`docs/build-plan.md`](docs/build-plan.md) | Stage plan |
| [`docs/portability.md`](docs/portability.md) | Moving the project between machines |
| [`RUNBOOK.md`](RUNBOOK.md) · [`COLAB.md`](COLAB.md) | Reproducing the training runs |

## Data licence

OULAD is released under CC-BY 4.0. Cite:

> J. Kuzilek, M. Hlosta, Z. Zdrahal, "Open University Learning Analytics dataset,"
> *Scientific Data*, vol. 4, 170171, 2017. doi:10.1038/sdata.2017.171
