# Project Context — read this first

Hand this to a fresh Claude Code session (or a collaborator, or your supervisor) and it
will have everything it needs. Every claim here is backed by a file in `results/`.

---

## What this is

An explainable learning-path recommender over the Open University Learning Analytics
Dataset (OULAD), built to support a research paper.

**The paper was rejected** by IEEE COMPUTINGCON 2026 (paper 1757, 31 Aug 2026), stated
reason *"Less Technical Contribution."* Three reviewers agreed on the cause: the paper
asserted results it had not produced, and described an algorithm without specifying it.
Novelty scored 1, 1, 2.

This repository is the work that answers those reviews. Every number the original draft
claimed has now been measured, and several were unreachable as stated.

---

## Current state

| Stage | Status |
|---|---|
| 0 · Data pipeline, mined concept graph | Complete |
| 1 · Knowledge tracing, 9 training runs, Table I | Complete |
| 2 · Explainability, planner, Table II | Complete |
| 4 · Web application with auth | Complete, running |
| 3 · Reinforcement learning | **Deferred** — greedy beats random by only d = 0.09 |
| 5 · User study | **Not started** |
| 6 · Paper rewrite | **Not started — this is the next task** |

`make test` → 32 passing. `make api` → http://localhost:8420

---

## Headline results (all measured, all in `results/`)

```
Proposed (KG-DKT)   AUC 0.9538 ± 0.0019   RMSE 0.2621 ± 0.0026   ECE 0.0054
GNN-based               0.9502 ± 0.0014   p = 0.0220
DKT (Piech et al.)      0.9435 ± 0.0017   p = 0.0009
SAKT                    0.9233 ± 0.0018   p < 0.0001
Majority (no-skill)     0.5000 ± 0.0000   (harness verified)
```

5-fold cross-validation, folds grouped by student, Holm-corrected paired t-tests.

### The finding the paper should be built around

Standard ablation says the knowledge graph does nothing:

```
proposed  0.9538      no_graph  0.9539      difference −0.0001
```

But that test is structurally blind. Accuracy is measured only at assessment positions,
and **only 89 of 237 concepts carry assessments** — all with thousands of examples each.
The graph's job is the other 148, which never enter the measurement.

Withholding *all* assessments for 27 of the 89 assessed concepts and scoring only those:

```
with graph     0.9004 ± 0.0059
without        0.8679 ± 0.0042
difference     +0.0325    p = 0.0003    Cohen's d = 5.06     (wins on every fold)
```

**Claim:** knowledge-graph propagation contributes nothing where direct supervision is
plentiful, and substantially where it is absent — and standard ablation protocols cannot
detect this.

That is a measured, non-obvious, testable finding. It answers the novelty criticism in a
way that "we combined DKT, GCN and RL" never will.

### Table II — path quality

906 decisions, 400 held-out successful learners, ground truth being what they actually
studied next.

```
greedy (proposed)  NDCG@5 0.1532    hit 0.3687    prereq violations 0.0000
weakest-first             0.1344         0.3642                     0.0000
random (legal)            0.1300         0.3455                     0.0000
popularity                0.1069         0.2936                     0.0000
curriculum                0.0612         0.1832                     0.0000
```

All significant after Holm correction, **but read the effect sizes**: vs random d = 0.09
(negligible), vs curriculum d = 0.36 (small). The honest reading is that the prerequisite
mask does most of the work and the learned ranking adds a small consistent improvement.

Zero prerequisite violations across all 906 decisions is a verified guarantee, not an
assumption.

### Explanation quality (objective, not survey)

```
fidelity 0.5318    sufficiency 0.0020    stability 0.5310
```

Sufficiency of 0.002 meant the explanation was true but radically incomplete — 99.8% of
the benefit came from transfer it never mentioned. The renderer now names beneficiaries.
No Likert survey would have surfaced that.

---

## What the data does not contain

Three things the original draft assumed:

1. **No concepts, no prerequisites.** The draft claimed 1,247 nodes and 3,891 edges from
   OULAD "prerequisite relationship links". No such field exists. Concepts are derived as
   *(module, week-of-study)* → **237**; edges are structural plus mined from behaviour →
   **724 prerequisite, 302 corequisite, 0 cycles**.
2. **No material titles.** Concepts cannot honestly be named by topic. Labels describe
   position and activity composition: *"DDD · Week 12 — middle, assessment week, mostly
   reading and course content."*
3. **No sub-day timestamps.** The draft's "30-minute inactivity sessionization" is not
   computable; one student-day is one session.

Placement proxy validated at **ρ = 0.975** against the 1,121 materials that do carry
metadata.

---

## Four data problems found and fixed

1. **Labels were 96% one class.** OULAD records only *submitted* assessments, so failures
   are missing. 150,013 of 323,925 expected submissions never happened. Treating
   non-submission as failure moved balance to **67.9 / 32.1**.
2. **82% of materials had no scheduling data.** Fell back to median observed access day,
   validated as above.
3. **Median imputation was fabricating scores** for students who never submitted. Caught
   because counts did not reconcile.
4. **Mined edges were half calendar artefacts.** 51.4% crossed module boundaries on
   co-enrolment ordering. Requiring lift and staying within-module cut 3,385 edges to 285
   — and cycles from 84 to **zero**.

---

## Known limitations, stated plainly

- The model predicts **successful completion**, not knowledge in the abstract. Because
  non-submission counts as failure, much of the signal is engagement. Right target for an
  intervention system; not the same claim as measuring understanding.
- Mastery is **bimodal** — engaged learners sit near 0.98 on everything, disengaged near
  0.00. Absolute thresholds are useless; the planner uses learner-relative percentiles.
- The proposed model sees student demographics that DKT and SAKT do not (those follow
  their original formulations). A `no_student` ablation would separate architecture from
  feature access. **Not yet run.**
- Only 89 of 237 concepts carry assessments, so mastery for the rest is uncalibrated and
  arrives through the graph.

---

## Repository map

```
elpr/          data · db · graph · mining · models · planner · explain · eval
sql/           00–12, the entire ETL as reviewable SQL (DuckDB)
scripts/       01 prepare · 02 graph · 03 sequences · 05 train · 06 table1
               07 labels · 08 recommend · 09 table2 · run_all_kt
api/           FastAPI service with scrypt auth and role-based access
web/           single-page app: login, student view, adviser view, results, method
tests/         32 tests
docs/          architecture · build-plan · what-we-found · paper-corrections
               claimed-vs-measured · portability
results/       every number in the paper, plus figures
```

## Rebuilding from scratch

```bash
make install
# download OULAD into data/raw/ (see docs/portability.md, checksum included)
.venv/bin/python scripts/01_prepare_data.py
.venv/bin/python scripts/02_build_graph.py
.venv/bin/python scripts/03_build_sequences.py
.venv/bin/python scripts/07_concept_labels.py
make test
```

Training runs on a free Colab T4 — see `RUNBOOK.md`. This machine is an Intel Mac with
no GPU; a full run is ~10 hours locally versus ~25 minutes per configuration on a T4.

---

## The next task

**Rewrite the paper.** Everything needed exists:

1. Restructure around the cold-concept finding rather than the architecture — this
   directly answers the novelty criticism.
2. Add the mathematics R3 asked for: MDP tuple, reward function, loss, GCN propagation,
   time-aware attention. The current draft contains **zero equations**.
3. Replace every number from `results/`. See `docs/claimed-vs-measured.md` for the
   complete claim-by-claim comparison.
4. Work `docs/paper-corrections.md` top to bottom — 40+ items by severity, including a
   wrong DKT expansion, a miscited dataset, five author-less references, and a missing
   ethics statement.
5. Add limitations, threats to validity, ethics, data availability, and the fairness
   analysis (the model consumes gender, disability and deprivation band with no subgroup
   analysis — a likely desk-reject in 2026).
