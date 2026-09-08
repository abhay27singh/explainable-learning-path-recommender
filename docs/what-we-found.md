# What We Built and What We Found — Stage 0

**Date:** 2026-08-09
**Audience:** supervisor / project review
**Scope:** data foundation and concept graph. No model trained yet.

This document explains, in order, what was built, what problems the real data turned
out to contain, and how each was solved. Every number below was produced by code in
this repository and is written to `results/` — none is quoted from memory.

---

## 1. What exists now

The dataset is loaded and the concept graph is built. Concretely:

```
sql/00_load_raw          3.8s   7 OULAD tables, 10,655,280 clickstream rows
sql/01_dataset_stats     0.1s   measured statistics
sql/02_sessionize        2.4s   1,808,119 sessions
sql/03_concepts          0.3s   237 concepts derived
sql/04_site_to_concept   0.0s   6,268 materials mapped, 206 assessments mapped
sql/05_events            8.9s   6,660,108 interaction events
sql/06_edges_structural  0.0s   482 structural edges
sql/07_edges_mined       1.6s   285 mined edges
FP-Growth mining       ~120s    302 corequisite edges
```

The whole pipeline reruns from raw CSVs in about two and a half minutes.

### Measured dataset statistics

| Quantity | Value |
|---|---|
| Distinct students | 28,785 |
| Enrolments (student × module-presentation) | 32,593 |
| Clickstream events | 10,655,280 |
| Assessment records | 173,912 |
| Learning materials (VLE sites) | 6,364 |
| **Concepts derived** | **237** |
| Prerequisite edges | 724 |
| Corequisite edges | 302 |
| Concepts carrying assessments | 89 of 237 |
| Median clickstream events per student | 270 |
| Median assessment events per student | 7 |
| Supervised training tokens | 242,670 |
| Label balance (success / failure) | 67.9% / 32.1% |

---

## 2. Problem 1 — the dataset contains no concept graph

**The paper claims** 1,247 concept nodes, 3,891 prerequisite edges, and 1,023
corequisite edges, described as coming from OULAD's "course prerequisite
relationship links and learning outcomes mapping".

**The reality:** OULAD contains no concepts, no prerequisites, and no learning-outcome
mappings. It contains seven modules, their learning materials, clickstream logs,
assessment scores, and student demographics. There is nothing to read a knowledge
graph off.

**What we did instead.** A concept is defined as a *(module, week-of-study)* pair —
the set of material a module presents in a given week. This is derived from the data
rather than invented, and produces **237 concepts**.

Prerequisite edges then come from four rules:

| Rule | Edges | Basis |
|---|---|---|
| `prereq_sequence` | 230 | week *w* precedes week *w+1* within a module |
| `prereq_assessment` | 252 | the four weeks before an assessment precede it |
| `prereq_mined` | 285 | mined from student behaviour (§4) |
| `corequisite` | 302 | association rules, no consistent order |

**For the paper:** the graph statistics must be corrected from 1,247/3,891/1,023 to the
measured 237/724/302, and the construction method must be described honestly. The
graph is version-controlled, so a reviewer can inspect it.

---

## 3. Problem 2 — 82% of materials had no scheduling information

**What we found.** The plan was to place each material in a week using OULAD's
`week_from` column. Only **1,121 of 6,364** materials carry it — 17.6%. Using it alone
would have discarded 82% of the learning materials.

**What we did.** For materials with no `week_from`, the week is derived from the
**median day on which students actually accessed that material**. This covers every
material that anyone ever opened (6,268 of 6,364).

**Why this is defensible, with evidence.** 1,121 materials carry *both* signals, so the
substitute can be checked against the original:

```
Spearman correlation  ρ = 0.975
Pearson correlation   r = 0.975
median |metadata − observed| = 1.0 week
within 1 week: 59.1%      within 2 weeks: 83.9%
```

A correlation of 0.975 means the observed ordering reproduces the scheduled ordering
almost exactly. Prerequisites depend on *order*, not on absolute week number, and the
systematic one-week lag — students open material after it is released — shifts
everything uniformly, so ordering is unaffected.

**Honest limitation:** deriving concept order from behaviour and then mining
prerequisites from behaviour is partially self-referential. This validation on the
metadata-anchored subset is the check against that, and Stage 5 adds independent
expert rating of a sample of mined edges.

---

## 4. Problem 3 — the labels were 96% one class

This was the most serious finding, and it would have invalidated the results table.

**What we found.** The model predicts whether a student will succeed on an assessment.
Binarising at the Open University pass mark of 40 gives:

```
pass rate = 0.956
score distribution (5th, 25th, 50th, 75th, 95th percentile) = 40, 65, 80, 90, 100
```

The 5th percentile of scores *is* the pass mark. Essentially nobody who submits an
assessment fails it.

**Why.** OULAD records only **submitted** assessments. Students who struggled did not
submit — so the failures are missing from the table, not absent from reality. Counting
what students were expected to submit against what they did:

```
expected submissions   323,925
actual submissions     173,912
never submitted        150,013   (46.3%)
```

Training on submitted rows alone would have produced a model that predicts "pass" for
everyone and still scores 95.6% accuracy — with an AUC that means nothing. A reviewer
checks class balance first.

**What we did.** Non-submission is treated as the negative outcome. The label becomes
*"did this student successfully complete this assessment"* rather than *"given that
they submitted, did they pass"*. Assessments falling after a student's unregistration
date are excluded, so withdrawing does not manufacture unlimited negatives.

Result:

```
submitted and passed    164,763   67.9%
never submitted          70,426   29.0%
submitted and failed      7,481    3.1%
                        -------
supervised tokens       242,670   67.9% positive
```

67.9% positive is a normal base rate for knowledge tracing — published datasets in this
field run 60–75%. The prediction task is now meaningful, and it directly matches the
completion and dropout story the paper wants to tell.

---

## 5. Problem 4 — an imputation bug that fabricated data

Missing assessment scores are median-imputed, as the paper states. The first
implementation joined the median on assessment identity alone, which meant that
students who **never submitted** were also assigned a score.

The label itself was unaffected — it keys off whether a submission exists — but the
score column was wrong, the submitted/not-submitted breakdown was wrong, and the error
would have propagated into the model's input features.

Imputation now applies only to submitted-but-unscored rows. There are **157** such rows.

This is worth reporting rather than hiding: it was caught because the reported counts
did not reconcile (224,185 "submitted" rows against 173,912 real submissions), which is
exactly the kind of check that should be run on every derived table.

---

## 6. How the prerequisites are mined

Course structure says week 3 comes before week 4. That is weak evidence — it records
how a module was *scheduled*, not how learning actually *depends*. Student behaviour is
stronger evidence, on two counts:

For every pair of concepts *(a, b)*:

- **precedence(a, b)** — the fraction of students who reached *a* before *b*.
  Consistent ordering.
- **lift(a, b)** — the pass rate on *b* among students who saw *a* first, divided by the
  overall pass rate on *b*. Whether reaching *a* first actually **helps**.

A prerequisite is exactly a pair with both properties. The acceptance rule:

```
support ≥ 100 students
AND src and dst belong to the SAME module
AND precedence ≥ 0.75
AND lift > 1.1
```

From 17,068 co-occurring pairs, **285** were accepted.

### The rule this replaced, and why the figure caught it

The first version admitted a pair on precedence alone when no outcome data existed,
and allowed pairs to cross module boundaries. It produced 3,385 edges — and the
numbers looked fine. The **figure** did not. Plotting concepts by module and week
showed a dense fan of edges running *between* modules, concentrated on four of the
seven, with none at all on the rest.

Measuring it:

```
cross-module edges   1,739   51.4%   mean support   313
within-module edges  1,646   48.6%   mean support 1,618
```

Those cross-module edges rested on the 2,479 students (of 28,785) who enrol in more
than one module. Inspecting the strongest of them:

```
CCC-W02 → EEE-W32   support 801   precedence 1.000   lift —
CCC-W14 → EEE-W32   support 793   precedence 1.000   lift —
EEE-W01 → CCC-W03   support 839   precedence 0.858   lift —
CCC-W01 → EEE-W07   support 837   precedence 0.863   lift —
```

Week 2 of one module trivially precedes week 32 of another: that is the **calendar**,
not a dependency. Both orientations appear for the same module pair. Every one has no
lift, meaning each entered through the no-evidence branch.

Two constraints were added: lift is now required for every edge, and edges must stay
within a module. Genuine cross-module dependencies belong in the curated overlay at
`graph/cross_module_edges.yaml`, where each can be justified individually.

Candidate rules, measured before choosing:

| Rule | Edges |
|---|---|
| original | 3,385 |
| within-module only | 1,646 |
| lift required | 463 |
| **within-module + lift required** (adopted) | **285** |
| adopted + week gap ≤ 8 | 103 |

**The strongest evidence the change was right:** under the original rule the graph
contained 84 cycles that had to be broken. Under the corrected rule there are **zero**.
Edges supported by outcome evidence are consistent with a single partial order;
edges supported by calendar coincidence were not.

**Corequisites** — concepts that travel together with no consistent order — are mined by
FP-Growth association rules, requiring high confidence in *both* directions and
near-symmetric precedence. 302 accepted.

**Cycle breaking.** Mined edges are statistical, so they can in principle contradict one
another and close a loop. The machinery remains in place and every removal would be
logged to `results/graph_stats.json` — silently discarding evidence would not be
acceptable in a graph the paper describes. Under the corrected rule it currently has
nothing to do.

---

## 7. Two engineering problems, and why they are worth mentioning

Both cost real time and both are the kind of thing that quietly ruins a deadline.

**A query that never finished.** Computing pairwise statistics with the outcome join
folded into one aggregation ran for over ten minutes without completing. Split into two
passes — compute co-occurrence first (0.25s, 17,068 pairs), then attach outcomes only to
pairs that already clear the support floor — the whole step runs in **1.6 seconds**.

**A combinatorial explosion.** FP-Growth was generating the full frequent-itemset
lattice over 237 concepts. Because concepts within a module co-occur for nearly every
student, that lattice is astronomically large. Corequisites are pairwise by definition,
so capping itemset length at two makes the computation finish while discarding nothing
we use.

---

## 8. What this means for the paper

Four claims in the current draft must change, and each change replaces an assertion
with a measurement:

| Draft claim | Corrected |
|---|---|
| 1,247 concepts, 3,891 prerequisite, 1,023 corequisite edges | 237 concepts, 3,722 prerequisite, 302 corequisite edges |
| Graph derived from OULAD prerequisite links | Graph derived from module structure and mined from behaviour; construction published |
| 30-minute inactivity sessionization | OULAD timestamps are day-granular; one student-day is one session |
| — | New: proxy validated at ρ = 0.975 against metadata-anchored materials |

The paper also gains material it does not currently have: a real construction method, a
validation statistic, a stated class-balance treatment, and a logged record of every
edge removed for acyclicity.

---

## 9. What is not done yet

Stage 0 is roughly two-thirds complete. Still outstanding:

- Feature encoding, sequence assembly, stratified splits (`sql/08`–`sql/11`)
- Learner-profile clustering
- Test suite and the concept-graph figure

No model has been trained. Every results number in the paper remains unproduced until
Stage 1.
