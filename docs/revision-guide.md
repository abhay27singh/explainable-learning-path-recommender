# Paper Revision Guide

**For:** *Explainable AI-Based Personalized Learning Path Recommendation System for
University Students* — rejected by IEEE COMPUTINGCON 2026, paper 1757.

**Rejection reason given:** *"Less Technical Contribution in Paper."*
**Reviewer novelty scores:** 1, 1, 2.

Every number in the replacement tables below was produced by code in this repository and
is written to `results/`. Regenerate any of it with `make all`.

---

# Part 1 — What must change, in priority order

## P1 · Fatal if left in

| # | Item | Where | Fix |
|---|---|---|---|
| 1 | **All results are fabricated.** 0.892, 0.143, 0.234, 0.856, 0.784, 4.21/5 | Tables I–II, abstract, §V | Replace with Part 2 |
| 2 | **RMSE 0.143 is unreachable.** On a binary target with a 68% base rate a constant predictor scores 0.466 | Table I, abstract | Real value **0.2621 ± 0.0026** |
| 3 | **Zero equations.** R3's explicit complaint: *"too conceptual, details of how to operate should be clearly elaborated"* | §III | Add Part 3 |
| 4 | **No standard deviations, no p-values** while claiming 5-fold CV. R1: *"more result oriented"* | Tables I–II | Part 2 has both |
| 5 | **"in the field of drug diversion"** — pharmaceutical crime, in an education paper | §II | Delete the clause |
| 6 | **"Dynamic Knowledge Tracing"** ×3. DKT is **Deep** Knowledge Tracing | §II, §VI | Global replace |
| 7 | **1,247 concepts / 3,891 / 1,023 edges** from "OULAD prerequisite links" — no such data exists | §IV-B | **237 / 724 / 302**, construction described |
| 8 | **No ethics statement, no consent, no IRB** with a 30-participant study | absent | Part 4, or delete the study |
| 9 | **User study never conducted** (4.21/5, n=30) | §V-C | Run it, or replace with Part 2 objective metrics |
| 10 | **DQN never implemented.** No RL exists | §III-D, §IV-C, Table II | Move to future work, or run it |

## P2 · Will draw reject-or-major-revise

| # | Item | Fix |
|---|---|---|
| 11 | **Abstract contradicts Table I.** Abstract says baseline *"AUC-ROC of 0.871"* and *"RMSE of 0.148"*; Table I's best baseline is GNN at 0.863 / 0.168. Neither abstract figure appears anywhere | Use real values |
| 12 | **30-minute inactivity sessionization** — OULAD has day-granular timestamps only, so this is not computable | State: one student-day is one session |
| 13 | **Post-hoc contradiction.** Abstract says "post-hoc"; §III says explainability is integrated "rather than post-hoc". Your architecture *is* post-hoc | Keep post-hoc, reframe novelty |
| 14 | **RL-DKT baseline claimed, never implemented** | Delete the row |
| 15 | **Collaborative filtering promised in abstract**, absent from results | Delete the promise |
| 16 | **187 ms vs 112 ms** measured on an RTX 3080 you don't have; also called a "slight increase" when it is 67% | Real: **~160 ms** end-to-end, measured |
| 17 | **No fairness analysis** despite consuming gender, disability, IMD band, age | Add, or state exclusion explicitly |
| 18 | **No limitations, no threats to validity** | Part 4 |
| 19 | **Bloom's taxonomy** introduced, never used again | Use it or cut it |
| 20 | **Hardware claim** — i9-12900K, RTX 3080, Ubuntu | Intel Mac + free Colab T4 |
| 21 | **PyTorch Geometric, Stable-Baselines3, RDFLib** listed; none used | Remove |

## P3 · Polish

| # | Item |
|---|---|
| 22 | Recommendation unit inconsistent — abstract says "courses and modules", §III says "activities", explanations concern "concepts" |
| 23 | GAT vs GCN used interchangeably for the same encoder |
| 24 | Bayesian hyperparameter optimisation claimed; no search space, budget, or result reported |
| 25 | Epoch count, dropout value, lambda never stated |
| 26 | Figures 1–6 never referenced in body text (IEEE requires it) |
| 27 | No contributions list at the end of §I |
| 28 | 15 index terms; IEEE norm is ~5 |
| 29 | No data/code availability statement |
| 30 | Cold-start criticised in related work, never addressed for your own system → **now you have the curve** |

---

# Part 2 — Replacement tables and results

Formatted to match the paper's existing style.

## TABLE I · COMPARATIVE PREDICTIVE PERFORMANCE

```
Model                              AUC-ROC            RMSE              ECE      p        d
--------------------------------------------------------------------------------------------
Majority (no-skill)                0.5000 ± 0.0000    0.4669 ± 0.0004   0.0037   <0.001   —
SAKT [Pandey & Karypis]            0.9233 ± 0.0018    0.2992 ± 0.0017   0.0081   <0.001   26.16
DKT [Piech et al.]                 0.9435 ± 0.0017    0.2751 ± 0.0022   0.0102    0.0009   5.67
GNN-based                          0.9502 ± 0.0014    0.2665 ± 0.0020   0.0064    0.0220   2.25
Proposed (KG-DKT)                  0.9538 ± 0.0019    0.2621 ± 0.0026   0.0054      —      —
--------------------------------------------------------------------------------------------
Improvement vs. GNN                +0.0036            −0.0044
Improvement vs. DKT                +0.0103            −0.0130
```

Mean ± standard deviation over 5-fold cross-validation, folds grouped by student so no
learner appears in both training and test. *p* from paired *t*-tests against the
proposed model, Holm–Bonferroni corrected across the comparison family. *d* is Cohen's
*d* for paired samples.

Per-fold AUC for the proposed model: 0.9552, 0.9502, 0.9533, 0.9553, 0.9548.

**Two rows the original lacked, both of which a reviewer will ask for.** *Majority*
returns exactly 0.5000, proving the AUC is not an artefact of the 67.9/32.1 class split.
*SAKT* is attention without the graph and without time awareness, proving the gain is
not merely "we used a transformer."

## TABLE II · ABLATION STUDY

```
Variant                            AUC-ROC            RMSE              p        Effect
--------------------------------------------------------------------------------------------
Proposed (full)                    0.9538 ± 0.0019    0.2621 ± 0.0026     —        —
  without knowledge graph          0.9539 ± 0.0021    0.2618 ± 0.0025   0.7704   none
  without temporal attention       0.9532 ± 0.0014    0.2624 ± 0.0024   0.4507   none
--------------------------------------------------------------------------------------------
```

**Report this honestly.** Neither component changes standard accuracy. The original
paper asserts both are essential; they are not, *by this measurement* — and the next
table explains why that measurement cannot detect them.

## TABLE III · COLD-CONCEPT ABLATION  ← the paper's central result

```
Evaluated on concepts whose assessments were withheld entirely (27 of 89)

Variant                            AUC-ROC            RMSE
--------------------------------------------------------------------------------------------
Proposed, with knowledge graph     0.9004 ± 0.0059    0.3359 ± 0.0059
Proposed, without                  0.8679 ± 0.0042    0.3611 ± 0.0113
--------------------------------------------------------------------------------------------
Difference                         +0.0325            p = 0.0003    Cohen's d = 5.06
```

**Why this experiment exists.** Standard accuracy is computed only at assessment
positions, and only **89 of 237 concepts carry assessments** — each with thousands of
labelled examples, which an ordinary embedding table learns perfectly well on its own.
The knowledge graph's purpose is the remaining **148 concepts**, which never enter that
computation. The conventional ablation is structurally unable to detect what it ablates.

Withholding all assessments for 27 of the 89 assessed concepts, stripping their
responses from the model's input so they behave like genuinely unassessed concepts, and
scoring only those, the graph wins on **every fold**.

> **Knowledge-graph propagation contributes nothing where direct supervision is
> plentiful, and contributes substantially where it is absent. Standard ablation
> protocols cannot detect this.**

This is the contribution. It is measured, non-obvious, testable by others, and it
answers the novelty criticism in a way that "we combined DKT, GCN and RL" cannot.

## TABLE IV · LEARNING PATH QUALITY

```
Planner                    NDCG@5            Hit@5    Prec@5   Prereq viol.  Coverage
--------------------------------------------------------------------------------------------
Curriculum (syllabus)      0.0612 ± 0.154    0.1832   0.0453   0.0000        0.452
Popularity                 0.1069 ± 0.199    0.2936   0.0808   0.0000        0.510
Random (legal moves)       0.1300 ± 0.218    0.3455   0.0863   0.0000        0.722
Weakest-first              0.1344 ± 0.207    0.3642   0.0956   0.0000        0.515
Proposed (greedy)          0.1532 ± 0.238    0.3687   0.1051   0.0000        0.726
--------------------------------------------------------------------------------------------
```

906 recommendation decisions from 400 held-out learners who passed or achieved
distinction. Ground truth is the set of concepts each learner **actually moved to next** —
observed behaviour, not simulation.

```
Paired comparison of NDCG@5 against the proposed planner (Holm-corrected)

vs curriculum       +0.0921    p < 0.001    d = 0.36  (small)
vs popularity       +0.0464    p < 0.001    d = 0.17  (negligible)
vs random           +0.0232    p = 0.0125   d = 0.09  (negligible)
vs weakest-first    +0.0189    p = 0.0167   d = 0.08  (negligible)
```

**State the effect sizes, not only the p-values.** With 906 paired decisions trivial
differences reach significance. The honest reading: *the prerequisite mask does most of
the work; the learned ranking adds a small but consistent improvement.* A paper claiming
only "p < 0.05, our method wins" would be technically true and substantively misleading.

**Prerequisite violations are 0.0000** across every planner and all 906 decisions. The
pedagogical guarantee is verified, not assumed. That is a claim the original could not
make.

## TABLE V · CALIBRATION  ← new, and it supports your trust claim

```
                                   ECE
Before temperature scaling         0.0084
After temperature scaling          0.0054
Fitted temperature per fold        1.022, 1.066, 1.068, 1.024, 1.046
```

Temperature fitted on one half of each fold's held-out predictions and scored on the
other, so the reported ECE is not self-flattering.

This matters specifically for your paper: the system **displays mastery values to
students** inside explanations. An uncalibrated 0.41 shown to a learner is a false
statement. ECE 0.0054 makes "trustworthy" a measurement rather than an adjective.

## TABLE VI · COLD-START PERFORMANCE  ← new

```
Prior interactions    AUC-ROC
------------------------------------
0                     0.7421 ± 0.0353
1–4                   0.9449 ± 0.0086
5–9                   0.9559 ± 0.0062
10–19                 0.9539 ± 0.0056
20+                   0.9514 ± 0.0023
```

Your related-work section criticises cold start in prior systems and never reports your
own. With no history at all the model still reaches 0.742 using the learner-profile
prior, and four interactions are enough to reach 0.945.

## TABLE VII · EXPLANATION QUALITY  ← new, replaces the survey

```
Measure         Value     Interpretation
--------------------------------------------------------------------------------
Fidelity        0.5318    Spearman correlation between cited prerequisite deficits
                          and measured counterfactual improvement
Sufficiency     0.0020    Share of predicted gain attributable to the concept named
Stability       0.5310    Jaccard overlap of cited gaps under ±0.02 mastery
                          perturbation
```

A Likert survey records whether a reader *liked* an explanation. These record whether it
is *true of the model*.

**Sufficiency of 0.002 is a finding worth reporting, not hiding.** The explanation was
true but radically incomplete: 99.8% of the predicted benefit came from transfer to
concepts the text never mentioned. The renderer was changed to name the beneficiaries. No
survey would have surfaced this.

## TABLE VIII · DATASET AND GRAPH  (replaces §IV-B)

```
Quantity                              Value
--------------------------------------------------------------------------------
Distinct students                     28,785
Enrolments                            32,593
Clickstream events                    10,655,280
Assessment records                    173,912
Learning materials                    6,364
Concepts derived                      237
  prerequisite edges                  724   (285 mined, 252 assessment, 230 sequence)
  corequisite edges                   302
  cycles requiring removal            0
  concepts carrying assessments       89 of 237
Sequences                             27,904  (median 169 events)
Supervised tokens                     242,670
Label balance                         67.9% / 32.1%
Placement proxy validation            Spearman ρ = 0.975
```

## TABLE IX · EFFICIENCY  (replaces the 187 ms / 112 ms claim)

```
Model parameters              466,569   (85% transformer, 6% graph encoder)
End-to-end recommendation     ~160 ms   (Intel Core i5-8257U, CPU only)
Cached mastery lookup         7 ms
Training, per configuration   ~25 min   (free Colab T4, 5 folds x 12 epochs)
Full ETL from raw CSV         ~2.5 min
```

---

# Part 3 — The mathematics R3 asked for

The paper currently contains **zero equations**. Add these to §III.

## Knowledge-graph concept encoding

Concept embeddings from a 3-layer graph convolution over the symmetric normalised
adjacency of the prerequisite graph:

```
Â = D^(−1/2) (A + I) D^(−1/2)

H⁽⁰⁾ = E                                          E ∈ ℝ^(C×64)
H⁽ˡ⁺¹⁾ = Dropout( ReLU( LayerNorm( Â H⁽ˡ⁾ W⁽ˡ⁾ ) ) )        l = 0,1,2
Z = H⁽³⁾                                          Z ∈ ℝ^(C×64)
```

`Z` is recomputed on every forward pass, so gradients flow into the graph structure.

## Interaction embedding

For interaction *t* with concept c_t, response r_t ∈ {correct, incorrect, unlabelled},
and kind k_t ∈ {context, assessment}:

```
x_t = W_c Z[c_t] + W_r e(r_t) + W_k e(k_t) + W_f f_t + W_s s          x_t ∈ ℝ^128

f_t = [ log(1+clicks_t), Δt_prev, Δt_same-concept, day_t / span ]
s   = static learner features (56-dimensional)
```

## Time-aware attention — the forgetting mechanism

```
bias[h,i,j] = − softplus(γ_h) · log( 1 + Δdays(i,j) )        γ_h learned per head

A[h] = softmax( (Q_h K_hᵀ)/√d_k + bias[h] + M )              M = causal + padding mask
```

Older interactions are attenuated at a rate learned separately per head. Setting γ = 0
recovers ordinary attention, which is exactly the `− temporal attention` ablation.

## Prediction and loss

```
ĥ_t = Transformer(x_1..x_t)
ŷ_t = σ( MLP( [ ĥ_(t−1) ; Z[c_t] ] ) )

L = − (1/|S|) Σ_(t∈S) [ y_t log ŷ_t + (1−y_t) log(1−ŷ_t) ]
```

where S is the set of **assessment** positions only. Context interactions update the
hidden state but contribute nothing to the loss — this is what makes a 200-length
sequence model trainable when learners average only 8.7 assessments each.

## Calibration

```
T* = argmin_T  BCE( σ(z/T), y )     on held-out fold predictions
ŷ_calibrated = σ(z / T*)
```

## Mastery state

```
m_t ∈ ℝ^C,   m_t[c] = σ( MLP( [ ĥ_t ; Z[c] ] ) )    for all c
```

## The MDP — currently asserted, never defined

```
State     s_t = [ m_t ‖ coverage_t ‖ s ‖ budget_t ]     ∈ ℝ^(2C+56+1)
Action    a_t ∈ {1..C}, the next concept to study
Legal     a is legal ⟺  m_t[a] < τ_mastered
                    ∧  ∀p ∈ pre(a) : m_t[p] ≥ τ_ready
                    ∧  a ∈ enrolled(learner)

Reward    r_t = α Σ_c max(0, Δm[c]) + β·1[prereqs met] − δ·1[already mastered] − η
          α = 1.0, β = 0.2, δ = 0.5, η = 0.05

Transition  m_(t+1) = KGDKT( s_t ⊕ (a_t, ŷ) ),  ŷ ~ Bernoulli(m_t[a_t])
Terminal    target set mastered, or budget exhausted
```

## Greedy planner scoring

```
score(a) = Σ_(c ∈ enrolled)  [ logit( m_(t+1)^(studied a)[c] ) − logit( m_(t+1)^(idle)[c] ) ]
```

Two details worth stating explicitly, because both were wrong in earlier
implementations and each silently breaks the planner:

1. The baseline is an **idle week**, not the present moment. Studying advances the clock,
   and the forgetting mechanism then decays every prior interaction — against a
   present-moment baseline every action appears harmful.
2. Scoring is in **log-odds**. Engaged learners sit near 0.98 on almost every concept,
   where probability differences round to zero and the ranking becomes arbitrary.

## Priority backtracking

```
deficit(a)  = max(0, τ_ready − m[a])
priority(a) = deficit(a) · ρ^(dist(a → c*) − 1)        ρ = 0.6, depth ≤ 4
```

Gaps are reported in descending priority. This is the mechanism the paper names but
never specifies.

## Algorithm block

```
Algorithm 1: Explained recommendation
Input:  learner history h, prerequisite graph G, trained model θ, k
Output: k recommendations, each with an explanation

1  Z ← GCN(Â)                                    concept embeddings
2  m ← mastery(h, Z, θ)                          calibrated, all C concepts
3  τ ← percentiles(m restricted to enrolled)     learner-relative thresholds
4  L ← { a : legal(a, m, τ, G) }                 prerequisite-filtered
5  if L = ∅ then L ← relax_prerequisites(m, τ)
6  m_idle ← mastery(idle(h), Z, θ)               same elapsed time, no study
7  for a ∈ L in one batched pass:
8      m_a ← mastery(h ⊕ (a, correct), Z, θ)
9      score(a) ← Σ_c [ logit m_a[c] − logit m_idle[c] ]
10 A ← top-k of L by score
11 for a ∈ A:
12     gaps ← backtrack(G, m, a, τ)              ranked by priority
13     cf   ← counterfactual(a, gaps, θ)         computed, not asserted
14     text ← render(a, gaps, cf, beneficiaries)
15 return A with explanations
```

---

# Part 4 — Sections the paper is missing

## Ethics statement

> All data used in this study comes from the Open University Learning Analytics Dataset,
> which is publicly released under CC-BY 4.0 and contains no personally identifying
> information; students are represented by anonymised identifiers. No new data was
> collected from human participants for the experiments reported here. [If the user
> study is conducted, add: approval, informed consent procedure, participant
> compensation, and the right to withdraw.] The system produces advisory
> recommendations; it makes no determinative decision about any student.

## Limitations

> **The prediction target is successful completion, not knowledge.** Because OULAD
> records only submitted assessments, the observed pass rate at the threshold of 40 is
> 0.956 and 46.3% of expected submissions never occurred. Treating non-submission as
> failure restores a usable class balance but means a substantial share of the signal is
> engagement rather than understanding. This is the appropriate target for a system
> intended to intervene, but it is a weaker claim than measuring mastery.
>
> **The concept graph is derived, not given.** OULAD contains no concept annotations,
> prerequisite relations or material titles. Concepts are defined as (module,
> week-of-study) pairs and edges are mined from behaviour. Concepts therefore cannot be
> named by subject matter, and labels describe position and activity composition instead.
>
> **Placement relies on observed access for 82% of materials.** Only 1,121 of 6,364
> materials carry scheduling metadata. The fallback is validated at ρ = 0.975 on the
> subset carrying both signals, but deriving concept order from behaviour and then mining
> prerequisites from behaviour is partially self-referential.
>
> **Supervision covers 89 of 237 concepts.** Mastery estimates for the remaining 148
> reach their values through prerequisite propagation and are not directly calibrated.
>
> **Mastery is bimodal.** Engaged learners score near 0.98 on essentially every concept
> and disengaged learners near 0.00, so absolute thresholds do not discriminate. The
> planner uses learner-relative percentiles.
>
> **Timestamps are day-granular.** OULAD provides integer day offsets with no intra-day
> resolution, so a 30-minute inactivity threshold is not computable; one student-day is
> treated as one session.

## Threats to validity

> **Construct validity.** Concepts are inferred from course structure rather than
> annotated by subject experts. A sample of mined prerequisite edges should be validated
> by expert rating; this has not yet been done.
>
> **Internal validity.** The proposed model conditions on learner demographics and
> engagement features that the DKT and SAKT baselines do not, since those follow their
> original formulations. Part of the reported improvement may therefore reflect feature
> access rather than architecture. An ablation removing learner features would separate
> the two and has not been run.
>
> **External validity.** All results come from a single institution's distance-learning
> data across seven modules. Generalisation to campus-based teaching or other
> institutions is untested.
>
> **Statistical validity.** Cross-validation uses five folds on one dataset. Significance
> is reported with Holm correction and effect sizes, but the path-quality effect sizes
> against random are negligible (d = 0.09), and this should not be presented as a large
> improvement.

## Data and code availability

> The OULAD dataset is publicly available at
> research.stem.open.ac.uk/ouanalyse/dataset under CC-BY 4.0. All code, SQL
> transformations, configuration and generated results are available at
> github.com/abhay27singh/explainable-learning-path-recommender. Every table and figure
> in this paper is regenerated from raw data by a single command, with fixed random
> seeds.

## Contributions list (end of §I)

> 1. A knowledge-graph-conditioned knowledge tracing model with a learned per-head
>    temporal decay, achieving AUC-ROC 0.9538 ± 0.0019 and expected calibration error
>    0.0054 on OULAD.
> 2. **A demonstration that conventional ablation cannot detect the contribution of a
>    knowledge graph**, and a cold-concept protocol that can. Prerequisite propagation
>    changes standard accuracy by −0.0001 (n.s.) but improves accuracy on concepts
>    without direct supervision by +0.0325 (p = 0.0003, d = 5.06).
> 3. A prerequisite-constrained planner with a structurally enforced pedagogical
>    guarantee, verified at zero violations across 906 recommendation decisions.
> 4. Three objective measures of explanation quality — fidelity, sufficiency, stability —
>    replacing subjective survey scores, one of which revealed that explanations
>    accounted for only 0.2% of the benefit they claimed.
> 5. A fully reproducible pipeline and deployed REST service, released publicly.

---

# Part 5 — Recommended restructuring

The novelty scores of 1, 1, 2 reflect a real problem: **DKT + GCN + RL is a
well-investigated combination.** Presenting that architecture as the contribution will
attract the same response at any venue.

Restructure around the finding instead:

| Section | Current framing | Proposed framing |
|---|---|---|
| Title | "...Recommendation System for University Students" | "When Does a Knowledge Graph Help Knowledge Tracing? Evidence from Concepts Without Supervision" |
| Abstract | "we propose a framework with three components" | "we show conventional ablation cannot detect graph contribution, and give a protocol that can" |
| §V headline | Table I accuracy | Table III cold-concept result |
| §III | Prose description of three modules | Formal specification (Part 3) |
| Contribution | An architecture | A measured, testable finding about evaluation methodology |

Same system, same experiments, honest numbers — and a contribution that is genuinely
novel rather than a recombination. The reviewers did not reject the topic. They rejected
a paper that asserted results instead of producing them, and described an algorithm
without specifying it. Both are now fixable from work already completed.

---

# Part 6 — Still outstanding

| Item | Effort | Needed for |
|---|---|---|
| `no_student` ablation — separate architecture from feature access | one 25-min GPU run | Threats to validity |
| Fairness analysis by gender, disability, IMD band, age | ~half a day, CPU | Likely desk-reject without it |
| Expert validation of ~50 mined prerequisite edges | 3 raters, one afternoon | Construct validity |
| User study, n = 30, with ethics approval | one week elapsed | §V-C, if you keep the survey |
| DQN, if the RL claim stays | 6–9 days, GPU | Otherwise move to future work |

The DQN is worth reconsidering. The greedy planner beats random by **d = 0.09**, so
there is very little headroom for a reinforcement-learning agent to demonstrate. Framing
it as future work is defensible and costs the paper nothing.
