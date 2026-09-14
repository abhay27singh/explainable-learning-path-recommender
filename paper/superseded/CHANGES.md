# What Changed in the Revision

Rejection: COMPUTINGCON 2026, paper 1757. Novelty scored 1, 1, 2; primary reason
"Less Technical Contribution". Every change below traces to that.

Title is unchanged — assigned by the department. The contribution is carried by the
abstract and the contributions list instead, which is where reviewers form the novelty
judgement.

---

## 1. Removed — fabricated numbers

Every one of these appeared in the submitted paper and none was produced by a run.

| Claim in submitted paper | Replaced with (measured) |
|---|---|
| AUC-ROC 0.892 | **0.9538 ± 0.0019** |
| RMSE 0.143 | **0.2621 ± 0.0026** |
| "baseline models' AUC-ROC of 0.871" — a figure absent from Table I | named baselines with real values and *p*-values |
| NDCG@10 0.856 | **NDCG@5 0.1532 ± 0.238** |
| Completion rate 0.784 | removed — never measured |
| Knowledge gain 0.234 | removed — never measured |
| Transparency 4.21 / 5, 42 participants | removed — study never run |
| Ablation deltas showing each component helping | **actual ablation: −0.0001 (p = 0.77) and −0.0006 (p = 0.45)** |

The abstract previously claimed 0.892 while citing a 0.871 baseline that appears in no
table. That internal contradiction alone is a desk-reject risk.

## 2. Removed — components that do not exist

- **DQN agent.** Not implemented. The MDP formalism is *kept* as the problem definition
  (state Eq. 11, admissible set Eq. 12, reward Eq. 13) because a reviewer asking for
  rigour wants it, but Section III-D now states plainly that **the policy evaluated is
  greedy**, and a learned policy is future work. Section V-D quantifies the headroom a
  learned policy would have to beat (*d* = 0.09 over random-among-admissible), which
  turns the omission into a stated research question rather than a gap.
- **PyTorch Geometric, Stable-Baselines3, RDFLib.** Removed from the implementation
  section. None is a dependency.
- **User study (42 participants).** Removed entirely, per your decision. Replaced by
  three objective measures — fidelity, sufficiency, stability — which is a stronger
  position: a rating scale cannot detect an explanation that omits something, and ours
  did omit something (see §4 below).

## 3. Added — the contribution that answers the novelty score

This is the change that matters. New Section V-B and V-C, new Table III, contribution 2,
and the centre of the abstract.

> Standard ablation says the knowledge graph does nothing: 0.9539 without vs 0.9538
> with, *p* = 0.77.
>
> That is a measurement artefact. Accuracy is scored only at assessment positions, and
> **only 89 of 237 concepts carry assessments.** The graph exists for the other 148,
> which never enter the metric.
>
> Cold-concept protocol — withhold *all* assessments for 27 of the 89 assessed
> concepts, strip their responses from the input, score only those:
> **0.9004 vs 0.8679, +0.0325, p = 0.0003, Cohen's d = 5.06, winning every fold.**

Stated as a claim about the literature, not just this model: component ablations should
come with evidence that the reporting metric can observe the component under test.

Section ordering in §V was changed to accuracy → why the obvious test fails → the test
that works, so the contribution is read before the incremental numbers.

## 4. Added — honest negative results

Reviewers reward these; they are what separate a real evaluation from a results table.

- **Sufficiency = 0.0020.** The concept named in an explanation accounted for 0.2% of the
  predicted benefit. Explanations were accurate but incomplete. Reported as a finding,
  with the fix noted.
- **Effect sizes alongside every *p*-value.** Against random-among-admissible the planner
  wins at *d* = 0.09 — negligible. The paper now says the prerequisite mask does most of
  the work and the ranking adds a small consistent improvement, rather than implying the
  model is responsible for the whole margin.
- **Baseline feature asymmetry.** DKT and SAKT follow their original formulations and do
  not see learner features; the proposed model does. Stated in Limitations, since part of
  the margin may be feature access rather than architecture.

## 5. Added — sections the submitted paper lacked entirely

| Section | Why |
|---|---|
| Contributions list (end of §I) | Reviewer 2: "details of the author's contribution are too little". IEEE reviewers look for this list. |
| §VI-B Limitations | Five stated, including that the target is *completion*, not knowledge. |
| §VI-C Ethical Considerations | Required for educational data. States that the model conditions on gender, disability, deprivation and age, that no subgroup analysis has been run, and that it must not be deployed for progression decisions until one is. |
| §VI-D Future Work | Fairness analysis (first), learned policy, user study, expert edge validation. |
| Reproducibility | Dataset URL, licence, repository, one-command regeneration. |
| Tables V–IX | Calibration, cold-start, explanation quality, dataset, efficiency. |

**Fairness is future work only, per your decision** — written into §VI-C and §VI-D as a
stated precondition for deployment, not run. That is defensible: an acknowledged,
scheduled gap reads far better than silence on the question.

## 6. Corrected — factual errors

| Error | Correction |
|---|---|
| "in the field of drug diversion" — copy-paste from another paper | deleted |
| "Dynamic Knowledge Tracing" (×3) | **Deep** Knowledge Tracing |
| "30-minute inactivity threshold" for sessionization | OULAD is day-granular; a session is one learner-day |
| Hardware overstated | Intel Mac (CPU) for data and inference, single NVIDIA T4 for training |
| §III claimed explainability is "integrated from the design phase, rather than post-hoc" while the abstract said post-hoc | kept post-hoc; novelty restated as *graph-grounded and objectively measured*, which is true |
| 15 index terms | 6 (IEEE convention ≈ 5) |
| Missing citations for DKT, SAKT, temperature scaling, Holm | added as [5], [8], [10], [11] |

## 7. Added — methodological detail reviewers will check

- Folds **grouped by student**, stated explicitly — no learner in both train and test.
- Holm–Bonferroni correction named, with the comparison family defined.
- Label construction explained: 150,013 of 323,925 expected submissions never occurred;
  non-submission treated as failure, restricted to the registered window; 95.6% → 67.9%.
- Concept placement validated at ρ = 0.975 on the 1,121 materials carrying both signals.
- Mining rule stated with its acceptance thresholds and the reason lift is required.
- Why log-odds scoring and an idle baseline are necessary (Eq. 14) — both are
  non-obvious and both were bugs before they were design decisions.
- Temperature fitted on one half of held-out predictions, scored on the other.

---

## Files

| File | Use |
|---|---|
| `paper.tex` | IEEEtran source — compile on Overleaf |
| `paper.md` | Same content, readable/editable |
| `Revised-Paper.pdf` | 12 pages, for reading and for your teacher |
| `Revised-Paper.docx` | If the department wants Word |

## Page budget

`paper.tex` runs past six pages with all nine tables. Trim in this order:

1. Table VI (cold-start)
2. Table V (calibration)
3. Table VIII (dataset)
4. Table IX (efficiency)

Never cut Tables I, III or IV. I is the headline, **III is the contribution**, IV is the
recommendation evidence.
