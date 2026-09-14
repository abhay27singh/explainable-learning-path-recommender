# Replacement Abstract, Contributions and §V Opening

Title is fixed by the department:

> **Explainable AI-Based Personalized Learning Path Recommendation System for
> University Students**

That is fine. A title names the topic; the abstract and the contributions list carry the
contribution. Reviewers form their novelty judgement from those, which is where the
original lost its 1, 1, 2.

---

## 1. Abstract — replaces the current one

> **Abstract** — Digital learning platforms in higher education generate large volumes
> of interaction data that could support personalised learning-path recommendation, but
> existing educational recommenders offer little transparency and are rarely evaluated in
> a way that establishes which of their components actually contribute. This paper
> presents an explainable learning-path recommendation framework combining
> knowledge-graph-conditioned knowledge tracing, prerequisite-constrained planning, and
> priority-backtracking explanation, evaluated on the Open University Learning Analytics
> Dataset (OULAD). The knowledge tracing model attains AUC-ROC 0.9538 ± 0.0019 and
> expected calibration error 0.0054 under five-fold cross-validation with folds grouped by
> student, significantly exceeding DKT (0.9435 ± 0.0017, p = 0.0009) and a
> graph-based baseline (0.9502 ± 0.0014, p = 0.022) after Holm correction.
> **A central finding concerns evaluation rather than architecture.** Removing
> knowledge-graph propagation altogether changes predictive accuracy by −0.0001
> (p = 0.77), which would ordinarily be read as the graph contributing nothing. That
> reading is an artefact of the measurement: accuracy is computed only at assessment
> positions, and only 89 of 237 concepts carry assessments. Under a cold-concept protocol
> that withholds all assessments for a subset of concepts and scores only those, the same
> graph improves accuracy by +0.0325 (p = 0.0003, Cohen's d = 5.06), winning on every
> fold. Knowledge-graph propagation therefore contributes nothing where direct supervision
> is plentiful and substantially where it is absent, and conventional ablation cannot
> detect the difference. Planning is evaluated against the concepts learners actually
> studied next, with zero prerequisite violations across 906 recommendation decisions, and
> explanation quality is assessed by objective fidelity, sufficiency and stability rather
> than by subjective rating. All code, data transformations and results are released
> publicly.

**Index Terms** — Explainable AI, knowledge tracing, knowledge graphs, learning path
recommendation, educational data mining, ablation methodology.

*(Six terms. The original had fifteen; IEEE convention is around five.)*

### What changed and why

| Original abstract | Replacement |
|---|---|
| "consists of three technical components" | leads with a finding, not a component list |
| AUC 0.892, RMSE 0.143 (fabricated) | 0.9538 ± 0.0019, ECE 0.0054 (measured) |
| "reducing the baseline models' AUC-ROC of 0.871" — a figure absent from Table I | named baselines with their real values and p-values |
| no mention of significance or variance | ± and Holm-corrected p throughout |
| NDCG 0.856, completion 0.784, knowledge gain 0.234 (fabricated) | replaced by the prerequisite guarantee and real evaluation protocol |
| transparency 4.21/5 from a study never run | objective fidelity / sufficiency / stability |
| DQN reinforcement learning agent (never built) | "prerequisite-constrained planning" — accurate |
| no claim a reviewer could call novel | the evaluation-methodology finding, stated plainly |

Note the DQN is gone. Describing a component you did not implement is the single most
dangerous sentence in the paper, and the planner you *do* have works and is measured.

---

## 2. Contributions list — add at the end of §I

The original §I ends without one. IEEE reviewers look for it, and its absence is part of
why "the details of the author's contribution are too little" (Reviewer 2).

> The contributions of this work are as follows.
>
> 1. A knowledge-graph-conditioned knowledge tracing model with a learned per-head
>    temporal decay, attaining AUC-ROC 0.9538 ± 0.0019 and expected calibration error
>    0.0054 on OULAD under student-grouped five-fold cross-validation.
> 2. **Evidence that conventional ablation cannot detect the contribution of a knowledge
>    graph, and a cold-concept protocol that can.** Prerequisite propagation changes
>    standard accuracy by −0.0001 (p = 0.77) yet improves accuracy on concepts lacking
>    direct supervision by +0.0325 (p = 0.0003, d = 5.06).
> 3. A prerequisite-constrained planner whose pedagogical guarantee is enforced
>    structurally rather than learned, verified at zero violations across 906
>    recommendation decisions evaluated against observed learner behaviour.
> 4. Three objective measures of explanation quality — fidelity, sufficiency and
>    stability — in place of subjective rating. One of these revealed that generated
>    explanations accounted for only 0.2% of the predicted benefit they described, a
>    defect no survey instrument would surface.
> 5. A reproducible pipeline from raw data to every reported figure, together with a
>    deployed REST service, released publicly.

Contribution 2 is the one that answers the novelty criticism. Put it second so it is
read.

---

## 3. §V opening — lead with the finding

Currently §V opens with Table I accuracy. Reorder so the contribution is encountered
before the incremental numbers.

> **V. RESULTS AND DISCUSSION**
>
> **A. Predictive Accuracy**
>
> Table I reports predictive performance under five-fold cross-validation, with folds
> grouped by student so that no learner appears in both training and test partitions. The
> proposed framework attains AUC-ROC 0.9538 ± 0.0019, exceeding DKT by 0.0103
> (p = 0.0009) and the graph-based baseline by 0.0036 (p = 0.022) after Holm–Bonferroni
> correction across the comparison family. A majority-class baseline returns exactly
> 0.5000, confirming that the reported discrimination is not an artefact of the
> 67.9/32.1 class distribution, and SAKT — attention without graph conditioning or
> temporal decay — reaches 0.9233, indicating that the improvement does not follow
> simply from the use of attention.
>
> **B. Why Conventional Ablation Is Insufficient**
>
> Table II reports the standard component ablation. Removing knowledge-graph propagation
> changes accuracy by −0.0001 (p = 0.77) and removing temporal attention by −0.0006
> (p = 0.45). Read at face value, neither component contributes.
>
> This conclusion is an artefact of the measurement. Predictive accuracy is computed only
> at assessment positions, and assessments exist for **89 of the 237 concepts**, each
> carrying on the order of 2,700 labelled examples. A free embedding table learns such
> concepts adequately without structural information. The role of the knowledge graph is
> to support the remaining **148 concepts**, which never enter the accuracy computation.
> The conventional ablation is therefore structurally incapable of detecting the component
> it removes.
>
> **C. Cold-Concept Evaluation**
>
> To measure the contribution directly, all assessments were withheld for 27 of the 89
> assessed concepts, their observed responses were removed from the model input so that
> they behave as genuinely unassessed concepts, and evaluation was restricted to those
> concepts alone. Table III reports the outcome: 0.9004 ± 0.0059 with graph propagation
> against 0.8679 ± 0.0042 without, a difference of +0.0325 (p = 0.0003, d = 5.06) in
> which the graph-conditioned model wins on every fold.
>
> Knowledge-graph propagation thus contributes nothing where direct supervision is
> plentiful and substantially where it is absent. This has a practical consequence for
> the recommender, which must rank across all 237 concepts, the majority of which carry
> no assessment of their own; and a methodological consequence for the literature, in
> which component ablations are routinely reported on metrics that cannot observe the
> component in question.

That subsection ordering — accuracy, then why the obvious test fails, then the test that
works — is what turns a null result into the paper's contribution.

---

## 4. One sentence to delete from §III

The paper contradicts itself on its own central claim:

- **Abstract and §I:** "post-hoc explanation generation"
- **§III-A / §III-C:** explainability "integrated... from the design phase, rather than
  in isolation", and "remove the post-hoc explainability constraints"

The architecture is post-hoc: backtracking runs after the planner selects. Keep the
abstract's wording and delete the §III claim to the contrary. The novelty is not that the
explanation is non-post-hoc; it is that it is **grounded in a prerequisite graph and in
counterfactuals the model can actually evaluate**, and that its quality is measured
objectively.
