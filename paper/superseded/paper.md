# Explainable AI-Based Personalized Learning Path Recommendation System for University Students

**Abhay Pratap Singh**, **Akshay Guleria**, **Uday Chaturvedi**, **Gursimar Makkar**, **Shikha Kamal**

*Department of Computer Science, Chandigarh University, Mohali, India*

`23BCS11784@cuchd.in` · `23BCS10517@cuchd.in` · `23BCS10807@cuchd.in` · `23BCS11455@cuchd.in` · `shikha.e12552@cumail.in`

---

## Abstract

Digital learning platforms in higher education generate large volumes of interaction data that could support personalised learning-path recommendation, but existing educational recommenders offer little transparency and are rarely evaluated in a way that establishes which of their components actually contribute. This paper presents an explainable learning-path recommendation framework combining knowledge-graph-conditioned knowledge tracing, prerequisite-constrained planning, and priority-backtracking explanation, evaluated on the Open University Learning Analytics Dataset (OULAD). The knowledge tracing model attains AUC-ROC 0.9538 ± 0.0019 and expected calibration error 0.0054 under five-fold cross-validation with folds grouped by student, significantly exceeding Deep Knowledge Tracing (0.9435 ± 0.0017, *p* = 0.0009) and a graph-based baseline (0.9502 ± 0.0014, *p* = 0.022) after Holm correction. A central finding concerns evaluation rather than architecture. Removing knowledge-graph propagation changes predictive accuracy by −0.0001 (*p* = 0.77), which would ordinarily be read as the graph contributing nothing. That reading is an artefact of the measurement: accuracy is computed only at assessment positions, and only 89 of 237 concepts carry assessments. Under a cold-concept protocol that withholds all assessments for a subset of concepts and scores only those, the same graph improves accuracy by +0.0325 (*p* = 0.0003, Cohen's *d* = 5.06), winning on every fold. Knowledge-graph propagation therefore contributes nothing where direct supervision is plentiful and substantially where it is absent, and conventional ablation cannot distinguish the two cases. Planning is evaluated against the concepts learners actually studied next, with zero prerequisite violations across 906 recommendation decisions, and explanation quality is assessed by objective fidelity, sufficiency and stability rather than by subjective rating. All code, data transformations and results are released publicly.

**Index Terms** — Explainable AI, knowledge tracing, knowledge graphs, learning path recommendation, educational data mining, ablation methodology.

---

## I. Introduction

The rapid digitisation of higher education has created unprecedented opportunities for data-driven personalisation, yet the one-size-fits-all curriculum remains common in most universities [1]. Students choosing between electives, or working through a compressed degree programme, rarely receive guidance matched to their prior experience, pace of study and intended direction. In cumulative subjects a gap in a foundational concept can produce persistent difficulty for the remainder of a programme.

Adaptive recommender systems are intended to address this, but two limitations recur. First, many systems operate as black boxes, offering no account of why a particular activity was suggested [12]. Students and academic advisers need that account in order to judge whether a suggestion fits their situation; without it, even well-founded recommendations are disregarded. Second, personalisation work has concentrated on adaptive feedback delivery rather than on modelling how a learner's knowledge evolves and constructing coherent sequences over it.

A third limitation is less often discussed and motivates the present work. Systems of this kind are routinely assembled from several components — a sequence model, a structural prior such as a knowledge graph, a planner — and each component is justified by an ablation showing that removing it degrades accuracy. We show that this evaluation can be systematically uninformative. When a structural prior exists precisely to support items for which direct supervision is scarce, and accuracy is measured only where supervision is abundant, the ablation measures the component in the one regime where it cannot matter.

This paper makes the following contributions.

1. A knowledge-graph-conditioned knowledge tracing model with a learned per-head temporal decay, attaining AUC-ROC 0.9538 ± 0.0019 and expected calibration error 0.0054 on OULAD under student-grouped five-fold cross-validation.

2. **Evidence that conventional ablation cannot detect the contribution of a knowledge graph, together with a cold-concept protocol that can.** Prerequisite propagation changes standard accuracy by −0.0001 (*p* = 0.77) yet improves accuracy on concepts lacking direct supervision by +0.0325 (*p* = 0.0003, *d* = 5.06).

3. A prerequisite-constrained planner whose pedagogical guarantee is enforced structurally rather than learned, verified at zero violations across 906 recommendation decisions evaluated against observed learner behaviour.

4. Three objective measures of explanation quality — fidelity, sufficiency and stability — in place of subjective rating. One of these revealed that generated explanations accounted for only 0.2% of the predicted benefit they described, a defect no rating scale would surface.

5. A reproducible pipeline from raw data to every reported figure, together with a deployed service, released publicly.

The remainder of the paper is organised as follows. Section II reviews related work; Section III specifies the framework; Section IV details the implementation; Section V reports results; Section VI concludes.

---

## II. Related Work

Educational recommender systems have progressed from undifferentiated delivery towards adaptive personalisation. Early systems relied on content-based and collaborative filtering, which suffer from cold-start difficulties, data sparsity and limited scalability [1], [2]. Hybrid models combining several recommendation strategies improved accuracy and robustness, but concentrated on predicting learner preference rather than modelling knowledge acquisition or optimising sequential pathways.

Deep Knowledge Tracing (DKT) [5] models a learner's latent knowledge state over time using recurrent networks. Unlike Bayesian Knowledge Tracing [4], which relies on static per-skill parameters, DKT learns temporal dependencies directly from interaction data. Subsequent work introduced memory structures [6] and attention [8], [7], the latter allowing a model to attend directly to any earlier interaction rather than propagating information through a recurrent chain [9].

Reinforcement learning has been applied to learning-path recommendation by formalising the problem as a Markov decision process and optimising for long-term learning gain rather than immediate engagement. Knowledge graphs offer a complementary route to structured and explainable recommendation by modelling concepts, course units and their dependencies explicitly [1]. Graph neural networks have proven effective for the relational structure such graphs induce.

Several gaps remain. Explainability is frequently added after the fact rather than treated as a design requirement, and where it is evaluated at all, evaluation is usually by participant rating — which measures whether readers found an explanation agreeable, not whether it is faithful to the model producing it. Temporal dynamics, particularly forgetting, are often omitted, which compromises the pedagogical validity of suggested trajectories. Most relevant to this paper, the contribution of structural priors is customarily established by an ablation on aggregate predictive accuracy, without examining whether that metric is capable of observing the component in question. We return to this point in Section V.

---

## III. Methodology

### A. Overview

The framework has three components: a knowledge tracing model that estimates a learner's mastery over every concept, a planner that selects the next concept subject to prerequisite constraints, and an explanation module that justifies the selection by reference to the prerequisite graph. Explanation is post-hoc: it operates on a decision already taken. What distinguishes it from a generic attribution method is that it is grounded in graph structure and in counterfactuals the model can be queried for directly, and that its quality is measured objectively rather than surveyed.

### B. Concept Graph Construction

OULAD contains no concept annotations, prerequisite relations, or titles for learning materials. A concept graph is therefore derived rather than read off. A concept is defined as a (module, week-of-study) pair: the set of material a module presents in a given week. Week placement uses the `week_from` field where present (17.6% of materials) and otherwise the median day on which learners accessed the material. On the 1,121 materials carrying both signals the two agree at Spearman ρ = 0.975, which we report as validation of the substitute.

Edges arise from three rules. Sequence edges connect consecutive weeks within a module. Assessment-coverage edges connect the four preceding weeks to the week an assessment falls in. Mined edges are admitted where learner behaviour shows both consistent ordering and a measurable effect: for concepts *a*, *b* within a module,

$$\mathrm{prec}(a,b) = \frac{\left|\{u : \tau_u(a) < \tau_u(b)\}\right|}{\left|\{u : a, b \in h_u\}\right|},
\qquad
\mathrm{lift}(a,b) = \frac{P(y_b \mid a \prec b)}{P(y_b)},
\tag{1}$$

where τ*ᵤ*(·) is learner *u*'s first interaction with a concept. An edge is accepted when support exceeds 100 learners, prec ≥ 0.75 and lift > 1.1. Requiring lift is essential: precedence alone admits pairs separated only by the calendar. The resulting graph is acyclic without intervention.

### C. Knowledge-Graph-Enhanced Knowledge Tracing

Concept representations come from a three-layer graph convolution over the symmetrically normalised adjacency of the prerequisite graph,

$$\hat{A} = D^{-1/2}(A + I)D^{-1/2},
\tag{2}$$

where *A* ∈ ℝ^(C×C) is the weighted adjacency over *C* = 237 concepts. With *H*⁽⁰⁾ = *E* a learned embedding table,

$$H^{(l+1)} = \mathrm{Drop}\!\left(\mathrm{ReLU}\!\left(\mathrm{LN}\!\left(\hat{A} H^{(l)} W^{(l)}\right)\right)\right), \quad l = 0,1,2,
\tag{3}$$

giving *Z* = *H*⁽³⁾ ∈ ℝ^(C×64). *Z* is recomputed at every forward pass, so gradients propagate into the graph structure.

For interaction *t* with concept *c_t*, response *r_t* and kind *k_t*,

$$x_t = W_c Z[c_t] + W_r e(r_t) + W_k e(k_t) + W_f f_t + W_s s,
\tag{4}$$

where *s* denotes static learner features and *f_t* carries click volume, elapsed time since the previous interaction, elapsed time since the same concept was last seen, and position through the presentation.

Forgetting is modelled as an additive attention bias decaying with elapsed time,

$$b^{(h)}_{ij} = -\,\mathrm{softplus}(\gamma_h)\,\log(1 + \Delta_{ij}),
\tag{5}$$

with Δ*ᵢⱼ* the days between interactions *i* and *j* and γ*_h* learned per head, so that one head may attend to recent activity while another retains longer memory. Attention is

$$A^{(h)} = \mathrm{softmax}\!\left(\frac{Q^{(h)}{K^{(h)}}^{\top}}{\sqrt{d_k}} + b^{(h)} + M\right),
\tag{6}$$

where *M* combines the causal and padding masks. Setting γ*_h* = 0 recovers ordinary attention and constitutes the temporal ablation of Table II.

Writing *h_t* for the encoder state after interaction *t*,

$$\hat{y}_t = \sigma\!\left(\mathrm{MLP}\!\left([\,h_{t-1}\,;\,Z[c_t]\,]\right)\right),
\tag{7}$$

conditioning on the identity of *c_t* but never its response. Training minimises binary cross-entropy over assessment positions 𝒮 alone,

$$\mathcal{L} = -\frac{1}{|\mathcal{S}|}\sum_{t \in \mathcal{S}} \left[\, y_t \log \hat{y}_t + (1 - y_t)\log(1 - \hat{y}_t)\,\right].
\tag{8}$$

Clickstream interactions update the hidden state without contributing a loss term. This dual-signal construction is what makes a sequence model of length 200 trainable on a dataset in which learners average 8.7 assessments each.

Because predicted mastery is displayed to learners, probabilities are calibrated by temperature scaling [10],

$$T^\star = \arg\min_{T} \mathrm{BCE}\!\left(\sigma(z/T), y\right), \qquad \hat{y} = \sigma(z/T^\star),
\tag{9}$$

fitted on one half of each fold's held-out predictions and evaluated on the other. The mastery vector consumed by the planner evaluates the prediction head against every concept,

$$m_t[c] = \sigma\!\left(\mathrm{MLP}\!\left([\,h_t\,;\,Z[c]\,]\right)\right), \quad c = 1,\dots,C.
\tag{10}$$

### D. Prerequisite-Constrained Path Optimisation

Recommendation is formalised as a Markov decision process. The state concatenates the mastery vector, a coverage mask of concepts already recommended, the static learner features and the remaining budget,

$$s_t = [\, m_t \,\|\, o_t \,\|\, s \,\|\, b_t \,] \in \mathbb{R}^{2C + 57}.
\tag{11}$$

An action *a* ∈ {1, …, *C*} is admissible only if

$$m_t[a] < \tau_{\mathrm{m}} \;\wedge\; \forall p \in \mathrm{pre}(a): m_t[p] \geq \tau_{\mathrm{r}} \;\wedge\; a \in \mathcal{E},
\tag{12}$$

with ℰ the concepts of the learner's enrolled modules. Encoding the pedagogical constraint in the admissible set, rather than expecting a learned policy to discover it, makes an unprepared recommendation impossible by construction; this is verified empirically in Table IV. The reward

$$r_t = \alpha \sum_c \max(0, \Delta m[c]) + \beta \mathbb{1}[\mathrm{prereq}] - \delta \mathbb{1}[\mathrm{redundant}] - \eta
\tag{13}$$

uses α = 1.0, β = 0.2, δ = 0.5, η = 0.05.

Predicted mastery is bimodal in this setting: because non-submission is treated as failure (Section IV), engaged learners score near 0.98 on nearly every concept and disengaged learners near 0.00. Fixed thresholds therefore fail to discriminate, and τ_m, τ_r are set to the 70th and 30th percentiles of each learner's own mastery distribution.

The policy evaluated here is greedy with respect to one-step predicted improvement,

$$\mathrm{score}(a) = \sum_{c \in \mathcal{E}} \left[\, \mathrm{logit}\, m^{(a)}_{t+1}[c] - \mathrm{logit}\, m^{(\varnothing)}_{t+1}[c] \,\right].
\tag{14}$$

Two choices in (14) are load-bearing. The baseline *m*^(∅) is an equal elapsed period **without** study rather than the present state: studying advances the clock, and the decay of (5) then attenuates every earlier interaction, so that measuring against the present moment makes every available action appear harmful. Scoring in log-odds rather than probability is necessary because engaged learners saturate near 0.98, where probability differences round to zero and the ranking degenerates. A learned policy over (11)–(13) is left to future work; Section V reports how much headroom remains for one.

### E. Explanation by Priority Backtracking

For a recommended concept *c**, each prerequisite ancestor *a* is scored by its shortfall, discounted by graph distance,

$$\pi(a) = \max(0,\ \tau_{\mathrm{r}} - m[a]) \cdot \rho^{\,d(a,c^\star) - 1},
\tag{15}$$

with ρ = 0.6 and traversal depth at most four, so that an unmet prerequisite immediately preceding the recommendation outranks an equal shortfall further back. Ancestors are reported in descending π. Counterfactual statements are computed by raising a gap concept's mastery to τ_m and re-evaluating (14), so the claimed effect of closing a gap is measured rather than asserted. Algorithm 1 states the full procedure.

**Algorithm 1** Explained recommendation

```
Require: history h, graph G, parameters θ, budget k
Ensure:  k recommendations, each with an explanation
 1: Z ← GCN(Â)                                        ▷ Eq. (3)
 2: m ← Mastery(h, Z, θ)                              ▷ Eq. (10), calibrated
 3: τ ← Percentiles(m|_ℰ)
 4: L ← {a : Admissible(a, m, τ, G)}                   ▷ Eq. (12)
 5: if L = ∅ then L ← Relax(m, τ)
 6: m^(∅) ← Mastery(Idle(h), Z, θ)
 7: for all a ∈ L do                                  ▷ single batched pass
 8:     m^(a) ← Mastery(h ⊕ (a, correct), Z, θ)
 9:     score(a) ← Σ_{c∈ℰ} [ logit m^(a)[c] − logit m^(∅)[c] ]
10: A ← TopK(L, score, k)
11: for all a ∈ A do
12:     𝒢 ← Backtrack(G, m, a, τ)                      ▷ Eq. (15)
13:     𝒞 ← Counterfactual(a, 𝒢, θ)
14:     Render(a, 𝒢, 𝒞)
15: return A with explanations
```

---

## IV. Implementation

### A. Environment

The framework is implemented in Python 3.11 with PyTorch. Data preparation is performed in SQL against an embedded DuckDB database, so that every transformation is auditable without reading application code. Training used a single NVIDIA T4; inference and all data processing run on CPU. Table IX reports measured cost.

### B. Dataset and Preprocessing

Evaluation uses the Open University Learning Analytics Dataset [3], comprising 32,593 enrolments, 10,655,280 clickstream events and 173,912 assessment records across seven modules. Categorical attributes are one-hot encoded and missing assessment scores are median-imputed. OULAD timestamps are day-granular with no intra-day resolution; a session is therefore taken to be one learner-day.

Two properties of the data required specific treatment.

**Label construction.** OULAD records only *submitted* assessments. Binarising at the pass mark of 40 yields a 95.6% positive rate, with the fifth percentile of submitted scores equal to the pass mark itself; the failures are absent from the records rather than from reality. Of 323,925 expected submissions, 150,013 never occurred. Treating non-submission as failure, restricted to assessments falling while a learner remained registered, yields a 67.9 / 32.1 balance and changes the prediction target to successful completion.

**Concept supervision is sparse and uneven.** Assessments exist for only 89 of the 237 concepts. Mastery estimates for the remaining 148 are reached through prerequisite propagation, which is the condition Section V examines directly.

Sequences are split 80/10/10 and evaluated by five-fold cross-validation stratified on final outcome and **grouped by learner**, so that no learner contributes to both training and test partitions. Table VIII summarises the resulting corpus.

### C. Model Configuration

The knowledge tracing encoder uses two transformer layers, hidden dimension 128, four attention heads and a maximum sequence length of 200. The graph encoder uses three convolution layers with layer normalisation and ReLU, producing 64-dimensional concept embeddings. Dropout is 0.2. Optimisation uses Adam at learning rate 10⁻³ with cosine annealing, weight decay 10⁻⁴, batch size 64 and at most 12 epochs with early stopping at patience 3. The model has 466,569 parameters. Random seeds are fixed and recorded with every result.

### D. Evaluation Protocol

Knowledge tracing is assessed by AUC-ROC, RMSE and expected calibration error over assessment positions on held-out learners. Statistical comparisons use paired *t*-tests over the five folds with Holm–Bonferroni correction [11] across the comparison family, and Cohen's *d* is reported alongside, since with large numbers of paired decisions statistical significance is attainable at negligible effect size.

Path quality is assessed against observed behaviour rather than simulation: each held-out learner's history is truncated at several points, each planner is asked for a recommendation, and the ground truth is the set of concepts that learner actually engaged with over the following interactions. Only learners who passed or achieved distinction contribute ground truth.

---

## V. Results and Discussion

### A. Predictive Accuracy

Table I reports predictive performance. The proposed framework attains AUC-ROC 0.9538 ± 0.0019, exceeding DKT by 0.0103 (*p* = 0.0009) and the graph-based baseline by 0.0036 (*p* = 0.022) after correction. A majority-class baseline returns exactly 0.5000, confirming that the reported discrimination is not an artefact of the 67.9 / 32.1 class distribution. SAKT — attention without graph conditioning or temporal decay — reaches 0.9233, indicating that the improvement does not follow simply from the use of attention.

Calibration is reported in Table V. Because predicted mastery is shown to learners inside explanations, a poorly calibrated probability is an incorrect statement rather than a cosmetic defect; expected calibration error of 0.0054 after temperature scaling supports the values the interface displays.

### B. Why Conventional Ablation Is Insufficient

Table II reports the standard component ablation. Removing knowledge-graph propagation changes accuracy by −0.0001 (*p* = 0.77) and removing temporal attention by −0.0006 (*p* = 0.45). Read at face value, neither component contributes.

That reading is an artefact of the measurement. Predictive accuracy is computed only at assessment positions, and assessments exist for 89 of the 237 concepts, each carrying on the order of 2,700 labelled examples. A free embedding table learns such concepts adequately without structural information; the graph is redundant there. Its function is to support the remaining 148 concepts, which never enter the accuracy computation. The conventional ablation is structurally incapable of observing the component it removes.

### C. Cold-Concept Evaluation

To measure the contribution directly, all assessments were withheld for 27 of the 89 assessed concepts and their observed responses removed from the model input, so that they behave as genuinely unassessed concepts; evaluation was then restricted to those concepts alone. Table III reports the outcome: 0.9004 ± 0.0059 with graph propagation against 0.8679 ± 0.0042 without, a difference of +0.0325 (*p* = 0.0003, *d* = 5.06), with the graph-conditioned model superior on every fold.

Knowledge-graph propagation therefore contributes nothing where direct supervision is plentiful and substantially where it is absent. This has a practical consequence for the recommender, which ranks across all 237 concepts, most of which carry no assessment; and a methodological consequence for the literature, in which component ablations are routinely reported on metrics unable to observe the component under test.

Table VI reports the related case of learners rather than concepts. With no interaction history the model reaches 0.7421 using a learner-profile prior obtained by clustering early behaviour, rising to 0.9449 after four interactions.

### D. Learning Path Quality

Table IV reports 906 recommendation decisions from 400 held-out successful learners. The proposed planner attains the highest NDCG@5 at 0.1532, ahead of weakest-first (0.1344), random selection among admissible actions (0.1300), popularity (0.1069) and syllabus order (0.0612). All differences are significant after correction.

The effect sizes qualify that conclusion and we report them rather than the *p*-values alone. Against syllabus order the effect is small (*d* = 0.36); against random selection among admissible actions it is negligible (*d* = 0.09). The honest interpretation is that the prerequisite mask performs most of the work and the learned ranking adds a small but consistent improvement. This also bounds the headroom available to a learned policy: an agent optimising over (13) must improve on a greedy baseline that itself exceeds random selection by *d* = 0.09.

Prerequisite violations are zero across all 906 decisions and every planner. The pedagogical guarantee of (12) is therefore verified rather than assumed. Concept coverage of 0.726 indicates that the planner does not collapse onto a small set of popular concepts, the characteristic failure mode that popularity-based recommendation exhibits here at 0.510.

### E. Explanation Quality

Table VII reports objective measures over 25 recommendation decisions, each repeated under three perturbations. The sample is small and we report it as such; the measures are nonetheless informative because each is computed against the model's own counterfactual response rather than against an opinion. Fidelity of 0.5318 indicates that cited prerequisite deficits are positively but imperfectly predictive of measured counterfactual improvement. Stability of 0.5310 indicates that roughly half the cited gaps persist under a ±0.02 perturbation of the mastery estimate.

Sufficiency of 0.0020 is the most informative of the three and we report it as a negative finding. It shows that the concept named in an explanation accounted for only 0.2% of the predicted benefit, the remainder arising from transfer to related concepts the explanation did not mention. The explanations were accurate but substantially incomplete. The renderer was subsequently amended to name the principal beneficiaries. No rating scale would have surfaced this: participants cannot assess a quantity the explanation omits.

---

## VI. Conclusion and Future Work

### A. Conclusion

This paper presented an explainable learning-path recommendation framework combining knowledge-graph-conditioned knowledge tracing, prerequisite-constrained planning and priority-backtracking explanation, evaluated on OULAD. The knowledge tracing model attains AUC-ROC 0.9538 ± 0.0019 with expected calibration error 0.0054, significantly exceeding DKT, SAKT and a graph-based baseline.

The principal finding concerns evaluation. A conventional ablation indicates that knowledge-graph propagation contributes nothing, yet under a protocol that evaluates concepts lacking direct supervision the same component improves accuracy by +0.0325 (*d* = 5.06). The discrepancy is not a property of this model but of the metric: aggregate accuracy measured where supervision is abundant cannot observe a structural prior whose purpose is to compensate for its absence. We suggest that component ablations in this literature be accompanied by evidence that the reporting metric is capable of detecting the component under test.

### B. Limitations

The prediction target is successful completion rather than knowledge. Because OULAD records only submitted assessments, treating non-submission as failure is necessary for a usable class balance but means a substantial share of the signal is engagement rather than understanding. This is appropriate for a system intended to prompt intervention, but it is a weaker claim than measuring mastery.

The concept graph is derived, not given. OULAD provides no concept annotations, prerequisite relations or material titles, so concepts are defined structurally and labelled by activity composition rather than subject matter. Placement for 82% of materials relies on observed access; the substitute is validated at ρ = 0.975 on the subset carrying both signals, but deriving concept order from behaviour and then mining prerequisites from behaviour is partially self-referential.

Supervision covers 89 of 237 concepts, so mastery for the remainder is not directly calibrated. Predicted mastery is bimodal, requiring learner-relative thresholds. The proposed model additionally conditions on learner features that the DKT and SAKT baselines do not, as those follow their original formulations; part of the reported improvement may therefore reflect feature access rather than architecture.

All results derive from a single institution's distance-learning data across seven modules, and generalisation to campus-based teaching is untested.

### C. Ethical Considerations

All data used in this study is drawn from OULAD, which is publicly released under CC-BY 4.0 and contains no personally identifying information; learners are represented by anonymised identifiers. No data was collected from human participants for the experiments reported here. The system produces advisory recommendations and makes no determinative decision concerning any learner.

The model conditions on attributes including gender, disability status, index of multiple deprivation and age band. A subgroup analysis of predictive accuracy and recommendation quality across these attributes has not been conducted and is identified below as required future work. Until it is, the system should not be deployed in any setting where its recommendations influence assessment or progression decisions.

### D. Future Work

Four directions follow directly from the limitations above.

**Fairness analysis.** Subgroup performance across gender, disability, deprivation band and age band must be established before deployment. This is the most immediate requirement.

**A learned policy.** Section V quantifies the available headroom: the greedy planner exceeds random selection among admissible actions by *d* = 0.09. A policy learned over (11)–(13), evaluated both in simulation and by off-policy estimation on logged trajectories, would establish whether that margin can be widened.

**User evaluation.** The objective measures reported here establish whether explanations are faithful to the model; they do not establish whether learners find them useful. A controlled study with ethical approval, comparing conditions with and without explanation, would address the complementary question.

**Expert validation of mined edges.** The prerequisite graph is inferred from behaviour. Rating a sample of mined edges by subject experts would provide an independent estimate of its precision.

---

## Reproducibility

OULAD is available at `research.stem.open.ac.uk/ouanalyse/dataset` under CC-BY 4.0. All source code, SQL transformations, configurations and generated results are available at `github.com/abhay27singh/explainable-learning-path-recommender`. Every table in this paper is regenerated from raw data by a single command with fixed random seeds.

---

## Tables

**TABLE I.** Comparative predictive performance. Mean ± standard deviation over five-fold cross-validation with folds grouped by learner. *p*-values are paired *t*-tests against the proposed model, Holm–Bonferroni corrected; *d* is Cohen's *d* for paired samples.

| Model | AUC-ROC | RMSE | ECE | *p* | *d* |
|---|---|---|---|---|---|
| Majority (no-skill) | 0.5000 ± 0.0000 | 0.4669 ± 0.0004 | 0.0037 | < 0.001 | — |
| SAKT [8] | 0.9233 ± 0.0018 | 0.2992 ± 0.0017 | 0.0081 | < 0.001 | 26.16 |
| DKT [5] | 0.9435 ± 0.0017 | 0.2751 ± 0.0022 | 0.0102 | 0.0009 | 5.67 |
| GNN-based | 0.9502 ± 0.0014 | 0.2665 ± 0.0020 | 0.0064 | 0.0220 | 2.25 |
| **Proposed** | **0.9538 ± 0.0019** | **0.2621 ± 0.0026** | **0.0054** | — | — |
| Improvement vs. GNN-based | +0.0036 | −0.0044 | | | |
| Improvement vs. DKT | +0.0103 | −0.0130 | | | |

**TABLE II.** Component ablation under the conventional protocol. Neither component produces a detectable change. Section V-B explains why this measurement cannot observe the knowledge graph.

| Variant | AUC-ROC | RMSE | *p* |
|---|---|---|---|
| Proposed (full) | 0.9538 ± 0.0019 | 0.2621 ± 0.0026 | — |
| w/o knowledge graph | 0.9539 ± 0.0021 | 0.2618 ± 0.0025 | 0.77 |
| w/o temporal attention | 0.9532 ± 0.0014 | 0.2624 ± 0.0024 | 0.45 |

**TABLE III.** Cold-concept ablation. All assessments are withheld for 27 of the 89 assessed concepts and their responses removed from the model input; evaluation is restricted to those concepts. The graph-conditioned model is superior on every fold.

| Variant | AUC-ROC | RMSE |
|---|---|---|
| With knowledge graph | **0.9004 ± 0.0059** | **0.3359 ± 0.0059** |
| Without | 0.8679 ± 0.0042 | 0.3611 ± 0.0113 |
| Difference | +0.0325, *p* = 0.0003, *d* = 5.06 | |

**TABLE IV.** Learning path quality over 906 recommendation decisions from 400 held-out learners who passed or achieved distinction. Ground truth is the set of concepts each learner actually engaged with next. All planners operate under the same admissible-action constraint of (12).

| Planner | NDCG@5 | Hit@5 | Prec@5 | Prereq. violations | Coverage |
|---|---|---|---|---|---|
| Syllabus order | 0.0612 ± 0.154 | 0.1832 | 0.0453 | 0.0000 | 0.452 |
| Popularity | 0.1069 ± 0.199 | 0.2936 | 0.0808 | 0.0000 | 0.510 |
| Random (admissible) | 0.1300 ± 0.218 | 0.3455 | 0.0863 | 0.0000 | 0.722 |
| Weakest-first | 0.1344 ± 0.207 | 0.3642 | 0.0956 | 0.0000 | 0.515 |
| **Proposed** | **0.1532 ± 0.238** | **0.3687** | **0.1051** | **0.0000** | **0.726** |

Paired comparison of NDCG@5 against the proposed planner (Holm-corrected):

| Comparison | Difference | *p* | Cohen's *d* |
|---|---|---|---|
| vs. syllabus order | +0.0921 | < 0.001 | 0.36 (small) |
| vs. popularity | +0.0464 | < 0.001 | 0.17 (negligible) |
| vs. random | +0.0232 | 0.0125 | 0.09 (negligible) |
| vs. weakest-first | +0.0189 | 0.0167 | 0.08 (negligible) |

**TABLE V.** Calibration of predicted mastery. Temperature is fitted on one half of each fold's held-out predictions and evaluated on the other.

| | Expected calibration error |
|---|---|
| Before temperature scaling | 0.0084 |
| After temperature scaling | **0.0054** |

**TABLE VI.** Predictive accuracy as a function of available interaction history.

| Prior interactions | AUC-ROC |
|---|---|
| 0 | 0.7421 ± 0.0353 |
| 1–4 | 0.9449 ± 0.0086 |
| 5–9 | 0.9559 ± 0.0062 |
| 10–19 | 0.9539 ± 0.0056 |
| ≥ 20 | 0.9514 ± 0.0023 |

**TABLE VII.** Objective explanation quality over 25 recommendation decisions, each with three perturbation replicates. These measures record whether an explanation is faithful to the model, rather than whether a reader found it agreeable.

| Measure | Value | Definition |
|---|---|---|
| Fidelity | 0.5318 | Rank correlation between cited deficits and measured counterfactual improvement |
| Sufficiency | 0.0020 | Share of predicted gain attributable to the concept named |
| Stability | 0.5310 | Overlap of cited gaps under ±0.02 perturbation |

**TABLE VIII.** Dataset and derived concept graph.

| Quantity | Value |
|---|---|
| Distinct learners | 28,785 |
| Enrolments | 32,593 |
| Clickstream events | 10,655,280 |
| Assessment records | 173,912 |
| Learning materials | 6,364 |
| Concepts derived | 237 |
| &nbsp;&nbsp;prerequisite edges | 724 |
| &nbsp;&nbsp;corequisite edges | 302 |
| &nbsp;&nbsp;concepts with assessments | 89 of 237 |
| Interaction sequences | 27,904 |
| Supervised tokens | 242,670 |
| Label balance | 67.9% / 32.1% |
| Placement validation | ρ = 0.975 |

**TABLE IX.** Computational cost, measured on the hardware used.

| Quantity | Value |
|---|---|
| Model parameters | 466,569 |
| Recommendation latency (CPU) | ≈ 160 ms |
| Cached mastery lookup | 7 ms |
| Training per configuration (T4) | ≈ 25 min |
| Data preparation from raw files | ≈ 2.5 min |

---

## References

[1] A. Li, Y. Li, and X. Gao, "Personalized learning path recommendation based on knowledge graphs: A survey," *Electronics*, vol. 15, no. 1, Art. no. 238, 2026, doi: 10.3390/electronics15010238.

[2] S. B. Aher and L. M. R. J. Lobo, "Combination of machine learning algorithms for recommendation of courses in e-learning system based on historical data," *Knowledge-Based Systems*, vol. 51, pp. 1–14, 2013, doi: 10.1016/j.knosys.2013.04.015.

[3] J. Kuzilek, M. Hlosta, and Z. Zdrahal, "Open University Learning Analytics dataset," *Scientific Data*, vol. 4, Art. no. 170171, 2017, doi: 10.1038/sdata.2017.171.

[4] A. T. Corbett and J. R. Anderson, "Knowledge tracing: Modeling the acquisition of procedural knowledge," *User Modeling and User-Adapted Interaction*, vol. 4, no. 4, pp. 253–278, 1994.

[5] C. Piech, J. Bassen, J. Huang, S. Ganguli, M. Sahami, L. Guibas, and J. Sohl-Dickstein, "Deep knowledge tracing," in *Advances in Neural Information Processing Systems*, vol. 28, 2015, pp. 505–513.

[6] J. Zhang, X. Shi, I. King, and D.-Y. Yeung, "Dynamic key-value memory networks for knowledge tracing," in *Proc. 26th Int. Conf. World Wide Web (WWW)*, 2017, pp. 765–774, doi: 10.1145/3038912.3052580.

[7] A. Ghosh, N. T. Heffernan, and A. S. Lan, "Context-aware attentive knowledge tracing," in *Proc. 26th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, 2020, pp. 2330–2339, doi: 10.1145/3394486.3403282.

[8] S. Pandey and G. Karypis, "A self-attentive model for knowledge tracing," in *Proc. 12th Int. Conf. Educational Data Mining (EDM)*, 2019, pp. 384–389.

[9] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 5998–6008.

[10] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th Int. Conf. Machine Learning (ICML)*, 2017, pp. 1321–1330.

[11] S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979.

[12] B. Ma, T. Yang, and B. Ren, "A survey on explainable course recommendation systems," in *Distributed, Ambient and Pervasive Interactions*, N. A. Streitz and S. Konomi, Eds. Cham: Springer, 2024, pp. 273–287, doi: 10.1007/978-3-031-60012-8_17.
