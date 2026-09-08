# Paper Correction List — Full Audit

**Paper:** Explainable AI-Based Personalized Learning Path Recommendation System for University Students
**Audited:** 2026-08-09
**Source:** `Explainable_AI_Based_Personalized_Learning_Path_Recommendation_System_for_University_Students_updated.pdf` (11 pages, text-extracted)

Severity tiers:

- **T1 — Desk-reject risk.** A reviewer or chair can reject on this alone.
- **T2 — Major revision.** Will draw a reject-or-major-revise score.
- **T3 — Minor.** Polish; individually harmless, collectively damaging.

> Note on extraction: this audit is based on text extracted from the PDF. Items marked **[VERIFY IN PDF]** may be extraction artifacts rather than real defects — open the rendered PDF and confirm before editing. Everything else is a genuine content defect.

---

## T1 — Desk-reject risk

### T1-1. "Dynamic Knowledge Tracing" — wrong name for DKT
**Location:** Section II (para 2), Section V-A/Conclusion
DKT is **Deep** Knowledge Tracing (Piech et al., 2015). The paper writes "Dynamic Knowledge Tracing (DKT)" in Section II and "Dynamic Knowledge Tracing" in the conclusion, while Section I correctly writes "Deep Knowledge Tracing (DKT)". Getting the name of your own core method wrong, inconsistently, in a knowledge-tracing paper is the fastest possible credibility loss with an expert reviewer.
**Fix:** "Deep Knowledge Tracing (DKT)" everywhere.

### T1-2. "particularly in the field of drug diversion"
**Location:** Section II, sentence beginning "In education, graph neural networks are one of the latest advances…"
Reads: *"graph neural networks are one of the latest advances in handling complex and dynamic relationships, particularly in the field of drug diversion [13]."* Drug diversion is pharmaceutical crime. Reference [13] is a paper on knowledge graphs and XAI for personalized learning. Wholly unrelated, wholly wrong, and it reads as machine-generated text nobody proofread.
**Fix:** Delete the clause or replace with the actual domain of [13].

### T1-3. OULAD is miscited
**Location:** Section III-B ("The Open University Learning Analytics Dataset (OULAD) … [14]"), Section IV-B
Reference [14] is *"AI-driven personalized learning paths for web-based vocational training, ScienceDirect, 2026"* — not the OULAD paper, and "ScienceDirect" is a platform, not a venue.
**Correct citation:** J. Kuzilek, M. Hlosta, Z. Zdrahal, "Open University Learning Analytics dataset," *Scientific Data*, vol. 4, 170171, 2017. doi:10.1038/sdata.2017.171
**Fix:** Cite the real dataset paper, and add the OULAD license (CC-BY 4.0) attribution.

### T1-4. No mathematical formalization anywhere
**Location:** Entire paper
There is not one equation in the paper. Missing: the MDP tuple (state space, action space, transition, reward, discount), the reward function, the Q-learning/DQN update, the knowledge-tracing prediction formulation, the GCN propagation rule, the attention formulation, the combined loss. Section III-D refers to "a hyperparameter lambda" that balances two losses — the loss is never written and lambda's value is never given.
For an AI/ML venue this is disqualifying on rigor: the method as written is not reproducible from the paper.
**Fix:** Add a formal Methodology subsection with numbered equations, plus an algorithm block for the training loop and one for the priority-backtracking interpreter.

### T1-5. Human-subjects study with no ethics statement
**Location:** Section IV-D, Section V-C
Study with 30 human participants. No ethics/IRB approval statement, no informed consent statement, no participant compensation, no protocol description, no questionnaire items, no participant demographics.
Many venues desk-reject human-subjects work without an ethics statement.
**Fix:** Add an ethics statement, consent procedure, full protocol, and the questionnaire instrument (appendix or supplementary).

### T1-6. Anonymous references
**Location:** References [6], [7], [8], [12], [14]
Five references have no authors at all. [8] and [16] are arXiv entries with no arXiv identifier. [12]'s venue is "Dimensions" (a research database, not a venue). [14]'s venue is "ScienceDirect" (a platform, not a venue).
**Fix:** Complete every entry with authors, venue, volume/pages, year, DOI or arXiv ID. Any reference you cannot complete is a reference you should not cite.

### T1-7. Results claimed but not statistically reported
**Location:** Tables I & II, Sections IV-D, V
Paper claims 5-fold cross-validation and paired t-tests at p<0.05, but reports single point values with **no standard deviations, no confidence intervals, and no p-values anywhere**. Claiming a significance test and not reporting its outcome is a standard reviewer objection.
**Fix:** Report mean ± std across folds for every metric, and the actual p-values for every claimed comparison.

### T1-8. Demographic/protected attributes used with no fairness analysis
**Location:** Section IV-B (one-hot encoding of gender, education level; OULAD also supplies disability, IMD deprivation band, age band, region)
A 2026 educational recommender that consumes protected attributes and then makes academic-path recommendations, with zero fairness, bias, or disparate-impact analysis, will draw a serious objection — especially in a paper whose entire selling point is trustworthy, transparent AI.
**Fix:** Either add a subgroup performance analysis (accuracy and recommendation quality by gender / disability / IMD band), or state explicitly which attributes were excluded from the model and why.

---

## T2 — Major revision

### T2-1. Post-hoc contradiction (the paper contradicts itself on its own core claim)
- **Abstract:** "by embedding **post-hoc** explanation generation…"
- **Section I:** "(3) a **post-hoc** explanation generation module…"
- **Section II-A / III-A:** "explainability is added as an afterthought to the existing systems rather than being an integral part…" and "The proposed framework integrates these aspects **from the design phase**, rather than in isolation as current techniques do"
- **Section III-C:** "This integrated design has been used in previous studies to **remove the post-hoc explainability constraints**."

The paper simultaneously claims its explanation module is post-hoc and that it is not post-hoc, and the not-post-hoc version is the claimed novelty. Your architecture backtracks the knowledge graph *after* the DQN selects an action — that is post-hoc.
**Fix:** Pick one and be consistent. Recommended: keep "post-hoc" and reframe the novelty as *knowledge-graph-grounded, pedagogically-structured* explanation, not *non-post-hoc* explanation.

### T2-2. Collaborative filtering baseline promised, never delivered
**Abstract:** "we evaluate the framework's performance and compare it to **baseline collaborative filtering** and non-explainable RL models."
Tables I and II contain DKT, RL-DKT, GNN-Based, Proposed. No collaborative filtering row.
**Fix:** Implement it or delete the claim.

### T2-3. Recommendation unit is inconsistent across the paper
- **Abstract:** recommends "best courses and sequence of courses and modules"
- **Section III-C / V-B:** recommends "learning activities"
- **Section III-C:** explanations concern "concepts" and prerequisite gaps

Three different units for the thing being recommended.
**Fix:** Define one unit (concept-level next-step, per the implementation design) and use it consistently in abstract, methodology, and results.

### T2-4. Explainability has no quantitative metric
**Location:** Section III-E, Section V-C
Section III-E's enumeration breaks mid-list: *"(2) path quality is evaluated by completion rate, NDCG, and average knowledge gain per recommended activity [7] **for explainability quality**, (3) comparison with state-of-the-art baselines…"* — the explainability metrics are never actually stated. Section V-C, titled "Explainability Quality Assessment", reports only Likert scores.
Explainability is this paper's headline contribution and it is evaluated only by subjective survey. XAI reviewers will require objective measures (fidelity, sufficiency, comprehensiveness, or explanation-stability).
**Fix:** Repair the enumeration, define explainability metrics, and report at least one objective fidelity measure alongside the Likert results.

### T2-5. Results and Conclusion disagree with each other
- **Section V-C:** pedagogical utility reported as "(08/5)" → 4.08/5
- **Section VI:** "pedagogical utility (4.32/5)"

Same quantity, two values, same paper.
**Fix:** One value, reported once, in one place.

### T2-6. Likert scale contradiction
**Section IV-D:** "using a 5-point Likert scale."
**Section V-C:** "usefulness to the classroom (08/7)" — a value out of 7.
**Fix:** A 5-point scale cannot produce an *x*/7 score. Correct the scale or the score.

### T2-7. "Slight increase" that is a 67% increase
**Section V-E:** *"a **slight** increase in computational complexity (inference time: 187 ms versus 112 ms for RL-DKT)."*
187/112 = 1.67. That is a 67% latency increase, described as slight.
**Fix:** Report it accurately, and re-measure on the hardware actually used.

### T2-8. Unsupportable cross-study comparison
**Section V-B:** average knowledge gain "0.234 – **much higher than in other studies**."
No other studies' values are given, and "knowledge gain" is not a standardized metric with comparable units across papers.
**Fix:** Delete the comparative claim, or cite specific values from specific papers using an identical metric definition.

### T2-9. Knowledge graph statistics are unobtainable from OULAD
**Section IV-B:** 1,247 concept nodes, 3,891 prerequisite edges, 1,023 co-requisite edges.
**Section III-B:** graph "generated from course prerequisite relationship links and learning outcomes mapping."
OULAD contains no concepts, no prerequisites, and no learning-outcome mappings. These numbers cannot come from OULAD.
**Fix:** Describe how the graph was actually constructed, publish it, and report its real size.

### T2-10. The two-graph description is incoherent
**Section IV-B:** *"There are two types of relationships: one is between concepts, where the nodes are concepts (n=1,247) and the edges are co-requisite relationships (n=1,023); the other is between concepts and **learning objective mappings**, where nodes are concepts (n=1,247) and the edges are prerequisites (n=3,891)."*
The second graph is said to hold between concepts and learning objectives, yet its nodes are only concepts. And "prerequisites" — the paper's central structure — is assigned to the *learning-objective* graph rather than the concept graph.
**Fix:** Rewrite. One concept graph with typed edges (prerequisite, co-requisite) is the coherent version.

### T2-11. Internal hyperparameter contradictions
| Parameter | Section III-D | Section IV-C |
|---|---|---|
| GCN layers | "(2, 4)" | "three graph convolution layers" |
| Replay buffer | "(99)" | "100,000 sized" |
| Attention heads | "(4-16)" | "4 attention head" |

The III-D values are meant to be Bayesian-optimization *search ranges* and IV-C the *selected* values, but this is never said, and "(99)" and "(2, 4)" are malformed either way.
**Fix:** Present a single hyperparameter table with columns: parameter | search range | selected value.

### T2-12. Encoder architecture is ambiguous
**Section III-C** says the model uses "**graph attention** mechanisms" and, two sentences later, "**Graph Convolutional Networks (GCN)**". These are different architectures (GAT vs GCN).
**Fix:** Pick one; if both are used, say which is used where.

### T2-13. Bayesian optimization claimed, never reported
**Section III-D** claims Bayesian hyperparameter optimization but gives no search budget, no objective function, and no resulting best configuration.
**Fix:** Report the search space, number of trials, objective, and selected configuration — or drop the claim.

### T2-14. Missing training details
Never reported anywhere: number of **epochs** (the sentence is truncated at "the number of3."), training time, random seeds, number of runs behind the reported values, early-stopping criterion, dropout value (garbled as "0(0."), lambda loss-balancing value.
**Fix:** Complete the implementation section. These are the minimum for reproducibility.

### T2-15. Baselines are not reproducibly described
**Section IV-D:** *"The 64-layer hidden dimension 2-layer GCN is used in the GNN-based recommender, the same DQN is employed in RL-DKT, and the conventional LSTM architecture is employed in DKT."*
"64-layer hidden dimension" is garbled (should be *64 hidden dimensions*). No baseline hyperparameters, no training budget, no citation of the exact implementations used.
**Fix:** Full baseline configuration table, and state whether baselines were tuned with the same budget as the proposed model — reviewers assume unfair tuning otherwise.

### T2-16. Undefined metrics
- **NDCG** — no ground-truth relevance definition. NDCG against *what* ranking?
- **Completion rate** — of what, measured how? No real student ever followed a recommended path, so this can only be simulated. The paper does not say it is simulated.
- **Average knowledge gain** — no units, no normalization, no definition.
**Fix:** Define each formally. Label simulated quantities as simulated.

### T2-17. Bloom's taxonomy is dangling and miscited
**Section II** introduces Bloom's taxonomy as "a theoretical foundation for the learning path design" citing [12] (a graph-neural-network paper). Bloom's taxonomy then never appears again — not in the method, not in the graph, not in the explanations.
**Fix:** Either use it in the method and cite Bloom (1956) / Anderson & Krathwohl (2001), or cut the sentence.

### T2-18. Citation-to-claim mismatches
| Claim | Cited | Problem |
|---|---|---|
| DKT with RNNs models latent knowledge state | [5] Badran & Preisach 2025 | Should be Piech et al., NeurIPS 2015 |
| RL-DKT reduced dropout and task completion time | [8][9] | [9] is a survey on explainable course recommendation; [8] is a question-scheduling paper. The Sci. Rep. RL+DKT paper is [6] |
| Bloom's taxonomy, six levels | [12] | [12] is about GNN recommendations |
| GNNs in "drug diversion" | [13] | [13] is about KG+GAN+XAI for learning |
| Use of paired t-tests | [3][7] | Methodological convention does not need domain citations |
**Fix:** Re-verify every citation against what the cited work actually says.

### T2-19. Duplicate reference
[1] and [11] are the same work: A. Li, Y. Li, X. Gao, "Personalized learning path recommendation based on knowledge graphs: A survey," *Electronics*, vol. 15, no. 1, p. 238, 2026. Cited as though they were two independent sources, including in "[1][11]" pairs that appear to be inflating support.
**Fix:** Merge to one reference; fix all "[1][11]" citations.

### T2-20. User-study population mismatch
Participants are 30 computer-science and engineering students. The system is trained on OULAD, whose modules are Open University social-science and STEM distance-learning courses. Participants are evaluating recommendations for a curriculum they are not enrolled in and a student population they are not part of.
**Fix:** Acknowledge as a threat to validity, or recruit participants matched to the domain.

### T2-21. "Preliminary results" vs. a full results section
**Abstract:** "**Preliminary** results indicate that the XAI enhanced system has improved…"
Section V presents complete tables, ablations, statistical tests, and a user study, and Section VI states definitive conclusions.
**Fix:** Remove "preliminary" or downgrade the conclusions. Not both.

---

## T3 — Minor, but they compound

### T3-1. Garbled decimals and truncated sentences throughout **[VERIFY IN PDF]**
Every one of these appears broken in the extracted text. Check the rendered PDF; repair whichever are genuinely broken:

| Location | As extracted | Should be |
|---|---|---|
| III-D | "initial learning rate is set to 0 … We will use 001" | learning rate = 0.001 |
| III-D | "experience replay buffer size (99)" | 100,000 |
| III-D | "the number of experience replay buffer heads (4-16)" | not a real quantity — likely attention heads |
| III-D | "L2 weight decay and dropout 0(0." | state the dropout value |
| IV-C | "dropout 0. During training, the batch size is 64 and the number of3." | dropout value + epoch count |
| IV-D | "paired t-tests (p¡0.05)" | p < 0.05 (broken `<` encoding) |
| IV-D | "5-fold cross validation runs using 05)." | p < 0.05 |
| V-A | "the 3. … 4% relative gain in AUC-ROC" | 3.4% |
| V-B | "the highest (0). … perseverance (784)." | 0.784 |
| V-B | "Compared to RL-DKT (0.821) and GNN-based . … the 809) methods" | 0.809 |
| V-B | "average knowledge gained … is 0, … 234" | 0.234 |
| V-C | "trust (mean = 4). … pedagogical utility (08/5)" | 4.08/5 |
| V-D | "AUC-ROC drops to 0. Without knowledge graph encoding, 861 is validated" | 0.861 |
| V-D | "completion rate is 0. … ignoring the time attribute of 723" | 0.723 |
| V-D | "(AUC-ROC: 0. … in 889). 08 to 3." | 0.889; trust 4.08 → 3.5 |
| VI | "not only reveal that 143) contributes" | RMSE 0.143 |
| VI | "transparency (4) through user studies21/5)" | 4.21/5 |
| IV-A | "Stable-Baselines3 1, one needs … wrappers. 7.0." | Stable-Baselines3 1.7.0 |
| IV-A | "The version 6 of RDFLib is required … 2.0 and NetworkX 2. 8" | RDFLib 6.2.0, NetworkX 2.8 |

Numbers split mid-sentence like this read as text that was reflowed or regenerated without proofreading. A reviewer who spots three of them stops reading for content.

### T3-2. Section headings out of order **[VERIFY IN PDF]**
- Page 2–3: "A. Research Gap" heading appears *after* its own body text and *after* the "III. METHODOLOGY" heading; Fig. 1 caption sits between them.
- Page 3: subsection headings appear as a block in the order A, B, C, **E, D** — Evaluation Strategy before Optimization and Training.
- Page 5: "D. Ablation Study" precedes "B. Learning Path Quality" and "C. Explainability Quality Assessment"; "A. Conclusion" appears inside Section V.
- Section II's final sentence ("Research on the application of XAI in education") is cut mid-sentence and resumes on page 3 after the Methodology heading.

If this reflects the real layout, the paper is structurally unreadable and will be rejected without a content review. Check the two-column float placement in your template.

### T3-3. Letter-spaced text **[VERIFY IN PDF]**
Abstract, index terms, all section titles, table headers, and figure captions extract with a space between every character ("A b s t r a c t", "I . I N T R O D U C T I O N", "TABLE I / C O M P A R A T I V E"). If the submitted PDF renders this way, it is a font-embedding or template failure and must be fixed before anything else.

### T3-4. Table titles corrupted **[VERIFY IN PDF]**
- Table I: "C O M P A R A T I V E P R E D I C T I V E **A** C C U R A C Y" — the A of ACCURACY is split off into its own cell.
- Table II: "LEARNING PATH **UALITY**COMPARISON" — the Q is detached, and no space before COMPARISON.

### T3-5. Grammar and wording
| Location | Problem |
|---|---|
| Section I | "There is a lot of systems exist today" — double verb |
| Section I | "the potential of closing the gap … is hoped" — unidiomatic passive |
| Section III-D | "We **will** use…" — future tense in a completed-work paper; tense mixes throughout |
| Section V-E | "Research lightweight explanation generation and actual use … in the future." — no subject or main verb |
| Section VI | "students in universities **and universities**" — duplication |
| Section VI | "contributes to improve learning path quality significantly higher, but also enhances the **learning path quality** significantly" — same claim twice in one sentence |
| Section VI | "First… Secondly… Thirdly… **Third**… Finally" — five items, ordinal used twice |
| Section VI | "individualized education. **.** Finally" — stray period |
| Section VI heading | "CONCLUSIONANDFUTUREWORK" — missing spaces |
| References | "EFERENCES" — dropped R |
| References | Missing spaces throughout: "A.Bekarystankyzy,A.Kassenkhan,andM.Iglikova"; "IntegratingReinforcementLearningwithDynamicKnowledgeTracing"; "T.Yang,B.Ren,B.Ma,M.A.Z.Khan" |
| Section II | "[15,16]" — comma style, inconsistent with "[15][16]" used everywhere else |
| Section V-D | "a study on user acceptance of open systems … the importance of **openness**" — off-topic; the concept is transparency, and the citation "in 12)" is malformed |

### T3-6. Author block
"Shikha kamal" — lowercase surname. Emails appear interleaved with the abstract text in extraction **[VERIFY IN PDF]**; if that is the real layout, the author block is overlapping the abstract.

### T3-7. Figures never referenced in text
Figures 1–6 have captions but the body never says "as shown in Fig. 2" etc. IEEE style requires every figure to be cited in text. Figure 5 has no caption at all in the extraction — it reads only "Fig. 5." followed by table data, suggesting a caption/table collision.

### T3-8. Index terms
15 index terms listed. IEEE conference norm is roughly 5. Trim.

### T3-9. Stale software stack
Python 3.9, PyTorch 1.12, PyTorch Geometric 2.2.0, Stable-Baselines3 1.7.0, Pandas 1.5.0, NumPy 1.23.0 — a 2021–2022 stack in a 2026 paper. Not fatal, but it signals the environment section was copied rather than recorded.

### T3-10. Unjustified dependencies
- **RDFLib** is listed as required "to support semantic relationship management", but no RDF, OWL, or SPARQL work is described anywhere in the paper.
- **Stable-Baselines3** is listed, but Section IV-C describes a hand-built 3-layer DQN. Which was used?

### T3-11. Redundancy
The 80/10/10 stratified split is described twice, in Section III-B and again in Section III-D, in near-identical words.

### T3-12. Ethics gap in future work
Future work proposes collecting **facial expressions** from students as multimodal input, with no privacy, consent, or surveillance discussion — in a paper about student trust. Reviewers will flag the irony.

---

## Missing sections that acceptance normally requires

1. **Contributions list** — Section I ends without an explicit bulleted list of contributions.
2. **Paper organization paragraph** — end of Section I.
3. **Formal problem statement** — before the method.
4. **Equations and algorithm blocks** — see T1-4.
5. **Complexity analysis** — claimed latency numbers but no analysis.
6. **Limitations section** — currently one sentence inside the discussion.
7. **Threats to validity** — simulated evaluation, curated graph, participant mismatch, single dataset.
8. **Ethics statement** — see T1-5.
9. **Data and code availability statement** — expected for a reproducibility-oriented contribution.
10. **Fairness analysis** — see T1-8.
11. **Cold-start discussion** — the related-work section criticizes cold-start in prior systems; the proposed system's own cold-start behavior is never addressed.

---

## Priority order for fixing

1. Verify the rendered PDF for T3-2/T3-3/T3-4 (layout, letter spacing, tables). If the real PDF looks like the extraction, nothing else matters until the template is fixed.
2. Fix T1-1, T1-2, T1-3 — three single-line edits that each independently destroy credibility.
3. Add formalization and ethics statement (T1-4, T1-5), fix references (T1-6).
4. Resolve the post-hoc contradiction (T2-1) — this is a claim-level decision, not an edit.
5. Run the implementation; replace every number, and add std/p-values (T1-7).
6. Add fairness analysis (T1-8) and the missing sections.
7. Sweep T2 contradictions, then T3 polish.
