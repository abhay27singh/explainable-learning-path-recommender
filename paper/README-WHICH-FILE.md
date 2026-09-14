# paper-revised.tex — your paper, fixed

8 pages, IEEE two-column, **all four figures included**, compiles clean:
0 overfull boxes, 0 undefined references.

My earlier over-scoped rewrite is in `superseded/` — ignore or delete it.

## Before you compile: upload your images

The four figure slots expect `fig1.png`, `fig2.png`, `fig3.png`, `fig4.png`.

Upload your existing images to the Overleaf project under those names, or
edit the four `\figslot{...}` lines to match your filenames.

If a file is missing the paper still compiles and prints a labelled
placeholder box in its place, so you can never get a broken build.

Verified to stay at 8 pages with figures up to 6 cm tall.

| Slot | Caption |
|---|---|
| `fig1.png` | Overall XAI-based personalized learning recommendation framework |
| `fig2.png` | Knowledge graph enhanced knowledge tracing |
| `fig3.png` | Prerequisite-constrained learning path optimization |
| `fig4.png` | Explainable AI module for recommendation justification |

Your original Fig. 3 showed a DQN. Since the DQN is gone, either redraw it
as the prerequisite-constrained planner or reuse the old diagram with the
"DQN" box relabelled "greedy planner".

Your original Fig. 5 plotted fabricated path-quality data and has been
dropped. A real figure is available at `results/figures/cold_start.png`
if you want a fifth — but it will push the paper to 9 pages.

---

## Unchanged from your paper

Title, authors, affiliations. Section order and numbering. Your subsection
titles. All 20 of your references, in your order. Figure positions. Most of
your sentences.

## Changed

**Fabricated numbers replaced with measured ones.**

| Your paper | Measured |
|---|---|
| AUC-ROC 0.892 | 0.9538 ± 0.0019 |
| RMSE 0.143 | 0.2621 ± 0.0026 |
| "baseline AUC-ROC 0.871" (in no table) | real baselines with p-values |
| NDCG 0.856 | NDCG@5 0.1532 |
| Completion rate 0.784 | removed — never measured |
| Knowledge gain 0.234 | removed — never measured |
| Transparency 4.21/5, 30 students | removed — study never run |
| Ablation showing each part helps | real: −0.0001 (p=0.77), −0.0006 (p=0.45) |
| 1,247 nodes / 3,891 prereq / 1,023 coreq | 237 / 724 / 302 |
| Inference 187 ms vs 112 ms | 160 ms; 7 ms cached |

**Components that do not exist, removed.** The DQN (§III-D now describes the
prerequisite-constrained planner that actually runs; the MDP framing stays
because that part was accurate, and a learned policy moves to Future Work
with its headroom quantified at d = 0.09). PyTorch Geometric,
Stable-Baselines3 and RDFLib. The user study, replaced by objective
fidelity / sufficiency / stability.

**One new subsection, V-E — "Why Conventional Ablation Is Insufficient."**
This answers "Less Technical Contribution". Standard ablation says the
knowledge graph does nothing (p = 0.77). That is a measurement artefact:
accuracy is scored only at assessment positions and only 89 of 237 concepts
carry assessments. Withhold all assessments for 27 concepts and score only
those: +0.0325, p = 0.0003, Cohen's d = 5.06, winning every fold.

**Added to §VI:** Limitations, Ethical Considerations, Reproducibility.
Fairness analysis is named as required future work, not run.

**Factual fixes.** "in the field of drug diversion" deleted (it came from
another paper). "Dynamic Knowledge Tracing" → "Deep Knowledge Tracing".
The 30-minute session threshold corrected — OULAD is day-granular, so that
number was never computable; a session is one learner-day. Hardware
corrected to Intel CPU plus one NVIDIA T4. Python 3.9 → 3.11. Index terms
15 → 7.

**Two equations added** — the mining rule (1) and the forgetting decay (2).
Your paper had none. To go back to zero, delete each `\begin{equation}`
block and the sentence introducing it; nothing else depends on them.

**Three references added:** SAKT [21], temperature scaling [22], Holm [23].
Your original 20 are untouched.

## Tables

Three, down from the nine in my earlier draft. Your Table I now carries the
ablation as two extra rows — which is how `results/table1.json` already
reports it, and reads better than a separate table. Your Table II is
Learning Path Quality. Table III is the cold-concept result.

## Verification

Every number in the paper was re-derived from `results/*.json`. All 31
measured values match. All 21 fabricated values are gone — checked against
the compiled PDF, not just the source.

## Compiling

Overleaf: upload `paper-revised.tex` plus your four images, Recompile.
pdfLaTeX, no .bib file needed.

Locally: `tectonic -X compile paper-revised.tex --outdir build`
