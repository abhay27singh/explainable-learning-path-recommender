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

The website, GyanGraph, runs from a fresh clone without retraining: the one trained model it loads,
`artifacts/models/kgdkt_proposed_fold0.pt` (1.9 MB), is committed. The dataset is not,
so the first run downloads it and rebuilds the processed data once.

```bash
make install                                    # Python 3.11 venv and dependencies

mkdir -p data/raw && cd data/raw                # 487 MB
curl -L -o oulad.zip https://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip
unzip oulad.zip && cd ../..
# sha256: 90dda45037939953f979072fa70a809ebe07e90ec783c408762af7698a3825ec

.venv/bin/python scripts/01_prepare_data.py     # ~18s
.venv/bin/python scripts/02_build_graph.py      # ~2.5 min
.venv/bin/python scripts/03_build_sequences.py
.venv/bin/python scripts/07_concept_labels.py

make test                                       # 284 tests
make api                                        # http://localhost:8420
make stop                                       # stops it from any terminal
```

[`HOW_TO_RUN.txt`](HOW_TO_RUN.txt) has the start and stop steps in plain words.

Accounts live in `data/app.db`, which is created on first start and never committed, so
every copy starts with no accounts. Students and advisers sign up on the site. An admin
can only be created from the command line, with the password typed at a hidden prompt:

```bash
.venv/bin/python scripts/08_admin.py create <username>
```

The remaining checkpoints (every fold and every baseline, 39 MB) are needed only to
reproduce the paper's tables, not to run the site. Training runs on a free Colab T4: see
[`RUNBOOK.md`](RUNBOOK.md).

## Going live

The site runs as one process behind a reverse proxy (Caddy or nginx) that holds the
HTTPS certificate. The padlock comes from that certificate, not from this code.

```bash
ELPR_SITE_URL=https://your-domain.in make serve   # no auto-reload, HTTPS settings on
.venv/bin/python scripts/08_admin.py create <username>
.venv/bin/python scripts/11_backup.py             # run daily from cron; copy backups off the machine
```

Advisers are invited, not open sign-ups. An admin makes a one-time invite code on the Accounts
page for a named person; they sign up with it, and the admin approves the account. An approved
adviser gets a class code and sees only the students who joined with it. A student can leave a
class at any time under My details.

| Setting | What it does |
|---|---|
| `ELPR_SITE_URL` | The public address. Link previews (WhatsApp, Telegram), `robots.txt` and `sitemap.xml` use it. |
| `ELPR_HTTPS=1` | Session cookies are marked Secure and browsers are told to stay on HTTPS (HSTS). Only behind HTTPS. |
| `ELPR_TRUST_PROXY=1` | Rate limits read the client address from `X-Forwarded-For`. Only behind a proxy. |
| `ELPR_API_DOCS=1` | Publishes `/docs`, `/redoc` and `/openapi.json`. Off by default. |

What is already in place: scrypt password hashes, 12-hour httpOnly sessions, per-username
sign-in throttling, rate limits on sign-up (5 an hour per address), the What-if (20 a
minute), self-check marking (30 a minute) and the API overall (300 a minute), security
headers with a narrow Content Security Policy, a database readable by its owner only,
model files loaded as data only, a 404 page, `favicon.ico`, `robots.txt` and
`sitemap.xml`. After launch, add the domain to Google Search Console and submit
`/sitemap.xml`. The pinned torch 2.2 is the last build for Intel Macs and has known
flaws in loading untrusted model files; the site loads only its own, with
`weights_only=True`, and a Linux server can take a newer torch once it has been tested
there.

## Web application

FastAPI service and a single-page app with scrypt-hashed accounts and role-based access.
One file, `web/index.html`, no build step.

- **The ladder.** Class 10 to class 12 or diploma, then undergraduate, then postgraduate.
  Each rung shows what is studied there with free study links, and a button for what comes
  next. Students pick their rung at sign-up and the site never offers a level they have
  already passed.
- **The weekly path.** Diploma and degree students follow their course from Week 1. Each
  step says why it comes next, what to revise first and what it opens up, all from the
  prerequisite graph and the model. A week moves on when it is marked studied, or when a
  quiz mark of 40 or more is entered. A bare "I passed" is refused by the API: a pass is
  evidence the model learns from, so it needs a mark.
- **Entrance exams.** JEE Main, NEET UG and CUET UG, each with its published syllabus
  outline, the official NTA link and a week by week revision plan. An exam only appears for
  a class 12 stream that can sit it. CUET's domain papers are built from the student's own
  subjects. No dates, cut-offs or ranks are stated anywhere.
- **Self-check questions.** 204 questions across all 104 JEE Main and NEET UG units, each
  with one right answer and the reason, marked on the server so the answers never reach
  the page first. Results are kept per student and never mixed into the data the model
  reads. The weekly course path has none, on purpose: its weeks come from an anonymised
  course and have no topic a question could honestly be about.
- **Course Finder.** Rule-based, not the model, and labelled as such. 64 courses: 48
  academic plus 16 NSQF skill courses, which NEP 2020 treats as an equal track. Eligibility
  by class 12 subjects or bachelor's degree, interest ranking, a shortlist, and a week by
  week plan for any course with a calendar file to download.
- **Finishing the course.** On Progress, weeks done over time and when the course would be
  finished at the student's own pace and at a week each week, worked out from their dates.
- **Progress and record.** A year of activity with the active days and both streaks
  above it, the way a contribution graph reads, plus milestones, quiz marks against the
  pass line, and a printable record of study. Every figure comes from what the
  student recorded. It says plainly that it is not a certificate. A student still at
  school sees their self-check results instead: units checked, passed on the latest try,
  and the units still to check.
- **Advisers** search all 25,101 dataset learners plus registered students, inspect any
  record, keep notes per learner, mark weeks a learner already knows, and compare planning
  strategies.
- **Admin** manages accounts and is the only role that sees the research results page,
  which reads `results/*.json` directly so it cannot drift from the paper.

The look follows the "Academic Navigator" layout in its icy-blue colours: sky blue on an
ice canvas, a glacial navy dark theme, amber for warnings, Plus Jakarta Sans headings, light glass panels (solid for anyone whose
system asks for reduced transparency), and a navy dark theme. Pages use the full width
of the window, and on a phone the main places sit in a tab bar at the bottom.

- **Five views of the path.** A timeline; a board with the course in four parts; the
  model's prerequisite graph, with the key week marked and a "whole chain" view; a table;
  and **What if**, where a student picks weeks they plan to pass and the model is run
  again on that record to show what would open up. Nothing is saved.
- **Course Finder as a planning form**: where you are (down to the engineering branch or
  degree subject) and what you want next, then what you enjoy. Every match is listed
  beside the questions, closest first; when nothing fits at one level, the matches at
  the others are offered. Explore is a catalogue with level, length and subject filters,
  a "For you" switch for signed-in students, a page per course with where the degree
  leads and the entrance exam's own revision plan, and a side by side comparison of up to
  three courses.

*Demonstration-grade authentication: an 8 character password minimum and sign-in
throttling, but no TLS and no account recovery. Do not reuse a real password.*

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
