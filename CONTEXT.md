# Project Context — read this first

Hand this to a fresh agent session (Claude Code, Antigravity, any IDE assistant), a
collaborator, or a supervisor and it will have everything it needs. Every research claim
here is backed by a file in `results/`.

Last updated 14 September 2026.

---

## What this is

Two things in one repository, deliberately kept apart:

1. **The research.** An explainable learning-path recommender over the Open University
   Learning Analytics Dataset (OULAD): knowledge tracing plus a prerequisite-aware
   planner, built for an IEEE paper.
2. **The product.** A web application for real students who sign up: study levels from
   class 10 to postgraduate, a week-by-week path through a course, and a rule-based
   Course Finder for Indian diploma, undergraduate and postgraduate courses.

The paper's measured results stay on the site as validation, on an admin-only Research
page. New product features are never added to the paper.

**Paper history.** Rejected by IEEE COMPUTINGCON 2026 (paper 1757, 31 Aug 2026), stated
reason *"Less Technical Contribution."* Three reviewers agreed on the cause: the paper
asserted results it had not produced, and described an algorithm without specifying it.
Everything the original draft claimed has since been measured, and several claims were
unreachable as stated.

---

## Current state

| Stage | Status |
|---|---|
| 0 · Data pipeline, mined concept graph | Complete |
| 1 · Knowledge tracing, 9 training runs, Table I | Complete |
| 2 · Explainability, planner, Table II | Complete |
| 4 · Web application | Complete and running |
| 6 · Paper revision | Complete: 8-page and 6-page versions in `paper/` |
| 3 · Reinforcement learning | **Deferred**: greedy beats random by only d = 0.09 |
| 5 · User study | **Not started** |
| 7 · Public deployment | **Not started**: see the launch checklist below |

`make test` runs 84 tests. `make api` serves http://localhost:8420.

The git remote is `https://github.com/abhay27singh/explainable-learning-path-recommender`
and work is on `main`.

---

## The web application

Single FastAPI service (`api/main.py`) plus one page, `web/index.html`, a vanilla-JS
single-page app with hash routing (`#/home`, `#/login`, `#/dashboard`, `#/finder`,
`#/research`, `#/admin`, `#/privacy`, `#/terms`). No build step, no framework.

**Roles.** `student`, `adviser`, `admin`. Admins are a superset of advisers. An admin can
only be created from the command line (`scripts/08_admin.py create <username>`), never
through the API, so a registered account can never escalate itself.

**Study levels.** Every student picks one at sign-up: `class_10`, `class_12`, `diploma`,
`ug`, `pg` (`elpr/course_finder.py: STAGES`). Diploma and degree students also choose a
course and get a weekly path; school students get the Course Finder instead. The Finder
offers only the level after the student's own (`NEXT_LEVELS`), enforced in the API, not
just hidden in the page. Levels are editable later under My details.

**The path.** `Service.course_path` walks the course in order from its first week.
A week counts as done when studied or passed; "I found it hard" keeps it in place
(`Service.done_concepts`). Each step is explained by the model with the earlier steps
marked as known, so every explanation is real model output under a stated assumption.
Advisers keep the model-ranked view (`Service.learning_path`, greedy planner) plus a
compare-planners tab.

**Week numbering.** The dataset counts a course's first week as 0. The page adds one for
display in `humanize()` in `web/index.html`, applied once per API response. Server, data
and tests all still use the dataset's numbering.

**Course Finder.** Rule-based, not the model, and labelled as such everywhere it appears.
43 courses in `elpr/course_finder.py` with eligibility by class 12 subjects or bachelor's
degree, interest ranking, an explore-everything view, and Google site-search links to
NPTEL and SWAYAM for each subject.

**Progress.** Streaks, a 12-week activity calendar, badges and a weekly summary, all
computed from recorded events only (`elpr/progress.py`). A student's first week makes no
comparison claims, because there is nothing real to compare with.

**Research page.** Admin only, in the page and in `/api/metrics`. It reads `results/*.json`
directly so it cannot drift from the paper.

**Security posture.** scrypt password hashing, httpOnly session cookies, server-side
access checks. No TLS, no rate limiting, no account recovery: demonstration grade, and
the privacy page says so. `data/app.db` holds real accounts and **must never be
committed**; it is git-ignored.

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

### The finding the paper is built around

Standard ablation says the knowledge graph does nothing:

```
proposed  0.9538      no_graph  0.9539      difference −0.0001
```

But that test is structurally blind. Accuracy is measured only at assessment positions,
and **only 89 of 237 concepts carry assessments**, all with thousands of examples each.
The graph's job is the other 148, which never enter the measurement.

Withholding *all* assessments for 27 of the 89 assessed concepts and scoring only those:

```
with graph     0.9004 ± 0.0059
without        0.8679 ± 0.0042
difference     +0.0325    p = 0.0003    Cohen's d = 5.06     (wins on every fold)
```

**Claim:** knowledge-graph propagation contributes nothing where direct supervision is
plentiful, and substantially where it is absent, and standard ablation protocols cannot
detect this.

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

All significant after Holm correction, **but read the effect sizes**: against random
d = 0.09 (negligible), against curriculum d = 0.36 (small). The honest reading is that
the prerequisite mask does most of the work and the learned ranking adds a small
consistent improvement.

Zero prerequisite violations across all 906 decisions is a verified guarantee, not an
assumption.

Note the tension worth stating out loud: the student-facing path deliberately follows
curriculum order, which scores worst in Table II. That is a product decision, because a
new student expects Week 1, and it is why the model is used for explanation and for the
adviser view rather than for reordering a beginner's first weeks.

### Explanation quality (objective, not survey)

```
fidelity 0.5318    sufficiency 0.0020    stability 0.5310
```

Sufficiency of 0.002 meant the explanation was true but radically incomplete: 99.8% of
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
   position and activity composition. Course names shown in the app are illustrative and
   chosen to match each module's published subject area (`elpr/modules.py`); the site says
   so on the Method section and at sign-up.
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
   co-enrolment ordering. Requiring lift and staying within-module cut 3,385 edges to 285,
   and cycles from 84 to **zero**.

---

## Known limitations, stated plainly

- The model predicts **successful completion**, not knowledge in the abstract. Because
  non-submission counts as failure, much of the signal is engagement. Right target for an
  intervention system; not the same claim as measuring understanding.
- Mastery is **bimodal**: engaged learners sit near 0.98 on everything, disengaged near
  0.00. Absolute thresholds are useless, so the planner uses learner-relative percentiles
  and the interface calls them Needs work, Getting there and Strong.
- The proposed model sees student demographics that DKT and SAKT do not (those follow
  their original formulations). A `no_student` ablation would separate architecture from
  feature access. **Not yet run.**
- Only 89 of 237 concepts carry assessments, so mastery for the rest is uncalibrated and
  arrives through the graph.
- The model consumes gender, disability and deprivation band with **no subgroup fairness
  analysis**. This is the most likely reviewer objection left.
- The Course Finder describes common Indian eligibility patterns that vary by university
  and board. No data about Indian students was used or collected.

---

## Repository map

```
elpr/          data · db · graph · mining · models · planner · explain · eval
               course_finder.py (rules, 43 courses) · progress.py (streaks)
               modules.py (illustrative course names) · profile.py (background fields)
sql/           00–12, the entire ETL as reviewable SQL (DuckDB)
scripts/       01 prepare · 02 graph · 03 sequences · 05 train · 06 table1
               07 labels · 08_admin (accounts) · 08 recommend · 09 table2 · run_all_kt
api/           service.py (model, paths, progress) · main.py (routes, access rules)
web/index.html one page: home, auth, dashboard, finder, research, admin, legal
tests/         84 tests
paper/         paper-revised.tex (8 pages) · paper-6page.tex + PDF · figures
docs/          architecture · build-plan · what-we-found · paper-corrections
               claimed-vs-measured · portability
results/       every number in the paper, plus figures
artifacts/     graph/ (committed) · models/ (only kgdkt_proposed_fold0.pt committed)
```

## Rebuilding from scratch

```bash
make install
# download OULAD into data/raw/ (see README, checksum included)
.venv/bin/python scripts/01_prepare_data.py     # ~18s
.venv/bin/python scripts/02_build_graph.py      # ~2.5 min
.venv/bin/python scripts/03_build_sequences.py  # ~33s
.venv/bin/python scripts/07_concept_labels.py
make test
make api
.venv/bin/python scripts/08_admin.py create <username>   # your admin account
```

Verified on a fresh clone on 14 September 2026: the committed checkpoint loads and the
site serves without retraining. Training itself runs on a free Colab T4, see `RUNBOOK.md`;
this machine has no GPU, so a full run is about 10 hours locally versus 25 minutes per
configuration on a T4.

Rebuilding rewrites `artifacts/graph/*.parquet` and two files in `results/` with the same
content in a different row order. Discard those diffs unless the numbers changed.

---

## Working agreement

Rules the owner has set. They are not negotiable, and they apply to anything user-facing.

- **No vibe-coded design.** No purple gradients, no pill-shaped buttons, no emoji icons,
  no over-the-top scroll animations, no cursor effects.
- **No invented content.** No fake reviews, testimonials, metrics, customer logos, AI
  stock photos or filler copy. Every number on the site traces to real data or the
  results files.
- **No em dashes in user-facing text.** Use a colon, a comma, or two sentences. The rule
  does not apply to the LaTeX paper.
- **Plain language.** The audience is students and teachers who did not build this.
  Prefer "chance of doing well" over "predicted mastery", name weeks rather than concept
  ids, and put technical detail behind a "show the working" toggle.
- **Never commit `data/app.db`**, and never put a real password in a file or a command.
- **Before launch:** custom domain, favicon (done), no "made with AI" badge, privacy
  policy (done) and terms page (done).
- Keep product features clearly separated from the paper: anything not evaluated in the
  paper is labelled on the site and never added to it.

---

## Next tasks

1. **Fairness analysis** for the paper: subgroup performance by gender, disability and
   deprivation band, plus the `no_student` ablation.
2. **Deployment**: a public host, a custom domain, TLS, rate limiting on sign-in, and a
   real password minimum before anyone outside the demo registers.
3. **User study** with real students, which the paper lists but has never run.
4. Optional product work: notifications or reminders, adviser notes per learner, and
   letting a student mark several weeks done at once.
