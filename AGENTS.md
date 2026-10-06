# Instructions for coding agents

Applies to every assistant working in this repository: Claude Code, Antigravity, or any
other IDE agent.

**Read [`CONTEXT.md`](CONTEXT.md) first.** It holds the full project state: what the
research measured, how the web application is built, the known limitations, and the next
tasks. This file is only the short version.

## What this project is

An explainable learning-path recommender over the OULAD dataset (the research, written up
in `paper/`) plus GyanGraph, a web application for Indian students: the ladder from class 10 to
postgraduate, a week-by-week course path, entrance exam revision plans (JEE, NEET, CUET),
and a rule-based Course Finder covering 64 academic and skill courses. The paper's results
stay on an admin-only Research page as validation. Product features are never added to the
paper.

## Commands

```bash
make test                                   # 248 tests, pytest
make api                                    # http://localhost:8420
make stop                                   # stops whatever is listening on 8420
.venv/bin/python -m pytest tests/test_stages.py -q
.venv/bin/python scripts/08_admin.py create <username>
```

Rebuilding the data needs the OULAD download; see the quick start in `README.md`.

## House rules

- **No vibe-coded design**: no purple gradients, pill buttons, emoji icons, heavy scroll
  animations, cursor effects or hidden system cursors.
- **No invented content**: no fake reviews, testimonials, metrics or filler copy. Every
  number shown to a user traces to real data or to `results/`.
- **No em dashes in user-facing text** (the LaTeX paper is exempt). Use a colon, a comma,
  or two sentences.
- **Plain language** for students and teachers who did not build this. Technical detail
  goes behind a "show the working" toggle.
- **Never commit `data/app.db`** (real accounts), and never write a real password into a
  file or a command.
- Match the surrounding code: comment density, naming and idiom. Tests are prose-named
  and explain why the case matters, and regression tests say what broke.

## Things that are easy to get wrong

- **Week numbering.** The dataset counts a course's first week as 0. Only the page adds
  one, in `humanize()` in `web/index.html`. Never shift weeks on the server or in tests.
- **Student path versus adviser path.** Students get `Service.course_path` (course order
  from Week 1, model used for the explanation). Advisers get `Service.learning_path`
  (model-ranked). Do not merge them.
- **Access rules live in the API**, not only in the page: `_require`, `_require_adviser`,
  `_require_admin`, and the level check in `/api/course-finder`.
- **The Course Finder is rule-based**, not the model, and every place it appears says so.
  The same goes for `elpr/exams.py`: syllabus outlines are published facts, not predictions,
  and no exam date, cut-off or rank is ever stated.
- **Both planners split a syllabus with `course_finder.spread`.** Courses and exams must
  deal their weeks the same way, because a student uses both in the same week.
- **Self-check questions only go where the topic is real.** JEE and NEET units are; a
  week of the course path is not (OULAD's content is anonymised). Never write questions
  for course weeks, and never record a self-check as a study event. A new question needs
  one right answer, three distinct wrong ones and a reason; `tests/test_selfcheck.py`
  checks the shape but not the facts, so check the facts yourself.
- **Study links depend on the level.** NPTEL and SWAYAM carry nothing for class 10, class
  12 or an entrance exam written on them, so school-level subjects link to NCERT, Khan
  Academy and YouTube instead (`study_links`).
- `web/index.html` is one file with no build step. After editing it, check the script with
  `node --check` on the extracted `<script>` block, then reload the page and read the
  browser console.
- **Colours come from tokens, never literals.** Three theme blocks have to stay in step:
  bare `:root`, the `prefers-color-scheme: dark` block, and `:root[data-theme="dark"]`. A
  colour defined in only one of them disappears in the other theme.
  `tests/test_theme_tokens.py` now fails if the two dark blocks differ or a value
  swallows the next token through a missing semicolon; both have happened here.
- **A field the page hides must be disabled too.** A hidden `required` field blocks its
  form without showing why; that is how adviser signup broke.
- **`hidden` always wins.** There is a global `[hidden]{display:none!important}`. Do not
  hide things with inline `display` when the attribute will do.
- **Restart the server after adding a route.** uvicorn runs without `--reload` here, so a
  new endpoint 404s until the process is restarted.
- **Accounts made while testing are real accounts.** Delete them from `data/app.db` when
  the check is finished.

## Working side by side with another agent

Two agents editing this repository at once will overwrite each other unless you separate
the work.

- Give each agent its own branch, or its own git worktree:
  `git worktree add ../elpr-antigravity -b antigravity-work`.
- Only one server can hold port 8420. The second agent should use another port:
  `.venv/bin/python -m uvicorn api.main:app --port 8421`.
- `data/app.db` is local to each working copy, so accounts made in one do not exist in the
  other. Each copy needs its own admin account.
- Before starting, run `git status` and `git log --oneline -3`, and say what you are about
  to change. Before committing, run `make test`.
