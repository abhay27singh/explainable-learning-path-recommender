# Instructions for coding agents

Applies to every assistant working in this repository: Claude Code, Antigravity, or any
other IDE agent.

**Read [`CONTEXT.md`](CONTEXT.md) first.** It holds the full project state: what the
research measured, how the web application is built, the known limitations, and the next
tasks. This file is only the short version.

## What this project is

An explainable learning-path recommender over the OULAD dataset (the research, written up
in `paper/`) plus a web application for real students (study levels from class 10 to
postgraduate, a week-by-week course path, and a rule-based Course Finder). The paper's
results stay on an admin-only Research page as validation. Product features are never
added to the paper.

## Commands

```bash
make test                                   # 84 tests, pytest
make api                                    # http://localhost:8420
.venv/bin/python -m pytest tests/test_stages.py -q
.venv/bin/python scripts/08_admin.py create <username>
```

Rebuilding the data needs the OULAD download; see the quick start in `README.md`.

## House rules

- **No vibe-coded design**: no purple gradients, pill buttons, emoji icons, heavy scroll
  animations or cursor effects.
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
- `web/index.html` is one file with no build step. After editing it, check the script with
  `node --check` on the extracted `<script>` block, then reload the page and read the
  browser console.

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
