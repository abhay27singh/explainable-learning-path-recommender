# Project Context — read this first

Hand this to a fresh agent session (Claude Code, Antigravity, any IDE assistant), a
collaborator, or a supervisor and it will have everything it needs. Every research claim
here is backed by a file in `results/`.

Last updated 2 October 2026.

---

## What this is

Two things in one repository, deliberately kept apart:

1. **The research.** An explainable learning-path recommender over the Open University
   Learning Analytics Dataset (OULAD): knowledge tracing plus a prerequisite-aware
   planner, built for an IEEE paper.
2. **The product.** GyanGraph, a web application for real students who sign up: study levels from
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
| 8 · Product work for Indian students | Ongoing: ladder, skill path, exams, planner |

`make test` runs 300 tests. `make api` serves http://localhost:8420 and `make stop` stops it;
`HOW_TO_RUN.txt` has the start and stop steps in plain words.

The git remote is `https://github.com/abhay27singh/explainable-learning-path-recommender`
and work is on `main`.

---

## The web application

Single FastAPI service (`api/main.py`) plus one page, `web/index.html`, a vanilla-JS
single-page app with hash routing (`#/home`, `#/login`, `#/signup`, `#/dashboard`,
`#/progress`, `#/record`, `#/finder`, `#/research`, `#/admin`, `#/privacy`, `#/terms`,
`#/policies`). The site is called GyanGraph (`SITE_NAME`); the repository and the paper keep
their own names.
No build step, no framework.

**Roles.** `student`, `adviser`, `admin`. Admins are a superset of advisers. An admin can
only be created from the command line (`scripts/08_admin.py create <username>`), never
through the API, so a registered account can never escalate itself.

**Layout.** Signed-in pages are a workspace: a rail on the left carrying who you are,
where you are in the site and where you are in your course; the work in the middle; the
things a student does often on the right (entering a mark, recent marks, milestones,
shortlist). Blocks that are wide by nature, the path and the charts, run under both
columns. Both rails fall away under 860px, where the score box returns to the next-step
card. Light and dark themes follow the system setting until the viewer picks one, stored
per browser under `elpr-theme`.

**Liquid glass.** One layer at the end of the stylesheet gives every page and role the
same finish: panels are a translucent fill with a backdrop blur, a lit top edge and a soft
shadow, over faint teal and green light (`--ground-art`) so the blur has something to
refract. A pane inside a pane gets a lighter fill and no second blur. Controls keep a
visible edge (`--control-edge`): a glass edge on a glass panel made buttons read as plain
text. Buttons stay rounded rectangles, not capsules, under the house rules. Anyone whose
system asks for reduced transparency gets solid panels. Pages run to 1760px, the same
edges as the header. On the dashboard the right rail spans two rows, so the path follows
the next step directly instead of after a blank strip.

**Academic Navigator.** The visual system adopted from a Stitch export (October 2026),
taking only what our data can back. Colours are the export's "icy blue" variant: an ice
canvas `#F3F9FE`, ink `#0F243D`, sky `#0369A1` for actions and links and bright sky
`#0EA5E9` only for bars, lines and glow (it is 2.8:1 against white, too faint for text);
the dark theme is the export's glacial navy `#0A1526` with cyan `#38BDF8` and dark text on
it. Every text pairing measures at least 4.5:1 in both themes; Plus Jakarta Sans for headings when available,
Inter otherwise; every page header is a small teal label, a large title, one line and
actions on the right; the header nav is a segmented control with square corners. Text on
a filled control uses `--on-accent`, which is dark in the dark theme. Not taken from the
export: its AI photos, salary, placement and review figures, star ratings, institutional
claims, product name and logo, pill buttons and gradient buttons.

**Prerequisite graph.** The dashboard's path block has five views, remembered per
browser: Timeline, Board, Prerequisite graph, Table and What if. Every split of a course
into Start, Early, Later and End comes from one helper, `courseParts`, so the graph,
board, Progress and the record always agree. The graph lays the course out in those four
columns from `/api/graph/{module}`, marks each week from the `done` list the path
response carries (never repeating the server's rule), and draws only the selected week's
links: a week can have sixteen and a course over a hundred. "Show the whole chain" lights
everything the week builds on and everything that follows from it. The key week is the
one the most later weeks build on directly. A week whose earlier weeks are unfinished is
"Not ready", never "Locked". The control band says "Prerequisites in order", or names
the first finished week that builds on an unfinished one (`orderClashes`).

**Your path** (the timeline view) says in words what each of the next five weeks builds on
and opens up, with "Ready to start" or "After Week N" beside it. **Which weeks look hard
for you** (`courseStrip`) is the course in its four parts, one block per week, shaded by
the model's relative level, with week numbers, a tick under done weeks and the next week
outlined. A sentence names the three hardest and three easiest weeks still ahead, and the
legend runs from "Harder for you" to "Easier for you". For advisers it keeps its old title
and stays clickable for "already knows this".

**Board** shows the four parts as columns, each marked Complete, In progress, Not
started or Current, with every week as a card. **What if** (`/api/me/what-if`,
`Service.what_if`) adds the weeks a student picks (up to six) to a copy of their history
as passed tests and runs the model again. It writes nothing and logs nothing. The page
shows weeks done, weeks ready and the model's average before and after, which weeks would
open, where the path would go on from, and a before and after dot per part. It says the
model lifts most weeks for a pass, which is the engagement signal the paper describes.

**Indian education policy, checked October 2026.** NEP 2020 as UGC applies it: the
four-year undergraduate exits carry UGC's credits (40, 80, 120, 160) and the fourth year
is honours, or honours with research above 75 percent (`NEP_UG_EXITS`, `NEP_EXIT_NOTE`);
UGC master's degrees are two years, or one after a four-year honours degree (`UGC_PG`);
B.Ed states NCTE's one-year option from 2026-27; the B.El.Ed, closed to new students from
2026-27, is replaced by NCTE's four-year ITEP through NCET. NEET UG follows the NMC
syllabus for 2026 (20, 20 and 10 units, Biology as one subject), JEE Main the unchanged
NTA syllabus (20, 20, 14), CUET UG the 2025 rules (up to five subjects, chosen freely,
universities setting their own; the paper is the General Aptitude Test). Check these each
academic year: NMC, NTA, UGC and NCTE all revise them. The public Education policies page
(`#/policies`) states each of these with its official source, and
`tests/test_policies_page.py` fails if its numbers drift from `course_finder.py` or
`exams.py`, so a syllabus change means editing the page too. NCTE's one-year B.Ed and the
end of the B.El.Ed come from its Regulations 2025, published as a draft; the page says so.
The privacy page names the DPDP Act 2023 and states that the DPDP Rules 2025 require a
parent's verifiable consent for under-18 users from 13 May 2027, which the site does not
have yet.

**Security and launch hygiene.** `guard` middleware in `api/main.py` applies rate limits
(`RATE_LIMITS`, in memory, per client address) before the work and security headers
(`SECURITY_HEADERS`, CSP included) after it. API docs are off unless `ELPR_API_DOCS=1`;
`ELPR_HTTPS=1` adds Secure cookies and HSTS; `ELPR_SITE_URL` fixes the address used in
link previews, `robots.txt` and `sitemap.xml`. A missing page gets `NOT_FOUND_PAGE` (the
API still answers 404 in JSON); inside the page, an unknown `#/` address shows
`view-notfound`. Every page sets its own tab title. Icons and the share picture come from
`scripts/10_icons.py`; backups from `scripts/11_backup.py` (SQLite's backup call plus an
integrity check). `tests/test_security.py` covers all of it. Visitors on a phone get the
same bottom bar as students (Home, Finder, Explore, Sign in).

**Full audit, 7 October 2026.** Every route was probed live as a visitor, a student, a
school student, an adviser and an admin, and every page swept at 375, 600, 768, 1024 and
1440 pixels. Fixed: anyone could sign up as an adviser and read every student's record, so
a new adviser now waits for an admin (`approved` column, `approve_adviser`, `_may_view`,
the Approve button on Accounts, a waiting page; existing advisers kept access). Behind the
proxy the rate limits read the visitor-written first `X-Forwarded-For` entry; now the last.
Requests over 64 KB are refused (`MAX_BODY`), and `/api/` answers carry
`Cache-Control: no-store`. Out-of-range week ids (`_overrides`), `upto=nan` (`UPTO`),
`k` below 1, and start dates past 2100 (`_monday_of`) answered 500 or a wrong record; a
school student's own state, recommendations and progress came back as an invented
Psychology record (`_own_course`, 409). `/static/index.html` now redirects to `/`. Dark
colours are screen-only, so printing gives dark text. Profile questions and the note box
have labels; the theme switch lost its capsule shape. All 204 self-check answers were
checked by hand; the dependency check found advisories only for torch 2.2.2, which is
pinned because it is the last build for Intel Macs and loads only this repository's own
files with `weights_only=True`. On a Linux server a newer torch is the better choice.

**Adviser invites and classes.** Approval alone left one wrong approval able to read every
student, so access now needs both sides. The admin's side: an adviser can only sign up with
a one-time invite (`adviser_invites`, only a SHA-256 of the code is stored, 7 days,
`INVITE_DAYS`) made for a named person, then waits for approval; the Accounts page shows
what each waiting adviser was invited as. The student's side: an approved adviser has a
class code (`users.class_code`, 8 characters without 0, O, 1, I or L) and sees only the
registered students who joined with it (`adviser_students`, checked in `_may_view`, the
learner list, the overview and the notes). Students join at sign-up or under My details and
can leave there; joining is rate-limited. The anonymised dataset learners stay open to every
approved adviser. On the first start with classes, existing approved advisers kept the
students they could already see (`_keep_existing_links`), so `demo_adviser` still has the
four demo students. ID uploads were considered and rejected: an ID shows who someone is,
not that they work at the school, an admin cannot tell a real card from an edited photo, and
storing ID documents (Aadhaar copies especially) would make the database far worse to leak.
The same audit found a school student's page squeezed into a 236 pixel column on phones
(`.workspace.two` outranked the phone rule) and footer links running together on touch
screens; both fixed, with tests.

**My learning, decluttered.** The page said "2 of 35 weeks done" four times (heading, a
tile row, the left rail, the band above the path). It is now one sentence in the heading
(`courseLine`); the tile row, the rail's course panel and the band's progress bar and
status chips are gone, and the band only speaks when weeks were done out of order. The
next step is titled by its week; the student's level on it moved behind "Show the
model's working" (advisers still see it on the card). Milestones and the quiz box are
written as sentences. On wide screens the right column stays in view like the left one:
its contents stick inside a column as tall as the rows beside the path
(`.railside.stick > .railside-in`), so they stop above the full-width strip.

Progress got the same treatment: one sentence in the heading, the activity card's numbers
and the streak as a sentence (said there only, `milestoneBlock(p, {streak: false})`), the
course card's counts as a sentence, recent activity as sentences with dates beside the
calendar, and the right column fixed like My learning's. The weekly summary from
`progress_for` no longer says "It fell ..." with nothing for "it" to mean, or "You passed
0 tests". The school Progress page lost its tile row for a sentence, and no rail shows
numbers any more (`railcourse` is gone).

**Four ideas taken from a reference site (cs-visualizer.com).**
- *Replay how you got here* on Progress: first, back, play, forward, last and a slider
  step through the student's own record (`/api/me/replay`, `Service.replay`, latest 60
  entries), redrawing the week map after each entry with one sentence about it. It shows
  what was recorded and how the path moved, never the model's raw figures: for a
  student's own record those sit near zero for most weeks (about 1 to 5 percent), which is
  also why the week map's tooltip now gives the level, not "Model: 1% chance".
- *See a real path* (`#/try`, public): the model's path and week strip for one real,
  anonymised OULAD learner halfway through their course (`Service.demo`, learner 599577,
  computed once, warmed at startup). It says it is the model's own ranking and how a
  student's own path differs. `pathList` for advisers now says "The model's next N choices,
  best first" instead of the wrong "in course order".
- *Real screens beside the words* on the home page: that learner's next step and a real
  course card, each labelled for what it is.
- *A section bar* on Privacy, Terms and Education policies: it stays under the header
  (whose height it measures, as the phone header wraps), lists the page's headings and
  marks the one being read.
Not taken: hover-only content (phones have no hover), scroll-triggered panels (against the
house rules), developer-style `// LABELS`, "coming soon" badges.

**Visitors keep the Course Finder and Explore.** Gating them was considered and rejected:
a student signs up for something they have seen work, results get shared with friends and
parents, and the privacy page promises they work without an account. What an account adds
is said where it is earned (`signupCta`, the self-check result), and a course a visitor
taps "Save this course" on is remembered in the tab (`PENDING_SAVE`) and saved as soon as
they sign up or sign in (`keepPendingSave`).

**Updates reach open tabs.** Moving between pages never reloads `index.html`, so a tab
left open kept running its first copy after an update. `/api/version` reports when the
file last changed; on a page change, at most every 30 seconds, the page asks and reloads
itself at the address being opened if it changed.

**My details** has its own address, `#/details`, drawn by the dashboard view. It used to
be a tab on `#/dashboard`, which kept reopening until a refresh. The student's name in the
header and the profile card in the sidebar open it; for advisers and admins the name is
plain text. It is drawn as its own page, "My details" with an account card on the right,
and nothing in the header is highlighted, because it once looked so like My learning that
students thought the click had failed.

**Header.** Brand, a context chip with the student's level and course (it opens My
details), the four places in the same order as the sidebar and the phone tab bar (My
learning, Progress, Course Finder, Explore), a search box that opens Explore with the words in it, the
theme switch, and a profile chip with the user's initial. No photos anywhere.

**Dashboard.** The page header carries the actions (Add to my calendar, Record of study,
Open the next week). Over the path, a control band holds the view switch, the course,
weeks done and whether the next week is ready.

**Course Finder steps.** The page is a planning form with two questions, both visible: A,
where you are (level, then stream and subjects for class 12, or degree and then branch or
subject for a graduate, from `DEGREE_GROUPS`) and what you want next (the next levels,
including the skill path); B, what you enjoy, as icon cards. "N of 2 answered" sits in
the page header. Results stay live. The right-hand column is pinned and lists every
match, the closest first; a row opens that course's page in Explore. When nothing at the
chosen level matches, the column shows what matches at the other levels, usually the
skill path, with one click to switch. Under 1180px the column folds away and the match
appears at the top of the list. The background fields redraw only when the rung, stream
or degree changes, so answering does not lose keyboard focus. There is no pace question:
every week plan carries its own length choice. The best-ranked course leads as the closest match, described by the interests it
covers and Strong or Partial, never a percentage. A trip to Explore and back keeps the
answers; a new rung or signing out clears them.

**Explore.** `#/explore` is a catalogue with its own address. The header gives the
course count and how many are on the skill path. Level, length (`duration_band`) and
subject-area filters carry real counts; a search card filters course names and subjects,
with a clear button and the `/` key to jump to it; the skill path and the six largest
subject areas are quick filters with a tick when on; every filter is a removable chip. A
signed-in student also gets a "For you" box naming the levels they can go on to (from
`next_levels`), a switch to show only those, and a "Next for you" flag on those courses.
A page per course: duration, how many subjects over how many parts, the NEP exit points
(only for the degrees in `NEP_EXIT_DEGREES`), every entrance exam named in its eligibility,
the syllabus with study links, a plan, "Where this leads" (`leads_to`: the postgraduate
courses its degree qualifies for by the Finder's own rules, split into those for this
degree and those open to any graduate), and other courses in the same area. When the
named exam is JEE Main or NEET UG, or the degree is one central universities fill through
CUET UG, the exam's own card sits on the page with its syllabus, plan and self-checks.
Up to three courses go into a comparison tray and open side by side, every row a field the
catalogue holds. Linked from the header, every student rail and both Finder buttons.

**Phone.** Below 860px a signed-in user gets a tab bar at the bottom (by role: My
learning, Progress, Finder, Explore for students; Learners, Finder, and for admins
Research and Admin) and the header nav steps aside.

**Study levels.** Every student picks one at sign-up: `class_10`, `class_12`, `diploma`,
`ug`, `pg` (`elpr/course_finder.py: STAGES`). Diploma and degree students also choose a
course and get a weekly path; school students get the ladder and the Course Finder. The
Finder offers only the level after the student's own (`NEXT_LEVELS`), enforced in the API,
not just hidden in the page. Levels are editable later under My details.

**The path.** `Service.course_path` walks the course in order from its first week.
A week counts as done when studied or passed; "I found it hard" keeps it in place
(`Service.done_concepts`). A quiz mark of 40 or more counts as a pass and anything lower
keeps the week in the path, matching the binarisation in `sql/05_events.sql`. There is no
bare "I passed" any more: `/api/me/study` refuses `correct: true` without a score, because
a pass is evidence the model learns from. "I found it hard" needs no mark, since it only
ever holds a week in place. Each step is
explained by the model with the earlier steps marked as known, so every explanation is
real model output under a stated assumption. The card reads as an instruction: a status,
a "Why this week" line from the prerequisite graph, and one "Do this" line. Advisers keep
the model-ranked view (`Service.learning_path`, greedy planner) plus a compare-planners
tab.

**Week numbering.** The dataset counts a course's first week as 0. The page adds one for
display in `humanize()` in `web/index.html`, applied once per API response. Server, data
and tests all still use the dataset's numbering.

**Course Finder.** Rule-based, not the model, and labelled as such everywhere it appears.
64 courses in `elpr/course_finder.py`: 48 academic and 16 NSQF skill courses, which NEP
2020 treats as an equal track alongside the academic one. Eligibility by class 12 subjects
or bachelor's degree (B.Ed follows NCTE: science, social science, humanities, commerce or
engineering), interest ranking, an explore-everything view, a saved shortlist, and
a week by week plan (`weekly_plan`) with dates and an iCalendar download. Study links are
chosen by level: NCERT, Khan Academy and YouTube for school, NPTEL, SWAYAM and YouTube for
college, Skill India and NSDC for the skill track.

**Entrance exams.** `elpr/exams.py`: JEE Main, NEET UG and CUET UG, each with the
published unit outline of its syllabus, the official NTA link, what it leads to, and the
same week by week planner (`exam_plan`) with an iCalendar download. An exam appears only
for a stream that can sit it, and the stream cards name the exams each stream opens. CUET
is assembled from the student's own class 12 subjects rather than a fixed list. No exam
date, cut-off or rank is stated anywhere: those change yearly, and a test enforces it.

**Self-checks.** `elpr/selfcheck.py`: three questions per topic, 68 topics, 204
questions, covering all 104 JEE Main and NEET UG units. Units map to topics by
(section, unit), because NEET has a Thermodynamics unit in both Physics and Chemistry.
Options are stored right-answer-first and shuffled on the way out by a crc32 seed, so the
right answer lands evenly across the four places and never moves on reload. Marking
happens on the server. Results go to their own `self_checks` table, never to
`study_events`, because an exam unit is not a week of the student's course. The page
shows the latest result per unit, not the best. The weekly course path has no questions
by design: OULAD's content is anonymised, so its weeks have no topic. CUET's language,
general test and commerce and humanities papers have no questions yet.

**Progress and record.** One activity card reads like a contribution graph: the count
for the year, active days, current and longest streak, then a year of days
(`CALENDAR_WEEKS = 53`). Beside it, this week's summary and milestones; below, "Your
course so far" (weeks done, ready to start and not ready yet, whole course and by part),
"Finishing the course" (weeks done over time from the student's own dates, with two
straight projections from today: their average pace since their first activity, needing a
week of history, and the course's pace of a week each week; arithmetic, not the model),
the quiz marks as horizontal bars against the pass mark, and a week map of the whole
course with the next week outlined and a dot for each quiz mark. The model's estimate
per week sits behind "How the model sees each week", on a scale that stops at the first
round step above the highest week. The old "Overall" level split is gone: levels are the
student's own 30th and 70th percentiles, so it always read about a third each. Status
colours are tokens (`--st-done`, `--st-ready`, `--st-low`), validated for colour
blindness and contrast in both themes. The printable record of study is laid out by part,
each Complete, In progress or Not started, with every week's last record, mark and date
(it asks `/api/me/state?full=true` for the whole history). All computed from recorded events only (`elpr/progress.py`). A
student's first week makes no comparison claims, because there is nothing real to compare
with. The record says plainly that it is not a certificate and is not issued by any
institution.

A student still at school has no course for the model, so their Progress page
(`renderSchoolProgress`) is built from self-checks alone: units checked out of the units
that have questions, how many pass on the latest try, and per exam and section the units
checked (with tries and the date) and the ones not checked yet, each with its button.
Advisers who open a school student are told there is no course path (409 from
`_course_student`) instead of being shown the model's fallback course. On a phone,
Explore's filters fold behind one Filters button that counts what is on.

**Research page.** Admin only, in the page and in `/api/metrics`. It reads `results/*.json`
directly so it cannot drift from the paper.

**Performance.** Mastery is cached per learner (dataset learners by `lru_cache`, registered
students keyed by their event count, last event, module and profile), so one page render
runs the model once rather than five times. GZip middleware, a background warm-up thread
on startup, and a self-hosted Inter woff2 instead of a font CDN. Measured: page 59 ms,
endpoints 2 to 34 ms, adviser path 297 ms (`tests/test_performance.py`).

**Security posture.** scrypt password hashing, httpOnly session cookies, server-side
access checks, an 8 character password minimum (`MIN_PASSWORD`), and sign-in throttling
after 8 failures in 15 minutes per username, which returns 429. Every account action is
written to `user_log`. No TLS and no account recovery: demonstration grade, and the
privacy page says so. `data/app.db` holds real accounts and **must never be committed**;
it is git-ignored.

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
               course_finder.py (rules, 64 courses, the ladder, weekly plans)
               exams.py (JEE, NEET, CUET syllabus outlines and revision plans)
               selfcheck.py (204 questions on JEE and NEET units, server marking)
               progress.py (streaks, calendar, milestones) · ics.py (calendar files)
               modules.py (illustrative course names) · profile.py (background fields)
sql/           00–12, the entire ETL as reviewable SQL (DuckDB)
scripts/       01 prepare · 02 graph · 03 sequences · 05 train · 06 table1
               07 labels · 08_admin (accounts) · 08 recommend · 09 table2 · run_all_kt
api/           service.py (model, paths, progress) · main.py (routes, access rules)
web/index.html one page: home, auth, dashboard, progress, record, finder,
               research, admin, legal
tests/         300 tests
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
  policy (done) and terms page (done). Parent consent for under-18 accounts before
  13 May 2027 (DPDP Rules 2025).
- **No custom cursors, and no hiding the real one.**
- Keep product features clearly separated from the paper: anything not evaluated in the
  paper is labelled on the site and never added to it.

---

## Next tasks

1. **Fairness analysis** for the paper: subgroup performance by gender, disability and
   deprivation band, plus the `no_student` ablation.
2. **Deployment**: a public host, a custom domain and TLS. The password minimum and
   sign-in throttling are done; account recovery is not.
3. **User study** with real students, which the paper lists but has never run.
4. **Product work still open**: self-check questions for CUET's language and general
   test papers, a printable PDF of a week plan, Hindi and regional languages, and
   self-service account deletion.
