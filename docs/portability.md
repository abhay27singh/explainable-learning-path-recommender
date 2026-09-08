# Moving This Project to Another Computer

Two separate things travel differently: **the project** and **the conversation**.

---

## 1. The project — use git

The working directory is **2.4 GB**, but almost none of that needs to move:

| Directory | Size | Travels? |
|---|---|---|
| `data/raw/` | 487 MB | No — freely re-downloadable |
| `data/oulad.duckdb` | 344 MB | No — regenerated in ~20s |
| `data/processed/` | 18 MB | No — regenerated |
| `artifacts/models/` | 39 MB | Separately (see below) |
| `data/app.db` | 192 KB | **Never** — real accounts and password hashes |
| Everything else | ~2 MB | Yes |

So the repository itself is about **2 MB**: source, SQL, configs, documentation, results
JSON, figures, and the concept graph. Everything else is either downloadable or
regenerable.

### Set it up

```bash
cd "/Users/abhay/Desktop/AI Based Personalized Leraning System"
git init
git add .
git commit -m "Explainable learning path recommender: pipeline, model, evaluation, demo"
```

Then create an empty repository on GitHub and push. **Make it private** unless you
intend to publish before the paper is submitted.

```bash
git remote add origin https://github.com/<you>/<repo>.git
git branch -M main
git push -u origin main
```

### On the other computer

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
make install                      # creates .venv, installs dependencies
```

Then rebuild the data — about three minutes total:

```bash
.venv/bin/python scripts/01_prepare_data.py    # needs data/raw/ first
.venv/bin/python scripts/02_build_graph.py
.venv/bin/python scripts/03_build_sequences.py
.venv/bin/python scripts/07_concept_labels.py
```

**Getting `data/raw/` back:**

```bash
mkdir -p data/raw && cd data/raw
curl -L -o oulad.zip https://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip
unzip oulad.zip
```

Verify it is the same data — this checksum matches the copy used for every result in
the paper:

```
90dda45037939953f979072fa70a809ebe07e90ec783c408762af7698a3825ec  oulad.zip
```

### The trained checkpoints

`artifacts/models/` is 39 MB — too large for a comfortable git repository and not
something git handles well. Three options, in order of preference:

1. **Google Drive.** They are already in `MyDrive/elpr_results/models/` from the Colab
   run. Download and unzip into `artifacts/models/`.
2. **A GitHub release.** Attach `trained.zip` to a release rather than committing it.
3. **Retrain.** `scripts/run_all_kt.py` reproduces every checkpoint on a Colab T4 in
   about two hours. Seeds are fixed, so the numbers come back the same.

Without checkpoints the demo still starts, but says plainly that it is running an
untrained model and that the numbers are meaningless.

### What must never be committed

`data/app.db` holds real usernames and password hashes for anyone who registers. It is
in `.gitignore`. If you ever commit it by accident, treat every password in it as
compromised and tell the people affected.

---

## 2. The conversation

Claude Code sessions are stored **locally**, not in the cloud:

```
~/.claude/projects/-Users-abhay-Desktop-AI-Based-Personalized-Leraning-System/
    f66108b9-3703-45a0-82a3-e2529fe52f7e.jsonl        (15 MB)
```

That file is this entire conversation. Three ways to have it elsewhere:

**Copy the file.** Put it in the matching path on the other machine:

```bash
~/.claude/projects/<same-encoded-project-path>/
```

The directory name is the project path with slashes replaced by dashes. Claude Code
will then list the session and `claude --resume` can pick it up. This is unofficial —
it depends on internal storage layout that may change.

**Use Claude Code on the web** (`claude.ai/code`) for work you want available from any
machine. Those sessions live in your account rather than on one laptop.

**Rely on the documentation instead — this is the one I would actually trust.**

Every decision, finding and number from this conversation is already written down:

| Document | What it holds |
|---|---|
| `docs/architecture.md` | System design, every component and why |
| `docs/build-plan.md` | Stage plan and what each stage delivers |
| `docs/what-we-found.md` | The four data problems and how each was solved |
| `docs/paper-corrections.md` | Every defect in the draft, by severity |
| `docs/claimed-vs-measured.md` | Every claim against its measured value |
| `docs/portability.md` | This file |
| `RUNBOOK.md` | Reproducing the training runs |
| `COLAB.md` | GPU training instructions |

A chat log is a transcript of how the thinking went. These are the conclusions, written
to be read by someone who was not here — including your supervisor, a reviewer, or you
in three months. If only one survives, it should be these.

---

## Quick check that a fresh machine is working

```bash
make test                                     # 32 tests
.venv/bin/python scripts/06_make_table1.py    # regenerate Table I
make api                                      # then open http://localhost:8420
```

If `make test` passes and Table I reproduces AUC 0.9538 ± 0.0019, the move worked.
