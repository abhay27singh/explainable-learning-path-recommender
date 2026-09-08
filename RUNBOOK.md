# Stage 1 — Clean Run

One session, one bundle, identical flags for every configuration. Earlier results were
produced across three different bundle versions, so they are discarded: a results table
whose rows came from different code is not defensible.

Nine configurations, roughly **two to three hours** total on a free T4.

---

## Before you start

**Runtime → Change runtime type → T4 GPU.** Confirm with cell 2 below before running
anything long.

---

## Cell 1 — mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/elpr_results
```

## Cell 2 — upload the bundle, unpack, verify the GPU

Drag `elpr_colab_bundle.zip` into the Files panel first.

```python
!unzip -q -o /content/elpr_colab_bundle.zip -d /content/elpr
%cd /content/elpr
!mkdir -p artifacts/models results
!pip -q install duckdb networkx mlxtend
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
!ls scripts
```

You should see `05_train_kt.py`, `run_all_kt.py`, and a T4.

## Cell 3 — smoke test

Never start a long run without confirming the loop works here.

```python
!python scripts/05_train_kt.py --folds 1 --epochs 2 --limit 1200 --workers 2
```

Expect AUC around 0.83. Then delete the throwaway result so it does not get mistaken
for a real run:

```python
!rm -f results/kt_proposed.json
```

## Cell 4 — run everything

```python
!python scripts/run_all_kt.py --backup /content/drive/MyDrive/elpr_results
```

This runs all nine configurations in order, copying `results/` and
`artifacts/models/` to Drive after each one.

**If the runtime disconnects**, re-run cells 1, 2 and 4. Completed configurations are
detected and skipped, so it resumes rather than restarting. To recover work from a
previous session first:

```python
!cp -r /content/drive/MyDrive/elpr_results/results /content/elpr/
!cp -r /content/drive/MyDrive/elpr_results/models /content/elpr/artifacts/
```

## Cell 5 — download

```python
!zip -qr trained.zip artifacts/models results
from google.colab import files
files.download('trained.zip')
```

---

## What runs, and why

| # | Configuration | Purpose |
|---|---|---|
| 1 | `proposed` | The full model. Table I headline row. |
| 2 | `no_graph` | Ablation: free embedding table instead of GCN propagation. |
| 3 | `no_time` | Ablation: time decay fixed at zero, elapsed-time features dropped. |
| 4 | `proposed --cold-concepts 0.3` | **The decisive test.** |
| 5 | `no_graph --cold-concepts 0.3` | **Its comparison.** |
| 6 | `dkt` | Piech LSTM. The reference every knowledge-tracing paper reports. |
| 7 | `sakt` | Self-attentive KT. Separates "attention helps" from "the graph helps". |
| 8 | `gnn` | GCN embeddings plus LSTM. Graph without time awareness. |
| 9 | `majority` | Predicts the base rate. Must score exactly AUC 0.5. |

### Why the cold-concept runs matter most

Ordinary AUC is computed only at assessment positions, and assessments exist for
**89 of 237 concepts**. Every headline number is therefore measured on concepts that
already carry thousands of labelled examples each.

The knowledge graph's job is to give sensible mastery estimates to the **148 concepts
with no assessments of their own**, by propagating from their prerequisites. Those
concepts never enter the AUC computation. The standard ablation is structurally blind
to the thing it ablates — which is why `no_graph` matched `proposed` exactly.

The cold-concept runs withhold every assessment for 27 of the 89 assessed concepts,
strip their responses from the model's input so they behave like genuinely unassessed
concepts, and then score **only** those held-out concepts.

- **Graph version clearly ahead** → prerequisite propagation works where it matters,
  and this becomes the paper's key result.
- **The two match** → the graph adds nothing even under conditions designed to favour
  it. The honest paper then presents knowledge tracing plus explainability, with the
  graph as structure for explanation rather than for accuracy.

Either outcome is publishable. Only a fabricated one is not.
