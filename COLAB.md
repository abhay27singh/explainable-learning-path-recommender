# Running Stage 1 on Colab

The pipeline is artifact-based, so nothing in the code changes between machines.
This laptop is an Intel Mac with no GPU: a full 5-fold run is roughly **17 hours per
variant** locally versus about **an hour** on a free Colab T4.

## 1. Build the bundle

```bash
.venv/bin/python scripts/package_for_colab.py
```

Produces `elpr_colab_bundle.zip` (~20 MB).

## 2. Open a GPU runtime

<https://colab.research.google.com> → new notebook → **Runtime → Change runtime type
→ T4 GPU**.

## 3. Paste these cells

**Cell 1 — upload**

```python
from google.colab import files
up = files.upload()          # choose elpr_colab_bundle.zip
!unzip -q -o elpr_colab_bundle.zip -d /content/elpr
%cd /content/elpr
!mkdir -p artifacts/models results
```

**Cell 2 — check the GPU**

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

**Cell 3 — dependencies**

Colab already ships torch, numpy, pandas and scikit-learn. Only the extras are needed.

```python
!pip -q install duckdb networkx mlxtend
```

**Cell 4 — smoke test first (about a minute)**

Never start a long run without confirming the loop works on the remote machine.

```python
!python scripts/05_train_kt.py --folds 1 --epochs 2 --limit 1200 --workers 2
```

Expect AUC around 0.83 even on this fraction of the data.

**Cell 5 — the real run**

```python
!python scripts/05_train_kt.py --variant proposed --folds 5 --epochs 12 --workers 2
```

**Cell 6 — ablations**

```python
!python scripts/05_train_kt.py --variant no_graph --folds 5 --epochs 12 --workers 2
!python scripts/05_train_kt.py --variant no_time  --folds 5 --epochs 12 --workers 2
```

`no_graph` is the important one: 148 of 237 concepts have no assessment of their own,
so removing prerequisite propagation removes their only source of signal.

**Cell 7 — bring the results back**

```python
!zip -qr trained.zip artifacts/models results
from google.colab import files
files.download('trained.zip')
```

## 4. Unpack locally

```bash
unzip -o ~/Downloads/trained.zip -d "/Users/abhay/Desktop/AI Based Personalized Leraning System"
```

Everything downstream — explanations, the planner, the demo — reads these checkpoints.

## Notes

- Free Colab disconnects after roughly 90 minutes idle. Keep the tab open.
- Each variant writes `results/kt_<variant>.json` with per-fold AUC, RMSE, the fitted
  temperature, ECE before and after calibration, and the cold-start curve.
- The temperature is fitted on one half of each fold's held-out predictions and
  scored on the other, so the reported ECE is not self-flattering.
