#!/usr/bin/env python
"""Stage 1: train KG-DKT with 5-fold cross-validation.

Runs unchanged on CPU or GPU. On this project's Intel Mac the CPU path is intended
for smoke tests (--folds 1 --epochs 1 --limit 2000); full runs go to a Colab T4.

    python scripts/05_train_kt.py --variant proposed
    python scripts/05_train_kt.py --variant no_graph   # ablation
    python scripts/05_train_kt.py --variant no_time    # ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elpr.data.dataset import (  # noqa: E402
    RESPONSE_UNLABELLED, SequenceDataset, collate, load_splits,
)
from elpr.eval.metrics import (  # noqa: E402
    auc_rmse, cold_start_auc, expected_calibration_error, fit_temperature,
    reliability_curve,
)
from elpr.models.baselines import BASELINES, BaselineConfig, MajorityBaseline  # noqa: E402
from elpr.models.kgdkt import KGDKT, KGDKTConfig  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

VARIANTS = {
    "proposed": dict(use_graph=True, use_time=True),
    "no_graph": dict(use_graph=False, use_time=True),
    "no_time": dict(use_graph=True, use_time=False),
    "no_graph_no_time": dict(use_graph=False, use_time=False),
}
ALL_VARIANTS = list(VARIANTS) + list(BASELINES)


def build_model(variant: str, n_concepts: int, n_features: int, max_len: int):
    """Return (model, config_dict) for either an ablation variant or a baseline."""
    if variant in VARIANTS:
        config = KGDKTConfig(
            n_concepts=n_concepts, n_student_features=n_features,
            max_len=max_len, **VARIANTS[variant],
        )
        return KGDKT(config), config.__dict__

    config = BaselineConfig(n_concepts=n_concepts, n_student_features=n_features)
    if variant == "sakt":
        return BASELINES[variant](config, max_len=max_len), config.__dict__
    return BASELINES[variant](config), config.__dict__


def device_of(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def apply_cold_concepts(batch: dict, cold: torch.Tensor, training: bool) -> dict:
    """Simulate concepts that carry no assessments at all.

    A concept is 'cold' if its assessments are withheld. To make it behave like the
    148 concepts that genuinely have none, two things must happen: the loss must skip
    it during training, AND its observed responses must be stripped from the input.
    Masking only the loss would leave the answer visible in the history, so the model
    could still learn the concept's difficulty without ever using the graph.

    At evaluation the mask is inverted: score ONLY the cold concepts, since they are
    the only place prerequisite propagation can contribute.
    """
    is_cold = cold[batch["concept_ids"]]
    supervised = batch["supervised"].bool()

    out = dict(batch)
    if training:
        out["supervised"] = (supervised & ~is_cold).float()
    else:
        out["supervised"] = (supervised & is_cold).float()

    # Cold interactions become unlabelled context in the input, in both phases.
    out["responses"] = torch.where(
        is_cold, torch.full_like(batch["responses"], RESPONSE_UNLABELLED), batch["responses"]
    )
    out["kinds"] = torch.where(is_cold, torch.zeros_like(batch["kinds"]), batch["kinds"])
    return out


def run_epoch(
    model, loader, adjacency, device, optimizer=None, scheduler=None, scaler=None, cold=None
):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    logits_all, labels_all, prior_all = [], [], []
    amp = scaler is not None and device.type == "cuda"

    for batch in loader:
        batch = move(batch, device)
        if cold is not None:
            batch = apply_cold_concepts(batch, cold, train)
            if batch["supervised"].sum() == 0:
                continue
        with torch.set_grad_enabled(train):
            # Mixed precision on the T4's tensor cores. Attention at L=200 is the bulk
            # of the compute and is the part that benefits; the loss stays in fp32.
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(batch, adjacency)
            logits = logits.float()
            supervised = batch["supervised"]
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits, batch["labels"].float(), reduction="none"
                )
                * supervised
            ).sum() / supervised.sum().clamp(min=1)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total += loss.item() * supervised.sum().item()
        n += supervised.sum().item()

        if not train:
            mask = supervised.bool()
            logits_all.append(logits[mask].detach().cpu().numpy())
            labels_all.append(batch["labels"][mask].detach().cpu().numpy())
            # How many interactions preceded each supervised position — the x axis of
            # the cold-start curve.
            positions = torch.arange(logits.size(1), device=device).expand_as(logits)
            prior_all.append(positions[mask].detach().cpu().numpy())

    if train:
        return total / max(n, 1), None
    return (
        total / max(n, 1),
        (np.concatenate(logits_all), np.concatenate(labels_all), np.concatenate(prior_all)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="proposed", choices=ALL_VARIANTS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="cap windows, for smoke tests")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument(
        "--cold-concepts", type=float, default=0.0,
        help="fraction of assessed concepts whose assessments are withheld from "
             "training and scored exclusively at evaluation",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = device_of(args.device)
    print(f"device: {device}   variant: {args.variant}")

    sequences, features = load_splits(PROCESSED)
    adjacency_file = torch.load(ARTIFACTS / "graph" / "adjacency.pt")
    adjacency = adjacency_file["prerequisite"].to(device)
    n_concepts = int(adjacency_file["n_concepts"])

    cold_mask = None
    if args.cold_concepts > 0:
        # Only concepts that actually carry assessments can be withheld; the other 148
        # have nothing to hold out.
        assessed = np.zeros(n_concepts, dtype=bool)
        for row in sequences.itertuples(index=False):
            ids = np.asarray(row.concept_ids)
            sup = np.asarray(row.is_supervised).astype(bool)
            assessed[ids[sup]] = True
        assessed_ids = np.flatnonzero(assessed)
        rng = np.random.default_rng(args.seed)
        n_cold = max(1, int(round(len(assessed_ids) * args.cold_concepts)))
        chosen = rng.choice(assessed_ids, size=n_cold, replace=False)
        cold = np.zeros(n_concepts, dtype=bool)
        cold[chosen] = True
        cold_mask = torch.from_numpy(cold).to(device)
        print(
            f"cold-concept ablation: withholding {n_cold} of {len(assessed_ids)} "
            f"assessed concepts ({args.cold_concepts:.0%}); evaluation scores only these"
        )

    fold_results = []
    for fold in range(args.folds):
        train_features = features[features.fold != fold]
        valid_features = features[features.fold == fold]

        train_ds = SequenceDataset(sequences, train_features, args.max_len)
        valid_ds = SequenceDataset(sequences, valid_features, args.max_len)
        if args.limit:
            train_ds.windows = train_ds.windows[: args.limit]
            train_ds.feature_index = train_ds.feature_index[: args.limit]
            valid_ds.windows = valid_ds.windows[: args.limit // 4]
            valid_ds.feature_index = valid_ds.feature_index[: args.limit // 4]

        pin = device.type == "cuda"
        train_dl = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
            num_workers=args.workers, drop_last=True, pin_memory=pin,
            persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None,
        )
        valid_dl = DataLoader(
            valid_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate,
            num_workers=args.workers, pin_memory=pin,
            persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None,
        )

        model, config_dict = build_model(
            args.variant, n_concepts, train_ds.student_matrix.shape[1], args.max_len
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(len(train_dl) * args.epochs, 1)
        )
        # GradScaler moved namespace in torch 2.4; support both so the same script runs
        # on this project's 2.2.2 (Intel-macOS ceiling) and on Colab's 2.11.
        if device.type == "cuda" and not args.no_amp:
            try:
                scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        best = {"auc": -1.0}
        stale = 0
        for epoch in range(args.epochs):
            t0 = time.perf_counter()
            train_loss, _ = run_epoch(
                model, train_dl, adjacency, device, optimizer, scheduler, scaler, cold_mask
            )
            valid_loss, (logits, labels, prior) = run_epoch(
                model, valid_dl, adjacency, device, cold=cold_mask
            )
            auc, rmse = auc_rmse(labels, 1 / (1 + np.exp(-logits)))
            print(
                f"  fold {fold} epoch {epoch:>2}  train {train_loss:.4f}  "
                f"valid {valid_loss:.4f}  AUC {auc:.4f}  RMSE {rmse:.4f}  "
                f"({time.perf_counter() - t0:.0f}s)"
            )
            if auc > best["auc"]:
                best = {
                    "auc": auc, "rmse": rmse, "epoch": epoch,
                    "logits": logits, "labels": labels, "prior": prior,
                }
                torch.save(
                    {"state_dict": model.state_dict(), "config": config_dict},
                    ARTIFACTS / "models" / f"kgdkt_{args.variant}_fold{fold}.pt",
                )
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"  fold {fold}: early stop at epoch {epoch}")
                    break

        # Calibration. The temperature must be fitted on data it is not then scored on,
        # or the reported ECE flatters itself: a temperature fitted on the same logits
        # is optimal for them by construction. The fold's held-out predictions are
        # split in half — fit on one, report on the other.
        rng = np.random.default_rng(args.seed + fold)
        order = rng.permutation(len(best["labels"]))
        fit_idx, score_idx = order[: len(order) // 2], order[len(order) // 2 :]

        temperature = fit_temperature(
            best["logits"][fit_idx], best["labels"][fit_idx].astype(float)
        )
        held_logits = best["logits"][score_idx]
        held_labels = best["labels"][score_idx]
        raw_prob = 1 / (1 + np.exp(-held_logits))
        cal_prob = 1 / (1 + np.exp(-held_logits / temperature))
        conf, acc, count = reliability_curve(held_labels, cal_prob)

        fold_results.append(
            {
                "fold": fold,
                "auc": best["auc"],
                "rmse": best["rmse"],
                "best_epoch": best["epoch"],
                "temperature": temperature,
                "ece_raw": expected_calibration_error(held_labels, raw_prob),
                "ece_calibrated": expected_calibration_error(held_labels, cal_prob),
                "cold_start_auc": cold_start_auc(
                    held_labels, cal_prob, best["prior"][score_idx]
                ),
                "reliability": {
                    "confidence": conf.tolist(), "accuracy": acc.tolist(),
                    "count": count.tolist(),
                },
                "n_supervised_eval": int(len(best["labels"])),
            }
        )

    summary = {
        "variant": args.variant,
        "seed": args.seed,
        "device": str(device),
        "args": vars(args),
        "folds": fold_results,
        "auc_mean": float(np.mean([f["auc"] for f in fold_results])),
        "auc_std": float(np.std([f["auc"] for f in fold_results])),
        "rmse_mean": float(np.mean([f["rmse"] for f in fold_results])),
        "rmse_std": float(np.std([f["rmse"] for f in fold_results])),
        "ece_raw_mean": float(np.mean([f["ece_raw"] for f in fold_results])),
        "ece_calibrated_mean": float(np.mean([f["ece_calibrated"] for f in fold_results])),
    }
    (ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
    suffix = f"_cold{args.cold_concepts:g}" if args.cold_concepts > 0 else ""
    out = RESULTS / f"kt_{args.variant}{suffix}.json"
    out.write_text(json.dumps(summary, indent=2))

    print(
        f"\n{args.variant}:  AUC {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}   "
        f"RMSE {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}   "
        f"ECE {summary['ece_raw_mean']:.4f} -> {summary['ece_calibrated_mean']:.4f}"
    )
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
