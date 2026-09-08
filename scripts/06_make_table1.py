#!/usr/bin/env python
"""Build Table I and the Stage 1 figures from results/kt_*.json.

Emits a LaTeX table with mean +/- std, Holm-corrected p-values and Cohen's d, plus
calibration, cold-start and ablation figures. Every number traces to a results file;
nothing here is typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elpr.eval.stats import compare_against_reference, interpret_d  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

DISPLAY = {
    "kt_majority": "Majority (no-skill)",
    "kt_dkt": "DKT (Piech et al.)",
    "kt_sakt": "SAKT",
    "kt_gnn": "GNN-based",
    "kt_no_graph": "Proposed, no knowledge graph",
    "kt_no_time": "Proposed, no temporal attention",
    "kt_proposed": "Proposed (KG-DKT)",
}
COLD = {
    "kt_proposed_cold0.3": "Proposed (KG-DKT)",
    "kt_no_graph_cold0.3": "No knowledge graph",
}


def load(stem: str) -> dict | None:
    path = RESULTS / f"{stem}.json"
    return json.loads(path.read_text()) if path.exists() else None


def fold_auc(data: dict) -> np.ndarray:
    return np.array([f["auc"] for f in data["folds"]])


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    loaded = {stem: load(stem) for stem in DISPLAY}
    available = {k: v for k, v in loaded.items() if v is not None}
    if "kt_proposed" not in available:
        print("results/kt_proposed.json missing — run Stage 1 first")
        return 1
    for stem in DISPLAY:
        if stem not in available:
            print(f"  note: {stem}.json not found, omitted from Table I")

    # ---- Table I ----------------------------------------------------------
    scores = {stem: fold_auc(data) for stem, data in available.items()}
    comparisons = {
        c.name: c for c in compare_against_reference(scores, "kt_proposed")
    }

    rows = []
    for stem, data in sorted(available.items(), key=lambda kv: kv[1]["auc_mean"]):
        comparison = comparisons.get(stem)
        rows.append(
            {
                "model": DISPLAY[stem],
                "auc": f"{data['auc_mean']:.4f} ± {data['auc_std']:.4f}",
                "rmse": f"{data['rmse_mean']:.4f} ± {data['rmse_std']:.4f}",
                "ece": f"{data['ece_calibrated_mean']:.4f}",
                "p": "—" if comparison is None else f"{comparison.p_holm:.4f}",
                "d": "—" if comparison is None else
                     f"{comparison.cohens_d:+.2f} ({interpret_d(comparison.cohens_d)})",
            }
        )

    width = max(len(r["model"]) for r in rows) + 2
    print("\n" + "=" * (width + 62))
    print(f"{'Model':<{width}}{'AUC-ROC':>18}{'RMSE':>18}{'ECE':>8}{'p (Holm)':>10}{'d':>22}")
    print("=" * (width + 62))
    for r in rows:
        print(f"{r['model']:<{width}}{r['auc']:>18}{r['rmse']:>18}{r['ece']:>8}{r['p']:>10}{r['d']:>22}")
    print("=" * (width + 62))
    print("p-values: paired t-test over 5 folds against the proposed model, "
          "Holm-corrected across the family.")

    latex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Comparative predictive accuracy. Mean $\pm$ standard deviation "
        r"over 5-fold cross-validation, folds grouped by student. $p$-values are "
        r"paired $t$-tests against the proposed model, Holm-corrected.}",
        r"\label{tab:accuracy}",
        r"\begin{tabular}{lccccc}", r"\hline",
        r"Model & AUC-ROC & RMSE & ECE & $p$ & Cohen's $d$ \\", r"\hline",
    ]
    pm = r"$\pm$"
    for r in rows:
        model = r["model"].replace("&", r"\&")
        auc = r["auc"].replace("±", pm)
        rmse = r["rmse"].replace("±", pm)
        latex.append(f"{model} & {auc} & {rmse} & {r['ece']} & {r['p']} & {r['d']} \\\\")
    latex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (TABLES / "table1.tex").write_text("\n".join(latex))

    summary = {
        "rows": rows,
        "comparisons": {
            name: {
                "difference": float(c.difference), "p_raw": float(c.p_raw),
                "p_holm": float(c.p_holm), "cohens_d": float(c.cohens_d),
                "significant": bool(c.significant),
            }
            for name, c in comparisons.items()
        },
    }
    (RESULTS / "table1.json").write_text(json.dumps(summary, indent=2))

    # ---- cold-concept comparison -----------------------------------------
    cold = {stem: load(stem) for stem in COLD}
    if all(v is not None for v in cold.values()):
        print("\nCold-concept ablation — AUC on concepts whose assessments were "
              "withheld entirely:")
        cold_scores = {stem: fold_auc(data) for stem, data in cold.items()}
        for stem, data in cold.items():
            print(f"  {COLD[stem]:<26}{data['auc_mean']:.4f} ± {data['auc_std']:.4f}")
        c = compare_against_reference(cold_scores, "kt_proposed_cold0.3")[0]
        print(f"  difference {c.difference:+.4f}   p = {c.p_raw:.4f}   "
              f"d = {c.cohens_d:+.2f} ({interpret_d(c.cohens_d)})")
        print("  " + (
            "Knowledge graph contributes where it can matter."
            if c.difference > 0 and c.p_raw < 0.05 else
            "No detectable contribution from the knowledge graph, even here."
        ))
        summary["cold_concept"] = {
            "proposed": cold["kt_proposed_cold0.3"]["auc_mean"],
            "no_graph": cold["kt_no_graph_cold0.3"]["auc_mean"],
            "difference": c.difference, "p_raw": c.p_raw, "cohens_d": c.cohens_d,
        }
        (RESULTS / "table1.json").write_text(json.dumps(summary, indent=2))

    # ---- figures ----------------------------------------------------------
    proposed = available["kt_proposed"]

    fig, ax = plt.subplots(figsize=(5.2, 5), dpi=160)
    ax.plot([0, 1], [0, 1], color="#8A94A0", lw=1, ls="--", label="perfect calibration")
    for fold in proposed["folds"]:
        rel = fold["reliability"]
        ax.plot(rel["confidence"], rel["accuracy"], color="#1F5E7A", alpha=0.5, lw=1.2,
                marker="o", ms=3)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(
        f"Calibration, 5 folds\nECE {proposed['ece_raw_mean']:.4f} → "
        f"{proposed['ece_calibrated_mean']:.4f} after temperature scaling", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png", bbox_inches="tight")

    buckets = list(proposed["folds"][0]["cold_start_auc"])
    values = np.array([[f["cold_start_auc"][b] for b in buckets] for f in proposed["folds"]])
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    ax.errorbar(range(len(buckets)), mean, yerr=std, color="#1F5E7A", marker="o",
                capsize=3, lw=1.5)
    ax.axhline(0.5, color="#9E3A2C", lw=1, ls="--", label="no-skill")
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets)
    ax.set_xlabel("Prior interactions in the sequence")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Cold-start performance", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "cold_start.png", bbox_inches="tight")

    print(f"\nWrote {(TABLES / 'table1.tex').relative_to(ROOT)}, "
          f"{(RESULTS / 'table1.json').relative_to(ROOT)}, and 2 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
