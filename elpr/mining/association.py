"""Association rule mining over per-student concept sets, for corequisite edges.

A corequisite is a pair of concepts that travel together without a consistent
order: students who reach one reach the other, but neither reliably comes first.
That is precisely a high-confidence association rule in both directions combined
with near-symmetric temporal precedence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth


def mine_corequisites(
    first_touch: pd.DataFrame,
    pair_stats: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.60,
    precedence_tolerance: float = 0.10,
) -> pd.DataFrame:
    """Return corequisite edges as (src, dst, edge_type, weight).

    Args:
        first_touch: one row per (id_student, concept_id).
        pair_stats: precedence per ordered pair, from sql/07.
        min_support: minimum fraction of students holding the itemset.
        min_confidence: rules must clear this in BOTH directions.
        precedence_tolerance: |precedence - 0.5| must stay within this, i.e. neither
            concept reliably precedes the other.
    """
    # Build the boolean basket directly. pivot_table over 635k rows is needlessly slow
    # here, and the result is a plain student x concept indicator matrix.
    students = np.sort(first_touch.id_student.unique())
    concepts = np.sort(first_touch.concept_id.unique())
    row = np.searchsorted(students, first_touch.id_student.to_numpy())
    col = np.searchsorted(concepts, first_touch.concept_id.to_numpy())
    matrix = np.zeros((len(students), len(concepts)), dtype=bool)
    matrix[row, col] = True
    basket = pd.DataFrame(matrix, columns=concepts)

    # max_len=2 is essential, not an optimisation. Concepts within a module co-occur
    # for nearly every student, so the frequent-itemset lattice explodes
    # combinatorially without a cap. Corequisites are pairwise by definition, so
    # itemsets larger than two are computed at great cost and then discarded.
    frequent = fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=2)
    if frequent.empty:
        return pd.DataFrame(columns=["src", "dst", "edge_type", "weight"])

    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    pairs = rules[
        (rules.antecedents.map(len) == 1) & (rules.consequents.map(len) == 1)
    ].copy()
    if pairs.empty:
        return pd.DataFrame(columns=["src", "dst", "edge_type", "weight"])

    pairs["a"] = pairs.antecedents.map(lambda s: next(iter(s)))
    pairs["b"] = pairs.consequents.map(lambda s: next(iter(s)))

    # Require confidence in both directions: join each rule to its mirror.
    mirror = pairs.set_index(["a", "b"]).confidence
    pairs["confidence_rev"] = [
        mirror.get((b, a), 0.0) for a, b in zip(pairs.a, pairs.b)
    ]
    pairs = pairs[pairs.confidence_rev >= min_confidence]

    # Require the absence of a consistent order.
    prec = pair_stats.set_index(["src", "dst"]).precedence
    pairs["precedence"] = [
        prec.get((min(a, b), max(a, b)), 0.5) for a, b in zip(pairs.a, pairs.b)
    ]
    pairs = pairs[(pairs.precedence - 0.5).abs() <= precedence_tolerance]

    out = (
        pairs.assign(
            src=pairs[["a", "b"]].min(axis=1),
            dst=pairs[["a", "b"]].max(axis=1),
            edge_type="corequisite",
            weight=pairs[["confidence", "confidence_rev"]].min(axis=1),
        )[["src", "dst", "edge_type", "weight"]]
        .drop_duplicates(subset=["src", "dst"])
        .reset_index(drop=True)
    )
    return out
