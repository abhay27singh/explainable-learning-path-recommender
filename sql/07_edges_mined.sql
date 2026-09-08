-- Sequential prerequisite mining.
--
-- Course structure says week w comes before week w+1. That is weak evidence of a
-- prerequisite: it records how the module was scheduled, not how learning depends.
-- Observed learner behaviour is stronger evidence, on two counts:
--
--   precedence(a,b)  the fraction of students who reached a before b. Directional
--                    asymmetry in WHEN concepts are engaged.
--   lift(a,b)        pass rate on b among students who saw a first, over the base
--                    pass rate on b. Whether reaching a first actually HELPS.
--
-- A prerequisite is exactly a pair with both properties: consistent ordering and a
-- measurable effect on downstream success.
--
-- Only 89 of the 237 concepts carry assessment events, so lift is computable for a
-- minority of targets. Pairs where lift cannot be computed must clear a stricter
-- precedence bar instead of being admitted for free.

CREATE OR REPLACE TABLE first_touch AS
SELECT
    id_student,
    concept_id,
    MIN(day)    AS first_day,
    SUM(clicks) AS total_clicks
FROM events
GROUP BY 1, 2;

-- Per-student outcome on each assessed concept: did they ultimately succeed there.
CREATE OR REPLACE TABLE concept_outcome AS
SELECT
    id_student,
    concept_id,
    MAX(label) AS passed
FROM events
WHERE kind = 'assessment'
GROUP BY 1, 2;

CREATE OR REPLACE TABLE concept_base_rate AS
SELECT
    concept_id,
    AVG(passed)  AS base_pass_rate,
    COUNT(*)     AS n_students
FROM concept_outcome
GROUP BY 1;

-- Ordered co-occurrence over every concept pair a student reached.
--
-- Computed in two passes on purpose. Folding the outcome join into this aggregation
-- makes the planner join first_touch against concept_outcome before restricting to
-- co-occurring pairs, which does not finish in reasonable time. Co-occurrence alone
-- runs in ~0.25s and yields ~17k pairs; restricting to those before touching
-- outcomes keeps the second pass small.
CREATE OR REPLACE TABLE pair_cooccurrence AS
SELECT
    a.concept_id                            AS src,
    b.concept_id                            AS dst,
    COUNT(*)                                AS support,
    AVG((a.first_day < b.first_day)::INT)   AS precedence
FROM first_touch a
JOIN first_touch b
  ON  a.id_student = b.id_student
  AND a.concept_id < b.concept_id           -- each unordered pair once
GROUP BY 1, 2;

-- Pass 2: outcome statistics, only for pairs that already clear the support floor.
CREATE OR REPLACE TABLE pair_outcome AS
SELECT
    p.src,
    p.dst,
    AVG(o.passed)   AS pass_rate_after_src,
    COUNT(o.passed) AS n_outcomes
FROM pair_cooccurrence p
JOIN first_touch a ON a.concept_id = p.src
JOIN first_touch b ON b.concept_id = p.dst AND b.id_student = a.id_student
JOIN concept_outcome o ON o.id_student = a.id_student AND o.concept_id = p.dst
WHERE p.support >= 100
  AND a.first_day < b.first_day
GROUP BY 1, 2;

CREATE OR REPLACE TABLE pair_stats AS
SELECT
    p.src,
    p.dst,
    p.support,
    p.precedence,
    o.pass_rate_after_src,
    COALESCE(o.n_outcomes, 0) AS n_outcomes
FROM pair_cooccurrence p
LEFT JOIN pair_outcome o USING (src, dst);

-- Evaluate both orientations of every pair, then keep those clearing the thresholds.
CREATE OR REPLACE TABLE edges_mined AS
WITH both_directions AS (
    SELECT src, dst, support, precedence,
           pass_rate_after_src, n_outcomes
    FROM pair_stats
    UNION ALL
    SELECT dst AS src, src AS dst, support, 1.0 - precedence AS precedence,
           NULL AS pass_rate_after_src, 0 AS n_outcomes
    FROM pair_stats
),
scored AS (
    SELECT
        d.src,
        d.dst,
        d.support,
        d.precedence,
        cs.code_module AS src_module,
        cd.code_module AS dst_module,
        CASE WHEN d.n_outcomes >= 100 AND br.base_pass_rate > 0
             THEN d.pass_rate_after_src / br.base_pass_rate END AS lift
    FROM both_directions d
    JOIN concepts cs ON cs.concept_id = d.src
    JOIN concepts cd ON cd.concept_id = d.dst
    LEFT JOIN concept_base_rate br ON br.concept_id = d.dst
)
-- ACCEPTANCE RULE. Two constraints, both learned the hard way from the data.
--
-- 1. Lift is REQUIRED, not optional. An earlier version admitted pairs on precedence
--    alone when no outcome data existed. That branch accepted edges like
--    CCC-W02 -> EEE-W32 at precedence 1.000: week 2 of any module trivially precedes
--    week 32 of any other. Precedence alone measures the calendar. Only lift shows
--    that reaching the source first actually improves success at the target.
--
-- 2. WITHIN-MODULE only. 51.4% of edges under the previous rule crossed modules,
--    resting on the 2,479 students (of 28,785) who enrol in more than one. Both
--    orientations appeared for the same module pair, which is co-enrolment ordering,
--    not dependency. Genuine cross-module prerequisites belong in the curated overlay
--    at graph/cross_module_edges.yaml, where they can be justified individually.
SELECT
    src,
    dst,
    'prereq_mined'  AS edge_type,
    support,
    ROUND(precedence, 4) AS precedence,
    ROUND(lift, 4)       AS lift,
    -- Confidence doubles as the cycle-breaking priority: the weakest edge in a cycle
    -- is the one removed.
    ROUND(precedence * lift, 4) AS weight
FROM scored
WHERE support >= 100
  AND src_module = dst_module
  AND lift IS NOT NULL
  AND precedence >= 0.75
  AND lift > 1.1;

CREATE OR REPLACE TABLE mining_stats AS
SELECT
    (SELECT COUNT(*) FROM pair_stats)                                       AS pairs_considered,
    (SELECT COUNT(*) FROM pair_stats WHERE support >= 100)                  AS pairs_above_support,
    (SELECT COUNT(*) FROM edges_mined)                                      AS edges_accepted,
    (SELECT COUNT(*) FROM edges_mined WHERE lift IS NOT NULL)               AS edges_with_lift,
    (SELECT COUNT(*) FROM edges_structural)                                 AS edges_structural;
