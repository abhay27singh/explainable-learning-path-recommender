-- Structural prerequisite edges, derived from course structure alone.
--
-- Two rules:
--   prereq_sequence    week w precedes week w+1 within a module. Curriculum order.
--   prereq_assessment  the concepts of the preceding weeks precede the concept an
--                      assessment falls in. An assessment tests what came before it.
--
-- Note on corequisites: the original design proposed "same week, different activity
-- type" as a corequisite rule. That rule is void here, because a concept IS a
-- (module, week) pair — materials of different activity types in the same week
-- collapse into one concept rather than into two related concepts. Corequisites are
-- therefore mined only (sql/07), via association rules.

CREATE OR REPLACE TABLE edges_structural AS
WITH ordered AS (
    SELECT
        concept_id,
        code_module,
        week,
        LEAD(concept_id) OVER (PARTITION BY code_module ORDER BY week) AS next_concept_id
    FROM concepts
),
sequence_edges AS (
    SELECT
        concept_id          AS src,
        next_concept_id     AS dst,
        'prereq_sequence'   AS edge_type,
        1.0                 AS weight
    FROM ordered
    WHERE next_concept_id IS NOT NULL
),
assessed AS (
    SELECT DISTINCT c.concept_id, c.code_module, c.week
    FROM concepts c
    JOIN assessment_to_concept a USING (concept_id)
),
assessment_edges AS (
    SELECT
        prior.concept_id        AS src,
        assessed.concept_id     AS dst,
        'prereq_assessment'     AS edge_type,
        1.0                     AS weight
    FROM assessed
    JOIN concepts prior
      ON  prior.code_module = assessed.code_module
      AND prior.week        < assessed.week
      AND prior.week        >= assessed.week - 4   -- a four-week lookback window
)
SELECT src, dst, ANY_VALUE(edge_type) AS edge_type, MAX(weight) AS weight
FROM (SELECT * FROM sequence_edges UNION ALL SELECT * FROM assessment_edges)
WHERE src <> dst
GROUP BY src, dst;
