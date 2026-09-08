-- Descriptive material for concept labels.
--
-- OULAD ships no titles for learning materials, so a concept cannot be given a real
-- topic name without inventing one. What CAN be derived honestly is its composition:
-- which kinds of activity a concept is made of, whether an assessment falls in its
-- week, and how heavily it is used.
--
-- That produces labels like "DDD · Week 12 — assessment week, quiz-heavy" instead of
-- the opaque "DDD-W12", without asserting subject matter the data does not contain.

CREATE OR REPLACE TABLE concept_activity_mix AS
WITH grouped AS (
    SELECT
        stc.concept_id,
        CASE
            WHEN stc.activity_type IN ('quiz', 'externalquiz')                       THEN 'quiz'
            WHEN stc.activity_type IN ('forumng', 'oucollaborate', 'ouwiki',
                                       'sharedsubpage', 'ouelluminate')              THEN 'discussion'
            WHEN stc.activity_type IN ('glossary', 'url', 'dataplus', 'questionnaire') THEN 'reference'
            ELSE 'content'
        END                                     AS activity_group,
        COUNT(*)                                AS n_sites
    FROM site_to_concept stc
    GROUP BY 1, 2
)
SELECT
    concept_id,
    activity_group,
    n_sites,
    ROUND(n_sites / SUM(n_sites) OVER (PARTITION BY concept_id), 4) AS share
FROM grouped;

CREATE OR REPLACE TABLE concept_descriptors AS
SELECT
    c.concept_id,
    c.code_module,
    c.week,
    c.concept_key,
    c.n_sites,
    c.n_students,
    -- The activity group contributing the most materials.
    (SELECT m.activity_group FROM concept_activity_mix m
      WHERE m.concept_id = c.concept_id
      ORDER BY m.n_sites DESC, m.activity_group LIMIT 1)            AS dominant_activity,
    (SELECT ROUND(m.share, 2) FROM concept_activity_mix m
      WHERE m.concept_id = c.concept_id
      ORDER BY m.n_sites DESC, m.activity_group LIMIT 1)            AS dominant_share,
    EXISTS (SELECT 1 FROM assessment_to_concept a
             WHERE a.concept_id = c.concept_id)                     AS has_assessment,
    (SELECT COUNT(*) FROM assessment_to_concept a
      WHERE a.concept_id = c.concept_id)                            AS n_assessments,
    -- Position through the module, so a label can say "early" or "final".
    ROUND(c.week / NULLIF(MAX(c.week) OVER (PARTITION BY c.code_module), 0), 3) AS progress
FROM concepts c;
