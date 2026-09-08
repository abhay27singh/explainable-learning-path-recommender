-- Map every placeable VLE site to exactly one concept, and every assessment to the
-- concept whose week it falls in.
--
-- Enforced by tests: the site mapping is total over placeable sites and unique.

CREATE OR REPLACE TABLE site_to_concept AS
SELECT
    sw.id_site,
    c.concept_id,
    sw.code_module,
    sw.activity_type,
    sw.week_source
FROM site_week sw
JOIN concepts c
  ON  c.code_module = sw.code_module
  AND c.week        = sw.week;

-- Assessments carry a due-date day. Final exams generally have a NULL date, so the
-- presentation length is used instead — an exam sits at the end of its presentation.
-- The resulting week is matched to the nearest concept week within the same module,
-- because a module need not have VLE material in the exact week an assessment falls.
CREATE OR REPLACE TABLE assessment_week AS
SELECT
    a.id_assessment,
    a.code_module,
    a.code_presentation,
    a.assessment_type,
    a.weight,
    GREATEST(0, CAST(FLOOR(COALESCE(a.date, co.presentation_length) / 7.0) AS INT)) AS week
FROM assessments a
JOIN courses co
  ON  co.code_module       = a.code_module
  AND co.code_presentation = a.code_presentation;

CREATE OR REPLACE TABLE assessment_to_concept AS
SELECT
    aw.id_assessment,
    aw.code_module,
    aw.code_presentation,
    aw.assessment_type,
    aw.weight,
    aw.week,
    c.concept_id,
    ABS(c.week - aw.week) AS week_distance
FROM assessment_week aw
JOIN concepts c ON c.code_module = aw.code_module
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY aw.id_assessment
    ORDER BY ABS(c.week - aw.week), c.week
) = 1;
