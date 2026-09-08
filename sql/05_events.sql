-- Unified interaction stream: the dual-signal sequence design.
--
--   kind='vle'        context tokens. Plentiful (~10.6M rows) but carry no
--                     correctness signal. They update the model's hidden state and
--                     contribute NOTHING to the loss.
--   kind='assessment' supervised tokens. Sparse (~174k rows, ~5 per student) but
--                     genuinely labelled. These alone produce loss, AUC and RMSE.
--
-- This is what makes a 200-length sequence model trainable on OULAD: assessment
-- events alone would give sequences of length ~5.

-- Missing scores are median-imputed per assessment, as the paper states.
CREATE OR REPLACE TABLE assessment_score_median AS
SELECT id_assessment, MEDIAN(score) AS median_score
FROM student_assessment
WHERE score IS NOT NULL
GROUP BY 1;

CREATE OR REPLACE TABLE events AS
WITH vle_events AS (
    SELECT
        sv.id_student,
        sv.code_module,
        sv.code_presentation,
        stc.concept_id,
        sv.date                             AS day,
        'vle'                               AS kind,
        SUM(sv.sum_click)                   AS clicks,
        CAST(NULL AS DOUBLE)                AS score,
        CAST(NULL AS TINYINT)               AS label,
        FALSE                               AS score_imputed
    FROM student_vle sv
    JOIN site_to_concept stc USING (id_site)
    GROUP BY 1, 2, 3, 4, 5, 6
),
-- Every assessment a student was expected to attempt, whether or not they did.
--
-- OULAD records only SUBMITTED assessments, so the observed pass rate at the
-- threshold of 40 is 0.956 — the 5th percentile of submitted scores is itself 40.
-- Training on submitted rows alone gives a 96/4 class split and a meaningless AUC:
-- the failures are missing from the table, not absent from reality.
--
-- 46.3% of expected submissions (150,013 of 323,925) never happened. Those are the
-- negatives. Treating non-submission as failure changes the target from "given they
-- submitted, did they pass" to "will this student successfully complete this
-- assessment", which is the question the system actually poses, and which restores
-- an approximately balanced label.
--
-- Assessments falling after a student's unregistration date are excluded, so that
-- withdrawing does not manufacture unbounded negatives.
expected_assessments AS (
    SELECT
        si.id_student,
        atc.id_assessment,
        atc.code_module,
        atc.code_presentation,
        atc.concept_id,
        COALESCE(a.date, co.presentation_length) AS due_day
    FROM student_info si
    JOIN assessment_to_concept atc
      ON  atc.code_module       = si.code_module
      AND atc.code_presentation = si.code_presentation
    JOIN assessments a  USING (id_assessment)
    JOIN courses co
      ON  co.code_module       = si.code_module
      AND co.code_presentation = si.code_presentation
    LEFT JOIN student_registration sr
      ON  sr.id_student        = si.id_student
      AND sr.code_module       = si.code_module
      AND sr.code_presentation = si.code_presentation
    WHERE sr.date_unregistration IS NULL
       OR COALESCE(a.date, co.presentation_length) <= sr.date_unregistration
),
assessment_events AS (
    SELECT
        e.id_student,
        e.code_module,
        e.code_presentation,
        e.concept_id,
        COALESCE(sa.date_submitted, e.due_day)  AS day,
        'assessment'                            AS kind,
        0                                       AS clicks,
        -- A student who never submitted has no score. Median imputation applies only
        -- to submitted-but-unscored rows; filling it for non-submitters would both
        -- fabricate data and corrupt the submitted/not-submitted breakdown.
        CASE WHEN sa.id_student IS NOT NULL
             THEN COALESCE(sa.score, m.median_score) END        AS score,
        (sa.id_student IS NOT NULL
         AND COALESCE(sa.score, m.median_score) >= 40)::TINYINT AS label,
        (sa.id_student IS NOT NULL AND sa.score IS NULL)        AS score_imputed
    FROM expected_assessments e
    LEFT JOIN student_assessment sa
      ON  sa.id_assessment = e.id_assessment
      AND sa.id_student    = e.id_student
    LEFT JOIN assessment_score_median m ON m.id_assessment = e.id_assessment
)
SELECT * FROM vle_events
UNION ALL
SELECT * FROM assessment_events;

-- Ordered per-student stream. Assessments are ordered after VLE activity on the
-- same day, so a prediction for an assessment may condition on that day's study.
CREATE OR REPLACE TABLE events_ordered AS
SELECT
    *,
    ROW_NUMBER() OVER (
        PARTITION BY id_student, code_module, code_presentation
        ORDER BY day, CASE kind WHEN 'vle' THEN 0 ELSE 1 END, concept_id
    ) - 1 AS position
FROM events;

CREATE OR REPLACE TABLE event_stats AS
SELECT
    kind,
    COUNT(*)                                    AS n_events,
    COUNT(DISTINCT id_student)                  AS n_students,
    COUNT(DISTINCT concept_id)                  AS n_concepts,
    ROUND(AVG(label), 4)                        AS mean_label,
    SUM(score_imputed::INT)                     AS n_imputed
FROM events
GROUP BY 1;

-- Label provenance: how the supervised class balance is composed.
CREATE OR REPLACE TABLE label_stats AS
SELECT
    CASE WHEN score IS NULL THEN 'not_submitted' ELSE 'submitted' END AS submission,
    label,
    COUNT(*)                                                          AS n,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct
FROM events
WHERE kind = 'assessment'
GROUP BY 1, 2
ORDER BY 1, 2;
