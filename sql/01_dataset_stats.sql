-- Measured dataset statistics.
--
-- Every dataset figure quoted in the paper must come from this table rather than
-- from documentation or memory. Emitted to results/dataset_stats.json.

CREATE OR REPLACE TABLE dataset_stats AS
WITH
vle_per_student AS (
    SELECT id_student, COUNT(*) AS n FROM student_vle GROUP BY 1
),
asm_per_student AS (
    SELECT id_student, COUNT(*) AS n FROM student_assessment GROUP BY 1
),
metrics(metric, value) AS (
    VALUES
        ('n_students',              (SELECT COUNT(DISTINCT id_student) FROM student_info)),
        ('n_enrolments',            (SELECT COUNT(*) FROM student_info)),
        ('n_modules',               (SELECT COUNT(DISTINCT code_module) FROM courses)),
        ('n_presentations',         (SELECT COUNT(*) FROM courses)),
        ('n_vle_sites',             (SELECT COUNT(*) FROM vle)),
        ('n_vle_sites_with_week',   (SELECT COUNT(week_from) FROM vle)),
        ('n_activity_types',        (SELECT COUNT(DISTINCT activity_type) FROM vle)),
        ('n_vle_events',            (SELECT COUNT(*) FROM student_vle)),
        ('n_assessments',           (SELECT COUNT(*) FROM assessments)),
        ('n_assessment_events',     (SELECT COUNT(*) FROM student_assessment)),
        ('vle_events_per_student_mean',   (SELECT ROUND(AVG(n), 2) FROM vle_per_student)),
        ('vle_events_per_student_median', (SELECT MEDIAN(n) FROM vle_per_student)),
        ('asm_events_per_student_mean',   (SELECT ROUND(AVG(n), 2) FROM asm_per_student)),
        ('asm_events_per_student_median', (SELECT MEDIAN(n) FROM asm_per_student)),
        ('n_scored_assessments',    (SELECT COUNT(score) FROM student_assessment)),
        ('n_missing_scores',        (SELECT COUNT(*) - COUNT(score) FROM student_assessment)),
        ('pass_rate_at_40',         (SELECT ROUND(AVG((score >= 40)::INT), 4) FROM student_assessment WHERE score IS NOT NULL))
)
SELECT metric, value::DOUBLE AS value FROM metrics;

-- Outcome distribution, reported separately because it is categorical.
CREATE OR REPLACE TABLE outcome_distribution AS
SELECT
    final_result,
    COUNT(*)                                                AS n,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)      AS pct
FROM student_info
GROUP BY 1
ORDER BY n DESC;
