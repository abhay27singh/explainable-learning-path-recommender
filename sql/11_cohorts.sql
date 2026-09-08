-- Fairness cohorts.
--
-- The model consumes protected attributes (gender, disability, deprivation band, age
-- band). A recommender that acts on those without a subgroup analysis is the single
-- most likely reviewer objection for a 2026 paper whose selling point is trust.
--
-- These cohorts are the grouping used to report per-subgroup accuracy and
-- recommendation quality. They are defined here rather than inside evaluation code so
-- that the definition is visible and fixed before any result is computed.

CREATE OR REPLACE TABLE cohorts AS
SELECT id_student, code_module, code_presentation, 'gender'     AS attribute, gender                        AS cohort FROM student_info
UNION ALL SELECT id_student, code_module, code_presentation, 'disability', disability                       FROM student_info
UNION ALL SELECT id_student, code_module, code_presentation, 'imd_band',   COALESCE(imd_band, 'unknown')    FROM student_info
UNION ALL SELECT id_student, code_module, code_presentation, 'age_band',   age_band                         FROM student_info
UNION ALL SELECT id_student, code_module, code_presentation, 'education',  highest_education                FROM student_info
UNION ALL SELECT id_student, code_module, code_presentation, 'module',     code_module                      FROM student_info;

CREATE OR REPLACE TABLE cohort_sizes AS
SELECT
    c.attribute,
    c.cohort,
    COUNT(*)                                                        AS n_enrolments,
    ROUND(AVG((si.final_result IN ('Pass', 'Distinction'))::INT), 4) AS base_success_rate
FROM cohorts c
JOIN student_info si USING (id_student, code_module, code_presentation)
GROUP BY 1, 2
ORDER BY 1, n_enrolments DESC;
