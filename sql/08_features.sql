-- Static per-enrolment features.
--
-- LEAKAGE CONTROL. The obvious behavioural features — total clicks, session count,
-- activity span — are computed over a student's whole enrolment. Feeding those to a
-- model that predicts outcomes within that same enrolment leaks the future: a student
-- who withdrew in week 5 has a short span *because* they failed.
--
-- Behavioural features are therefore restricted to an EARLY WINDOW of the first 28
-- days. They describe how a student started, which is legitimately available at
-- prediction time and is what a cold-start prior needs.
--
-- Demographic features carry no such problem: they are known at registration.

CREATE OR REPLACE TABLE early_engagement AS
WITH early AS (
    SELECT * FROM sessions WHERE day < 28
),
gaps AS (
    SELECT
        id_student, code_module, code_presentation, day,
        day - LAG(day) OVER (
            PARTITION BY id_student, code_module, code_presentation ORDER BY day
        ) AS gap
    FROM early
)
SELECT
    e.id_student,
    e.code_module,
    e.code_presentation,
    COUNT(*)                                        AS early_sessions,
    COALESCE(SUM(e.total_clicks), 0)                AS early_clicks,
    ROUND(AVG(e.total_clicks), 3)                   AS early_clicks_per_session,
    COALESCE(MAX(e.day) - MIN(e.day), 0)            AS early_span,
    ROUND(COALESCE(AVG(g.gap), 0), 3)               AS early_mean_gap,
    ROUND(COALESCE(STDDEV_POP(g.gap), 0), 3)        AS early_gap_std,
    COUNT(DISTINCT e.day)                           AS early_active_days
FROM early e
LEFT JOIN gaps g
       ON  g.id_student        = e.id_student
       AND g.code_module       = e.code_module
       AND g.code_presentation = e.code_presentation
       AND g.day               = e.day
GROUP BY 1, 2, 3;

-- Categorical vocabularies, emitted so that encoding is reproducible and inspectable
-- rather than hidden inside a fitted Python object.
CREATE OR REPLACE TABLE feature_vocab AS
SELECT 'gender'            AS feature, gender            AS value, DENSE_RANK() OVER (ORDER BY gender) - 1            AS code FROM (SELECT DISTINCT gender FROM student_info)
UNION ALL SELECT 'region',            region,            DENSE_RANK() OVER (ORDER BY region) - 1            FROM (SELECT DISTINCT region FROM student_info)
UNION ALL SELECT 'highest_education', highest_education, DENSE_RANK() OVER (ORDER BY highest_education) - 1 FROM (SELECT DISTINCT highest_education FROM student_info)
UNION ALL SELECT 'imd_band',          COALESCE(imd_band, 'unknown'), DENSE_RANK() OVER (ORDER BY COALESCE(imd_band, 'unknown')) - 1 FROM (SELECT DISTINCT imd_band FROM student_info)
UNION ALL SELECT 'age_band',          age_band,          DENSE_RANK() OVER (ORDER BY age_band) - 1          FROM (SELECT DISTINCT age_band FROM student_info)
UNION ALL SELECT 'disability',        disability,        DENSE_RANK() OVER (ORDER BY disability) - 1        FROM (SELECT DISTINCT disability FROM student_info)
UNION ALL SELECT 'code_module',       code_module,       DENSE_RANK() OVER (ORDER BY code_module) - 1       FROM (SELECT DISTINCT code_module FROM student_info);

CREATE OR REPLACE TABLE student_features AS
SELECT
    si.id_student,
    si.code_module,
    si.code_presentation,
    -- categorical codes
    vg.code   AS gender_code,
    vr.code   AS region_code,
    ve.code   AS education_code,
    vi.code   AS imd_code,
    va.code   AS age_code,
    vd.code   AS disability_code,
    vm.code   AS module_code,
    -- numeric, known at registration
    si.num_of_prev_attempts,
    si.studied_credits,
    COALESCE(sr.date_registration, 0)   AS date_registration,
    (sr.date_unregistration IS NOT NULL)::TINYINT AS unregistered,
    -- behavioural, first 28 days only
    COALESCE(ee.early_sessions, 0)           AS early_sessions,
    COALESCE(ee.early_clicks, 0)             AS early_clicks,
    COALESCE(ee.early_clicks_per_session, 0) AS early_clicks_per_session,
    COALESCE(ee.early_span, 0)               AS early_span,
    COALESCE(ee.early_mean_gap, 0)           AS early_mean_gap,
    COALESCE(ee.early_gap_std, 0)            AS early_gap_std,
    COALESCE(ee.early_active_days, 0)        AS early_active_days,
    -- target, used for stratification and for the fairness analysis, never as input
    si.final_result
FROM student_info si
LEFT JOIN student_registration sr USING (id_student, code_module, code_presentation)
LEFT JOIN early_engagement ee     USING (id_student, code_module, code_presentation)
LEFT JOIN feature_vocab vg ON vg.feature='gender'            AND vg.value = si.gender
LEFT JOIN feature_vocab vr ON vr.feature='region'            AND vr.value = si.region
LEFT JOIN feature_vocab ve ON ve.feature='highest_education' AND ve.value = si.highest_education
LEFT JOIN feature_vocab vi ON vi.feature='imd_band'          AND vi.value = COALESCE(si.imd_band, 'unknown')
LEFT JOIN feature_vocab va ON va.feature='age_band'          AND va.value = si.age_band
LEFT JOIN feature_vocab vd ON vd.feature='disability'        AND vd.value = si.disability
LEFT JOIN feature_vocab vm ON vm.feature='code_module'       AND vm.value = si.code_module;
