-- Sessionize VLE activity.
--
-- LIMITATION, to be stated in the paper: OULAD timestamps are day-granular. The
-- `date` column is an integer day offset relative to the start of the module
-- presentation, with no intra-day resolution. A 30-minute inactivity threshold is
-- therefore not computable from this data, and one student-day resolves to one
-- session. The paper's current draft implies finer resolution than OULAD provides.

CREATE OR REPLACE TABLE sessions AS
SELECT
    id_student,
    code_module,
    code_presentation,
    date                                        AS day,
    COUNT(DISTINCT id_site)                     AS n_sites,
    COUNT(*)                                    AS n_interactions,
    SUM(sum_click)                              AS total_clicks
FROM student_vle
GROUP BY 1, 2, 3, 4;

-- Per-student engagement summary, used for learner-profile clustering (sql/08)
-- and as the behavioural half of the cold-start prior.
CREATE OR REPLACE TABLE student_engagement AS
WITH gaps AS (
    SELECT
        id_student, code_module, code_presentation, day,
        day - LAG(day) OVER (PARTITION BY id_student, code_module, code_presentation ORDER BY day) AS gap
    FROM sessions
)
SELECT
    s.id_student,
    s.code_module,
    s.code_presentation,
    COUNT(*)                                    AS n_sessions,
    SUM(s.total_clicks)                         AS total_clicks,
    ROUND(AVG(s.total_clicks), 2)               AS clicks_per_session,
    MIN(s.day)                                  AS first_day,
    MAX(s.day)                                  AS last_day,
    MAX(s.day) - MIN(s.day)                     AS active_span,
    ROUND(AVG(g.gap), 2)                        AS mean_gap_days,
    ROUND(STDDEV_POP(g.gap), 2)                 AS std_gap_days
FROM sessions s
LEFT JOIN gaps g
       ON  g.id_student        = s.id_student
       AND g.code_module       = s.code_module
       AND g.code_presentation = s.code_presentation
       AND g.day               = s.day
GROUP BY 1, 2, 3;
